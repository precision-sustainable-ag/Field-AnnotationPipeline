import sqlite3
import json
import logging
import exifread
from pathlib import Path
from typing import List, Optional, Dict, Any, Callable
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from tqdm import tqdm

log = logging.getLogger(__name__)

# --- Conversion helpers ---
def parse_range(val: str) -> Optional[List[float]]:
    if not isinstance(val, str):
        return None
    s = val.replace('–', '-').replace('—', '-').replace('m', '').replace('%', '')
    parts = s.split('-')
    try:
        nums = [float(p.strip()) for p in parts if p.strip()]
        return nums if len(nums) == 2 else None
    except Exception:
        return None

def convert_bool(val: Any) -> int:
    return 1 if str(val).strip().lower() == 'true' else 0

def convert_int(val: Any) -> Optional[int]:
    try:
        return int(val)
    except Exception:
        return None

def convert_float(val: Any) -> Optional[float]:
    try:
        return float(val)
    except Exception:
        return None

def convert_json(val: Any) -> str:
    return json.dumps(val) if isinstance(val, (list, dict)) else val

class FieldDataImporter:
    def __init__(self, cfg: DictConfig):
        self.db_path = Path(cfg.paths.agir_field_db)
        self.schema_path = Path(cfg.paths.agir_field_db_schema)
        self.csv_path = str(cfg.paths.persistent_data_table)
        self.csv_to_db_map = cfg.update_db.table_to_db_map
        self.db_columns = cfg.update_db.db_columns
        self.exif_meta_fields = cfg.update_db.exif_meta_fields
        self.category_field_map = cfg.update_db.category_field_map
        self.conn = None
        self.cur = None
        self.species_lookup = self.build_species_lookup(cfg.paths.field_species_info)
        self.lts_dir = Path(cfg.paths.longterm_storage)

        self.db_insert_dt = pd.Timestamp.now(tz='UTC').isoformat()

        self.converters: Dict[str, Callable[[Any], Any]] = {
            'has_matching_jpg_and_raw': convert_bool,
            'image_index': convert_int,
            'size_mib': convert_float,
            'flower_fruit_or_seeds': convert_bool,
            'height': lambda v: convert_json(parse_range(v)) or v,
            'ground_cover': lambda v: convert_json(parse_range(v)) or v
        }

    def connect(self):
        self.conn = sqlite3.connect(self.db_path)
        self.cur = self.conn.cursor()
        log.info(f"Connected to {self.db_path}")

    def create_table(self):
        with open(self.schema_path, "r") as f:
            schema_sql = f.read()
        self.cur.executescript(schema_sql)
        self.conn.commit()
        log.info("Table created from schema.")
        self.auto_migrate_columns()

    def auto_migrate_columns(self):
        """Ensure all db_columns exist in the SQLite table, adding any that are missing."""
        # Get existing columns
        self.cur.execute(f"PRAGMA table_info(field_data);")
        existing_cols = {row[1] for row in self.cur.fetchall()}

        # Loop through desired columns
        for col in self.db_columns:
            if col not in existing_cols:
                # Default to TEXT type for new columns (customize per column if needed)
                log.info(f"Adding missing column '{col}' to field_data table.")
                try:
                    self.cur.execute(f"ALTER TABLE field_data ADD COLUMN {col} TEXT;")
                except Exception as e:
                    log.warning(f"Could not add column '{col}': {e}")
        self.conn.commit()

    def _load_and_clean_csv(self) -> pd.DataFrame:
        df = pd.read_csv(self.csv_path, low_memory=False)
        
        # remove the extension column if it exists
        if 'Extension' in df.columns:
            log.info("Removing existing 'extension' column from DataFrame.")
            df.drop(columns=['Extension'], inplace=True)

        # Create new extension column by taking the extension in Name column
        df['Extension'] = df['Name'].apply(lambda x: Path(x).suffix if pd.notna(x) else None)
        
        # Only for testing. Remove this afterward
        # df = df[(df['Extension'].str.lower() == '.jpg')].sample(10000)
        
        return df

    def _build_row(self, row: pd.Series) -> tuple:
        vals = []
        cat_info = self.get_category_info(row.get('Species', ''))
        for col in self.db_columns:
            # Priority: category_field_map > csv_to_db_map > None
            if col in self.category_field_map.values() and cat_info:
                # Find which JSON field maps to this DB col
                for json_field, db_field in self.category_field_map.items():
                    if db_field == col:
                        value = cat_info.get(json_field)
                        if isinstance(value, (list, dict)):
                            value = json.dumps(value)
                        vals.append(value)
                        break
                else:
                    vals.append(None)
            elif col in self.csv_to_db_map.values():
                # Find which CSV col maps to this DB col
                for csv_col, db_col in self.csv_to_db_map.items():
                    if db_col == col:
                        val = row[csv_col] if csv_col in row else None
                        # Optionally, add your type conversions here
                        converter = self.converters.get(db_col, lambda x: x)
                        vals.append(converter(val))
                        break
                else:
                    vals.append(None)
            
            elif col == 'is_preprocessed':
                stem = row.get('Stem', None)
                batch_id = row.get('BatchID', None)
                val = self.lts_dir / "field-batches" / str(batch_id) / "developed-images" / f"{stem}.jpg"
                vals.append(int(1) if val.exists() else int(0))

            elif col == "developed_image_path":
                stem = row.get('Stem', None)
                batch_id = row.get('BatchID', None)
                if stem and batch_id:
                    developed_image_path = self.lts_dir / "field-batches" / str(batch_id) / "developed-images" / f"{stem}.jpg"
                    if developed_image_path.exists():
                        short_developed_path = Path("field-batches") / str(batch_id) / "developed-images" / f"{stem}.jpg"
                        vals.append(str(short_developed_path))
                    else:
                        vals.append(None)
                else:
                    vals.append(None)

            elif col == "raw_image_path":
                stem = row.get('Stem', None)
                batch_id = row.get('BatchID', None)
                sub_batch_index = row.get('SubBatchIndex', None)
                if stem and batch_id and pd.notna(sub_batch_index):
                    raw_path = self.lts_dir / "field-batches" / str(batch_id) / "raws" / f"0{int(sub_batch_index)}" / f"{stem}.ARW"
                    if raw_path.exists():
                        short_raw_path = Path("field-batches") / str(batch_id) / "raws" / f"0{int(sub_batch_index)}" / f"{stem}.ARW"
                        vals.append(str(short_raw_path))
                    else:
                        vals.append(None)
                else:
                    vals.append(None)
            
            elif col == "db_insert_datetime":
                # Use current UTC time for db_insert_datetime
                vals.append(str(self.db_insert_dt))


            elif col == "exif_meta":
                stem = row.get('Stem', None)
                batch_id = row.get('BatchID', None)
                if stem and batch_id:
                    developed_image_path = self.lts_dir / "field-batches" / str(batch_id) / "developed-images" / f"{stem}.jpg"
                    if developed_image_path.exists():
                        # extract exif data using exifread
                        with open(developed_image_path, "rb") as f:
                            # process_file returns a dict of tags
                            tags = exifread.process_file(f, details=True)
                        # Convert tags to JSON string
                        exif_data_json_str = json.dumps({tag: str(value) for tag, value in tags.items() if tag in self.exif_meta_fields})
                        vals.append(exif_data_json_str)
                    else:
                        vals.append(None)

            else:
                vals.append(None)
        
        return tuple(vals)

    def _validate_rows(self, data: List[tuple]):
        if not data:
            log.warning("No data to insert!")
            return
        expected = len(self.db_columns)
        for i, r in enumerate(data):
            if len(r) != expected:
                log.error(f"Row {i} has {len(r)} values, expected {expected}. Row data length: {len(r)}")
                raise ValueError(
                    f"Row {i} has {len(r)} values, but {expected} columns expected"
                )

    def _insert_data(self, data: List[tuple]):
        insert_sql = f"""
            INSERT OR IGNORE INTO field_data ({', '.join(self.db_columns)})
            VALUES ({', '.join(['?'] * len(self.db_columns))})
        """
        log.info(f"Prepared to insert {len(data)} rows into field_data table.")
        self.cur.executemany(insert_sql, data)
        self.conn.commit()
        log.info(f"Imported {len(data)} rows.")

    def build_species_lookup(self, species_info_path: str) -> Dict[str, Any]:
        with open(species_info_path, 'r') as f:
            data = json.load(f)
        species_info = data['species']
        lookup = {}
        for sp_dict in species_info.values():
            if sp_dict.get('common_name'):
                lookup[sp_dict['common_name'].strip().lower()] = sp_dict
            for alias in sp_dict.get('alias', []):
                lookup[alias.strip().lower()] = sp_dict
        return lookup

    def get_category_info(self, species_val) -> Optional[Dict[str, Any]]:
        key = str(species_val).strip().lower()
        return self.species_lookup.get(key)

    def close(self):
        if self.conn:
            self.conn.close()
            log.info("Database connection closed.")

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def run(self):
        df = self._load_and_clean_csv()
        data = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
            data.append(self._build_row(row))

        self._validate_rows(data)
        self._insert_data(data)

def main(cfg):
    try:
        with FieldDataImporter(cfg) as importer:
            importer.create_table()
            importer.run()
    except Exception as e:
        log.exception(f"An error occurred: {e}")
        raise