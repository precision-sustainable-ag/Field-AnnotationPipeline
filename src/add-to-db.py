import sqlite3
import json
import re
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any

log = logging.getLogger(__name__)

class FieldMetadataDB:
    def __init__(self, db_path: Path, schema_path: Path, table_name: str = "field_metadata"):
        self.db_path = db_path
        self.schema_path = schema_path
        self.table_name = table_name
        self.conn = sqlite3.connect(db_path)
        self.ensure_schema()
        self.add_missing_columns()

    def ensure_schema(self):
        with self.schema_path.open("r") as f:
            self.conn.executescript(f.read())

    def get_columns(self) -> List[str]:
        cur = self.conn.execute(f"PRAGMA table_info({self.table_name});")
        return [row[1] for row in cur.fetchall()]

    def add_missing_columns(self):
        with self.schema_path.open("r") as f:
            schema_sql = f.read()
        m = re.search(rf"CREATE TABLE IF NOT EXISTS {self.table_name}\s*\((.*?)\);", schema_sql, re.S)
        if not m:
            raise ValueError(f"Table {self.table_name} definition not found in schema.sql")
        col_defs = m.group(1)
        schema_cols = {}
        for line in col_defs.splitlines():
            line = line.strip().rstrip(",")
            if not line or line.upper().startswith("PRIMARY KEY"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                schema_cols[parts[0]] = parts[1]
        db_cols = self.get_columns()
        for col, typ in schema_cols.items():
            if col not in db_cols:
                log.info(f"Adding missing column: {col} {typ}")
                self.conn.execute(f"ALTER TABLE {self.table_name} ADD COLUMN {col} {typ}")

    def insert_row(self, row: Dict[str, Any]):
        cols = ", ".join(row.keys())
        placeholders = ", ".join(["?"] * len(row))
        sql = f"INSERT OR REPLACE INTO {self.table_name} ({cols}) VALUES ({placeholders})"
        self.conn.execute(sql, tuple(row.values()))

    def close(self):
        self.conn.close()

    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

class JSONRowParser:
    VERSION_NUMBER = 1

    def __init__(self, db_columns: List[str]):
        self.db_columns = db_columns

    def parse(self, json_path: Path) -> Optional[Dict[str, Any]]:
        """Load a JSON file and return a dict mapped to db_columns."""
        try:
            with json_path.open("r") as f:
                data = json.load(f)
            # Gather sections safely
            image_info = data.get("image_info", {}) or {}
            plant_info = data.get("plant_field_info", {}) or {}
            annotation = data.get("annotation", {}) or {}
            category = data.get("category", {}) or {}
            exif = data.get("exif_meta", {}) or {}
            bbox = annotation.get("bbox") or [None]*4
            bbox_x, bbox_y, bbox_w, bbox_h = bbox if len(bbox) == 4 else (None, None, None, None)
            # Serialize list fields for DB
            rgb = json.dumps(category.get("rgb") or [])
            alias = json.dumps(category.get("alias") or [])
            row = {
                "image_id": image_info.get("Name"),
                "extension": image_info.get("Extension"),
                "batch_id": image_info.get("Batch_id"),
                "image_url": image_info.get("ImageURL"),
                "upload_datetime_utc": image_info.get("UploadDateTimeUTC"),
                "camera_datetime": image_info.get("CameraInfo_DateTime"),
                "size_mib": image_info.get("SizeMiB"),
                "has_matching_jpg_and_raw": image_info.get("HasMatchingJpgAndRaw"),
                "image_index": int(image_info.get("ImageIndex", -1)),
                "us_state": image_info.get("UsState"),
                "note": image_info.get("Note"),
                "version": data.get("version", JSONRowParser.VERSION_NUMBER),
                # Plant fields
                "plant_type": plant_info.get("PlantType"),
                "cloud_cover": plant_info.get("CloudCover"),
                "ground_residue": plant_info.get("GroundResidue"),
                "ground_cover": plant_info.get("GroundCover"),
                "cover_crop_family": plant_info.get("CoverCropFamily"),
                "growth_stage": plant_info.get("GrowthStage"),
                "cotton_variety": plant_info.get("CottonVariety"),
                "crop_or_fallow": plant_info.get("CropOrFallow"),
                "crop_type_secondary": plant_info.get("CropTypeSecondary"),
                "species": plant_info.get("Species"),
                "height": plant_info.get("Height"),
                "size_class": plant_info.get("SizeClass"),
                "flower_fruit_or_seeds": plant_info.get("FlowerFruitOrSeeds"),
                # Annotation
                "bbox_x": bbox_x,
                "bbox_y": bbox_y,
                "bbox_w": bbox_w,
                "bbox_h": bbox_h,
                "det_pred_conf": annotation.get("det_pred_conf"),
                "has_mat_pred": annotation.get("HasMatPred"),
                "has_mat_pred_conf": annotation.get("HasMatPredConf"),
                # Category
                "class_id": category.get("class_id"),
                "usda_symbol": category.get("USDA_symbol"),
                "eppo": category.get("EPPO"),
                "taxonomic_group": category.get("group"),
                "taxonomic_class": category.get("class"),
                "taxonomic_subclass": category.get("subclass"),
                "taxonomic_order": category.get("order"),
                "taxonomic_family": category.get("family"),
                "taxonomic_genus": category.get("genus"),
                "taxonomic_species_name": category.get("species"),
                "taxonomic_subspecies": category.get("subspecies"),
                "taxonomic_authority": category.get("authority"),
                "common_name": category.get("common_name"),
                "growth_habit": category.get("growth_habit"),
                "duration": category.get("duration"),
                "multi_species_USDA_symbol": category.get("multi_species_USDA_symbol"),
                "link": category.get("link"),
                "rgb": rgb,
                "alias": alias,
                # Exif
                "make": exif.get("Make"),
                "model": exif.get("Model"),
                "software": exif.get("Software"),
                "exposure_time": exif.get("ExposureTime"),
                "fnumber": exif.get("FNumber"),
                "iso": exif.get("ISOSpeedRatings"),
                "flash": exif.get("Flash"),
                "focal_length": exif.get("FocalLength"),
                "lens_model": exif.get("LensModel")
            }
            # Return only db columns
            return {k: row.get(k) for k in self.db_columns}
        except Exception as e:
            log.warning(f"❌ Failed to parse {json_path.name}: {e}")
            return None

class BulkImporter:
    def __init__(self, json_dir: Path, db: FieldMetadataDB):
        self.json_dir = json_dir
        self.db = db
        self.parser = JSONRowParser(self.db.get_columns())

    def bulk_insert(self, rows, table="field_metadata"):
        first_row = rows[0]
        cols = list(first_row.keys())
        col_str = ", ".join(cols)
        placeholder_str = ", ".join(["?"] * len(cols))
        # Prepare SQL statement
        sql = f"INSERT OR REPLACE INTO {table} ({col_str}) VALUES ({placeholder_str})"
        data = [tuple(row[col] for col in cols) for row in rows]

        with self.db.conn:
            self.db.conn.executemany(sql, data)
        log.info(f"✅ Successfully inserted {len(rows)} rows into the database.")

    def run(self):
        log.info("Parsing JSON files...")
        rows = [
            r for p in self.json_dir.glob("*.json")
            if (r := self.parser.parse(p))
        ]
        if not rows:
            log.warning("No valid rows to insert.")
            return

        try: 
            log.info(f"Inserting {len(rows)} rows into the database...")
            self.bulk_insert(rows)
        except Exception as e:
            log.error(f"Error preparing to insert rows: {e}")
            raise
        

    def close(self):
        self.db.close()

def main(cfg):
    json_directory = Path(cfg.paths.batch_dir) / "cutouts"
    database_path = Path(cfg.paths.agir_field_db)
    schema_path = Path(cfg.paths.agir_field_db_schema)
    try:
        with FieldMetadataDB(database_path, schema_path) as db:
            importer = BulkImporter(json_directory, db)
            importer.run()
            importer.close()
            log.info("All JSON files processed successfully.")
    except Exception as e:
        log.error(f"An error occurred: {e}")
        raise
