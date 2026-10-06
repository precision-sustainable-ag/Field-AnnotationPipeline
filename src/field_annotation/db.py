from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

log = logging.getLogger(__name__)

_SCHEMA_PATH = Path(__file__).parent / "schema.sql"

# Every column in `cutouts` except the autoincrement `id`. Kept as one tuple
# so upsert_cutout builds its INSERT/ON CONFLICT lists from a single source
# of truth instead of duplicating the column list -- also imported by
# pipeline.py to make sure every metadata JSON has every field (null where
# not applicable), not just whichever ones happened to be set.
CUTOUT_COLUMNS = (
    "base_name",
    "cutout_index",
    "master_ref_id",
    "batch_label",
    "location_code",
    "plant_type",
    "species",
    "height",
    "size_class",
    "growth_stage",
    "cotton_variety",
    "crop_or_fallow",
    "crop_type_secondary",
    "cover_crop_family",
    "flower_fruit_or_seeds",
    "cloud_cover",
    "ground_residue",
    "ground_cover",
    "class_id",
    "status",
    "det_pred_conf",
    "detection_bbox_x",
    "detection_bbox_y",
    "detection_bbox_w",
    "detection_bbox_h",
    "final_bbox_x",
    "final_bbox_y",
    "final_bbox_w",
    "final_bbox_h",
    "crop_path",
    "mask_path",
    "cutout_path",
    "metadata_json_path",
    "error_message",
    "detection_model",
    "segmentation_model",
    "processed_at",
)

# CREATE TABLE IF NOT EXISTS in schema.sql only creates a table on a brand-new
# DB -- it's a no-op for a column added to a table that already exists on the
# live, shared field_exploration.db. Mirrors Field-DataExploration's own
# connection.py migration pattern for the same reason.
_COLUMN_MIGRATIONS = (
    "detection_bbox_x INTEGER",
    "detection_bbox_y INTEGER",
    "detection_bbox_w INTEGER",
    "detection_bbox_h INTEGER",
    "final_bbox_x INTEGER",
    "final_bbox_y INTEGER",
    "final_bbox_w INTEGER",
    "final_bbox_h INTEGER",
    "height TEXT",
    "size_class TEXT",
    "growth_stage TEXT",
    "cotton_variety TEXT",
    "crop_or_fallow TEXT",
    "crop_type_secondary TEXT",
    "cover_crop_family TEXT",
    "flower_fruit_or_seeds TEXT",
    "cloud_cover TEXT",
    "ground_residue TEXT",
    "ground_cover TEXT",
)


# Renamed to detection_bbox_*/final_bbox_* above -- dropped rather than left
# as dead columns, since this table (unlike file_status/file_locations) is
# owned entirely by this pipeline and DROP COLUMN has been supported since
# SQLite 3.35 (2021).
_COLUMNS_TO_DROP = ("bbox_x", "bbox_y", "bbox_w", "bbox_h")


def _apply_column_migrations(conn: sqlite3.Connection) -> None:
    existing = {row[1] for row in conn.execute("PRAGMA table_info(cutouts)")}
    for column in _COLUMN_MIGRATIONS:
        name = column.split()[0]
        if name not in existing:
            conn.execute(f"ALTER TABLE cutouts ADD COLUMN {column}")
    for name in _COLUMNS_TO_DROP:
        if name in existing:
            conn.execute(f"ALTER TABLE cutouts DROP COLUMN {name}")

# The "needs annotation" query joins on file_locations.batch_label (the same row
# that supplies jpg_path) rather than file_status.batch_label, and filters on
# file_status.processed_jpg_in_nfs=1 (the developed JPG exists on NFS) -- NOT
# file_status.needs_processing, which actually means "raw uploaded but not yet
# developed to JPG" and has nothing to do with annotation readiness.
_NEEDS_ANNOTATION_QUERY = """
    SELECT
        fs.base_name, fs.master_ref_id, fs.location_code, fs.plant_type, fs.species,
        fs.height, fs.size_class, fs.growth_stage, fs.cotton_variety, fs.crop_or_fallow,
        fs.crop_type_secondary, fs.cover_crop_family, fs.flower_fruit_or_seeds,
        fs.cloud_cover, fs.ground_residue, fs.ground_cover,
        fl.batch_label, fl.path AS jpg_path
    FROM file_status fs
    JOIN file_locations fl
      ON fl.base_name = fs.base_name
     AND fl.artifact_kind = 'processed_jpg'
     AND fl.storage_location = 'nfs'
    LEFT JOIN cutouts c
      ON c.base_name = fs.base_name
     AND c.cutout_index = 0
    WHERE fs.processed_jpg_in_nfs = 1
      AND c.id IS NULL
      {batch_filter}
      {plant_type_filter}
    ORDER BY RANDOM()
    {limit_clause}
"""


# Fills developed_images: one row per developed JPG on NFS, joined to its
# file_status row (phenotype), its raw `images` row (identity) and its cutout,
# if any. Run right after `DELETE FROM developed_images` in the same
# transaction -- see refresh_developed_images and docs/developed_images_design.md.
# jpg_path is stored relative to the LTS field-batches/ root, like the path
# columns in cutouts.
_REFRESH_DEVELOPED_IMAGES_SQL = """
    INSERT INTO developed_images (
        base_name, master_ref_id, batch_id, batch_label, location_code, sub_batch_index,
        raw_image_id, raw_blob_name, raw_image_url, exif_datetime,
        jpg_path, jpg_size_bytes, jpg_mtime_utc,
        plant_type, species, height, size_class, growth_stage, cotton_variety,
        crop_or_fallow, crop_type_secondary, cover_crop_family, flower_fruit_or_seeds,
        cloud_cover, ground_residue, ground_cover,
        has_cutout, refreshed_at
    )
    SELECT
        fs.base_name, fs.master_ref_id, fs.batch_id, fl.batch_label, fs.location_code, fs.sub_batch_index,
        i.id, i.blob_name, i.image_url, i.exif_datetime,
        CASE
            WHEN instr(fl.path, '/field-batches/') > 0
            THEN substr(fl.path, instr(fl.path, '/field-batches/') + length('/field-batches/'))
            ELSE fl.path
        END,
        fl.size_bytes, fl.mtime_utc,
        fs.plant_type, fs.species, fs.height, fs.size_class, fs.growth_stage, fs.cotton_variety,
        fs.crop_or_fallow, fs.crop_type_secondary, fs.cover_crop_family, fs.flower_fruit_or_seeds,
        fs.cloud_cover, fs.ground_residue, fs.ground_cover,
        CASE WHEN c.id IS NULL THEN 0 ELSE 1 END,
        strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
    FROM file_status fs
    JOIN file_locations fl
      ON fl.base_name = fs.base_name
     AND fl.artifact_kind = 'processed_jpg'
     AND fl.storage_location = 'nfs'
    LEFT JOIN temp.raw_images i
      ON i.base_name = fs.base_name
    LEFT JOIN cutouts c
      ON c.base_name = fs.base_name
     AND c.cutout_index = 0
    WHERE fs.processed_jpg_in_nfs = 1
"""

class CutoutsDb:
    """Access layer for the `cutouts` table this pipeline owns inside the
    shared field_exploration.db (maintained otherwise by Field-DataExploration).
    """

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path).expanduser()

    def connect(self) -> sqlite3.Connection:
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database does not exist: {self.db_path}")
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.executescript(_SCHEMA_PATH.read_text())
        _apply_column_migrations(conn)
        return conn

    def fetch_images_needing_annotation(
        self,
        conn: sqlite3.Connection,
        batch_label: str | None = None,
        plant_type: str | None = None,
        limit: int | None = None,
    ) -> list[sqlite3.Row]:
        batch_filter = ""
        plant_type_filter = ""
        params: dict[str, object] = {}
        if batch_label:
            batch_filter = "AND fl.batch_label = :batch_label"
            params["batch_label"] = batch_label
        if plant_type:
            plant_type_filter = "AND fs.plant_type = :plant_type"
            params["plant_type"] = plant_type

        limit_clause = ""
        if limit:
            limit_clause = "LIMIT :limit"
            params["limit"] = limit

        query = _NEEDS_ANNOTATION_QUERY.format(
            batch_filter=batch_filter, plant_type_filter=plant_type_filter, limit_clause=limit_clause
        )
        return conn.execute(query, params).fetchall()

    def batch_summary_needing_annotation(
        self, conn: sqlite3.Connection, plant_type: str | None = None
    ) -> list[tuple[str, int]]:
        plant_type_filter = ""
        params: dict[str, object] = {}
        if plant_type:
            plant_type_filter = "AND fs.plant_type = :plant_type"
            params["plant_type"] = plant_type

        query = _NEEDS_ANNOTATION_QUERY.format(batch_filter="", plant_type_filter=plant_type_filter, limit_clause="")
        rows = conn.execute(
            f"SELECT batch_label, COUNT(*) AS n FROM ({query}) GROUP BY batch_label ORDER BY batch_label", params
        ).fetchall()
        return [(row["batch_label"], row["n"]) for row in rows]

    def upsert_cutout(self, conn: sqlite3.Connection, row: dict) -> None:
        columns = CUTOUT_COLUMNS
        placeholders = ", ".join(f":{col}" for col in columns)
        update_clause = ", ".join(f"{col}=excluded.{col}" for col in columns if col not in ("base_name", "cutout_index"))
        values = {col: row.get(col) for col in columns}
        conn.execute(
            f"""
            INSERT INTO cutouts ({", ".join(columns)})
            VALUES ({placeholders})
            ON CONFLICT(base_name, cutout_index) DO UPDATE SET {update_clause}
            """,
            values,
        )
    
    
    def refresh_developed_images(self, conn: sqlite3.Connection) -> dict[str, int]:
        """Rebuild developed_images from scratch. The DELETE and INSERT run in
        one transaction (`with conn:` commits at the end, or rolls back if
        anything fails), so readers see either the old table or the new one,
        never a half-written one. Returns row counts for a quick sanity check.
        """
        # `images` has no index on base_name (and isn't ours to add one to), so
        # joining it directly scans all ~245k rows for every developed JPG.
        # Copy just the raw rows into an indexed TEMP table first -- it lives
        # only in this connection and never touches the shared DB's schema.
        conn.executescript(
            """
            DROP TABLE IF EXISTS temp.raw_images;
            CREATE TEMP TABLE raw_images AS
                SELECT id, base_name, blob_name, image_url, exif_datetime
                FROM images
                WHERE extension = 'arw';
            CREATE INDEX temp.idx_raw_images_base_name ON raw_images(base_name);
            """
        )
        with conn:
            conn.execute("DELETE FROM developed_images")
            conn.execute(_REFRESH_DEVELOPED_IMAGES_SQL)
        row = conn.execute(
            """
            SELECT
                COUNT(*) AS total,
                COALESCE(SUM(has_cutout = 0), 0) AS pending,
                COALESCE(SUM(jpg_size_bytes = 0), 0) AS zero_byte
            FROM developed_images
            """
        ).fetchone()
        return {"total": row["total"], "pending": row["pending"], "zero_byte": row["zero_byte"]}
