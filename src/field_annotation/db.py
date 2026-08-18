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
    ORDER BY fl.batch_label, fs.base_name
    {limit_clause}
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
