import sqlite3

import pytest

from field_annotation.db import CutoutsDb


@pytest.fixture
def db_path(tmp_path):
    """A throwaway sqlite file with minimal file_status/file_locations tables,
    standing in for the shared field_exploration.db this pipeline reads from
    but does not own (that DB is populated by Field-DataExploration)."""
    path = tmp_path / "field_exploration.db"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE file_status (
            base_name TEXT PRIMARY KEY,
            master_ref_id TEXT,
            location_code TEXT,
            plant_type TEXT,
            species TEXT,
            processed_jpg_in_nfs BOOLEAN,
            needs_processing BOOLEAN,
            batch_label TEXT
        );
        CREATE TABLE file_locations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            base_name TEXT,
            extension TEXT,
            artifact_kind TEXT,
            storage_location TEXT,
            path TEXT,
            batch_label TEXT
        );
        """
    )
    conn.executemany(
        "INSERT INTO file_status (base_name, master_ref_id, location_code, plant_type, species, "
        "processed_jpg_in_nfs, needs_processing, batch_label) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [
            ("ALB001", "ref-1", "AL", "WEEDS", "Palmer amaranth", 1, 0, "AL_2024-08-19"),
            ("ALB002", "ref-2", "AL", "WEEDS", "prickly sida", 1, 0, "AL_2024-08-19"),
            # needs_processing=1 (raw not yet developed) must NOT show up as pending annotation.
            ("ALB003", "ref-3", "AL", "WEEDS", "waterhemp", 0, 1, "AL_2024-08-19"),
            ("MDB001", "ref-4", "MD", "CASHCROPS", "cotton", 1, 0, "MD_2024-06-25"),
        ],
    )
    conn.executemany(
        "INSERT INTO file_locations (base_name, extension, artifact_kind, storage_location, path, batch_label) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        [
            ("ALB001", "jpg", "processed_jpg", "nfs", "/lts/AL_2024-08-19/developed-images/ALB001.jpg", "AL_2024-08-19"),
            ("ALB002", "jpg", "processed_jpg", "nfs", "/lts/AL_2024-08-19/developed-images/ALB002.jpg", "AL_2024-08-19"),
            ("ALB001", "arw", "raw", "nfs", "/lts/AL_2024-08-19/raws/ALB001.ARW", "AL_2024-08-19"),
            ("MDB001", "jpg", "processed_jpg", "nfs", "/lts/MD_2024-06-25/developed-images/MDB001.jpg", "MD_2024-06-25"),
        ],
    )
    conn.commit()
    conn.close()
    return path


def test_fetch_images_needing_annotation_excludes_undeveloped(db_path):
    db = CutoutsDb(db_path)
    conn = db.connect()
    rows = db.fetch_images_needing_annotation(conn)
    base_names = {row["base_name"] for row in rows}
    assert base_names == {"ALB001", "ALB002", "MDB001"}


def test_fetch_images_needing_annotation_filters_by_plant_type(db_path):
    db = CutoutsDb(db_path)
    conn = db.connect()

    weeds = db.fetch_images_needing_annotation(conn, plant_type="WEEDS")
    assert {row["base_name"] for row in weeds} == {"ALB001", "ALB002"}

    cashcrops = db.fetch_images_needing_annotation(conn, plant_type="CASHCROPS")
    assert {row["base_name"] for row in cashcrops} == {"MDB001"}

    covercrops = db.fetch_images_needing_annotation(conn, plant_type="COVERCROPS")
    assert covercrops == []


def test_fetch_images_needing_annotation_uses_jpg_path_not_raw(db_path):
    db = CutoutsDb(db_path)
    conn = db.connect()
    rows = db.fetch_images_needing_annotation(conn)
    for row in rows:
        assert row["jpg_path"].endswith(".jpg")


def test_upsert_cutout_excludes_from_future_scans(db_path):
    db = CutoutsDb(db_path)
    conn = db.connect()

    db.upsert_cutout(
        conn,
        {
            "base_name": "ALB001",
            "cutout_index": 0,
            "batch_label": "AL_2024-08-19",
            "status": "detected_segmented",
            "processed_at": "2026-08-17T00:00:00.000000Z",
        },
    )
    conn.commit()

    rows = db.fetch_images_needing_annotation(conn)
    base_names = {row["base_name"] for row in rows}
    assert base_names == {"ALB002", "MDB001"}


def test_upsert_cutout_is_idempotent(db_path):
    db = CutoutsDb(db_path)
    conn = db.connect()

    row = {
        "base_name": "ALB001",
        "cutout_index": 0,
        "batch_label": "AL_2024-08-19",
        "status": "detected_segmented",
        "processed_at": "2026-08-17T00:00:00.000000Z",
    }
    db.upsert_cutout(conn, row)
    db.upsert_cutout(conn, {**row, "status": "error", "error_message": "retry after fix"})
    conn.commit()

    result = conn.execute("SELECT status, error_message FROM cutouts WHERE base_name = 'ALB001'").fetchall()
    assert len(result) == 1
    assert result[0]["status"] == "error"
    assert result[0]["error_message"] == "retry after fix"


def test_batch_summary_needing_annotation(db_path):
    db = CutoutsDb(db_path)
    conn = db.connect()
    summary = db.batch_summary_needing_annotation(conn)
    assert summary == [("AL_2024-08-19", 2), ("MD_2024-06-25", 1)]


def test_batch_summary_needing_annotation_filters_by_plant_type(db_path):
    db = CutoutsDb(db_path)
    conn = db.connect()
    summary = db.batch_summary_needing_annotation(conn, plant_type="CASHCROPS")
    assert summary == [("MD_2024-06-25", 1)]
