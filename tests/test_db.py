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
            height TEXT,
            size_class TEXT,
            growth_stage TEXT,
            cotton_variety TEXT,
            crop_or_fallow TEXT,
            crop_type_secondary TEXT,
            cover_crop_family TEXT,
            flower_fruit_or_seeds TEXT,
            cloud_cover TEXT,
            ground_residue TEXT,
            ground_cover TEXT,
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



@pytest.fixture
def developed_db_path(db_path):
    """db_path plus the extra columns and the `images` table that
    refresh_developed_images reads. MDB001's developed JPG is a zero-byte file,
    and only ALB001 has a raw (and a preview) in `images`."""
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        ALTER TABLE file_status ADD COLUMN batch_id INTEGER;
        ALTER TABLE file_status ADD COLUMN sub_batch_index TEXT;
        ALTER TABLE file_locations ADD COLUMN size_bytes INTEGER;
        ALTER TABLE file_locations ADD COLUMN mtime_utc TEXT;
        UPDATE file_locations
           SET path = '/mnt/lts/field-batches/' || batch_label || '/developed-images/' || base_name || '.jpg',
               size_bytes = 1000,
               mtime_utc = '2026-05-11 05:01:46'
         WHERE artifact_kind = 'processed_jpg';
        UPDATE file_locations SET size_bytes = 0 WHERE base_name = 'MDB001';
        CREATE TABLE images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            blob_name TEXT,
            base_name TEXT,
            extension TEXT,
            image_url TEXT,
            exif_datetime TEXT
        );
        INSERT INTO images (blob_name, base_name, extension, image_url, exif_datetime) VALUES
            ('ALB001.ARW', 'ALB001', 'arw', 'https://example.blob/ALB001.ARW', '2024-08-19 10:00:00'),
            ('ALB001.JPG', 'ALB001', 'jpg', 'https://example.blob/ALB001.JPG', '2024-08-19 10:00:00');
        """
    )
    conn.commit()
    conn.close()
    return db_path


def test_refresh_developed_images_one_row_per_developed_jpg(developed_db_path):
    db = CutoutsDb(developed_db_path)
    conn = db.connect()
    counts = db.refresh_developed_images(conn)

    rows = {row["base_name"]: row for row in conn.execute("SELECT * FROM developed_images")}
    # ALB003 has no developed JPG yet (processed_jpg_in_nfs = 0), so it's left out.
    assert set(rows) == {"ALB001", "ALB002", "MDB001"}
    assert counts == {"total": 3, "pending": 3, "zero_byte": 1}
    assert rows["ALB001"]["jpg_path"] == "AL_2024-08-19/developed-images/ALB001.jpg"
    assert rows["ALB001"]["species"] == "Palmer amaranth"


def test_refresh_developed_images_links_raw_image_only(developed_db_path):
    db = CutoutsDb(developed_db_path)
    conn = db.connect()
    db.refresh_developed_images(conn)

    rows = {row["base_name"]: row for row in conn.execute("SELECT * FROM developed_images")}
    # Linked to the raw, not the preview JPG -- and still one row, not two.
    assert rows["ALB001"]["raw_blob_name"] == "ALB001.ARW"
    # No raw in `images` -> the image still gets a row, with raw_* left NULL.
    assert rows["ALB002"]["raw_image_id"] is None


def test_refresh_developed_images_flags_cutouts_and_is_repeatable(developed_db_path):
    db = CutoutsDb(developed_db_path)
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

    db.refresh_developed_images(conn)
    counts = db.refresh_developed_images(conn)  # a second rebuild must not duplicate rows

    flags = {row["base_name"]: row["has_cutout"] for row in conn.execute("SELECT * FROM developed_images")}
    assert flags == {"ALB001": 1, "ALB002": 0, "MDB001": 0}
    assert counts["total"] == 3
    assert counts["pending"] == 2