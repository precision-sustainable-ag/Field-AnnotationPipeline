-- Owned by this pipeline, not by Field-DataExploration (which owns file_status /
-- file_locations / samples / batches in this same shared DB). Created idempotently
-- on every connect() via `CREATE TABLE IF NOT EXISTS` -- never touches the tables
-- above, only reads them.
CREATE TABLE IF NOT EXISTS cutouts (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    base_name           TEXT NOT NULL,
    cutout_index        INTEGER NOT NULL DEFAULT 0,
    master_ref_id       TEXT,
    batch_label         TEXT,
    location_code       TEXT,
    plant_type          TEXT,
    species             TEXT,
    height              TEXT,
    size_class          TEXT,
    growth_stage        TEXT,
    cotton_variety      TEXT,
    crop_or_fallow      TEXT,
    crop_type_secondary TEXT,
    cover_crop_family   TEXT,
    flower_fruit_or_seeds TEXT,
    cloud_cover         TEXT,
    ground_residue      TEXT,
    ground_cover        TEXT,
    class_id            INTEGER,
    status              TEXT NOT NULL, -- 'detected_segmented' | 'no_detection' | 'segmented' | 'error'
                                        -- 'detected_segmented' = run_detection was true, found
                                        -- something, and segmentation succeeded (both steps ran).
                                        -- 'segmented' = run_detection was false (full-frame,
                                        -- no detection attempted, segmentation only).
                                        -- 'no_detection' = run_detection was true but found nothing.
    det_pred_conf       REAL, -- set only when run_detection produced this row
    -- Raw YOLO output, pre-padding -- set only when run_detection was true
    -- (there's no such thing as a "detection bbox" for a full-frame segment).
    detection_bbox_x    INTEGER,
    detection_bbox_y    INTEGER,
    detection_bbox_w    INTEGER,
    detection_bbox_h    INTEGER,
    -- The region actually used for crop/mask/cutout: padded detection bbox
    -- or the full frame, then tightened to the segmented mask unless
    -- inference.tighten_to_mask is false. Always set.
    final_bbox_x        INTEGER,
    final_bbox_y        INTEGER,
    final_bbox_w        INTEGER,
    final_bbox_h        INTEGER,
    crop_path           TEXT,
    mask_path           TEXT,
    cutout_path         TEXT,
    metadata_json_path  TEXT,
    error_message       TEXT,
    detection_model     TEXT,
    segmentation_model  TEXT,
    processed_at        TEXT NOT NULL,
    UNIQUE(base_name, cutout_index)
);

CREATE INDEX IF NOT EXISTS idx_cutouts_base_name   ON cutouts(base_name);
CREATE INDEX IF NOT EXISTS idx_cutouts_batch_label ON cutouts(batch_label);
CREATE INDEX IF NOT EXISTS idx_cutouts_status      ON cutouts(status);


-- One row per developed (color-corrected) JPG on NFS, with the image's
-- identity, phenotype and path in one place. Owned by this pipeline, like
-- `cutouts`, and rebuilt in full by `field-annotation refresh-developed-images`
-- (CutoutsDb.refresh_developed_images). See docs/developed_images_design.md.
CREATE TABLE IF NOT EXISTS developed_images (
    base_name             TEXT PRIMARY KEY,
    master_ref_id         TEXT,
    batch_id              INTEGER,
    batch_label           TEXT NOT NULL,
    location_code         TEXT,
    sub_batch_index       TEXT,
    raw_image_id          INTEGER,
    raw_blob_name         TEXT,
    raw_image_url         TEXT,
    exif_datetime         TEXT,
    jpg_path              TEXT NOT NULL,
    jpg_size_bytes        INTEGER NOT NULL,
    jpg_mtime_utc         TEXT NOT NULL,
    plant_type            TEXT,
    species               TEXT,
    height                TEXT,
    size_class            TEXT,
    growth_stage          TEXT,
    cotton_variety        TEXT,
    crop_or_fallow        TEXT,
    crop_type_secondary   TEXT,
    cover_crop_family     TEXT,
    flower_fruit_or_seeds TEXT,
    cloud_cover           TEXT,
    ground_residue        TEXT,
    ground_cover          TEXT,
    has_cutout            INTEGER NOT NULL, -- 1 if a cutouts row exists (cutout_index = 0), else 0
    refreshed_at          TEXT NOT NULL     -- ISO 8601 UTC time of the last rebuild
);

CREATE INDEX IF NOT EXISTS idx_developed_images_batch_label   ON developed_images(batch_label);
CREATE INDEX IF NOT EXISTS idx_developed_images_has_cutout    ON developed_images(has_cutout);
CREATE INDEX IF NOT EXISTS idx_developed_images_species       ON developed_images(species);
CREATE INDEX IF NOT EXISTS idx_developed_images_master_ref_id ON developed_images(master_ref_id);