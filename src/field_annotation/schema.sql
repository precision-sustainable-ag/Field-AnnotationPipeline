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
