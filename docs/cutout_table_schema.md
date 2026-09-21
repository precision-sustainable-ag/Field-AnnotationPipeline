# `cutouts` Table Schema

Owned by this pipeline inside the shared `field_exploration.db` SQLite database.
The database also holds several tables owned by Field-DataExploration
(`samples`, `batches`, `images`, `file_status`, `file_locations`, and others) —
see [`docs/db_overview.md`](db_overview.md) for the full list. Defined in
[`src/field_annotation/schema.sql`](../src/field_annotation/schema.sql).

One row per cutout produced from a source image (`base_name` +
`cutout_index`, unique together — an image can yield multiple cutouts).

| Column | Type | Notes |
|---|---|---|
| `id` | INTEGER | Primary key, autoincrement |
| `base_name` | TEXT | Source image identifier, not null |
| `cutout_index` | INTEGER | Index of this cutout within the image, default `0` |
| `master_ref_id` | TEXT | |
| `batch_label` | TEXT | |
| `location_code` | TEXT | |
| `plant_type` | TEXT | |
| `species` | TEXT | |
| `class_id` | INTEGER | |
| `status` | TEXT | Not null. One of `detected_segmented`, `no_detection`, `segmented`, `error` |
| `det_pred_conf` | REAL | Set only when `run_detection` produced this row |
| `detection_bbox_x/y/w/h` | INTEGER | Raw YOLO detection bbox, pre-padding. Set only when `run_detection` was true |
| `final_bbox_x/y/w/h` | INTEGER | Region actually used for crop/mask/cutout (padded detection bbox or full frame, tightened to mask unless disabled). Always set |
| `crop_path` | TEXT | |
| `mask_path` | TEXT | |
| `cutout_path` | TEXT | |
| `metadata_json_path` | TEXT | |
| `error_message` | TEXT | |
| `detection_model` | TEXT | |
| `segmentation_model` | TEXT | |
| `processed_at` | TEXT | Not null |
| `height` | TEXT | Plant height bucket, copied from the sample record. Added after the table's initial creation (appears after `processed_at` in `.schema` — a later `ALTER TABLE ADD COLUMN`) |
| `size_class` | TEXT | Plant size class, copied from the sample record. Added later, same as above |
| `growth_stage` | TEXT | Copied from the sample record. Added later |
| `cotton_variety` | TEXT | Copied from the sample record. Added later |
| `crop_or_fallow` | TEXT | Copied from the sample record. Added later |
| `crop_type_secondary` | TEXT | Copied from the sample record. Added later |
| `cover_crop_family` | TEXT | Copied from the sample record. Added later |
| `flower_fruit_or_seeds` | TEXT | Copied from the sample record. Added later |
| `cloud_cover` | TEXT | Copied from the sample record. Added later |
| `ground_residue` | TEXT | Copied from the sample record. Added later |
| `ground_cover` | TEXT | Copied from the sample record. Added later |

**Status values**

- `detected_segmented` — detection ran, found something, and segmentation succeeded.
- `segmented` — detection was skipped (full-frame); segmentation only.
- `no_detection` — detection ran but found nothing.
- `error` — see `error_message`.

**Constraints & indexes**

- `UNIQUE(base_name, cutout_index)`
- Indexes on `base_name`, `batch_label`, `status`

Column list and upsert logic live in
[`src/field_annotation/db.py`](../src/field_annotation/db.py) (`CUTOUT_COLUMNS`, `CutoutsDb.upsert_cutout`).