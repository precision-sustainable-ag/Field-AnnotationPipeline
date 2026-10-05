# `field_exploration.db` Overview

Shared SQLite database used across the Field-* pipelines. Tables are owned
by different repos (noted below); each repo only reads tables it doesn't own.
Row counts are approximate and will keep growing.

| Table | Owner | ~Rows | What it's for |
|---|---|---|---|
| `locations` | Field-DataExploration | 15 | Lookup table of location codes (state/site codes) with a display name and an optional parent code, so locations can nest (e.g. a site under a state). |
| `samples` | Field-DataExploration | 22,852 | One row per unique plant/sample, keyed by `master_ref_id`. Holds the plant and field-condition metadata collected for it (species, growth stage, size, crop/cover-crop info, ground cover, etc.), plus bookkeeping fields (`username`, and several `wir*_rowkey`/`wir*_timestamp` pairs) that trace each field back to the source record it was synced from. |
| `batches` | Field-DataExploration | 770 | One row per image batch — a location + date + label grouping that images and cutouts are organized under. |
| `images` | Field-DataExploration | 244,512 | Registry of image files in blob storage, one row per blob. Most images have two rows: the raw (`extension = 'arw'`) and the camera preview JPG (`'jpg'`). Links to `samples` and `batches`, and records size, upload/EXIF timestamps and the blob URL. The color-corrected JPGs are not here; they're on NFS (see `file_locations`). |
| `raw_sample_attributes` | Field-DataExploration | 45,200 | Staging table holding the original, unprocessed attribute records pulled from source systems for each sample (one row per `source` + `master_ref_id`), including a raw JSON blob. Effectively an audit trail behind `samples`. |
| `file_locations` | Field-DataExploration | 201,370 | Inventory of files on disk storage, built by scanning NFS: one row per file path, with size and modified time. `artifact_kind = 'raw'` is a raw `.ARW` file under `raws/`; `'processed_jpg'` is a developed (color-corrected) JPG under `developed-images/`. |
| `file_status` | Field-DataExploration | 132,793 | One row per image (`base_name`), tracking where its files are: flags for whether the raw, preview and developed JPG exist in blob, NFS and Juno storage. `needs_processing = 1` means the raw was uploaded but hasn't been developed into a JPG yet; it says nothing about annotation. Also carries a copy of the sample's plant/field metadata, which is where the cutouts pipeline reads it from. |
| `planned_batches` | Field-DataExploration | 179 | Staging table for batches that have been planned (e.g. during import/scanning) but not yet fully registered in `batches`/`file_locations`. |
| `location_code_corrections` | Field-DataExploration | 4,753 | Audit log of location-code corrections — records when a sample's or batch's location code was changed, from what to what, and when. |
| `cutouts` | Field-AnnotationPipeline (this repo) | 61,647 | One row per detected+segmented cutout produced from a source image. See [`docs/cutout_table_schema.md`](cutout_table_schema.md) for full column detail. |
| `sqlite_sequence` | SQLite (internal) | 6 | Auto-managed by SQLite itself to track `AUTOINCREMENT` counters for tables that use them. Not application data. |

## Views

| View | Defined over | What it's for |
|---|---|---|
| `images_needing_processing` | `file_status` | The rows of `file_status` where `needs_processing = 1`: raws waiting to be developed into JPGs. This is not the annotation queue; that's `_NEEDS_ANNOTATION_QUERY` in [`src/field_annotation/db.py`](../src/field_annotation/db.py). Exposes `BaseName`, `MasterRefID`, `UsState`, `Species`, `BatchId`, `BatchLabel`, `SubBatchIndex`, `FilePath`. |

## Known data quirks

- Nothing is on Juno yet: every `file_locations` row has `storage_location = 'nfs'`,
  and `file_status.raw_in_juno` / `processed_jpg_in_juno` are 0 for every row.
- `file_locations.first_seen_at = '1970-01-01T00:00:00+00:00'` (190,768 rows) is a
  placeholder for files recorded before first-seen tracking started, not a real date.
- `file_locations.size_bytes` and `images.size_mib` both have a minimum of 0, so
  some zero-byte files are recorded.
- `batches` has 10 rows with a NULL `location_code`, and at least one label built
  from a missing code (`nan_2025-08-12`).
- 913 `images` rows have no `base_name` or `extension`, so they can't be linked
  to the other tables.
- `sub_batch_index` is formatted differently by table: `01`, `02_1` in
  `file_locations` / `file_status`, but `1.0`, `2.0` in `images`.
  `images.image_index` is also stored as float-style text (`0.0`, `1.0`).
- To join `images` without doubling rows, match on `base_name` and `extension`
  together; each `base_name` has at most one `arw` row and one `jpg` row.