# `field_exploration.db` Overview

Shared SQLite database used across the Field-* pipelines. Tables are owned
by different repos (noted below); each repo only reads tables it doesn't own.
Row counts are a snapshot and will keep growing.

| Table | Owner | ~Rows | What it's for |
|---|---|---|---|
| `locations` | Field-DataExploration | 15 | Lookup table of location codes (state/site codes) with a display name and an optional parent code, so locations can nest (e.g. a site under a state). |
| `samples` | Field-DataExploration | 22,852 | One row per unique plant/sample, keyed by `master_ref_id`. Holds the plant and field-condition metadata collected for it (species, growth stage, size, crop/cover-crop info, ground cover, etc.), plus bookkeeping fields (`username`, and several `wir*_rowkey`/`wir*_timestamp` pairs) that trace each field back to the source record it was synced from. |
| `batches` | Field-DataExploration | 770 | One row per image batch — a location + date + label grouping that images and cutouts are organized under. |
| `images` | Field-DataExploration | 244,512 | Registry of individual image files (raw or processed), one row per blob. Links to `samples` and `batches`, and records size, upload/EXIF timestamps, and whether a matching raw+JPG pair exists. |
| `raw_sample_attributes` | Field-DataExploration | 45,200 | Staging table holding the original, unprocessed attribute records pulled from source systems for each sample (one row per `source` + `master_ref_id`), including a raw JSON blob. Effectively an audit trail behind `samples`. |
| `file_locations` | Field-DataExploration | 194,752 | Inventory of where each raw/processed file physically lives on disk (`nfs` or `juno` storage), built by scanning storage — one row per storage location + path, with size and modified time. |
| `file_status` | Field-DataExploration | 132,793 | One row per image (`base_name`), tracking its progress through the pipeline — flags for whether the raw/preview/processed files exist in blob, NFS, and Juno storage, and whether it still `needs_processing`. Also carries a copy of the sample's plant/field metadata for quick filtering. |
| `planned_batches` | Field-DataExploration | 179 | Staging table for batches that have been planned (e.g. during import/scanning) but not yet fully registered in `batches`/`file_locations`. |
| `location_code_corrections` | Field-DataExploration | 4,753 | Audit log of location-code corrections — records when a sample's or batch's location code was changed, from what to what, and when. |
| `cutouts` | Field-AnnotationPipeline (this repo) | 61,647 | One row per detected+segmented cutout produced from a source image. See [`docs/cutout_table_schema.md`](cutout_table_schema.md) for full column detail. |
| `sqlite_sequence` | SQLite (internal) | 6 | Auto-managed by SQLite itself to track `AUTOINCREMENT` counters for tables that use them. Not application data. |

**Views**

| View | Defined over | What it's for |
|---|---|---|
| `images_needing_processing` | `file_status` | Convenience query: the subset of `file_status` rows where `needs_processing = 1`, i.e. the pipeline's work queue. Exposes `BaseName`, `MasterRefID`, `UsState`, `BatchId`, `BatchLabel`, `SubBatchIndex`, `FilePath`. |