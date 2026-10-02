# Design: `developed_images` table

## Purpose

One row per developed (color-corrected) JPG on NFS, combining the image's
identity, its phenotype description and its file path in one place. Today
that information is spread over `file_status`, `file_locations`, `images` and
`cutouts`, and the pipeline rebuilds the join every time it looks for work
(`_NEEDS_ANNOTATION_QUERY` in
[`src/field_annotation/db.py`](../src/field_annotation/db.py)).

The table answers questions like "which developed images don't have cutouts
yet?" with a single `WHERE has_cutout = 0`.

Snapshot (October 2026): 90,608 developed JPGs on NFS; 61,647 have cutouts;
28,961 don't.

## Proposed name

`developed_images`. It matches the `developed-images/` folder the JPGs live in,
and it doesn't clash with the existing `images` table, which tracks raw and
preview files in blob storage.

Alternative considered: `annotation_queue`, if the table should hold only the
images still waiting for cutouts (see Open questions).

## Schema

Owned by Field-AnnotationPipeline, like `cutouts`.

| Column | Type | Source | NULL | Notes |
|---|---|---|---|---|
| **Image identity** | | | | |
| `base_name` | TEXT | `file_status.base_name` | `NOT NULL`, primary key | Unique: no `base_name` has more than one developed JPG. |
| `master_ref_id` | TEXT | `file_status.master_ref_id` | Allowed | Links to `samples`. |
| `batch_id` | INTEGER | `file_status.batch_id` | Allowed | Links to `batches.id`. |
| `batch_label` | TEXT | `file_locations.batch_label` | `NOT NULL` | Same source the pipeline uses today. |
| `location_code` | TEXT | `file_status.location_code` | Allowed | |
| `sub_batch_index` | TEXT | `file_status.sub_batch_index` | Allowed | `01`, `02_1`, ... (from the raw's folder). |
| `raw_image_id` | INTEGER | `images.id` (raw row) | Allowed | NULL if `images` has no matching raw. |
| `raw_blob_name` | TEXT | `images.blob_name` (raw row) | Allowed | e.g. `ALA00050.ARW` |
| `raw_image_url` | TEXT | `images.image_url` (raw row) | Allowed | Blob storage URL. |
| `exif_datetime` | TEXT | `images.exif_datetime` (raw row) | Allowed | Capture time from the camera. |
| **Path** | | | | |
| `jpg_path` | TEXT | `file_locations.path` | `NOT NULL` | Relative to the LTS `field-batches/` root, matching the path columns in `cutouts`, e.g. `AL_2023-08-01/developed-images/ALA00050.jpg`. |
| `jpg_size_bytes` | INTEGER | `file_locations.size_bytes` | `NOT NULL` | Used to flag zero-byte files. |
| `jpg_mtime_utc` | TEXT | `file_locations.mtime_utc` | `NOT NULL` | Last modified time on NFS. |
| **Phenotype** | | | | |
| `plant_type`, `species`, `height`, `size_class`, `growth_stage`, `cotton_variety`, `crop_or_fallow`, `crop_type_secondary`, `cover_crop_family`, `flower_fruit_or_seeds`, `cloud_cover`, `ground_residue`, `ground_cover` | TEXT | `file_status` | Allowed | Same names and values as the matching columns in `cutouts`, which are copied from `file_status` when a cutout is made. Value sets are documented in [`cutout_table_schema.md`](cutout_table_schema.md). |
| **Status** | | | | |
| `has_cutout` | INTEGER | `cutouts` | `NOT NULL` | `1` if a `cutouts` row exists for this `base_name` (`cutout_index = 0`), else `0`. |
| `refreshed_at` | TEXT | — | `NOT NULL` | ISO 8601 UTC time of the last rebuild. |

Phenotype comes from `file_status` instead of `cutouts`, because images that
don't have cutouts yet have no `cutouts` row. It's the same data either way.

**Indexes:** `batch_label`, `has_cutout`, `species`, `master_ref_id`.

```sql
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
    has_cutout            INTEGER NOT NULL,
    refreshed_at          TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_developed_images_batch_label   ON developed_images(batch_label);
CREATE INDEX IF NOT EXISTS idx_developed_images_has_cutout    ON developed_images(has_cutout);
CREATE INDEX IF NOT EXISTS idx_developed_images_species       ON developed_images(species);
CREATE INDEX IF NOT EXISTS idx_developed_images_master_ref_id ON developed_images(master_ref_id);
```

## How it's populated from NFS

### NFS layout

Each batch has its own folder under the LTS root
`/mnt/research-projects/r/raatwell/longterm_images3/field-batches/`:

```
field-batches/
  <batch_label>/                      e.g. AL_2023-08-01
    raws/<sub_batch_index>/<base_name>.ARW
    developed-images/<base_name>.jpg  <- the color-corrected JPGs this table tracks
    cutouts/<base_name>_<n>.jpg / _mask.png / .png / .json
```

- `batch_label` is `<location_code>_<YYYY-MM-DD>`, where `location_code` is two
  letters plus an optional two-digit site suffix (`TX`, `TX02`, `NC01`).
- `developed-images/` is flat: the sub-batch folder only appears under `raws/`.
- File names on NFS are all letter-prefixed (`ALA00050`). The timestamp-style
  names in blob storage (`20220622_112638`) don't appear on NFS.

### Source: `file_locations`, not a new NFS scan

Field-DataExploration already walks this tree and records every file in
`file_locations`. For each file, the path is parsed into `batch_label`,
`artifact_kind` (`processed_jpg` for `developed-images/*.jpg`) and
`sub_batch_index`. The annotation pipeline already relies on that table to
find developed JPGs. This design reads it too, instead of walking about 90,000
files over NFS again.

The path pattern that scan follows, for reference and for validating rows:

```
^.*/field-batches/(?P<batch_label>(?P<location_code>[A-Z]{2}\d{0,2})_(?P<batch_date>\d{4}-\d{2}-\d{2}))/developed-images/(?P<base_name>[^/]+)\.jpg$
```

### Rebuild query

The table is rebuilt in full inside one transaction, so readers always see
either the old contents or the new ones, never a half-written table.
`:field_batches_root` is the LTS root from config, so stored paths are
relative.

```sql
BEGIN;
DELETE FROM developed_images;
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
    substr(fl.path, length(:field_batches_root) + 1), fl.size_bytes, fl.mtime_utc,
    fs.plant_type, fs.species, fs.height, fs.size_class, fs.growth_stage, fs.cotton_variety,
    fs.crop_or_fallow, fs.crop_type_secondary, fs.cover_crop_family, fs.flower_fruit_or_seeds,
    fs.cloud_cover, fs.ground_residue, fs.ground_cover,
    c.id IS NOT NULL,
    strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
FROM file_status fs
JOIN file_locations fl
  ON fl.base_name = fs.base_name
 AND fl.artifact_kind = 'processed_jpg'
 AND fl.storage_location = 'nfs'
LEFT JOIN images i
  ON i.base_name = fs.base_name
 AND i.extension = 'arw'
LEFT JOIN cutouts c
  ON c.base_name = fs.base_name
 AND c.cutout_index = 0
WHERE fs.processed_jpg_in_nfs = 1;
COMMIT;
```

How the joins behave:
- `JOIN file_locations`: only images whose developed JPG is on NFS get a row.
- `LEFT JOIN images`: an image with no raw in `images` still gets a row, with
  the `raw_*` columns NULL. Each `base_name` has at most one `arw` row, so
  this can't duplicate rows.
- `LEFT JOIN cutouts`: sets `has_cutout` without dropping images that don't
  have a cutout yet.

### When it runs

As a new CLI command (for example `field-annotation refresh-developed-images`),
run after Field-DataExploration refreshes `file_locations` / `file_status`,
and again after an annotation run so `has_cutout` stays current. Once the table
exists, `_NEEDS_ANNOTATION_QUERY` can become a simple
`SELECT ... FROM developed_images WHERE has_cutout = 0`.

### Checks after a rebuild

- Row count equals the number of `processed_jpg` rows in `file_locations`
  (90,608 at the October 2026 snapshot).
- `has_cutout = 0` count equals the pending count (28,961 at the snapshot).
- Rows with `jpg_size_bytes = 0` are logged: `file_locations` contains some
  zero-byte files, and they can't be annotated.

## Open questions

1. **All developed images or only pending ones?** This design keeps every
   developed JPG and flags `has_cutout`. A pending-only `annotation_queue`
   would lose rows as cutouts are made and couldn't answer "what's been done."
2. **Table or view?** Every column already exists in other tables, so a SQL
   view would always be current and need no rebuild. A table is faster to
   query and gives a stable snapshot, but it can go stale between rebuilds.
3. **Which `images` row to link?** This design links the raw (`ARW`), since the
   developed JPG is made from it. The camera preview JPG (`extension = 'jpg'`)
   could be added as `preview_image_id` if needed.
4. **The crimson clover filter.** `_NEEDS_ANNOTATION_QUERY` currently includes
   `fs.species like '%crimson%'` and `fs.flower_fruit_or_seeds like 'True'`. If
   that query moves to this table, those filters should become CLI options
   instead of being hard-coded.