# `cutouts` Table Schema

Owned by this pipeline inside the shared `field_exploration.db` SQLite database.
The database also holds several tables owned by Field-DataExploration
(`samples`, `batches`, `images`, `file_status`, `file_locations`, and others) —
see [`docs/db_overview.md`](db_overview.md) for the full list. Defined in
[`src/field_annotation/schema.sql`](../src/field_annotation/schema.sql).

One row per cutout produced from a source image (`base_name` + `cutout_index`,
unique together). The schema allows several cutouts per image, but today every
image has exactly one row and `cutout_index` is always `0`. Images where
detection found nothing still get a row (`status = 'no_detection'`).

Counts below are from an October 2026 snapshot (61,647 rows).

## Data types

SQLite doesn't enforce declared types (the table isn't `STRICT`), so a column
can hold a value of a different type than declared. The values documented
here were checked with `typeof()`.

| Type | Holds | Limits |
|---|---|---|
| `INTEGER` | Whole number | 64-bit signed: −9,223,372,036,854,775,808 to 9,223,372,036,854,775,807 |
| `REAL` | Decimal number | 8-byte IEEE floating point, about 15 significant digits |
| `TEXT` | UTF-8 string | No per-column length limit; SQLite's default maximum is 1,000,000,000 bytes |

## Pipeline columns

| Column | Type | NULL | Values / range | Notes |
|---|---|---|---|---|
| `id` | INTEGER | Never (primary key) | 1 to 61,657 | Autoincrement. Gaps mean rows were deleted, e.g. to force reprocessing. |
| `base_name` | TEXT | Never (`NOT NULL`) | 8 to 12 chars, e.g. `ALA00049`, `VA_08957` | Source image file name without extension; matches `images.base_name` and `file_locations.base_name`. The letter prefix doesn't reliably match the batch location (e.g. `NCA03595` is in batch `AL_2023-05-09`). |
| `cutout_index` | INTEGER | Never (`NOT NULL`, default `0`) | Always `0` today | Position of this cutout within the image. |
| `master_ref_id` | TEXT | Allowed; 0 rows | 36-char UUID, e.g. `3548adc5-aef4-4f04-8911-941125e583b6` | Links to `samples.master_ref_id`. 6,616 distinct; many images share one sample. |
| `batch_label` | TEXT | Allowed; 0 rows | `<location_code>_<YYYY-MM-DD>`, 13 to 15 chars, e.g. `AL_2023-08-01` | 359 distinct batches. Also the first folder in every path column. |
| `location_code` | TEXT | Allowed; 0 rows | 11 codes, see [Value sets](#value-sets) | State postal code; a numeric suffix (`TX01`, `TX02`, `NC01`) marks a separate site in that state. |
| `plant_type` | TEXT | Allowed; 0 rows | `WEEDS`, `COVERCROPS` | `samples` also has `CASHCROPS`, `SOILS`, `COTTONFLOWERS`; none have been processed into cutouts yet. |
| `species` | TEXT | 12 rows | 42 lower-case common names, 5 to 31 chars, e.g. `palmer amaranth` | Full list in [Value sets](#value-sets). |
| `class_id` | INTEGER | 145 rows | 37 distinct values between 1 and 54 | Numeric class for `species`. NULL for the 12 rows with no species, `devil’s claw` (130) and `weed2` (3). Some species appear under two names that share a class (see [Value sets](#value-sets)). |
| `status` | TEXT | Never (`NOT NULL`) | `detected_segmented`, `no_detection`, `segmented`, `error` | See [Status values](#status-values). |
| `det_pred_conf` | REAL | When `status` is `no_detection` or `segmented` (9,262 rows) | 0.500 to 0.984 | YOLO detection confidence. Nothing is below 0.5, matching `detection.conf_threshold: 0.50` in `conf/config.yaml`. |
| `detection_bbox_x` / `_y` / `_w` / `_h` | INTEGER | Same 9,262 rows as `det_pred_conf` | Pixels. x 0 to 8,871 · y 0 to 9,124 · w 250 to 9,550 · h 419 to 7,964 | Raw YOLO box before padding. |
| `final_bbox_x` / `_y` / `_w` / `_h` | INTEGER | When `status = 'no_detection'` (8,748 rows) | Pixels. x 0 to 8,912 · y 0 to 9,123 · w 179 to 9,590 · h 341 to 8,095 | Region actually used for the crop, mask and cutout: the padded detection box, or the full frame for `segmented` rows, tightened to the mask unless `tighten_to_mask` is disabled. |
| `crop_path` | TEXT | When `status = 'no_detection'` | `<batch_label>/cutouts/<base_name>_<cutout_index>.jpg`, 36 to 42 chars | Rectangular crop of the source image. |
| `mask_path` | TEXT | When `status = 'no_detection'` | `<batch_label>/cutouts/<base_name>_<cutout_index>_mask.png` | Segmentation mask. |
| `cutout_path` | TEXT | When `status = 'no_detection'` | `<batch_label>/cutouts/<base_name>_<cutout_index>.png` | The segmented plant cut out of the crop. |
| `metadata_json_path` | TEXT | Allowed; 0 rows | `<batch_label>/cutouts/<base_name>_<cutout_index>.json` | Written for every row, including `no_detection`. |
| `error_message` | TEXT | Always, so far | — | Filled only when `status = 'error'`; no such rows exist yet. |
| `detection_model` | TEXT | When `status = 'segmented'` (514 rows) | One value: `detect_train_2026-08-16_20_44_02/weights/best.pt` | Detection checkpoint used. |
| `segmentation_model` | TEXT | Allowed; 0 rows | One value: `train_unet_mitb4_v4/checkpoints/epoch=51-step=3484-val_loss=0.00.ckpt` | Segmentation checkpoint used. |
| `processed_at` | TEXT | Never (`NOT NULL`) | ISO 8601 UTC with microseconds, 27 chars, e.g. `2026-08-18T18:12:25.248717Z`. Range 2026-08-18 to 2026-09-01 | When the row was written. |

Path columns are relative to the LTS batch root,
`/mnt/research-projects/r/raatwell/longterm_images3/field-batches/`, so
`AL_2023-08-01/cutouts/ALA00050_0.png` is stored at
`.../field-batches/AL_2023-08-01/cutouts/ALA00050_0.png`, next to that batch's
`developed-images/` and `raws/` folders (see `lts_cutouts_dir` in
[`src/field_annotation/pipeline.py`](../src/field_annotation/pipeline.py)).

## Sample metadata columns

These come from the image's `file_status` row, read when the image is picked
for annotation (`_NEEDS_ANNOTATION_QUERY` in
[`src/field_annotation/db.py`](../src/field_annotation/db.py)), and hold the
same plant and field-condition fields as `samples`. They were added after the
table was first created (`_apply_column_migrations` in `db.py`), which is why
they come after `processed_at` in `.schema`.

| Column | Type | NULL | Values |
|---|---|---|---|
| `height` | TEXT | 51,161 rows (83%) | `0.3 – 0.6m`, `0.61 – 0.9m`, `0.91 – 1.2m`, `1.21 – 1.5m` |
| `size_class` | TEXT | 524 rows | `SMALL`, `MEDIUM`, `LARGE` |
| `growth_stage` | TEXT | All rows today | None yet. `samples` uses the cotton stages `Vegetative`, `Squaring`, `First Flower`, `Flowering`, `Open Boll`, `Post Defoliation`. |
| `cotton_variety` | TEXT | All rows today | None yet. `samples` has `UA 107 Okra`, `FM Hairy`, `ST 5707 B2XF`, `ST 5091 B3XF`, `DG 3528 B3XF`, `DP 2038 B3XF`, `DP 2038`, `PHY 415 W3FE`, `PHY 411 W3FE`, `PHY 443 W3FE`. |
| `crop_or_fallow` | TEXT | 524 rows | `Fallow`, `Crop` |
| `crop_type_secondary` | TEXT | 524 rows | `N/A`, `Cotton`, `Soybean`, `Corn` |
| `cover_crop_family` | TEXT | 61,135 rows | `Legume`, `Grass` (`samples` also has `Brassicas`) |
| `flower_fruit_or_seeds` | TEXT | 12 rows | The text values `True` / `False`, not 0/1 |
| `cloud_cover` | TEXT | 2 rows | `Clear`, `Few Clouds`, `Scattered`, `Completely Obscured` |
| `ground_residue` | TEXT | 2 rows | 29 distinct values, see [Value sets](#value-sets) |
| `ground_cover` | TEXT | 2 rows | `0 – 25`, `26 – 50`, `51 – 75`, `76 – 100` (percent) |

## Matching values exactly

- `height` and `ground_cover` use an en dash (`–`, U+2013) with a space on each
  side, not a hyphen. `WHERE ground_cover = '51-75'` matches nothing; use `'51 – 75'`.
- `flower_fruit_or_seeds` holds text, so filter with `= 'True'`, not `= 1`.
  (The flag columns in `file_status` are the opposite: integers 0/1.)
- `N/A`, `None` and `Unknown` are real text values, different from NULL.
  `crop_type_secondary IS NULL` and `crop_type_secondary = 'N/A'` return different rows.
- `ground_residue` mixes a fixed list with free text entered as
  `Other : <description>`. The free text has case variants (`Sicklepod` /
  `sicklepod`), typos (`netsedge`, `weesa`) and trailing spaces:
  `'Other : sicklepod '` (790 rows) and `'Other : sicklepod'` (50 rows) are
  different values, and so are `'Other : weeds '` (120) and `'Other : weeds'` (231).
  Use `TRIM(ground_residue)` when grouping or comparing.
- `species` has a few naming inconsistencies: `waterhemp` / `common waterhemp`,
  `jungle rice` / `junglerice` and `large crabgrass` /
  `crabgrass (large or other spp.)` each share a `class_id`. `devil’s claw` uses
  a curly apostrophe (’, U+2019), so `'devil''s claw'` with a straight
  apostrophe won't match. `weed2` looks like a placeholder value.

## Status values

| Value | Rows | Meaning |
|---|---|---|
| `detected_segmented` | 52,385 | Detection ran, found something, and segmentation succeeded. |
| `no_detection` | 8,748 | Detection ran but found nothing. No bbox, crop, mask or cutout; only the metadata JSON. |
| `segmented` | 514 | Detection was skipped and the full frame was segmented. Every `COVERCROPS` row has this status; `WEEDS` rows are always `detected_segmented` or `no_detection`. |
| `error` | 0 | Processing failed; see `error_message`. |

## Value sets

**`location_code`:** `TX` (19,221), `NC` (11,583), `MD` (10,104), `GA` (6,660),
`IL` (5,857), `TX02` (3,359), `VA` (1,814), `KS` (1,566), `AL` (863),
`NC01` (460), `TX01` (160). Other tables also have `OH`, `MS` and `DV`.

**`species`** (42 names, with `class_id` and row count):

| species | class_id | rows |
|---|---|---|
| *(NULL)* | — | 12 |
| barley | 39 | 2 |
| barnyardgrass | 10 | 460 |
| broadleaf signalgrass | 7 | 2,980 |
| canada thistle | 53 | 580 |
| cereal rye | 35 | 2 |
| common cocklebur | 4 | 3,632 |
| common lambsquarters | 19 | 2,385 |
| common ragweed | 2 | 1,000 |
| common sunflower | 14 | 3,508 |
| common waterhemp | 9 | 3,430 |
| crabgrass (large or other spp.) | 5 | 70 |
| crimson clover | 31 | 504 |
| devil’s claw | — | 130 |
| fall panicum | 20 | 1,309 |
| field bindweed | 52 | 20 |
| giant foxtail | 24 | 1,265 |
| giant ragweed | 50 | 1,070 |
| goosegrass | 6 | 1,810 |
| hophornbeam copperleaf | 48 | 2,574 |
| horseweed | 25 | 2,417 |
| jimson weed | 21 | 1,239 |
| johnsongrass | 16 | 3,484 |
| jungle rice | 11 | 44 |
| junglerice | 11 | 10 |
| kochia | 13 | 1,451 |
| large crabgrass | 5 | 391 |
| palmer amaranth | 1 | 4,066 |
| prickly sida | 43 | 439 |
| ragweed parthenium | 15 | 3,515 |
| redroot pigweed | 54 | 178 |
| ryegrass | 47 | 1,420 |
| sicklepod | 3 | 1,780 |
| silverleaf nightshade | 51 | 1,519 |
| smooth pigweed | 18 | 725 |
| spiny amaranth | 45 | 1,120 |
| texas millet | 12 | 3,947 |
| velvetleaf | 22 | 189 |
| waterhemp | 9 | 1,972 |
| weed2 | — | 3 |
| winter wheat | 37 | 4 |
| yellow foxtail | 23 | 2,046 |
| yellow nutsedge | 49 | 2,945 |

**`ground_residue`:** standard values `Grass` (33,765), `Broadleaf` (8,301),
`Corn` (6,476), `None` (5,212), `Unknown` (2,367), `Other` (2,242),
`Cotton` (909), `Soybean` (804). Free text (21 distinct values, 1,569 rows):
`Other : sicklepod`, `Other : Sicklepod`, `Other : sicklepod/palmer`,
`Other : weeds`, `Other : mixed weeds`, `Other : weeds (morning glory)`,
`Other : weeds(morning Glory)`, `Other : weeds (copperleaf; amaranth)`,
`Other : morning glory`, `Other : Morning Glory`, `Other : nutsedge`,
`Other : netsedge`, `Other : palmer`, `Other : Grass, waterhemp, smellmelon`,
`Other : grass, waterhemp, smellmelon`, `Other : Rye Grass (terminated)`,
`Other : dead grass`, `Other : mix grass and broadleaf`, `Other : weesa`.
`Other : sicklepod` and `Other : weeds` each appear twice, with and without a
trailing space.

## Constraints & indexes

- `UNIQUE(base_name, cutout_index)`
- Indexes on `base_name`, `batch_label`, `status`

Column list and upsert logic live in
[`src/field_annotation/db.py`](../src/field_annotation/db.py) (`CUTOUT_COLUMNS`, `CutoutsDb.upsert_cutout`).