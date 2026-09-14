# Field-AnnotationPipeline

Detection + segmentation annotation pipeline for field imagery.

Scans `field_exploration.db` (maintained by [Field-DataExploration](https://github.com/precision-sustainable-ag/Field-DataExploration))
for developed JPGs that haven't been annotated yet, stages them from long-term
storage (LTS) batch by batch, runs weed detection + segmentation (model code and
checkpoints from [Field-SegmentationTraining](https://github.com/precision-sustainable-ag/Field-SegmentationTraining))
to produce cutouts + metadata, records the results in the DB, and copies the
cutout files back to LTS.

```
scan DB for images needing annotation
  -> copy developed JPGs from LTS to local temp, batch by batch
  -> detect (YOLO) + segment (SMP UNet) -> cutout + mask + metadata
  -> upsert results into the DB (`cutouts` table)
  -> copy cutouts + metadata back to LTS
```

## Install

```bash
bash setup.sh
```

Creates `.venv` (Python 3.12 by default; override with `PYTHON_VERSION=3.11
bash setup.sh`) and installs PyTorch/torchvision from the CUDA 12.6 wheel
index pinned in `pyproject.toml` (`[tool.uv.sources]` -> `pytorch-cu126`).
Re-run any time to rebuild the environment from scratch. If your host's GPU
driver needs a different CUDA version, update that index in
`pyproject.toml`.

If the environment already exists and you just changed a dependency, `uv
sync` alone is enough.

## Configure

Edit [conf/config.yaml](conf/config.yaml). Top-level: `database.path`, shared
`device`, `paths` (local temp dir, log dir, species info, and the two model
checkpoint paths -- `det_weights`, `seg_weights`), `batching`. Model and
inference knobs are grouped by stage:
- `detection`: `enabled` (false = skip YOLO, segment the full frame instead),
  `conf_threshold`, `padding` (pad the detected ROI before segmenting)
- `segmentation`: `model` (architecture -- `arch`/`encoder`/...),
  `threshold`, `pad_to_divisor`, `tile` (tiling for large crops),
  `clean_disconnected_mask` (drop mask speckles), `tighten_to_mask` (re-crop
  to the segmented foreground)

## Run

`save_to_lts` in `conf/config.yaml` controls whether `run` commits its
results. With `save_to_lts: false` (the default), it writes cutouts locally
for review and leaves the DB/LTS untouched; with `save_to_lts: true`, it also
copies cutouts to LTS and records them in the DB.

```bash
# see what's pending
uv run field-annotation list-pending

# preview what would be processed, no inference run yet
uv run field-annotation run --batch-label AL_2023-05-11 --dry-run

# run inference (save_to_lts: false in config) -- writes cutouts locally only
uv run field-annotation run --batch-label AL_2023-05-11
# -> review data/temp/AL_2023-05-11/cutouts/*.jpg / _mask.png / .png / .json

# happy with the results? set save_to_lts: true in conf/config.yaml, then:
uv run field-annotation run --batch-label AL_2023-05-11
# (this re-runs inference -- reviewed local outputs aren't reused)

# process everything pending, saving as it goes
uv run field-annotation run
```

`run` accepts `--batch-label`, `--plant-type`, `--limit`, `--device {cuda,cpu}`,
and `--dry-run`.

## Idempotency

Any `base_name` already recorded in `cutouts` (any status: `detected_segmented`,
`no_detection`, `segmented`, `error`) is skipped on future runs. To force a
specific image or batch to be reprocessed, delete its row(s) first:

```sql
DELETE FROM cutouts WHERE base_name = 'ALB001';
DELETE FROM cutouts WHERE batch_label = 'AL_2023-05-11';
```

## Test

```bash
uv run pytest
```
