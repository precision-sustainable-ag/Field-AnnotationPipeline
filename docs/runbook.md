# Runbook: annotating field images

How to turn developed field JPGs into cutouts with this pipeline. For what
the pipeline does and how it's configured, see the [README](../README.md).

Run every command from the repo root. The config and data paths are relative
to it.

## One-time setup

1. Install the environment: `bash setup.sh`.
2. Check that the NFS share is mounted:
   `ls /mnt/research-projects/r/raatwell/longterm_images3/field-db/`.
3. Check that the checkpoints named in `paths.det_weights` and
   `paths.seg_weights` in [`conf/config.yaml`](../conf/config.yaml) exist
   under `data/models/`.

## Before each run: check the config

These four keys in [`conf/config.yaml`](../conf/config.yaml) decide what a
run does:

| Key | What it controls |
|---|---|
| `save_to_lts` | `true` writes results to the shared DB and copies cutouts to LTS. `false` only writes local files for review. |
| `plant_type_filter` | Which plant type gets processed (`WEEDS`, `COVERCROPS`, ...). `null` means all of them. |
| `detection.enabled` | `true` runs YOLO first and segments the detected region. `false` segments the whole frame. |
| `device` | `cuda` picks the least-busy GPU. `cuda:N` uses a specific GPU. |

## Steps

### 1. See what's pending

```bash
uv run field-annotation list-pending
```

This prints each batch label with its number of images that have no
cutout yet.

### 2. Preview a batch

```bash
uv run field-annotation run --batch-label AL_2023-05-11 --dry-run
```

This prints what would be processed. It doesn't run inference.

### 3. Test on a few images

Set `save_to_lts: false`, then run:

```bash
uv run field-annotation run --batch-label AL_2023-05-11 --limit 10
```

Look at the files in `data/temp/AL_2023-05-11/cutouts/`. Each image gets
four: `<name>_0.jpg` (crop), `_0_mask.png`, `_0.png` (cutout) and `_0.json`
(metadata).

When you're done, delete `data/temp/AL_2023-05-11/`. Otherwise the saving
run in step 4 copies these review files to LTS along with its own.

### 4. Run for real

Set `save_to_lts: true`. Start the run inside `tmux` so it keeps going if
you disconnect:

```bash
tmux new -s annotate
uv run field-annotation run --batch-label AL_2023-05-11   # one batch
uv run field-annotation run                               # everything pending
```

Detach with `Ctrl-b d` and reattach with `tmux attach -t annotate`.

The saving run redoes inference. It doesn't reuse the test output from
step 3.

### 5. Monitor

- Each run writes a timestamped log to `logging/`. To follow the latest one:
  `tail -f "$(ls -t logging/*.log | head -1)"`
- Check GPU use with `nvidia-smi`.

### 6. Check the results

The run ends by printing counts by status. To check one batch in the DB
without changing it:

```bash
sqlite3 "file:/mnt/research-projects/r/raatwell/longterm_images3/field-db/field_exploration.db?mode=ro" \
  "SELECT status, COUNT(*) FROM cutouts WHERE batch_label = 'AL_2023-05-11' GROUP BY status;"
```

The cutout files go to
`/mnt/research-projects/r/raatwell/longterm_images3/field-batches/<batch_label>/cutouts/`.

## Statuses

| Status | Meaning |
|---|---|
| `detected_segmented` | YOLO found a plant and it was segmented. |
| `segmented` | Detection was off, so the whole frame was segmented. |
| `no_detection` | YOLO found nothing. A JSON file is written but no images. |
| `error` | Detection, reading or segmentation failed. See `error_message` and the log. |

## Troubleshooting

- **The run stopped partway.** Run the same command again. Results are
  committed every 100 images, so committed images are skipped and the rest
  are picked up. Clear any leftovers in `data/temp/<batch_label>/` first.
- **The log says "cuda requested but not available, falling back to cpu".**
  The run continues on the CPU and is much slower. Stop it and check
  `nvidia-smi`.
- **The log says "source not found on LTS".** The JPG listed in the DB isn't
  on NFS. That image gets no row and stays pending until the file is there.
- **An image or batch needs redoing.** Every image with a `cutouts` row is
  skipped, `error` rows included. Redoing one means deleting its row from
  the shared DB (see the [README](../README.md#idempotency)). Agree on it
  with the team before you do.
- **Even `list-pending` writes to the DB.** Opening the connection applies
  the `cutouts` schema and column changes to the database.
