from __future__ import annotations

import argparse
import logging
from collections import Counter
from pathlib import Path

import torch

from field_annotation.config import load_config
from field_annotation.db import CutoutsDb
from field_annotation.detection import WeedDetector
from field_annotation.pipeline import group_by_batch, process_batch
from field_annotation.segmentation import build_model, load_weights
from field_annotation.species import load_species_info
from field_annotation.utils import configure_logging

log = logging.getLogger(__name__)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    cfg = load_config(args.config)
    log_path = configure_logging(args.log_level, cfg["paths"].get("log_dir"))
    if log_path:
        log.info("Logging to %s", log_path)

    if args.command == "run":
        cmd_run(cfg, args)
    elif args.command == "list-pending":
        cmd_list_pending(cfg, args)
    else:
        parser.error(f"Unknown command: {args.command}")


def cmd_list_pending(cfg: dict, args: argparse.Namespace) -> None:
    plant_type = args.plant_type or cfg.get("plant_type_filter")
    db = CutoutsDb(cfg["database"]["path"])
    conn = db.connect()
    rows = db.batch_summary_needing_annotation(conn, plant_type=plant_type)
    if not rows:
        print("No images need annotation.")
        return
    total = 0
    for batch_label, count in rows:
        print(f"{batch_label:20s} {count}")
        total += count
    print(f"\n{len(rows)} batches, {total} images pending.")


def cmd_run(cfg: dict, args: argparse.Namespace) -> None:
    device = _resolve_device(args.device or cfg["device"])
    cfg = {**cfg, "device": device}
    plant_type = args.plant_type or cfg.get("plant_type_filter")

    db = CutoutsDb(cfg["database"]["path"])
    conn = db.connect()
    rows = db.fetch_images_needing_annotation(
        conn, batch_label=args.batch_label, plant_type=plant_type, limit=args.limit
    )
    if not rows:
        print("No images need annotation.")
        return

    grouped = group_by_batch(rows)

    if args.dry_run:
        total = 0
        for batch_label, batch_rows in grouped.items():
            print(f"{batch_label:20s} {len(batch_rows)}")
            total += len(batch_rows)
        print(f"\n[dry-run] {len(grouped)} batches, {total} images would be processed.")
        return

    run_detection = cfg["detection"].get("enabled", True)
    detector = None
    if run_detection:
        log.info("Loading detection model: %s", cfg["paths"]["det_weights"])
        detector = WeedDetector(
            Path(cfg["paths"]["det_weights"]),
            conf_threshold=cfg["detection"]["conf_threshold"],
            device=device,
        )
    else:
        log.info("detection.enabled is false -- segmenting full frames, no detection model loaded")

    log.info("Loading segmentation model: %s", cfg["paths"]["seg_weights"])
    seg_model = build_model(cfg, device=device)
    load_weights(seg_model, Path(cfg["paths"]["seg_weights"]))

    species_info = load_species_info(Path(cfg["paths"]["species_info"]))

    save_to_lts = cfg.get("save_to_lts", False)

    total_counts: Counter = Counter()
    for batch_label, batch_rows in grouped.items():
        counts = process_batch(
            cfg, db, conn, batch_label, batch_rows, detector, seg_model, species_info,
            save_to_lts=save_to_lts, run_detection=run_detection,
        )
        total_counts.update(counts)

    if save_to_lts:
        print(f"Saved to LTS: {dict(total_counts)}")
    else:
        local_root = Path(cfg["paths"]["local_temp_dir"]).resolve()
        print(f"Processed (not saved): {dict(total_counts)}")
        print(f"Review outputs under: {local_root}/<batch_label>/cutouts/")
        print("Nothing was written to the DB or LTS. Set save_to_lts: true in "
              f"{args.config} once you're happy with the results, then rerun "
              "(this re-runs inference).")


def _resolve_device(requested: str) -> str:
    if requested == "cuda" and not torch.cuda.is_available():
        log.warning("cuda requested but not available, falling back to cpu")
        return "cpu"
    return requested


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="field-annotation")
    parser.add_argument("--config", type=Path, default=Path("conf/config.yaml"))
    parser.add_argument("--log-level", default="INFO")

    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser(
        "run",
        help="Scan for images needing annotation and run detection+segmentation",
    )
    run.add_argument("--batch-label", help="Restrict to a single batch_label")
    run.add_argument(
        "--plant-type",
        help="Restrict to a single file_status.plant_type, e.g. WEEDS, CASHCROPS, or COVERCROPS",
    )
    run.add_argument("--limit", type=int, help="Restrict to at most N images (across batches)")
    run.add_argument("--device", choices=["cuda", "cpu"], help="Overrides device from config")
    run.add_argument("--dry-run", action="store_true", help="Only print what would be processed")

    list_pending = subparsers.add_parser("list-pending", help="List batches with images that still need annotation")
    list_pending.add_argument(
        "--plant-type",
        help="Restrict to a single file_status.plant_type, e.g. WEEDS, CASHCROPS, or COVERCROPS",
    )

    return parser


if __name__ == "__main__":
    main()
