from __future__ import annotations

import logging
import sqlite3
from collections import Counter
from itertools import islice
from pathlib import Path
from typing import Iterable, Iterator

import cv2
import numpy as np

from field_annotation.cutout_writer import build_class_mask, write_cutout_files, write_metadata_json
from field_annotation.db import CUTOUT_COLUMNS, CutoutsDb
from field_annotation.detection import WeedDetector, pad_bbox
from field_annotation.segmentation import clean_disconnected_mask, predict_mask, tight_bbox_from_mask
from field_annotation.species import get_class_id
from field_annotation.transfer import cleanup_local_dir, stage_batch_images, stage_cutouts_out
from field_annotation.utils import utcnow_iso

log = logging.getLogger(__name__)


def _chunked(items: list, size: int) -> Iterator[list]:
    it = iter(items)
    while chunk := list(islice(it, size)):
        yield chunk


def _complete_row(row: dict) -> dict:
    """Fills in every CUTOUT_COLUMNS field, defaulting to None for whatever
    a given code path didn't set -- so every metadata JSON (and DB row) has
    the same fixed set of keys regardless of status, instead of silently
    omitting fields that weren't applicable (e.g. detection_bbox_* when
    detection didn't run).
    """
    return {col: row.get(col) for col in CUTOUT_COLUMNS}


def _model_label(path: str | Path) -> str:
    """Filename plus its 2 parent directories, e.g.
    ".../train_unet_mitb4_v4/checkpoints/epoch=51-...ckpt" ->
    "train_unet_mitb4_v4/checkpoints/epoch=51-...ckpt" -- a bare checkpoint
    filename like "best.pt" alone doesn't identify which training run/model
    variant produced it, the parent dirs do.
    """
    parts = Path(path).parts
    return str(Path(*parts[-3:])) if len(parts) >= 3 else str(Path(path))


def process_one_image(
    row: dict,
    detector: WeedDetector | None,
    seg_model,
    species_info: dict,
    cutouts_local_dir: Path,
    cfg: dict,
    run_detection: bool = True,
) -> dict:
    """Never raises -- detection and segmentation are each wrapped so one bad
    image can't abort a batch. Returns a dict shaped like a `cutouts` row.

    When run_detection is False, `detector` is ignored (may be None) and the
    whole image is segmented directly -- no bbox/crop/padding step, since
    there's no detected ROI to pad. clean_disconnected_mask and the
    tighten-to-mask step afterward still apply either way.

    Files are written under `cutouts_local_dir` (this is only where the batch
    stages its outputs before process_batch copies them to LTS in bulk), but
    every path recorded in the DB/JSON is relative to the LTS field-batches
    root (`<batch_label>/cutouts/<file>`), not an absolute filesystem path --
    the local staging dir is deleted at the end of the batch anyway, and a
    relative path stays valid regardless of where field-batches is mounted.
    """
    base_name = row["base_name"]
    batch_label = row.get("batch_label")
    # Looked up once here (not just on a successful segment) since it only
    # depends on file_status.species -- nothing to do with whether detection
    # or segmentation succeeded, so it should be populated for no_detection
    # and error rows too rather than left null just because they returned early.
    class_id = get_class_id(species_info, row.get("species"))
    common = {
        "base_name": base_name,
        "cutout_index": 0,
        "master_ref_id": row.get("master_ref_id"),
        "batch_label": batch_label,
        "location_code": row.get("location_code"),
        "plant_type": row.get("plant_type"),
        "species": row.get("species"),
        "class_id": class_id,
        "detection_model": _model_label(cfg["paths"]["det_weights"]) if run_detection else None,
        "segmentation_model": _model_label(cfg["paths"]["seg_weights"]),
        "processed_at": utcnow_iso(),
    }
    lts_json_path = f"{batch_label}/cutouts/{base_name}_0.json"

    # detection_bbox_x/y/w/h + det_pred_conf are the raw YOLO output,
    # pre-padding -- only meaningful (and only ever set) when run_detection
    # actually ran; writing them (even as null) for a full-frame segment
    # would misleadingly imply a detection happened. final_bbox_x/y/w/h
    # below is the complementary field: always set, the region actually used
    # for crop/mask/cutout.
    det = None
    detection_bbox_fields: dict = {}
    if run_detection:
        try:
            det = detector.detect(Path(row["local_jpg_path"]))
        except Exception as exc:
            log.exception("Detection failed for %s", base_name)
            return _complete_row({**common, "status": "error", "error_message": f"detect: {exc}"})

        if det is None:
            result = _complete_row({**common, "status": "no_detection", "metadata_json_path": lts_json_path})
            write_metadata_json(cutouts_local_dir, base_name, result)
            return result

        raw_x, raw_y, raw_w, raw_h = det["bbox"]
        detection_bbox_fields = {
            "detection_bbox_x": raw_x, "detection_bbox_y": raw_y,
            "detection_bbox_w": raw_w, "detection_bbox_h": raw_h,
            "det_pred_conf": det["det_pred_conf"],
        }

    lts_paths = {
        "crop_path": f"{batch_label}/cutouts/{base_name}_0.jpg",
        "mask_path": f"{batch_label}/cutouts/{base_name}_0_mask.png",
        "cutout_path": f"{batch_label}/cutouts/{base_name}_0.png",
    }

    try:
        image_bgr = cv2.imread(row["local_jpg_path"])
        if image_bgr is None:
            raise RuntimeError(f"cv2 could not read {row['local_jpg_path']}")
    except Exception as exc:
        log.exception("Reading image failed for %s", base_name)
        return _complete_row({**common, **detection_bbox_fields, "status": "error", "error_message": f"read: {exc}"})

    img_h, img_w = image_bgr.shape[:2]

    if run_detection:
        x, y, w, h = raw_x, raw_y, raw_w, raw_h
        pad_cfg = cfg["detection"].get("padding") or {}
        if pad_cfg.get("enabled") and pad_cfg.get("pad_px", 0) > 0:
            x, y, w, h = pad_bbox((x, y, w, h), pad_cfg["pad_px"], img_w, img_h)
    else:
        # No detected ROI to crop to (or pad) -- segment the full frame.
        x, y, w, h = 0, 0, img_w, img_h
    final_bbox_fields = {"final_bbox_x": x, "final_bbox_y": y, "final_bbox_w": w, "final_bbox_h": h}

    try:
        crop_bgr = image_bgr[y:y + h, x:x + w]
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)

        seg_cfg = cfg["segmentation"]
        bin_mask = predict_mask(
            seg_model,
            crop_rgb,
            device=cfg["device"],
            threshold=seg_cfg["threshold"],
            pad_to_divisor=seg_cfg["pad_to_divisor"],
            tile_if_larger_than=seg_cfg["tile"]["max_untiled_side"],
            tile_size=seg_cfg["tile"]["tile_size"],
            overlap=seg_cfg["tile"]["overlap"],
            blend_method=seg_cfg["tile"]["blend_method"],
        )

        clean_cfg = seg_cfg.get("clean_disconnected_mask") or {}
        if clean_cfg.get("enable"):
            bin_mask = clean_disconnected_mask(bin_mask, max_gap_px=clean_cfg.get("min_area_px", 500))

        # Tighten crop+mask down to where the segmentation actually found
        # foreground pixels, replacing the padded/full-frame region --
        # crop_bgr and bin_mask are re-sliced together so they stay spatially
        # aligned; mask_class/cutout_bgr below are then derived from those
        # already-tightened arrays, so they come out tightened too with no
        # separate bookkeeping. final_bbox_fields (set above) is left as-is
        # if the mask has no foreground pixels at all (tight is None),
        # already fills the crop, or segmentation.tighten_to_mask is false.
        if seg_cfg.get("tighten_to_mask", True):
            tight = tight_bbox_from_mask(bin_mask)
            if tight is not None:
                tx1, ty1, tx2, ty2 = tight
                if (tx1, ty1, tx2, ty2) != (0, 0, w, h):
                    crop_bgr = crop_bgr[ty1:ty2, tx1:tx2]
                    bin_mask = bin_mask[ty1:ty2, tx1:tx2]
                    final_bbox_fields = {
                        "final_bbox_x": x + tx1, "final_bbox_y": y + ty1,
                        "final_bbox_w": tx2 - tx1, "final_bbox_h": ty2 - ty1,
                    }

        mask_class = build_class_mask(bin_mask, class_id)
        mask_3c = np.repeat((bin_mask == 1)[:, :, None], 3, axis=2)
        cutout_bgr = np.where(mask_3c, crop_bgr, 0).astype(np.uint8)

        write_cutout_files(cutouts_local_dir, base_name, crop_bgr, mask_class, cutout_bgr)
    except Exception as exc:
        log.exception("Segmentation failed for %s", base_name)
        return _complete_row({
            **common, **detection_bbox_fields, **final_bbox_fields,
            "status": "error", "error_message": f"segment: {exc}",
        })

    # "detected_segmented" means both steps actually happened -- with
    # run_detection false there was no detection step at all, just a
    # full-frame segment, so that status would misrepresent what ran.
    # "segmented" is the detection-less counterpart, parallel to how
    # "no_detection" already covers the detection-ran-but-found-nothing case.
    status = "detected_segmented" if run_detection else "segmented"

    result = _complete_row({
        **common,
        **detection_bbox_fields,
        **final_bbox_fields,
        "status": status,
        **lts_paths,
        "metadata_json_path": lts_json_path,
    })
    write_metadata_json(cutouts_local_dir, base_name, result)

    return result


def process_batch(
    cfg: dict,
    db: CutoutsDb,
    conn: sqlite3.Connection,
    batch_label: str,
    rows: list[dict],
    detector: WeedDetector | None,
    seg_model,
    species_info: dict,
    save_to_lts: bool = False,
    run_detection: bool = True,
) -> dict:
    """Runs detection+segmentation for `rows` and always writes cutout files
    under the local temp dir. Only when `save_to_lts` is set does it also
    upsert results into the DB, copy cutouts to LTS, and clean up -- with it
    unset, outputs are left in place under local_temp_dir/<batch_label>/cutouts/
    for manual review, untouched by the DB anti-join, so a later re-run of the
    same batch (with or without save_to_lts) reprocesses it from scratch.

    detector may be None when run_detection is False (nothing loaded it since
    it's never called) -- process_one_image segments the full frame instead.
    """
    local_dir = Path(cfg["paths"]["local_temp_dir"]) / batch_label
    cutouts_local_dir = local_dir / "cutouts"

    # Derived from a row's own jpg_path rather than a configured batches-root
    # + batch_label join, so it stays correct even if batch_label naming ever
    # drifts from the on-disk directory naming. Every row in `rows` belongs to
    # the same batch_label, so they all share this same LTS directory.
    lts_cutouts_dir = Path(rows[0]["jpg_path"]).parent.parent / "cutouts"

    chunk_size = cfg["batching"]["images_per_copy_chunk"]
    counts: Counter = Counter()
    total = len(rows)
    processed = 0

    for chunk in _chunked(rows, chunk_size):
        staged = stage_batch_images(chunk, local_dir)

        for row in staged:
            result = process_one_image(
                row, detector, seg_model, species_info, cutouts_local_dir, cfg,
                run_detection=run_detection,
            )
            processed += 1
            counts[result["status"]] += 1
            if result["status"] == "detected_segmented":
                log.info(
                    "[%d/%d] Detected+segmented %s (batch=%s, conf=%.3f, class_id=%s)",
                    processed, total, result["base_name"], batch_label,
                    result["det_pred_conf"], result["class_id"],
                )
            elif result["status"] == "segmented":
                log.info(
                    "[%d/%d] Segmented %s (batch=%s, class_id=%s)",
                    processed, total, result["base_name"], batch_label, result["class_id"],
                )
            if save_to_lts:
                db.upsert_cutout(conn, result)

        if save_to_lts:
            conn.commit()
            stage_cutouts_out(cutouts_local_dir, lts_cutouts_dir)
            cleanup_local_dir(cutouts_local_dir)

        # Staged source JPGs aren't needed for review (the crop/mask/cutout
        # files already carry the relevant image content), so by default
        # clear them to bound local disk usage regardless of save_to_lts.
        # batching.cleanup_staged_images: false keeps them around instead
        # (e.g. to compare a cutout against its full-size source image).
        if cfg["batching"].get("cleanup_staged_images", True):
            cleanup_local_dir(local_dir / "developed-images")

    if save_to_lts:
        cleanup_local_dir(local_dir)

    log.info("Batch %s (save_to_lts=%s): %s", batch_label, save_to_lts, dict(counts))
    return dict(counts)


def group_by_batch(rows: Iterable[sqlite3.Row]) -> dict[str, list[dict]]:
    """Groups rows (already SQL-ordered by batch_label) into
    {batch_label: [row, ...]} preserving arrival order."""
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        row_dict = dict(row)
        grouped.setdefault(row_dict["batch_label"], []).append(row_dict)
    return grouped
