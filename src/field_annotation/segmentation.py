from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F

log = logging.getLogger(__name__)


# --- weight loading, ported verbatim from
# Field-SegmentationTraining/src/inference_utils/weight_loader.py --
# handles both a Lightning `.ckpt` ({"state_dict": {...}}, keys prefixed
# "model.") and a bare-state_dict `.pth` export.

def _strip_prefix(state_dict: dict, prefixes: tuple[str, ...] = ("model.", "module.")) -> dict:
    out = {}
    for k, v in state_dict.items():
        nk = k
        for pref in prefixes:
            if nk.startswith(pref):
                nk = nk[len(pref):]
        out[nk] = v
    return out


def load_state_dict_flex(model: torch.nn.Module, weights_path: str | Path, strict: bool = False) -> Tuple[list, list]:
    """Load a checkpoint 'state_dict' saved from Lightning into a bare SMP model."""
    p = Path(weights_path)
    obj = torch.load(str(weights_path), map_location="cpu", weights_only=False)

    sd = obj.get("state_dict", obj) if isinstance(obj, dict) else obj
    if not isinstance(sd, dict):
        raise RuntimeError(f"Unsupported checkpoint format at {p}")
    sd = _strip_prefix(sd)
    missing, unexpected = model.load_state_dict(sd, strict=strict)
    return missing, unexpected


def build_model(cfg, device: str) -> torch.nn.Module:
    model_cfg = cfg["segmentation"]["model"]
    kwargs = {
        "arch":            model_cfg["arch"],
        "encoder_name":    model_cfg["encoder"],
        "encoder_weights": model_cfg["encoder_weights"],
        "in_channels":     model_cfg["in_channels"],
        "classes":         model_cfg["classes"],
    }
    if "decoder_attention_type" in model_cfg:
        kwargs["decoder_attention_type"] = model_cfg["decoder_attention_type"]
    if model_cfg.get("encoder_freeze", False):
        kwargs["encoder_freeze"] = True

    model = smp.create_model(**kwargs)
    return model.to(device).eval()


def load_weights(model: torch.nn.Module, weights_path: Path) -> None:
    missing, unexpected = load_state_dict_flex(model, weights_path, strict=False)
    if missing or unexpected:
        log.warning("Loaded %s with missing=%s unexpected=%s", weights_path, missing, unexpected)


# --- tensor helpers, ported from
# Field-SegmentationTraining/src/inference_utils/inference_pipeline.py --

def _to_tensor01(rgb: np.ndarray) -> torch.Tensor:
    t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    return t.unsqueeze(0)


def _pad_to_divisor(x: torch.Tensor, divisor: Optional[int]) -> tuple[torch.Tensor, tuple[int, int, int, int]]:
    if not divisor or divisor <= 1:
        return x, (0, 0, 0, 0)
    _, _, h, w = x.shape
    pad_h = (divisor - (h % divisor)) % divisor
    pad_w = (divisor - (w % divisor)) % divisor
    pad = (0, pad_w, 0, pad_h)  # (left, right, top, bottom)
    x_pad = F.pad(x, pad, mode="constant", value=0.0)
    return x_pad, (pad[2], pad[3], pad[0], pad[1])  # (top, bottom, left, right)


def _unpad(np_img: np.ndarray, pads: tuple[int, int, int, int]) -> np.ndarray:
    t, b, l, r = pads
    h, w = np_img.shape[:2]
    return np_img[t:h - b if b > 0 else h, l:w - r if r > 0 else w]


def _predict_mask(model: torch.nn.Module, x: torch.Tensor, thr: float, device: str) -> np.ndarray:
    with torch.inference_mode():
        logits = model(x.to(device))
        prob = torch.sigmoid(logits)
    mask = (prob > thr).float()
    return mask.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.uint8)


def _hann2d(h: int, w: int) -> np.ndarray:
    w2d = np.outer(np.hanning(h), np.hanning(w))
    return (w2d / (w2d.max() + 1e-8)).astype(np.float32)


def _gaussian2d(h: int, w: int, sigma_rel: float = 0.3) -> np.ndarray:
    cy, cx = h / 2.0, w / 2.0
    sigma_y, sigma_x = sigma_rel * h, sigma_rel * w
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    g = np.exp(-0.5 * (((yy - cy) / sigma_y) ** 2 + ((xx - cx) / sigma_x) ** 2))
    return (g / (g.max() + 1e-8)).astype(np.float32)


def _predict_mask_tiled(
    model: torch.nn.Module,
    img_rgb: np.ndarray,
    device: str,
    tile_size: int = 2048,
    overlap: int = 256,
    divisor: Optional[int] = 32,
    thr: float = 0.5,
    blend_method: str = "hann",
) -> np.ndarray:
    h_img, w_img = img_rgb.shape[:2]
    step = tile_size - overlap
    use_max = blend_method.lower() == "max"

    if use_max:
        fused = np.zeros((h_img, w_img), dtype=np.float32)
    else:
        acc = np.zeros((h_img, w_img), dtype=np.float32)
        wsum = np.zeros((h_img, w_img), dtype=np.float32)

    y = 0
    while y < h_img:
        x = 0
        while x < w_img:
            y1c, y2c = y, min(y + tile_size, h_img)
            x1c, x2c = x, min(x + tile_size, w_img)
            tile = img_rgb[y1c:y2c, x1c:x2c]
            th, tw = tile.shape[:2]

            tile_pad, pads = _pad_to_divisor(_to_tensor01(tile), divisor)
            with torch.inference_mode():
                if device == "cuda":
                    with torch.autocast(device_type="cuda"):
                        logits = model(tile_pad.to(device))
                        probs = torch.sigmoid(logits).squeeze(0).squeeze(0).detach().cpu().numpy()
                else:
                    logits = model(tile_pad.to(device))
                    probs = torch.sigmoid(logits).squeeze(0).squeeze(0).detach().cpu().numpy()
            if divisor:
                probs = _unpad(probs, pads)

            if use_max:
                fused[y1c:y2c, x1c:x2c] = np.maximum(fused[y1c:y2c, x1c:x2c], probs)
            else:
                win = _gaussian2d(th, tw) if blend_method.lower() == "gaussian" else _hann2d(th, tw)
                win = cv2.resize(win, (probs.shape[1], probs.shape[0]), interpolation=cv2.INTER_LINEAR)
                acc[y1c:y2c, x1c:x2c] += probs * win
                wsum[y1c:y2c, x1c:x2c] += win

            x += step
        y += step

    out = fused if use_max else (acc / np.clip(wsum, 1e-6, None))
    return (out >= thr).astype(np.uint8)


def clean_disconnected_mask(bin_mask: np.ndarray, max_gap_px: float) -> np.ndarray:
    """Drops connected components of `bin_mask` that are farther than
    `max_gap_px` from the largest component -- removes speckle noise while
    keeping fragments genuinely close to the main plant. Ported verbatim from
    Field-SegmentationTraining/src/inference_utils/inference_pipeline.py's
    _clean_disconnected_mask.
    """
    kernel = np.ones((3, 3), np.uint8)
    # Used only to decide connectivity/grouping -- breaks 1-2px noise bridges
    # without affecting which original pixels end up in the final mask.
    label_src = cv2.morphologyEx(bin_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(label_src, connectivity=4)
    if num <= 2:
        return bin_mask

    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = 1 + int(np.argmax(areas))
    main_mask = (labels == largest_label).astype(np.uint8)
    dist_to_main = cv2.distanceTransform(1 - main_mask, cv2.DIST_L2, 5)

    # Re-label the ORIGINAL (unopened) mask so we don't lose real pixels that
    # the opening removed, then decide per-original-component using distance
    # from the (opened) main_mask.
    num_o, labels_o, _, _ = cv2.connectedComponentsWithStats(bin_mask, connectivity=4)
    cleaned = np.zeros_like(bin_mask)
    for label in range(1, num_o):
        comp_pixels = labels_o == label
        min_dist = dist_to_main[comp_pixels].min()
        if min_dist <= max_gap_px:
            cleaned[comp_pixels] = 1
    return cleaned


def tight_bbox_from_mask(mask: np.ndarray) -> Optional[tuple[int, int, int, int]]:
    """Tight bounding box (x1, y1, x2, y2), end-exclusive, around all non-zero
    pixels of a crop-space binary mask. Returns None if the mask has no
    foreground pixels at all. Ported verbatim from
    Field-SegmentationTraining/src/inference_utils/inference_pipeline.py's
    _tight_bbox_from_mask.
    """
    bin_mask = mask > 0
    rows = np.any(bin_mask, axis=1)
    cols = np.any(bin_mask, axis=0)
    if not rows.any() or not cols.any():
        return None
    y_idx = np.where(rows)[0]
    x_idx = np.where(cols)[0]
    y1, y2 = int(y_idx[0]), int(y_idx[-1]) + 1
    x1, x2 = int(x_idx[0]), int(x_idx[-1]) + 1
    return x1, y1, x2, y2


def predict_mask(
    model: torch.nn.Module,
    crop_rgb: np.ndarray,
    device: str,
    threshold: float = 0.5,
    pad_to_divisor: int = 32,
    tile_if_larger_than: int = 4000,
    tile_size: int = 2048,
    overlap: int = 256,
    blend_method: str = "hann",
) -> np.ndarray:
    """Binary {0,1} mask, same H,W as crop_rgb. Dispatches to tiled inference
    above `tile_if_larger_than` on either side -- mirrors the exact 4000px
    cutoff the old pipeline used (unet_segment_weeds.py::UNetInference), so
    default behavior matches what's already proven on these field images.
    """
    h, w = crop_rgb.shape[:2]
    if max(h, w) > tile_if_larger_than:
        return _predict_mask_tiled(model, crop_rgb, device, tile_size=tile_size, overlap=overlap,
                                    divisor=pad_to_divisor, thr=threshold, blend_method=blend_method)

    x_pad, pads = _pad_to_divisor(_to_tensor01(crop_rgb), pad_to_divisor)
    mask = _predict_mask(model, x_pad, threshold, device)
    return _unpad(mask, pads) if pad_to_divisor else mask
