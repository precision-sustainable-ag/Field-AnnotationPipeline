from __future__ import annotations

from pathlib import Path
from typing import Optional

from ultralytics import YOLO


class WeedDetector:
    """Single-class bbox detector. Ported from
    Field-SegmentationTraining/src/mask_gen_utils/detect.py::WeedDetector --
    always reduces multiple detections to the single highest-confidence box.
    """

    def __init__(self, yolo_model_path: Path, conf_threshold: float = 0.25, device: str = "cpu"):
        self.model = YOLO(str(yolo_model_path))
        self.conf_threshold = conf_threshold
        self.device = device

    def detect(self, image_path: Path) -> Optional[dict]:
        results = self.model(str(image_path), conf=self.conf_threshold, device=self.device, verbose=False)
        if not results or not results[0].boxes.xyxy.tolist():
            return None

        boxes = results[0].boxes.xyxy.tolist()
        confs = results[0].boxes.conf.tolist()
        max_conf_idx = confs.index(max(confs)) if len(boxes) > 1 else 0

        x_min, y_min, x_max, y_max = map(round, boxes[max_conf_idx])
        bbox = [x_min, y_min, x_max - x_min, y_max - y_min]  # xywh
        det_pred_conf = round(confs[max_conf_idx], 6)
        return {"bbox": bbox, "det_pred_conf": det_pred_conf}


def pad_bbox(bbox_xywh: tuple[int, int, int, int], pad_px: int, image_width: int, image_height: int) -> list[int]:
    """Expands an xywh bbox by pad_px on every side, clamped to image bounds.
    Ported from Field-SegmentationTraining/src/inference_utils/inference_pipeline.py's
    detection-padding step in _detect_roi_if_enabled -- there it pads the raw
    xyxy box before returning it; here the same clamped expansion is applied
    to the xywh box WeedDetector.detect returns.
    """
    x, y, w, h = bbox_xywh
    x1 = max(0, x - pad_px)
    y1 = max(0, y - pad_px)
    x2 = min(image_width, x + w + pad_px)
    y2 = min(image_height, y + h + pad_px)
    return [x1, y1, x2 - x1, y2 - y1]
