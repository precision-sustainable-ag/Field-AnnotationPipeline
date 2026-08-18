import numpy as np

from field_annotation.segmentation import clean_disconnected_mask, tight_bbox_from_mask


def _mask_with_blobs(shape, blobs):
    mask = np.zeros(shape, dtype=np.uint8)
    for y0, y1, x0, x1 in blobs:
        mask[y0:y1, x0:x1] = 1
    return mask


def test_clean_disconnected_mask_keeps_nearby_blob():
    mask = _mask_with_blobs((100, 100), [
        (40, 60, 40, 60),  # main blob, 20x20
        (62, 66, 40, 44),  # small blob a few px below main
    ])
    cleaned = clean_disconnected_mask(mask, max_gap_px=10)
    assert cleaned[40:60, 40:60].all()
    assert cleaned[62:66, 40:44].all()


def test_clean_disconnected_mask_drops_far_blob():
    mask = _mask_with_blobs((100, 100), [
        (40, 60, 40, 60),  # main blob
        (0, 4, 0, 4),      # far speckle, opposite corner
    ])
    cleaned = clean_disconnected_mask(mask, max_gap_px=10)
    assert cleaned[40:60, 40:60].all()
    assert not cleaned[0:4, 0:4].any()


def test_clean_disconnected_mask_single_blob_is_unchanged():
    mask = _mask_with_blobs((50, 50), [(10, 20, 10, 20)])
    cleaned = clean_disconnected_mask(mask, max_gap_px=5)
    assert np.array_equal(cleaned, mask)


def test_tight_bbox_from_mask_bounds_foreground_exactly():
    mask = _mask_with_blobs((100, 100), [(20, 30, 40, 55)])  # rows 20-29, cols 40-54
    assert tight_bbox_from_mask(mask) == (40, 20, 55, 30)


def test_tight_bbox_from_mask_spans_multiple_blobs():
    mask = _mask_with_blobs((100, 100), [(10, 15, 10, 15), (80, 85, 90, 95)])
    assert tight_bbox_from_mask(mask) == (10, 10, 95, 85)


def test_tight_bbox_from_mask_returns_none_for_empty_mask():
    mask = np.zeros((50, 50), dtype=np.uint8)
    assert tight_bbox_from_mask(mask) is None
