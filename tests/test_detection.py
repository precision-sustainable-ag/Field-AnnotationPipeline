from field_annotation.detection import pad_bbox


def test_pad_bbox_expands_on_every_side():
    assert pad_bbox((100, 100, 50, 50), pad_px=10, image_width=1000, image_height=1000) == [90, 90, 70, 70]


def test_pad_bbox_clamps_to_image_bounds():
    # Box near the top-left and bottom-right corners of a small image --
    # padding should clip rather than go negative or past the edge.
    assert pad_bbox((5, 5, 20, 20), pad_px=10, image_width=30, image_height=30) == [0, 0, 30, 30]


def test_pad_bbox_zero_padding_is_a_no_op():
    assert pad_bbox((10, 20, 30, 40), pad_px=0, image_width=1000, image_height=1000) == [10, 20, 30, 40]
