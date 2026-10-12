import numpy as np
import pytest

from wood_stitch.stitch import to_uint8


def test_uint8_passes_through_unchanged():
    img = np.array([[[0, 128, 255]]], dtype=np.uint8)
    out = to_uint8(img)
    assert out.dtype == np.uint8
    assert np.array_equal(out, img)


def test_uint16_12bit_uses_4bit_right_shift():
    img = np.array([[[0, 16, 4095]]], dtype=np.uint16)
    out = to_uint8(img)
    assert out.dtype == np.uint8
    assert out.tolist() == [[[0, 1, 255]]]


def test_uint16_above_12bit_range_is_rejected():
    img = np.array([[[0, 4096, 65535]]], dtype=np.uint16)
    with pytest.raises(ValueError):
        to_uint8(img)


def test_other_dtypes_are_rejected():
    img = np.zeros((2, 2, 3), dtype=np.float32)
    with pytest.raises((ValueError, TypeError)):
        to_uint8(img)
