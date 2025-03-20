import numpy as np
import pytest

from numcodecs.pcodec import PCodec

try:
    # initializing codec triggers ImportError
    PCodec()
except ImportError:  # pragma: no cover
    pytest.skip("pcodec not available", allow_module_level=True)

from numcodecs.tests.common import (
    check_backwards_compatibility,
    check_config,
    check_encode_decode_array_to_bytes,
    check_err_decode_object_buffer,
    check_err_encode_object_buffer,
    check_repr,
)

codecs = [
    PCodec(),
    PCodec(level=1),
    PCodec(level=5),
    PCodec(level=9),
    PCodec(mode_spec='classic'),
    PCodec(equal_pages_up_to=300),
]


# mix of dtypes: integer, float
# mix of shapes: 1D, 2D
# mix of orders: C, F
arrays = [
    np.arange(1000, dtype="u4"),
    np.arange(1000, dtype="u8"),
    np.arange(1000, dtype="i4"),
    np.arange(1000, dtype="i8"),
    np.linspace(1000, 1001, 1000, dtype="f4"),
    np.linspace(1000, 1001, 1000, dtype="f8"),
    np.random.normal(loc=1000, scale=1, size=(100, 10)),
    np.asfortranarray(np.random.normal(loc=1000, scale=1, size=(100, 10))),
    np.random.randint(0, 2**60, size=1000, dtype="u8"),
    np.random.randint(-(2**63), -(2**63) + 20, size=1000, dtype="i8"),
]


@pytest.mark.parametrize("arr", arrays)
@pytest.mark.parametrize("codec", codecs)
def test_encode_decode(arr, codec):
    check_encode_decode_array_to_bytes(arr, codec)


def test_config():
    codec = PCodec(level=3)
    check_config(codec)


def test_invalid_config_error():
    with pytest.raises(ValueError):
        codec = PCodec(mode_spec='bogus')
        check_encode_decode_array_to_bytes(arrays[0], codec)


def test_repr():
    check_repr(
        "PCodec(delta_encoding_order=None, equal_pages_up_to=262144, level=3, mode_spec='auto')"
    )


def test_backwards_compatibility():
    check_backwards_compatibility(PCodec.codec_id, arrays, codecs)


def test_err_decode_object_buffer():
    check_err_decode_object_buffer(PCodec())


def test_err_encode_object_buffer():
    check_err_encode_object_buffer(PCodec())
