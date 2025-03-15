import itertools

import numpy as np
import pytest

from numcodecs.base64 import Base64
from numcodecs.tests.common import (
    check_backwards_compatibility,
    check_encode_decode,
    check_err_decode_object_buffer,
    check_err_encode_object_buffer,
    check_repr,
)

codecs = [
    Base64(),
]


# mix of dtypes: integer, float, bool, string
# mix of shapes: 1D, 2D, 3D
# mix of orders: C, F
arrays = [
    np.arange(1000, dtype="i4"),
    np.linspace(1000, 1001, 1000, dtype="f8"),
    np.random.normal(loc=1000, scale=1, size=(100, 10)),
    np.random.randint(0, 2, size=1000, dtype=bool).reshape(100, 10, order="F"),
    np.random.choice([b"a", b"bb", b"ccc"], size=1000).reshape(10, 10, 10),
    np.random.randint(0, 2**60, size=1000, dtype="u8").view("M8[ns]"),
    np.random.randint(0, 2**60, size=1000, dtype="u8").view("m8[ns]"),
    np.random.randint(0, 2**25, size=1000, dtype="u8").view("M8[m]"),
    np.random.randint(0, 2**25, size=1000, dtype="u8").view("m8[m]"),
    np.random.randint(-(2**63), -(2**63) + 20, size=1000, dtype="i8").view("M8[ns]"),
    np.random.randint(-(2**63), -(2**63) + 20, size=1000, dtype="i8").view("m8[ns]"),
    np.random.randint(-(2**63), -(2**63) + 20, size=1000, dtype="i8").view("M8[m]"),
    np.random.randint(-(2**63), -(2**63) + 20, size=1000, dtype="i8").view("m8[m]"),
]


def test_encode_decode():
    for arr, codec in itertools.product(arrays, codecs):
        check_encode_decode(arr, codec)


def test_repr():
    check_repr("Base64()")


def test_eq():
    assert Base64() == Base64()
    assert not Base64() != Base64()
    assert Base64() != "foo"
    assert "foo" != Base64()
    assert not Base64() == "foo"


def test_backwards_compatibility():
    check_backwards_compatibility(Base64.codec_id, arrays, codecs)


def test_err_decode_object_buffer():
    check_err_decode_object_buffer(Base64())


def test_err_encode_object_buffer():
    check_err_encode_object_buffer(Base64())


def test_err_encode_list():
    data = ["foo", "bar", "baz"]
    for codec in codecs:
        with pytest.raises(TypeError):
            codec.encode(data)


def test_err_encode_non_contiguous():
    # non-contiguous memory
    arr = np.arange(1000, dtype="i4")[::2]
    for codec in codecs:
        with pytest.raises(ValueError):
            codec.encode(arr)
