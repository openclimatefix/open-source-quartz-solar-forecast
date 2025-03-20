import numpy as np
import pytest
from numpy.testing import assert_array_equal

from numcodecs.delta import Delta
from numcodecs.tests.common import (
    check_backwards_compatibility,
    check_config,
    check_encode_decode,
    check_repr,
)

# mix of dtypes: integer, float
# mix of shapes: 1D, 2D, 3D
# mix of orders: C, F
arrays = [
    np.arange(1000, dtype='<i4'),
    np.linspace(1000, 1001, 1000, dtype='<f4').reshape(100, 10),
    np.random.normal(loc=1000, scale=1, size=(10, 10, 10)).astype('<f8'),
    np.random.randint(0, 200, size=1000, dtype='u2').astype('<u2').reshape(100, 10, order='F'),
]


def test_encode_decode():
    for arr in arrays:
        codec = Delta(dtype=arr.dtype)
        check_encode_decode(arr, codec)


def test_encode():
    dtype = 'i8'
    astype = 'i4'
    codec = Delta(dtype=dtype, astype=astype)
    arr = np.arange(10, 20, 1, dtype=dtype)
    expect = np.array([10] + ([1] * 9), dtype=astype)
    actual = codec.encode(arr)
    assert_array_equal(expect, actual)
    assert np.dtype(astype) == actual.dtype


def test_config():
    codec = Delta(dtype='<i4', astype='<i2')
    check_config(codec)


def test_repr():
    check_repr("Delta(dtype='<i4', astype='<i2')")


def test_backwards_compatibility():
    for arr in arrays:
        codec = Delta(dtype=arr.dtype)
        check_backwards_compatibility(Delta.codec_id, [arr], [codec], prefix=str(arr.dtype))


def test_errors():
    with pytest.raises(ValueError):
        Delta(dtype=object)
    with pytest.raises(ValueError):
        Delta(dtype='i8', astype=object)
