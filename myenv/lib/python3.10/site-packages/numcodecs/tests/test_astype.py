import numpy as np
from numpy.testing import assert_array_equal

from numcodecs.astype import AsType
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
    np.arange(1000, dtype='i4'),
    np.linspace(1000, 1001, 1000, dtype='f8').reshape(100, 10),
    np.random.normal(loc=1000, scale=1, size=(10, 10, 10)),
    np.random.randint(0, 200, size=1000, dtype='u2').reshape(100, 10, order='F'),
]


def test_encode_decode():
    for arr in arrays:
        codec = AsType(encode_dtype=arr.dtype, decode_dtype=arr.dtype)
        check_encode_decode(arr, codec)


def test_decode():
    encode_dtype, decode_dtype = '<i4', '<i8'
    codec = AsType(encode_dtype=encode_dtype, decode_dtype=decode_dtype)
    arr = np.arange(10, 20, 1, dtype=encode_dtype)
    expect = arr.astype(decode_dtype)
    actual = codec.decode(arr)
    assert_array_equal(expect, actual)
    assert np.dtype(decode_dtype) == actual.dtype


def test_encode():
    encode_dtype, decode_dtype = '<i4', '<i8'
    codec = AsType(encode_dtype=encode_dtype, decode_dtype=decode_dtype)
    arr = np.arange(10, 20, 1, dtype=decode_dtype)
    expect = arr.astype(encode_dtype)
    actual = codec.encode(arr)
    assert_array_equal(expect, actual)
    assert np.dtype(encode_dtype) == actual.dtype


def test_config():
    encode_dtype, decode_dtype = '<i4', '<i8'
    codec = AsType(encode_dtype=encode_dtype, decode_dtype=decode_dtype)
    check_config(codec)


def test_repr():
    check_repr("AsType(encode_dtype='<i4', decode_dtype='<i2')")


def test_backwards_compatibility():
    # integers
    arrs = [
        np.arange(1000, dtype='<i4'),
        np.random.randint(0, 200, size=1000, dtype='i4').astype('<i4').reshape(100, 10, order='F'),
    ]
    codec = AsType(encode_dtype='<i2', decode_dtype='<i4')
    check_backwards_compatibility(AsType.codec_id, arrs, [codec], prefix='i')

    # floats
    arrs = [
        np.linspace(1000, 1001, 1000, dtype='<f8').reshape(100, 10, order='F'),
        np.random.normal(loc=1000, scale=1, size=(10, 10, 10)).astype('<f8'),
    ]
    codec = AsType(encode_dtype='<f4', decode_dtype='<f8')
    check_backwards_compatibility(AsType.codec_id, arrs, [codec], precision=[3], prefix='f')
