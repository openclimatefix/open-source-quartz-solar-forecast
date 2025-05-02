import unittest

import numpy as np
import pytest

try:
    from numcodecs.vlen import VLenArray
except ImportError as e:  # pragma: no cover
    raise unittest.SkipTest("vlen-array not available") from e
from numcodecs.tests.common import (
    assert_array_items_equal,
    check_backwards_compatibility,
    check_config,
    check_encode_decode_array,
    check_repr,
)

arrays = [
    np.array([np.array([1, 2, 3]), np.array([4]), np.array([5, 6])] * 300, dtype=object),
    np.array([np.array([1, 2, 3]), np.array([4]), np.array([5, 6])] * 300, dtype=object).reshape(
        90, 10
    ),
]


codecs = [
    VLenArray('<i1'),
    VLenArray('<i2'),
    VLenArray('<i4'),
    VLenArray('<i8'),
    VLenArray('<u1'),
    VLenArray('<u2'),
    VLenArray('<u4'),
    VLenArray('<u8'),
]


def test_encode_decode():
    for arr in arrays:
        for codec in codecs:
            check_encode_decode_array(arr, codec)


def test_config():
    codec = VLenArray('<i8')
    check_config(codec)


def test_repr():
    check_repr("VLenArray(dtype='<i8')")


def test_backwards_compatibility():
    check_backwards_compatibility(VLenArray.codec_id, arrays, codecs)


def test_encode_errors():
    codec = VLenArray('<i8')
    with pytest.raises(ValueError):
        codec.encode('foo')
    with pytest.raises(ValueError):
        codec.encode(['foo', 'bar'])


def test_decode_errors():
    codec = VLenArray('<i8')
    with pytest.raises(TypeError):
        codec.decode(1234)
    # these should look like corrupt data
    with pytest.raises(ValueError):
        codec.decode(b'foo')
    with pytest.raises(ValueError):
        codec.decode(np.arange(2, 3, dtype='i4'))
    with pytest.raises(ValueError):
        codec.decode(np.arange(10, 20, dtype='i4'))
    with pytest.raises(TypeError):
        codec.decode('foo')

    # test out parameter
    enc = codec.encode(arrays[0])
    with pytest.raises(TypeError):
        codec.decode(enc, out=b'foo')
    with pytest.raises(TypeError):
        codec.decode(enc, out='foo')
    with pytest.raises(TypeError):
        codec.decode(enc, out=123)
    with pytest.raises(ValueError):
        codec.decode(enc, out=np.zeros(10, dtype='i4'))


def test_encode_none():
    a = np.array([[1, 3], None, [4, 7]], dtype=object)
    codec = VLenArray(int)
    enc = codec.encode(a)
    dec = codec.decode(enc)
    expect = np.array([np.array([1, 3]), np.array([]), np.array([4, 7])], dtype=object)
    assert_array_items_equal(expect, dec)
