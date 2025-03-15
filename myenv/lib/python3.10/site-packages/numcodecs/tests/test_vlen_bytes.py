import unittest

import numpy as np
import pytest

try:
    from numcodecs.vlen import VLenBytes
except ImportError as e:  # pragma: no cover
    raise unittest.SkipTest("vlen-bytes not available") from e
from numcodecs.tests.common import (
    assert_array_items_equal,
    check_backwards_compatibility,
    check_config,
    check_encode_decode_array,
    check_repr,
    greetings,
)

greetings_bytes = [g.encode('utf-8') for g in greetings]


arrays = [
    np.array([b'foo', b'bar', b'baz'] * 300, dtype=object),
    np.array(greetings_bytes * 100, dtype=object),
    np.array([b'foo', b'bar', b'baz'] * 300, dtype=object).reshape(90, 10),
    np.array(greetings_bytes * 1000, dtype=object).reshape(
        len(greetings_bytes), 100, 10, order='F'
    ),
]


def test_encode_decode():
    for arr in arrays:
        codec = VLenBytes()
        check_encode_decode_array(arr, codec)


def test_config():
    codec = VLenBytes()
    check_config(codec)


def test_repr():
    check_repr("VLenBytes()")


def test_backwards_compatibility():
    check_backwards_compatibility(VLenBytes.codec_id, arrays, [VLenBytes()])


def test_encode_errors():
    codec = VLenBytes()
    with pytest.raises(TypeError):
        codec.encode(1234)
    with pytest.raises(TypeError):
        codec.encode([1234, 5678])
    with pytest.raises(TypeError):
        codec.encode(np.ones(10, dtype='i4'))


def test_decode_errors():
    codec = VLenBytes()
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
    a = np.array([b'foo', None, b'bar'], dtype=object)
    codec = VLenBytes()
    enc = codec.encode(a)
    dec = codec.decode(enc)
    expect = np.array([b'foo', b'', b'bar'], dtype=object)
    assert_array_items_equal(expect, dec)
