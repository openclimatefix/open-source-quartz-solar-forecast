import unittest

import numpy as np
import pytest

try:
    from numcodecs.vlen import VLenUTF8
except ImportError as e:  # pragma: no cover
    raise unittest.SkipTest("vlen-utf8 not available") from e
from numcodecs.tests.common import (
    assert_array_items_equal,
    check_backwards_compatibility,
    check_config,
    check_encode_decode_array,
    check_repr,
    greetings,
)

arrays = [
    np.array(['foo', 'bar', 'baz'] * 300, dtype=object),
    np.array(greetings * 100, dtype=object),
    np.array(['foo', 'bar', 'baz'] * 300, dtype=object).reshape(90, 10),
    np.array(greetings * 1000, dtype=object).reshape(len(greetings), 100, 10, order='F'),
]


def test_encode_decode():
    for arr in arrays:
        codec = VLenUTF8()
        check_encode_decode_array(arr, codec)


def test_config():
    codec = VLenUTF8()
    check_config(codec)


def test_repr():
    check_repr("VLenUTF8()")


def test_backwards_compatibility():
    check_backwards_compatibility(VLenUTF8.codec_id, arrays, [VLenUTF8()])


def test_encode_errors():
    codec = VLenUTF8()
    with pytest.raises(TypeError):
        codec.encode(1234)
    with pytest.raises(TypeError):
        codec.encode([1234, 5678])
    with pytest.raises(TypeError):
        codec.encode(np.ones(10, dtype='i4'))


def test_decode_errors():
    codec = VLenUTF8()
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


@pytest.mark.parametrize("writable", [True, False])
def test_encode_utf8(writable):
    a = np.array(['foo', None, 'bar'], dtype=object)
    if not writable:
        a.setflags(write=False)
    codec = VLenUTF8()
    enc = codec.encode(a)
    dec = codec.decode(enc)
    expect = np.array(['foo', '', 'bar'], dtype=object)
    assert_array_items_equal(expect, dec)
