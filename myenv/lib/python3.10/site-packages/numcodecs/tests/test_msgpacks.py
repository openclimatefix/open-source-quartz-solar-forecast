import unittest

import numpy as np
import pytest

try:
    from numcodecs.msgpacks import MsgPack
except ImportError as e:  # pragma: no cover
    raise unittest.SkipTest("msgpack not available") from e


from numcodecs.tests.common import (
    check_backwards_compatibility,
    check_config,
    check_encode_decode_array,
    check_repr,
    greetings,
)

# object array with strings
# object array with mix strings / nans
# object array with mix of string, int, float
# ...
arrays = [
    np.array(['foo', 'bar', 'baz'] * 300, dtype=object),
    np.array([['foo', 'bar', np.nan]] * 300, dtype=object),
    np.array(['foo', 1.0, 2] * 300, dtype=object),
    np.arange(1000, dtype='i4'),
    np.array(['foo', 'bar', 'baz'] * 300),
    np.array(['foo', ['bar', 1.0, 2], {'a': 'b', 'c': 42}] * 300, dtype=object),
    np.array(greetings * 100),
    np.array(greetings * 100, dtype=object),
    np.array([b'foo', b'bar', b'baz'] * 300, dtype=object),
    np.array([g.encode('utf-8') for g in greetings] * 100, dtype=object),
    np.array([[0, 1], [2, 3]], dtype=object),
]


def test_encode_decode():
    for arr in arrays:
        check_encode_decode_array(arr, MsgPack())


def test_config():
    check_config(MsgPack())


def test_repr():
    check_repr("MsgPack(raw=False, use_bin_type=True, use_single_float=False)")
    check_repr("MsgPack(raw=True, use_bin_type=False, use_single_float=True)")


def test_backwards_compatibility():
    codec = MsgPack()
    check_backwards_compatibility(codec.codec_id, arrays, [codec])


@pytest.mark.parametrize(
    "input_data, dtype",
    [
        ([0, 1], None),
        ([[0, 1], [2, 3]], None),
        ([[0], [1], [2, 3]], object),
        ([[[0, 0]], [[1, 1]], [[2, 3]]], None),
        (["1"], None),
        (["11", "11"], None),
        (["11", "1", "1"], None),
        ([{}], None),
        ([{"key": "value"}, ["list", "of", "strings"]], object),
        ([b"1"], None),
        ([b"11", b"11"], None),
        ([b"11", b"1", b"1"], None),
        ([{b"key": b"value"}, [b"list", b"of", b"strings"]], object),
    ],
)
def test_non_numpy_inputs(input_data, dtype):
    codec = MsgPack()
    # numpy will infer a range of different shapes and dtypes for these inputs.
    # Make sure that round-tripping through encode preserves this.
    actual = codec.decode(codec.encode(input_data))
    expect = np.array(input_data, dtype=dtype)
    assert expect.shape == actual.shape
    assert np.array_equal(expect, actual)


def test_encode_decode_shape_dtype_preserved():
    codec = MsgPack()
    for arr in arrays:
        actual = codec.decode(codec.encode(arr))
        assert arr.shape == actual.shape
        assert arr.dtype == actual.dtype


def test_bytes():
    # test msgpack behaviour with bytes and str (unicode)
    bytes_arr = np.array([b'foo', b'bar', b'baz'], dtype=object)
    unicode_arr = np.array(['foo', 'bar', 'baz'], dtype=object)

    # raw=False (default)
    codec = MsgPack()
    # works for bytes array, round-trips bytes to bytes
    b = codec.decode(codec.encode(bytes_arr))
    assert np.array_equal(bytes_arr, b)
    assert isinstance(b[0], bytes)
    assert b[0] == b'foo'
    # works for unicode array, round-trips unicode to unicode
    b = codec.decode(codec.encode(unicode_arr))
    assert np.array_equal(unicode_arr, b)
    assert isinstance(b[0], str)
    assert b[0] == 'foo'

    # raw=True
    codec = MsgPack(raw=True)
    # works for bytes array, round-trips bytes to bytes
    b = codec.decode(codec.encode(bytes_arr))
    assert np.array_equal(bytes_arr, b)
    assert isinstance(b[0], bytes)
    assert b[0] == b'foo'
    # broken for unicode array, round-trips unicode to bytes
    b = codec.decode(codec.encode(unicode_arr))
    assert not np.array_equal(unicode_arr, b)
    assert isinstance(b[0], bytes)
    assert b[0] == b'foo'
