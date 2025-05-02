import array
import json as _json
import os
from glob import glob

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal

# star import needed for repr tests so eval finds names
from numcodecs import *  # noqa: F403
from numcodecs.compat import ensure_bytes, ensure_ndarray
from numcodecs.registry import get_codec

greetings = [
    '¡Hola mundo!',
    'Hej Världen!',
    'Servus Woid!',
    'Hei maailma!',
    'Xin chào thế giới',
    'Njatjeta Botë!',
    'Γεια σου κόσμε!',
    'こんにちは世界',
    '世界，你好！',
    'Helló, világ!',
    'Zdravo svete!',
    'เฮลโลเวิลด์',
]


def compare_arrays(arr, res, precision=None):
    # ensure numpy array with matching dtype
    res = ensure_ndarray(res).view(arr.dtype)

    # convert to correct shape
    if arr.flags.f_contiguous:
        order = 'F'
    else:
        order = 'C'
    res = res.reshape(arr.shape, order=order)

    # exact compare
    if precision is None:
        assert_array_equal(arr, res)

    # fuzzy compare
    else:
        assert_array_almost_equal(arr, res, decimal=precision)


def check_encode_decode(arr, codec, precision=None):
    # N.B., watch out here with blosc compressor, if the itemsize of
    # the source buffer is different then the results of encoding
    # (i.e., compression) may be different. Hence we *do not* require that
    # the results of encoding be identical for all possible inputs, rather
    # we just require that the results of the encode/decode round-trip can
    # be compared to the original array.

    # encoding should support any object exporting the buffer protocol

    # test encoding of numpy array
    enc = codec.encode(arr)
    dec = codec.decode(enc)
    compare_arrays(arr, dec, precision=precision)

    # test encoding of bytes
    buf = arr.tobytes(order='A')
    enc = codec.encode(buf)
    dec = codec.decode(enc)
    compare_arrays(arr, dec, precision=precision)

    # test encoding of bytearray
    buf = bytearray(arr.tobytes(order='A'))
    enc = codec.encode(buf)
    dec = codec.decode(enc)
    compare_arrays(arr, dec, precision=precision)

    # test encoding of array.array
    buf = array.array('b', arr.tobytes(order='A'))
    enc = codec.encode(buf)
    dec = codec.decode(enc)
    compare_arrays(arr, dec, precision=precision)

    # decoding should support any object exporting the buffer protocol,

    # setup
    enc_bytes = ensure_bytes(enc)

    # test decoding of raw bytes
    dec = codec.decode(enc_bytes)
    compare_arrays(arr, dec, precision=precision)

    # test decoding of bytearray
    dec = codec.decode(bytearray(enc_bytes))
    compare_arrays(arr, dec, precision=precision)

    # test decoding of array.array
    buf = array.array('b', enc_bytes)
    dec = codec.decode(buf)
    compare_arrays(arr, dec, precision=precision)

    # test decoding of numpy array
    buf = np.frombuffer(enc_bytes, dtype='u1')
    dec = codec.decode(buf)
    compare_arrays(arr, dec, precision=precision)

    # test decoding directly into numpy array
    out = np.empty_like(arr)
    codec.decode(enc_bytes, out=out)
    compare_arrays(arr, out, precision=precision)

    # test decoding directly into bytearray
    out = bytearray(arr.nbytes)
    codec.decode(enc_bytes, out=out)
    # noinspection PyTypeChecker
    compare_arrays(arr, out, precision=precision)


def check_encode_decode_partial(arr, codec, precision=None):
    # N.B., watch out here with blosc compressor, if the itemsize of
    # the source buffer is different then the results of encoding
    # (i.e., compression) may be different. Hence we *do not* require that
    # the results of encoding be identical for all possible inputs, rather
    # we just require that the results of the encode/decode round-trip can
    # be compared to the original array.

    itemsize = arr.itemsize
    start, nitems = 5, 10
    compare_arr = arr[start : start + nitems]
    # test encoding of numpy array
    enc = codec.encode(arr)
    dec = codec.decode_partial(enc, start, nitems)
    compare_arrays(compare_arr, dec, precision=precision)

    # out = np.empty_like(compare_arr)
    out = np.empty_like(compare_arr)
    print(len(out))

    # test partial decode of encoded bytes
    buf = arr.tobytes(order='A')
    enc = codec.encode(buf)
    dec = codec.decode_partial(enc, start * itemsize, nitems * itemsize, out=out)
    compare_arrays(compare_arr, dec, precision=precision)

    # test partial decode of encoded bytearray
    buf = bytearray(arr.tobytes(order='A'))
    enc = codec.encode(buf)
    dec = codec.decode_partial(enc, start * itemsize, nitems * itemsize, out=out)
    compare_arrays(compare_arr, dec, precision=precision)

    # test partial decode of encoded array.array
    buf = array.array('b', arr.tobytes(order='A'))
    enc = codec.encode(buf)
    dec = codec.decode_partial(enc, start * itemsize, nitems * itemsize, out=out)
    compare_arrays(compare_arr, dec, precision=precision)

    # # decoding should support any object exporting the buffer protocol,

    # # setup
    enc_bytes = ensure_bytes(enc)

    # test decoding of raw bytes into numpy array
    dec = codec.decode_partial(enc_bytes, start * itemsize, nitems * itemsize, out=out)
    compare_arrays(compare_arr, dec, precision=precision)

    # test partial decoding of bytearray
    dec = codec.decode_partial(bytearray(enc_bytes), start * itemsize, nitems * itemsize, out=out)
    compare_arrays(compare_arr, dec, precision=precision)

    # test partial decoding of array.array
    buf = array.array('b', enc_bytes)
    dec = codec.decode_partial(buf, start * itemsize, nitems * itemsize, out=out)
    compare_arrays(compare_arr, dec, precision=precision)

    # test decoding of numpy array into numpy array
    buf = np.frombuffer(enc_bytes, dtype='u1')
    dec = codec.decode_partial(buf, start * itemsize, nitems * itemsize, out=out)
    compare_arrays(compare_arr, dec, precision=precision)

    # test decoding directly into bytearray
    out = bytearray(compare_arr.nbytes)
    codec.decode_partial(enc_bytes, start * itemsize, nitems * itemsize, out=out)
    # noinspection PyTypeChecker
    compare_arrays(compare_arr, out, precision=precision)


def assert_array_items_equal(res, arr):
    assert isinstance(res, np.ndarray)
    res = res.reshape(-1, order='A')
    arr = arr.reshape(-1, order='A')
    assert res.shape == arr.shape
    assert res.dtype == arr.dtype

    # numpy asserts don't compare object arrays
    # properly; assert that we have the same nans
    # and values
    arr = arr.ravel().tolist()
    res = res.ravel().tolist()
    for a, r in zip(arr, res, strict=True):
        if isinstance(a, np.ndarray):
            assert_array_equal(a, r)
        elif a != a:
            assert r != r
        else:
            assert a == r


def check_encode_decode_array(arr, codec):
    enc = codec.encode(arr)
    dec = codec.decode(enc)
    assert_array_items_equal(arr, dec)

    out = np.empty_like(arr)
    codec.decode(enc, out=out)
    assert_array_items_equal(arr, out)

    enc = codec.encode(arr)
    dec = codec.decode(ensure_ndarray(enc))
    assert_array_items_equal(arr, dec)


def check_encode_decode_array_to_bytes(arr, codec):
    enc = codec.encode(arr)
    dec = codec.decode(enc)
    assert_array_items_equal(arr, dec)

    out = np.empty_like(arr)
    codec.decode(enc, out=out)
    assert_array_items_equal(arr, out)


def check_config(codec):
    config = codec.get_config()
    # round-trip through JSON to check serialization
    config = _json.loads(_json.dumps(config))
    assert codec == get_codec(config)


def check_repr(stmt):
    # check repr matches instantiation statement
    codec = eval(stmt)
    actual = repr(codec)
    assert stmt == actual


def check_backwards_compatibility(codec_id, arrays, codecs, precision=None, prefix=None):
    # setup directory to hold data fixture
    if prefix:
        fixture_dir = os.path.join('fixture', codec_id, prefix)
    else:
        fixture_dir = os.path.join('fixture', codec_id)
    if not os.path.exists(fixture_dir):  # pragma: no cover
        os.makedirs(fixture_dir)

    # save fixture data
    for i, arr in enumerate(arrays):
        arr_fn = os.path.join(fixture_dir, f'array.{i:02d}.npy')
        if not os.path.exists(arr_fn):  # pragma: no cover
            np.save(arr_fn, arr)

    # load fixture data
    for arr_fn in glob(os.path.join(fixture_dir, 'array.*.npy')):
        # setup
        i = int(arr_fn.split('.')[-2])
        arr = np.load(arr_fn, allow_pickle=True)
        arr_bytes = arr.tobytes(order='A')
        if arr.flags.f_contiguous:
            order = 'F'
        else:
            order = 'C'

        for j, codec in enumerate(codecs):
            if codec is None:
                pytest.skip("codec has been removed")

            # setup a directory to hold encoded data
            codec_dir = os.path.join(fixture_dir, f'codec.{j:02d}')
            if not os.path.exists(codec_dir):  # pragma: no cover
                os.makedirs(codec_dir)

            # file with codec configuration information
            codec_fn = os.path.join(codec_dir, 'config.json')
            # one time save config
            if not os.path.exists(codec_fn):  # pragma: no cover
                with open(codec_fn, mode='w') as cf:
                    _json.dump(codec.get_config(), cf, sort_keys=True, indent=4)
            # load config and compare with expectation
            with open(codec_fn) as cf:
                config = _json.load(cf)
                assert codec == get_codec(config)

            enc_fn = os.path.join(codec_dir, f'encoded.{i:02d}.dat')

            # one time encode and save array
            if not os.path.exists(enc_fn):  # pragma: no cover
                enc = codec.encode(arr)
                enc = ensure_bytes(enc)
                with open(enc_fn, mode='wb') as ef:
                    ef.write(enc)

            # load and decode data
            with open(enc_fn, mode='rb') as ef:
                enc = ef.read()
                dec = codec.decode(enc)
                dec_arr = ensure_ndarray(dec).reshape(-1, order='A')
                dec_arr = dec_arr.view(dtype=arr.dtype).reshape(arr.shape, order=order)
                if precision and precision[j] is not None:
                    assert_array_almost_equal(arr, dec_arr, decimal=precision[j])
                elif arr.dtype == 'object':
                    assert_array_items_equal(arr, dec_arr)
                else:
                    assert_array_equal(arr, dec_arr)
                    assert arr_bytes == ensure_bytes(dec)


def check_err_decode_object_buffer(compressor):
    # cannot decode directly into object array, leads to segfaults
    a = np.arange(10)
    enc = compressor.encode(a)
    out = np.empty(10, dtype=object)
    with pytest.raises(TypeError):
        compressor.decode(enc, out=out)


def check_err_encode_object_buffer(compressor):
    # compressors cannot encode object array
    a = np.array(['foo', 'bar', 'baz'], dtype=object)
    with pytest.raises(TypeError):
        compressor.encode(a)


def check_max_buffer_size(codec):
    for max_buffer_size in (4, 64, 1024):
        old_max_buffer_size = codec.max_buffer_size
        try:
            codec.max_buffer_size = max_buffer_size
            # Just up the max_buffer_size is fine.
            codec.encode(np.zeros(max_buffer_size - 1, dtype=np.int8))
            codec.encode(np.zeros(max_buffer_size, dtype=np.int8))

            buffers = [
                bytes(b"x" * (max_buffer_size + 1)),
                np.zeros(max_buffer_size + 1, dtype=np.int8),
                np.zeros(max_buffer_size + 2, dtype=np.int8),
                np.zeros(max_buffer_size, dtype=np.int16),
                np.zeros(max_buffer_size, dtype=np.int32),
            ]
            for buf in buffers:
                with pytest.raises(ValueError):
                    codec.encode(buf)
                with pytest.raises(ValueError):
                    codec.decode(buf)
        finally:
            codec.max_buffer_size = old_max_buffer_size
