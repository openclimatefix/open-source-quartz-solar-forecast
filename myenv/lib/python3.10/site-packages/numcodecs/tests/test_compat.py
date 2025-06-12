import array
import mmap

import numpy as np
import pytest

from numcodecs.compat import ensure_bytes, ensure_contiguous_ndarray, ensure_text


def test_ensure_text():
    bufs = [
        b'adsdasdas',
        'adsdasdas',
        np.asarray(memoryview(b'adsdasdas')),
        array.array('B', b'qwertyuiqwertyui'),
    ]
    for buf in bufs:
        b = ensure_text(buf)
        assert isinstance(b, str)


def test_ensure_bytes():
    bufs = [
        b'adsdasdas',
        bytes(20),
        np.arange(100),
        array.array('l', b'qwertyuiqwertyui'),
    ]
    for buf in bufs:
        b = ensure_bytes(buf)
        assert isinstance(b, bytes)


def test_ensure_contiguous_ndarray_shares_memory():
    typed_bufs = [
        ('u', 1, b'adsdasdas'),
        ('u', 1, bytes(20)),
        ('i', 8, np.arange(100, dtype=np.int64)),
        ('f', 8, np.linspace(0, 1, 100, dtype=np.float64)),
        ('i', 4, array.array('i', b'qwertyuiqwertyui')),
        ('u', 4, array.array('I', b'qwertyuiqwertyui')),
        ('f', 4, array.array('f', b'qwertyuiqwertyui')),
        ('f', 8, array.array('d', b'qwertyuiqwertyui')),
        ('i', 1, array.array('b', b'qwertyuiqwertyui')),
        ('u', 1, array.array('B', b'qwertyuiqwertyui')),
        ('u', 1, mmap.mmap(-1, 10)),
    ]
    for expected_kind, expected_itemsize, buf in typed_bufs:
        a = ensure_contiguous_ndarray(buf)
        assert isinstance(a, np.ndarray)
        assert expected_kind == a.dtype.kind
        if isinstance(buf, array.array):
            assert buf.itemsize == a.dtype.itemsize
        else:
            assert expected_itemsize == a.dtype.itemsize
        assert np.shares_memory(a, memoryview(buf))


def test_ensure_bytes_invalid_inputs():
    # object array not allowed
    a = np.array(['Xin chào thế giới'], dtype=object)
    for e in (a, memoryview(a)):
        with pytest.raises(TypeError):
            ensure_bytes(e)


def test_ensure_contiguous_ndarray_invalid_inputs():
    # object array not allowed
    a = np.array(['Xin chào thế giới'], dtype=object)
    for e in (a, memoryview(a)):
        with pytest.raises(TypeError):
            ensure_contiguous_ndarray(e)

    # non-contiguous arrays not allowed
    with pytest.raises(ValueError):
        ensure_contiguous_ndarray(np.arange(100)[::2])

    # unicode array.array not allowed
    a = array.array('u', 'qwertyuiqwertyui')
    with pytest.raises(TypeError):
        ensure_contiguous_ndarray(a)


def test_ensure_contiguous_ndarray_writeable():
    # check that the writeability of the underlying buffer is preserved
    for writeable in (False, True):
        a = np.arange(100)
        a.setflags(write=writeable)
        m = ensure_contiguous_ndarray(a)
        assert m.flags.writeable == writeable
        m = ensure_contiguous_ndarray(memoryview(a))
        assert m.flags.writeable == writeable


def test_ensure_contiguous_ndarray_max_buffer_size():
    for max_buffer_size in (4, 64, 1024):
        ensure_contiguous_ndarray(np.zeros(max_buffer_size - 1, dtype=np.int8), max_buffer_size)
        ensure_contiguous_ndarray(np.zeros(max_buffer_size, dtype=np.int8), max_buffer_size)
        buffers = [
            bytes(b"x" * (max_buffer_size + 1)),
            np.zeros(max_buffer_size + 1, dtype=np.int8),
            np.zeros(max_buffer_size + 2, dtype=np.int8),
            np.zeros(max_buffer_size, dtype=np.int16),
            np.zeros(max_buffer_size, dtype=np.int32),
        ]
        for buf in buffers:
            with pytest.raises(ValueError):
                ensure_contiguous_ndarray(buf, max_buffer_size=max_buffer_size)
