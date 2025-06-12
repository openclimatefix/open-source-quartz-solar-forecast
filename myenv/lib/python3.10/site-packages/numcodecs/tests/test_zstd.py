import itertools

import numpy as np
import pytest

try:
    from numcodecs.zstd import Zstd
except ImportError:  # pragma: no cover
    pytest.skip("numcodecs.zstd not available", allow_module_level=True)


from numcodecs.tests.common import (
    check_backwards_compatibility,
    check_config,
    check_encode_decode,
    check_err_decode_object_buffer,
    check_err_encode_object_buffer,
    check_repr,
)

codecs = [
    Zstd(),
    Zstd(level=-1),
    Zstd(level=0),
    Zstd(level=1),
    Zstd(level=10),
    Zstd(level=22),
    Zstd(level=100),
    Zstd(checksum=True),
    Zstd(level=0, checksum=True),
    Zstd(level=22, checksum=True),
]


# mix of dtypes: integer, float, bool, string
# mix of shapes: 1D, 2D, 3D
# mix of orders: C, F
arrays = [
    np.arange(1000, dtype="i4"),
    np.linspace(1000, 1001, 1000, dtype="f8"),
    np.random.normal(loc=1000, scale=1, size=(100, 10)),
    np.random.randint(0, 2, size=1000, dtype=bool).reshape(100, 10, order='F'),
    np.random.choice([b'a', b'bb', b'ccc'], size=1000).reshape(10, 10, 10),
    np.random.randint(0, 2**60, size=1000, dtype='u8').view('M8[ns]'),
    np.random.randint(0, 2**60, size=1000, dtype='u8').view('m8[ns]'),
    np.random.randint(0, 2**25, size=1000, dtype='u8').view('M8[m]'),
    np.random.randint(0, 2**25, size=1000, dtype='u8').view('m8[m]'),
    np.random.randint(-(2**63), -(2**63) + 20, size=1000, dtype='i8').view('M8[ns]'),
    np.random.randint(-(2**63), -(2**63) + 20, size=1000, dtype='i8').view('m8[ns]'),
    np.random.randint(-(2**63), -(2**63) + 20, size=1000, dtype='i8').view('M8[m]'),
    np.random.randint(-(2**63), -(2**63) + 20, size=1000, dtype='i8').view('m8[m]'),
]


def test_encode_decode():
    for arr, codec in itertools.product(arrays, codecs):
        check_encode_decode(arr, codec)


def test_config():
    for codec in codecs:
        check_config(codec)


def test_repr():
    check_repr("Zstd(level=3)")


def test_backwards_compatibility():
    check_backwards_compatibility(Zstd.codec_id, arrays, codecs)


def test_err_decode_object_buffer():
    check_err_decode_object_buffer(Zstd())


def test_err_encode_object_buffer():
    check_err_encode_object_buffer(Zstd())


def test_checksum():
    data = np.arange(0, 64, dtype="uint8")
    assert len(Zstd(level=0, checksum=False).encode(data)) + 4 == len(
        Zstd(level=0, checksum=True).encode(data)
    )


def test_native_functions():
    # Note, these assertions might need to be changed for new versions of zstd
    assert Zstd.default_level() == 3
    assert Zstd.min_level() == -131072
    assert Zstd.max_level() == 22
