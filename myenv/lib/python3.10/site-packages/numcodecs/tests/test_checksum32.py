import itertools

import numpy as np
import pytest

from numcodecs.checksum32 import CRC32, Adler32
from numcodecs.tests.common import (
    check_backwards_compatibility,
    check_config,
    check_encode_decode,
    check_err_encode_object_buffer,
    check_repr,
)

# mix of dtypes: integer, float, bool, string
# mix of shapes: 1D, 2D, 3D
# mix of orders: C, F
arrays = [
    np.arange(1000, dtype='i4'),
    np.linspace(1000, 1001, 1000, dtype='f8'),
    np.random.normal(loc=1000, scale=1, size=(100, 10)),
    np.random.randint(0, 2, size=1000, dtype=bool).reshape(100, 10, order='F'),
    np.random.choice([b'a', b'bb', b'ccc'], size=1000).reshape(10, 10, 10),
]

codecs = [CRC32(), Adler32()]


def test_encode_decode():
    for codec, arr in itertools.product(codecs, arrays):
        check_encode_decode(arr, codec)


def test_errors():
    for codec, arr in itertools.product(codecs, arrays):
        enc = codec.encode(arr)
        with pytest.raises(RuntimeError):
            codec.decode(enc[:-1])


def test_config():
    for codec in codecs:
        check_config(codec)


def test_repr():
    check_repr("CRC32()")
    check_repr("Adler32()")


def test_backwards_compatibility():
    check_backwards_compatibility(CRC32.codec_id, arrays, [CRC32()])
    check_backwards_compatibility(Adler32.codec_id, arrays, [Adler32()])


def test_err_encode_object_buffer():
    check_err_encode_object_buffer(CRC32())
    check_err_encode_object_buffer(Adler32())
