import numpy as np

from numcodecs.packbits import PackBits
from numcodecs.tests.common import (
    check_backwards_compatibility,
    check_config,
    check_encode_decode,
    check_repr,
)

arrays = [
    np.random.randint(0, 2, size=1000, dtype=bool),
    np.random.randint(0, 2, size=(100, 10), dtype=bool),
    np.random.randint(0, 2, size=(10, 10, 10), dtype=bool),
    np.random.randint(0, 2, size=1000, dtype=bool).reshape(10, 10, 10, order='F'),
]


def test_encode_decode():
    codec = PackBits()
    for arr in arrays:
        check_encode_decode(arr, codec)
    # check different number of left-over bits
    arr = np.random.randint(0, 2, size=1000, dtype=bool)
    for size in list(range(1, 17)):
        check_encode_decode(arr[:size], codec)


def test_config():
    codec = PackBits()
    check_config(codec)


def test_repr():
    check_repr("PackBits()")


def test_backwards_compatibility():
    check_backwards_compatibility(PackBits.codec_id, arrays, [PackBits()])
