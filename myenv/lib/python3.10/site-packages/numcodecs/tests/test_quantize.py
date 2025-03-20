import itertools

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal

from numcodecs.quantize import Quantize
from numcodecs.tests.common import (
    check_backwards_compatibility,
    check_config,
    check_encode_decode,
    check_repr,
)

arrays = [
    np.linspace(100, 200, 1000, dtype='<f8'),
    np.random.normal(loc=0, scale=1, size=1000).astype('<f8'),
    np.linspace(100, 200, 1000, dtype='<f8').reshape(100, 10),
    np.linspace(100, 200, 1000, dtype='<f8').reshape(100, 10, order='F'),
    np.linspace(100, 200, 1000, dtype='<f8').reshape(10, 10, 10),
]


codecs = [
    Quantize(digits=-1, dtype='<f8', astype='<f2'),
    Quantize(digits=0, dtype='<f8', astype='<f2'),
    Quantize(digits=1, dtype='<f8', astype='<f2'),
    Quantize(digits=5, dtype='<f8', astype='<f4'),
    Quantize(digits=12, dtype='<f8', astype='<f8'),
]


def test_encode_decode():
    for arr, codec in itertools.product(arrays, codecs):
        check_encode_decode(arr, codec, precision=codec.digits)


def test_encode():
    for arr, codec in itertools.product(arrays, codecs):
        if arr.flags.f_contiguous:
            order = 'F'
        else:
            order = 'C'
        enc = codec.encode(arr).reshape(arr.shape, order=order)
        assert_array_almost_equal(arr, enc, decimal=codec.digits)


def test_decode():
    # decode is a no-op
    for arr, codec in itertools.product(arrays, codecs):
        enc = codec.encode(arr)
        dec = codec.decode(enc)
        assert_array_equal(enc, dec)


def test_config():
    for codec in codecs:
        check_config(codec)


def test_repr():
    check_repr("Quantize(digits=2, dtype='<f8', astype='<f2')")


def test_errors():
    with pytest.raises(ValueError):
        Quantize(digits=2, dtype='i4')
    with pytest.raises(ValueError):
        Quantize(digits=2, dtype=object)
    with pytest.raises(ValueError):
        Quantize(digits=2, dtype='f8', astype=object)


def test_backwards_compatibility():
    precision = [codec.digits for codec in codecs]
    check_backwards_compatibility(Quantize.codec_id, arrays, codecs, precision=precision)
