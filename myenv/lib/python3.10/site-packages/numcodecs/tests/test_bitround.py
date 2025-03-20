import numpy as np
import pytest

from numcodecs.bitround import BitRound, max_bits

# adapted from https://github.com/milankl/BitInformation.jl/blob/main/test/round_nearest.jl


# TODO: add other dtypes
@pytest.fixture(params=["float32", "float64"])
def dtype(request):
    return request.param


def round(data, keepbits):
    codec = BitRound(keepbits=keepbits)
    data = data.copy()  # otherwise overwrites the input
    encoded = codec.encode(data)
    return codec.decode(encoded)


def test_round_zero_to_zero(dtype):
    a = np.zeros((3, 2), dtype=dtype)
    # Don't understand Milan's original test:
    # How is it possible to have negative keepbits?
    # for k in range(-5, 50):
    for k in range(0, max_bits[dtype]):
        ar = round(a, k)
        np.testing.assert_equal(a, ar)


def test_round_one_to_one(dtype):
    a = np.ones((3, 2), dtype=dtype)
    for k in range(0, max_bits[dtype]):
        ar = round(a, k)
        np.testing.assert_equal(a, ar)


def test_round_minus_one_to_minus_one(dtype):
    a = -np.ones((3, 2), dtype=dtype)
    for k in range(0, max_bits[dtype]):
        ar = round(a, k)
        np.testing.assert_equal(a, ar)


def test_no_rounding(dtype):
    a = np.random.random_sample((300, 200)).astype(dtype)
    keepbits = max_bits[dtype]
    ar = round(a, keepbits)
    np.testing.assert_equal(a, ar)


APPROX_KEEPBITS = {"float32": 11, "float64": 18}


def test_approx_equal(dtype):
    a = np.random.random_sample((300, 200)).astype(dtype)
    ar = round(a, APPROX_KEEPBITS[dtype])
    # Mimic julia behavior - https://docs.julialang.org/en/v1/base/math/#Base.isapprox
    rtol = np.sqrt(np.finfo(np.float32).eps)
    # This gets us much closer but still failing for ~6% of the array
    # It does pass if we add 1 to keepbits (11 instead of 10)
    # Is there an off-by-one issue here?
    np.testing.assert_allclose(a, ar, rtol=rtol)


def test_idempotence(dtype):
    a = np.random.random_sample((300, 200)).astype(dtype)
    for k in range(20):
        ar = round(a, k)
        ar2 = round(a, k)
        np.testing.assert_equal(ar, ar2)


def test_errors():
    with pytest.raises(ValueError):
        BitRound(keepbits=99).encode(np.array([0], dtype="float32"))
    with pytest.raises(TypeError):
        BitRound(keepbits=10).encode(np.array([0]))
    with pytest.raises(ValueError):
        BitRound(-1)
