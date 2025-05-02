import pytest

from numcodecs.ndarray_like import DType, FlagsObj, NDArrayLike


@pytest.mark.parametrize("module", ["numpy", "cupy"])
def test_is_ndarray_like(module):
    m = pytest.importorskip(module)
    a = m.arange(10)
    assert isinstance(a, NDArrayLike)


def test_is_not_ndarray_like():
    assert not isinstance([1, 2, 3], NDArrayLike)
    assert not isinstance(b"1,2,3", NDArrayLike)


@pytest.mark.parametrize("module", ["numpy", "cupy"])
def test_is_dtype_like(module):
    m = pytest.importorskip(module)
    d = m.dtype("u8")
    assert isinstance(d, DType)


def test_is_not_dtype_like():
    assert not isinstance([1, 2, 3], DType)
    assert not isinstance(b"1,2,3", DType)


@pytest.mark.parametrize("module", ["numpy", "cupy"])
def test_is_flags_like(module):
    m = pytest.importorskip(module)
    d = m.arange(10).flags
    assert isinstance(d, FlagsObj)


def test_is_not_flags_like():
    assert not isinstance([1, 2, 3], FlagsObj)
    assert not isinstance(b"1,2,3", FlagsObj)


@pytest.mark.parametrize("module", ["numpy", "cupy"])
def test_cached_isinstance_check(module):
    m = pytest.importorskip(module)
    a = m.arange(10)
    assert isinstance(a, NDArrayLike)
    assert not isinstance(a, DType)
    assert not isinstance(a, FlagsObj)
