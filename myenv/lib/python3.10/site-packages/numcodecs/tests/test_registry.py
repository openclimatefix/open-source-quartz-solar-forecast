import inspect

import pytest

import numcodecs
from numcodecs.registry import get_codec


def test_registry_errors():
    with pytest.raises(ValueError):
        get_codec({'id': 'foo'})


def test_get_codec_argument():
    # Check that get_codec doesn't modify its argument.
    arg = {"id": "json2"}
    before = dict(arg)
    get_codec(arg)
    assert before == arg


def test_all_classes_registered():
    """
    find all Codec subclasses in this repository and check that they
    have been registered.

    see #346 for more info
    """
    missing = set()
    for _name, submod in inspect.getmembers(numcodecs, inspect.ismodule):
        for _name, obj in inspect.getmembers(submod):
            if inspect.isclass(obj):
                if issubclass(obj, numcodecs.abc.Codec):
                    if obj.codec_id not in numcodecs.registry.codec_registry:
                        missing.add(obj.codec_id)

    # remove `None``
    missing.remove(None)
    if missing:
        raise Exception(f"these codecs are missing: {missing}")  # pragma: no cover
