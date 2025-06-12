import importlib
import os.path
import sys
from multiprocessing import Process

import pytest

import numcodecs.registry

if not importlib.util.find_spec("importlib_metadata").loader:  # pragma: no cover
    pytest.skip(
        "This test module requires importlib_metadata to be installed",
        allow_module_level=True,
    )

here = os.path.abspath(os.path.dirname(__file__))


def get_entrypoints_with_importlib_metadata_loaded():
    # importlib_metadata patches importlib.metadata, which can lead to breaking changes
    # to the APIs of EntryPoint objects used when registering entrypoints. Attempt to
    # isolate those changes to just this test.
    import importlib_metadata  # noqa: F401

    sys.path.append(here)
    numcodecs.registry.run_entrypoints()
    cls = numcodecs.registry.get_codec({"id": "test"})
    assert cls.codec_id == "test"


def test_entrypoint_codec_with_importlib_metadata():
    p = Process(target=get_entrypoints_with_importlib_metadata_loaded)
    p.start()
    p.join()
    assert p.exitcode == 0
