import argparse

from psp.scripts.inspect_data_file import main


def test_inspect_data_file():
    """Trivial test that makes sure the script runs."""
    args = argparse.Namespace(input="psp/tests/fixtures/pv_data.nc", engine=None)
    main(args)
