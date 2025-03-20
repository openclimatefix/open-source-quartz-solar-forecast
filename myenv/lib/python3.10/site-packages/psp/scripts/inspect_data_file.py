"""Inspect a netcdf/zarr data file.

Print a summary view of the data files along with some statistics on the data.
"""

import argparse
import pathlib

import ocf_blosc2  # noqa: F401
import xarray


def inspect(path: pathlib.Path, *, engine: str):
    ds = xarray.open_dataset(path, engine=engine)
    print(ds)

    for coord in ds.coords:
        print()
        print("Coord", coord)
        data = ds.coords[coord]
        print(" min:", data.min().values)
        print(" max:", data.max().values)


def _parse_args():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("input", help="path to the file to inspect")
    parser.add_argument("--engine", help="tell xarray what engine to use")
    return parser.parse_args()


def main(args: argparse.Namespace):
    inspect(args.input, engine=args.engine)


if __name__ == "__main__":
    args = _parse_args()
    main(args)
