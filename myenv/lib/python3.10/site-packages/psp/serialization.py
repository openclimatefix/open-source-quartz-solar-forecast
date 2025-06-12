import pathlib
import pickle

import fsspec

from psp.models.base import PvSiteModel


def save_model(model: PvSiteModel, filepath: pathlib.Path | str):
    # This is the main reason why we don't use pickle directly: so that models can customize what
    # they save. This can be different than the default behaviour of pickling (with __get_state__).
    # For example, we might want to save something different when models are pickled by pytorch's
    # DataLoader for multiprocessing.

    # This is inspired from:
    # https://docs.python.org/3/library/pickle.html#pickling-class-instances
    state = (model.__class__, model.get_state())
    with open(filepath, "wb") as f:
        pickle.dump(state, f)


def load_model(filepath: pathlib.Path | str) -> PvSiteModel:
    # Use fsspec to support loading models from the cloud, using paths like "s3://..".
    with fsspec.open(str(filepath), "rb") as f:
        (cls, attrs) = pickle.load(f)

    model = cls.__new__(cls)
    model.set_state(attrs)

    return model
