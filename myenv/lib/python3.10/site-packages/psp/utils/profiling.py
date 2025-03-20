import contextlib
import logging
import time

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def profile(name: str):
    t0 = time.time()
    yield
    t1 = time.time()
    logger.debug(f'Executed "{name}" in {t1 - t0:.3f}s')
