from datetime import timedelta
from typing import Optional, Tuple, TypeVar, Union

import requests
from requests.adapters import HTTPAdapter
from urllib3 import Retry

T = TypeVar("T", bound=requests.Session)


class TSession(requests.Session):
    """A session that has a timeout for all of its requests."""

    def __init__(self, timeout: Union[int, timedelta] = 5):
        """

        Args:
            timeout: Time that requests will wait to receive the first
                     response bytes (not the whole time) from the server.
                     An int in seconds or a timedelta object.
        """
        super().__init__()
        self.timeout = timeout if isinstance(timeout, int) else timeout.seconds

    def request(self, method, url, *args, **kwargs):
        kwargs.setdefault("timeout", self.timeout)
        return super().request(method, url, *args, **kwargs)


class RSession(TSession):
    """A session that has a timeout and a ``raises_for_status``
     for all of its requests.
    """

    def __init__(self, timeout: Union[int, timedelta] = 5):
        super().__init__(timeout)
        self.hooks["response"] = lambda r, *args, **kwargs: r.raise_for_status()


def retry(
    session: Optional[T] = None,
    retries: int = 3,
    backoff_factor: float = 1,
    status_to_retry: Tuple[int, ...] = (500, 502, 504),
    prefixes: Tuple[str, ...] = ("http://", "https://"),
    **kwargs
) -> Union[T, TSession]:
    """
    Configures the passed-in session to retry on failed requests
    due to connection errors, specific HTTP response codes and
    30X redirections.

    Args:
        session: A session to allow to retry. None creates a new Session.
        retries: The number of maximum retries before raising an
                 exception.
        backoff_factor: A factor used to compute the waiting time
                        between retries.
                        See :arg:`urllib3.util.retry.Retry.backoff_factor`.
        status_to_retry: A tuple of status codes that trigger the reply
                         behaviour.
        prefixes: A tuple of URL prefixes that this retry configuration
                  affects. By default, ``https`` and ``https``.
        **kwargs: Extra arguments that are passed to
                  :class:`urllib3.util.retry.Retry`.

    Returns:
        A session object with the retry setup.
    """
    session = session or TSession()

    # Retry too in non-idempotent methods like POST
    kwargs.setdefault("allowed_methods", None)

    r = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_to_retry,
        **kwargs
    )
    adapter = HTTPAdapter(max_retries=r)
    for prefix in prefixes:
        session.mount(prefix, adapter)
    return session
