import hashlib
from typing import Any


def naive_hash(x: Any) -> int:
    """Naive hash that has the particularity of being consistent between session, unlike
    `hash("some string")`
    """
    return int(hashlib.sha1(str(x).encode()).hexdigest(), 16)
