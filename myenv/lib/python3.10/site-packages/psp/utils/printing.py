from psp.typings import PvId


def pv_list_to_short_str(x: list[PvId]) -> str:
    """Util to format a list of PV ids into a small string."""
    if len(x) < 4:
        return str(x)
    else:
        return f"[{repr(x[0])}, {repr(x[1])}, ..., {repr(x[-1])}]"
