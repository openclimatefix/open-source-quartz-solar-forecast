from typing import Literal, Optional

import numcodecs
import numcodecs.abc
from numcodecs.compat import ensure_contiguous_ndarray

try:
    from pcodec import ChunkConfig, ModeSpec, PagingSpec, standalone
except ImportError:  # pragma: no cover
    standalone = None


DEFAULT_MAX_PAGE_N = 262144


class PCodec(numcodecs.abc.Codec):
    """
    PCodec (or pco, pronounced "pico") losslessly compresses and decompresses
    numerical sequences with high compression ratio and fast speed.

    See `PCodec Repo <https://github.com/mwlon/pcodec>`_ for more information.

    PCodec supports only the following numerical dtypes: uint16, uint32, uint64,
    int16, int32, int64, float16, float32, and float64.

    Parameters
    ----------
    level : int
        A compression level from 0-12, where 12 take the longest and compresses
        the most.
    delta_encoding_order : init or None
        Either a delta encoding level from 0-7 or None. If set to None, pcodec
        will try to infer the optimal delta encoding order.
    mode_spec : {'auto', 'classic'}
        Configures whether Pcodec should try to infer the best "mode" or
        structure of the data (e.g. approximate multiples of 0.1) to improve
        compression ratio, or skip this step and just use the numbers as-is
        (Classic mode).
    equal_pages_up_to : int
        Divide the chunk into equal pages of up to this many numbers.
    """

    codec_id = "pcodec"

    def __init__(
        self,
        level: int = 8,
        delta_encoding_order: Optional[int] = None,
        equal_pages_up_to: int = 262144,
        # TODO one day, add support for the Try* mode specs
        mode_spec: Literal['auto', 'classic'] = 'auto',
    ):
        if standalone is None:  # pragma: no cover
            raise ImportError("pcodec must be installed to use the PCodec codec.")

        # note that we use `level` instead of `compression_level` to
        # match other codecs
        self.level = level
        self.delta_encoding_order = delta_encoding_order
        self.equal_pages_up_to = equal_pages_up_to
        self.mode_spec = mode_spec

    def encode(self, buf):
        buf = ensure_contiguous_ndarray(buf)

        match self.mode_spec:
            case 'auto':
                mode_spec = ModeSpec.auto()
            case 'classic':
                mode_spec = ModeSpec.classic()
            case _:
                raise ValueError(f"unknown value for mode_spec: {self.mode_spec}")
        paging_spec = PagingSpec.equal_pages_up_to(self.equal_pages_up_to)

        config = ChunkConfig(
            compression_level=self.level,
            delta_encoding_order=self.delta_encoding_order,
            mode_spec=mode_spec,
            paging_spec=paging_spec,
        )
        return standalone.simple_compress(buf, config)

    def decode(self, buf, out=None):
        if out is not None:
            out = ensure_contiguous_ndarray(out)
            standalone.simple_decompress_into(buf, out)
            return out
        else:
            return standalone.simple_decompress(buf)
