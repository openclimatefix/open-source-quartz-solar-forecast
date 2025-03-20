from numcodecs.registry import register_codec
from numcodecs.abc import Codec
from numcodecs.compat import ensure_contiguous_ndarray
import blosc2


class Blosc2(Codec):
    """Codec providing compression using the Blosc2 meta-compressor.

    Parameters
    ----------
    cname : string, optional
        A string naming one of the compression algorithms available within blosc2, e.g.,
        'zstd', 'blosc2'
    clevel : integer, optional
        An integer between 0 and 9 specifying the compression level.
    See Also
    --------
    numcodecs.zstd.Zstd, numcodecs.lz4.LZ4
    """

    codec_id = "blosc2"
    max_buffer_size = 2**31 - 1

    def __init__(self, cname="blosc2", clevel=5):
        self.cname = cname
        if cname == "zstd":
            self._codec = blosc2.Codec.ZSTD
        elif cname == "blosc2":
            self._codec = blosc2.Codec.BLOSCLZ
        self.clevel = clevel

    def encode(self, buf):
        buf = ensure_contiguous_ndarray(buf, self.max_buffer_size)
        return blosc2.compress(buf, codec=self._codec, clevel=self.clevel)

    def decode(self, buf, out=None):
        buf = ensure_contiguous_ndarray(buf, self.max_buffer_size)
        return blosc2.decompress(buf, out)

    def __repr__(self):
        r = "%s(cname=%r, clevel=%r)" % (
            type(self).__name__,
            self.cname,
            self.clevel,
        )
        return r


register_codec(Blosc2)

