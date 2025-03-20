from numcodecs.abc import Codec


class TestCodec(Codec):
    codec_id = "test"

    def encode(self, buf):  # pragma: no cover
        pass

    def decode(self, buf, out=None):  # pragma: no cover
        pass
