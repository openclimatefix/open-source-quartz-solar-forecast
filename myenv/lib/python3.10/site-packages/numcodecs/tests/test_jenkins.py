import numpy as np
import pytest

from numcodecs.checksum32 import JenkinsLookup3
from numcodecs.jenkins import jenkins_lookup3


def test_jenkins_lookup3():
    h = jenkins_lookup3(b"", 0)
    assert h == 0xDEADBEEF
    h = jenkins_lookup3(b"", 0xDEADBEEF)
    assert h == 0xBD5B7DDE
    h = jenkins_lookup3(b"Four score and seven years ago", 0)
    assert h == 0x17770551
    h = jenkins_lookup3(b"Four score and seven years ago", 1)
    assert h == 0xCD628161

    # jenkins-cffi example
    h = jenkins_lookup3(b"jenkins", 0)
    assert h == 202276345

    h_last = [0]

    h = jenkins_lookup3(b"", h_last[-1])
    assert h not in h_last
    h_last.append(h)

    h = jenkins_lookup3(b"", h_last[-1])
    assert h not in h_last
    h_last.append(h)

    h = jenkins_lookup3(b"", h_last[-1])
    assert h not in h_last
    h_last.append(h)

    h = jenkins_lookup3(b"", h_last[-1])
    assert h not in h_last
    h_last.append(h)

    h = jenkins_lookup3(b"", h_last[-1])
    assert h not in h_last
    h_last.append(h)

    h = jenkins_lookup3(b"", h_last[-1])
    assert h not in h_last
    h_last.append(h)

    h = jenkins_lookup3(b"", h_last[-1])
    assert h not in h_last
    h_last.append(h)

    h = jenkins_lookup3(b"", h_last[-1])
    assert h not in h_last
    h_last.append(h)

    h = jenkins_lookup3(b"", h_last[-1])
    assert h not in h_last
    h_last.append(h)

    a = np.frombuffer(b"Four score and seven years ago", dtype="uint8")
    h = jenkins_lookup3(a, 0)
    assert h == 0x17770551


def test_jenkins_lookup3_codec():
    s = b"Four score and seven years ago"
    j = JenkinsLookup3()
    result = j.encode(s)
    assert result[-4:] == b'\x51\x05\x77\x17'
    assert bytes(j.decode(result)) == s

    j = JenkinsLookup3(initval=0xDEADBEEF)
    result = j.encode(s)
    assert bytes(j.decode(result)) == s

    j = JenkinsLookup3(initval=1230)
    result = j.encode(s)
    assert result[-4:] == b'\xd7Z\xe2\x0e'
    assert bytes(j.decode(result)) == s

    j = JenkinsLookup3(initval=1230, prefix=b"Hello world")
    result = j.encode(s)
    assert bytes(j.decode(result)) == s

    chunk_index = (
        b"\x00\x08\x00\x00\x00\x00\x00\x00"
        b"\xf7\x0f\x00\x00\x00\x00\x00\x00"
        b"\xf7\x17\x00\x00\x00\x00\x00\x00"
        b"\xf7\x0f\x00\x00\x00\x00\x00\x00"
        b"\xee'\x00\x00\x00\x00\x00\x00"
        b"\xf7\x0f\x00\x00\x00\x00\x00\x00"
        b"\xe57\x00\x00\x00\x00\x00\x00"
        b"\xf7\x0f\x00\x00\x00\x00\x00\x00"
        b"\xdcG\x00\x00\x00\x00\x00\x00"
        b"\xf7\x0f\x00\x00\x00\x00\x00\x00"
        b"\xd3W\x00\x00\x00\x00\x00\x00"
        b"\xf7\x0f\x00\x00\x00\x00\x00\x00"
        b"\xcag\x00\x00\x00\x00\x00\x00"
        b"\xf7\x0f\x00\x00\x00\x00\x00\x00"
        b"\xc1w\x00\x00\x00\x00\x00\x00"
        b"\xf7\x0f\x00\x00\x00\x00\x00\x00"
        b"\xb8\x87\x00\x00\x00\x00\x00\x00"
        b"\xf7\x0f\x00\x00\x00\x00\x00\x00"
        b"\xaf\x97\x00\x00\x00\x00\x00\x00"
        b"\xf7\x0f\x00\x00\x00\x00\x00\x00"
        b"\xa6\xa7\x00\x00\x00\x00\x00\x00"
        b"\xf7\x0f\x00\x00\x00\x00\x00\x00"
        b"\x9d\xb7\x00\x00\x00\x00\x00\x00"
        b"\xf7\x0f\x00\x00\x00\x00\x00\x00"
        b"\x94\xc7\x00\x00\x00\x00\x00\x00"
        b"\xf7\x0f\x00\x00\x00\x00\x00\x00"
        b"\x8b\xd7\x00\x00\x00\x00\x00\x00"
        b"\xf7\x0f\x00\x00\x00\x00\x00\x00"
        b"\x82\xe7\x00\x00\x00\x00\x00\x00"
        b"\xf7\x0f\x00\x00\x00\x00\x00\x00"
        b"y\xf7\x00\x00\x00\x00\x00\x00"
        b"\xf7\x0f\x00\x00\x00\x00\x00\x00"
        b"n\x96\x07\x85"
    )
    hdf5_fadb_prefix = b'FADB\x00\x01\xcf\x01\x00\x00\x00\x00\x00\x00'
    j = JenkinsLookup3(prefix=hdf5_fadb_prefix)
    result = j.encode(chunk_index[:-4])
    j.decode(result)
    assert result == chunk_index


@pytest.mark.parametrize("dtype", ["uint8", "int32", "float32"])
def test_with_data(dtype):
    data = np.arange(100, dtype=dtype)
    j = JenkinsLookup3()
    arr = np.frombuffer(j.decode(j.encode(data)), dtype=dtype)
    assert (arr == data).all()


def test_error():
    data = np.arange(100)
    j = JenkinsLookup3()
    enc = j.encode(data)
    enc2 = bytearray(enc)
    enc2[0] += 1
    with pytest.raises(RuntimeError) as e:
        j.decode(enc2)
    assert "Bob Jenkin's lookup3 checksum" in str(e.value)


def test_out():
    data = np.frombuffer(bytearray(b"Hello World"), dtype="uint8")
    j = JenkinsLookup3()
    result = j.encode(data)
    j.decode(result, out=data)
