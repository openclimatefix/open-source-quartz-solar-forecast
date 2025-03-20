#!/usr/bin/env python
#
# Copyright (c) 2018 Donfig Developers
# Copyright (c) 2014-2018, Anaconda, Inc. and contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import os
import site
import stat
import subprocess
import sys
from collections import OrderedDict
from contextlib import contextmanager

import cloudpickle
import pytest
import yaml

from donfig.config_obj import (
    Config,
    canonical_name,
    collect_env,
    collect_yaml,
    deserialize,
    expand_environment_variables,
    merge,
    serialize,
    update,
)
from donfig.utils import tmpfile

CONFIG_NAME = "mytest"
ENV_PREFIX = CONFIG_NAME.upper() + "_"


def test_canonical_name():
    c = {"foo-bar": 1, "fizz_buzz": 2}
    assert canonical_name("foo-bar", c) == "foo-bar"
    assert canonical_name("foo_bar", c) == "foo-bar"
    assert canonical_name("fizz-buzz", c) == "fizz_buzz"
    assert canonical_name("fizz_buzz", c) == "fizz_buzz"
    assert canonical_name("new-key", c) == "new-key"
    assert canonical_name("new_key", c) == "new_key"


def test_update():
    a = {"x": 1, "y": {"a": 1}}
    b = {"x": 2, "z": 3, "y": OrderedDict({"b": 2})}
    update(b, a)
    assert b == {"x": 1, "y": {"a": 1, "b": 2}, "z": 3}

    a = {"x": 1, "y": {"a": 1}}
    b = {"x": 2, "z": 3, "y": {"a": 3, "b": 2}}
    update(b, a, priority="old")
    assert b == {"x": 2, "y": {"a": 3, "b": 2}, "z": 3}


def test_update_new_defaults():
    d = {"x": 1, "y": 1, "z": {"a": 1, "b": 1}}
    o = {"x": 1, "y": 2, "z": {"a": 1, "b": 2}, "c": 2, "c2": {"d": 2}}
    n = {"x": 3, "y": 3, "z": OrderedDict({"a": 3, "b": 3}), "c": 3, "c2": {"d": 3}}
    assert update(o, n, priority="new-defaults", defaults=d) == {
        "x": 3,
        "y": 2,
        "z": {"a": 3, "b": 2},
        "c": 2,
        "c2": {"d": 2},
    }
    assert update(o, n, priority="new-defaults", defaults=o) == update(o, n, priority="new")
    assert update(o, n, priority="new-defaults", defaults=None) == update(o, n, priority="old")


def test_update_defaults():
    defaults = [
        {"a": 1, "b": {"c": 1}},
        {"a": 2, "b": {"d": 2}},
    ]
    config = Config(CONFIG_NAME, defaults=defaults)
    current = {"a": 2, "b": {"c": 1, "d": 3}, "extra": 0}
    config.update(current)
    new = {"a": 0, "b": {"c": 0, "d": 0}, "new-extra": 0}
    config.update_defaults(new)

    assert defaults == [
        {"a": 1, "b": {"c": 1}},
        {"a": 2, "b": {"d": 2}},
        {"a": 0, "b": {"c": 0, "d": 0}, "new-extra": 0},
    ]
    assert config.to_dict() == {"a": 0, "b": {"c": 0, "d": 3}, "extra": 0, "new-extra": 0}


def test_merge():
    a = {"x": 1, "y": {"a": 1}}
    b = {"x": 2, "z": 3, "y": {"b": 2}}

    expected = {"x": 2, "y": {"a": 1, "b": 2}, "z": 3}

    c = merge(a, b)
    assert c == expected


def test_collect_yaml_paths():
    a = {"x": 1, "y": {"a": 1}}
    b = {"x": 2, "z": 3, "y": {"b": 2}}

    expected = {
        "x": 2,
        "y": {"a": 1, "b": 2},
        "z": 3,
    }

    with tmpfile(extension="yaml") as fn1:
        with tmpfile(extension="yaml") as fn2:
            with open(fn1, "w") as f:
                yaml.dump(a, f)
            with open(fn2, "w") as f:
                yaml.dump(b, f)

            config = merge(*collect_yaml(paths=[fn1, fn2]))
            assert config == expected


def test_collect_yaml_dir():
    a = {"x": 1, "y": {"a": 1}}
    b = {"x": 2, "z": 3, "y": {"b": 2}}

    expected = {
        "x": 2,
        "y": {"a": 1, "b": 2},
        "z": 3,
    }

    with tmpfile() as dirname:
        os.mkdir(dirname)
        with open(os.path.join(dirname, "a.yaml"), mode="w") as f:
            yaml.dump(a, f)
        with open(os.path.join(dirname, "b.yaml"), mode="w") as f:
            yaml.dump(b, f)

        config = merge(*collect_yaml(paths=[dirname]))
        assert config == expected


@contextmanager
def no_read_permissions(path):
    perm_orig = stat.S_IMODE(os.stat(path).st_mode)
    perm_new = perm_orig ^ stat.S_IREAD
    try:
        os.chmod(path, perm_new)
        yield
    finally:
        os.chmod(path, perm_orig)


@pytest.mark.skipif(sys.platform == "win32", reason="Can't make writeonly file on windows")
@pytest.mark.parametrize("kind", ["directory", "file"])
def test_collect_yaml_permission_errors(tmpdir, kind):
    a = {"x": 1, "y": 2}
    b = {"y": 3, "z": 4}

    dir_path = str(tmpdir)
    a_path = os.path.join(dir_path, "a.yaml")
    b_path = os.path.join(dir_path, "b.yaml")

    with open(a_path, mode="w") as f:
        yaml.dump(a, f)
    with open(b_path, mode="w") as f:
        yaml.dump(b, f)

    if kind == "directory":
        cant_read = dir_path
        expected = {}
    else:
        cant_read = a_path
        expected = b

    with no_read_permissions(cant_read):
        config = merge(*collect_yaml(paths=[dir_path]))
        assert config == expected


def test_collect_yaml_malformed_file(tmpdir):
    dir_path = str(tmpdir)
    fil_path = os.path.join(dir_path, "a.yaml")

    with open(fil_path, mode="wb") as f:
        f.write(b"{")

    with pytest.raises(ValueError) as rec:
        collect_yaml(paths=[dir_path])
    assert repr(fil_path) in str(rec.value)
    assert "is malformed" in str(rec.value)
    assert "original error message" in str(rec.value)


def test_collect_yaml_no_top_level_dict(tmpdir):
    dir_path = str(tmpdir)
    fil_path = os.path.join(dir_path, "a.yaml")

    with open(fil_path, mode="wb") as f:
        f.write(b"[1234]")

    with pytest.raises(ValueError) as rec:
        collect_yaml(paths=[dir_path])
    assert repr(fil_path) in str(rec.value)
    assert "is malformed" in str(rec.value)
    assert "must have a dict" in str(rec.value)


def test_env():
    env = {
        ENV_PREFIX + "A_B": "123",
        ENV_PREFIX + "C": "True",
        ENV_PREFIX + "D": "hello",
        ENV_PREFIX + "E__X": "123",
        ENV_PREFIX + "E__Y": "456",
        ENV_PREFIX + "F": '[1, 2, "3"]',
        ENV_PREFIX + "G": "/not/parsable/as/literal",
        "FOO": "not included",
    }

    expected = {
        "a_b": 123,
        "c": True,
        "d": "hello",
        "e": {"x": 123, "y": 456},
        "f": [1, 2, "3"],
        "g": "/not/parsable/as/literal",
    }

    res = collect_env(CONFIG_NAME.upper() + "_", env)
    assert res == expected


def test_collect():
    a = {"x": 1, "y": {"a": 1}}
    b = {"x": 2, "z": 3, "y": {"b": 2}}
    env = {ENV_PREFIX + "W": 4}

    expected = {
        "w": 4,
        "x": 2,
        "y": {"a": 1, "b": 2},
        "z": 3,
    }

    config = Config(CONFIG_NAME)
    with tmpfile(extension="yaml") as fn1:
        with tmpfile(extension="yaml") as fn2:
            with open(fn1, "w") as f:
                yaml.dump(a, f)
            with open(fn2, "w") as f:
                yaml.dump(b, f)

            config = config.collect([fn1, fn2], env=env)
            assert config == expected


def test_collect_env_none():
    os.environ[ENV_PREFIX + "FOO"] = "bar"
    config = Config(CONFIG_NAME)
    try:
        config = config.collect([])
        assert config == {"foo": "bar"}
    finally:
        del os.environ[ENV_PREFIX + "FOO"]


def test_get():
    test_config = Config(CONFIG_NAME)
    test_config.config = {"x": 1, "y": {"a": 2}}

    assert test_config.get("x") == 1
    assert test_config["x"] == 1
    assert test_config.get("y.a") == 2
    assert test_config["y.a"] == 2
    assert test_config.get("y.b", 123) == 123
    with pytest.raises(KeyError):
        test_config.get("y.b")
    with pytest.raises(KeyError):
        test_config["y.b"]


def test_contains():
    test_config = Config(CONFIG_NAME)
    test_config.config = {"x": 1, "y": {"a": 2}}

    assert "x" in test_config
    assert "y.a" in test_config
    assert "y.b" not in test_config


def test_ensure_file(tmpdir):
    a = {"x": 1, "y": {"a": 1}}
    b = {"x": 123}

    source = os.path.join(str(tmpdir), "source.yaml")
    dest = os.path.join(str(tmpdir), "dest")
    destination = os.path.join(dest, "source.yaml")

    with open(source, "w") as f:
        yaml.dump(a, f)

    config = Config(CONFIG_NAME)
    config.ensure_file(source=source, destination=dest, comment=False)

    with open(destination) as f:
        result = yaml.safe_load(f)
    assert result == a

    # don't overwrite old config files
    with open(source, "w") as f:
        yaml.dump(b, f)

    config.ensure_file(source=source, destination=dest, comment=False)

    with open(destination) as f:
        result = yaml.safe_load(f)
    assert result == a

    os.remove(destination)

    # Write again, now with comments
    config.ensure_file(source=source, destination=dest, comment=True)

    with open(destination) as f:
        text = f.read()
    assert "123" in text

    with open(destination) as f:
        result = yaml.safe_load(f)
    assert not result


def test_set():
    config = Config(CONFIG_NAME)
    with config.set(abc=123):
        assert config.config["abc"] == 123
        with config.set(abc=456):
            assert config.config["abc"] == 456
        assert config.config["abc"] == 123

    assert "abc" not in config.config

    with config.set({"abc": 123}):
        assert config.config["abc"] == 123
    assert "abc" not in config.config

    with config.set({"abc.x": 1, "abc.y": 2, "abc.z.a": 3}):
        assert config.config["abc"] == {"x": 1, "y": 2, "z": {"a": 3}}
    assert "abc" not in config.config

    config.config = {}
    config.set({"abc.x": 123})
    assert config.config["abc"]["x"] == 123


def test_set_kwargs():
    config = Config(CONFIG_NAME)
    with config.set(foo__bar=1, foo__baz=2):
        assert config.config["foo"] == {"bar": 1, "baz": 2}
    assert "foo" not in config.config

    # Mix kwargs and dict, kwargs override
    with config.set({"foo.bar": 1, "foo.baz": 2}, foo__buzz=3, foo__bar=4):
        assert config.config["foo"] == {"bar": 4, "baz": 2, "buzz": 3}
    assert "foo" not in config.config

    # Mix kwargs and nested dict, kwargs override
    with config.set({"foo": {"bar": 1, "baz": 2}}, foo__buzz=3, foo__bar=4):
        assert config.config["foo"] == {"bar": 4, "baz": 2, "buzz": 3}
    assert "foo" not in config.config


def test_set_nested():
    config = Config(CONFIG_NAME)
    with config.set({"abc": {"x": 123}}):
        assert config.config["abc"] == {"x": 123}
        with config.set({"abc.y": 456}):
            assert config.config["abc"] == {"x": 123, "y": 456}
        assert config.config["abc"] == {"x": 123}
    assert "abc" not in config.config


def test_set_hard_to_copyables():
    import threading

    config = Config(CONFIG_NAME)
    with config.set(x=threading.Lock()):
        with config.set(y=1):
            pass


@pytest.mark.parametrize("mkdir", [True, False])
def test_ensure_file_directory(mkdir, tmpdir):
    a = {"x": 1, "y": {"a": 1}}

    source = os.path.join(str(tmpdir), "source.yaml")
    dest = os.path.join(str(tmpdir), "dest")

    with open(source, "w") as f:
        yaml.dump(a, f)

    if mkdir:
        os.mkdir(dest)

    config = Config(CONFIG_NAME)
    config.ensure_file(source=source, destination=dest)

    assert os.path.isdir(dest)
    assert os.path.exists(os.path.join(dest, "source.yaml"))


def test_ensure_file_defaults_to_TEST_CONFIG_directory(tmpdir):
    a = {"x": 1, "y": {"a": 1}}
    source = os.path.join(str(tmpdir), "source.yaml")
    with open(source, "w") as f:
        yaml.dump(a, f)

    config = Config("test")
    destination = os.path.join(str(tmpdir), "test")
    PATH = config.main_path
    try:
        config.main_path = destination
        config.ensure_file(source=source)
    finally:
        config.main_path = PATH

    assert os.path.isdir(destination)
    [fn] = os.listdir(destination)
    assert os.path.split(fn)[1] == os.path.split(source)[1]


def test_rename():
    config = Config(CONFIG_NAME)
    aliases = {"foo_bar": "foo.bar"}
    config.config = {"foo-bar": 123}
    config.rename(aliases)
    assert config.config == {"foo": {"bar": 123}}


def test_refresh():
    defaults = []
    config = Config(CONFIG_NAME, defaults=defaults)

    config.update_defaults({"a": 1})
    assert config.config == {"a": 1}

    config.refresh(paths=[], env={ENV_PREFIX + "B": "2"})
    assert config.config == {"a": 1, "b": 2}

    config.refresh(paths=[], env={ENV_PREFIX + "C": "3"})
    assert config.config == {"a": 1, "c": 3}


@pytest.mark.parametrize(
    "inp,out",
    [
        ("1", "1"),
        (1, 1),
        ("$FOO", "foo"),
        ([1, "$FOO"], [1, "foo"]),
        ((1, "$FOO"), (1, "foo")),
        ({1, "$FOO"}, {1, "foo"}),
        ({"a": "$FOO"}, {"a": "foo"}),
        ({"a": "A", "b": [1, "2", "$FOO"]}, {"a": "A", "b": [1, "2", "foo"]}),
    ],
)
def test_expand_environment_variables(inp, out):
    try:
        os.environ["FOO"] = "foo"
        assert expand_environment_variables(inp) == out
    finally:
        del os.environ["FOO"]


def test_env_var_canonical_name(monkeypatch):
    value = 3
    monkeypatch.setenv(ENV_PREFIX + "A_B", str(value))
    config = Config(CONFIG_NAME)

    assert config.get("a_b") == value
    assert config.get("a-b") == value


def test_get_set_canonical_name():
    c = {"x-y": {"a_b": 123}}
    config = Config(CONFIG_NAME)
    config.update(c)

    keys = ["x_y.a_b", "x-y.a-b", "x_y.a-b"]
    for k in keys:
        assert config.get(k) == 123

    with config.set({"x_y": {"a-b": 456}}):
        for k in keys:
            assert config.get(k) == 456

    # No change to new keys in sub dicts
    with config.set({"x_y": {"a-b": {"c_d": 1}, "e-f": 2}}):
        assert config.get("x_y.a-b") == {"c_d": 1}
        assert config.get("x_y.e_f") == 2


@pytest.mark.parametrize("key", ["custom_key", "custom-key"])
def test_get_set_roundtrip(key):
    value = 123
    config = Config(CONFIG_NAME)
    with config.set({key: value}):
        assert config.get("custom_key") == value
        assert config.get("custom-key") == value


def test_merge_none_to_dict():
    assert merge({"a": None, "c": 0}, {"a": {"b": 1}}) == {"a": {"b": 1}, "c": 0}


def test_pprint(capsys):
    test_config = Config(CONFIG_NAME)
    test_config.config = {"x": 1, "y": {"a": 2}}
    test_config.pprint()
    cap_out = capsys.readouterr()[0]
    assert cap_out == """{'x': 1, 'y': {'a': 2}}\n"""


def test_to_dict():
    test_config = Config(CONFIG_NAME)
    test_config.config = {"x": 1, "y": {"a": 2}}
    d = test_config.to_dict()
    assert d == test_config.config
    # make sure we copied
    d["z"] = 3
    d["y"]["b"] = 4
    assert d != test_config.config
    assert d["y"] != test_config.config["y"]


def test_path_includes_site_prefix():
    command = (
        "import site, os; "
        'prefix = os.path.join("include", "this", "path"); '
        "site.PREFIXES.append(prefix); "
        "from donfig import Config; "
        f"config = Config('{CONFIG_NAME}'); "
        "print(config.paths); "
        f'assert os.path.join(prefix, "etc", "{CONFIG_NAME}") in config.paths'
    )

    subprocess.check_call([sys.executable, "-c", command])


def test__get_paths(monkeypatch):
    # These settings, if present, would interfere with these tests
    # We temporarily remove them to avoid interference from the
    # machine where tests are being run.
    monkeypatch.delenv("MYPKG_CONFIG", raising=False)
    monkeypatch.delenv("MYPKG_ROOT_CONFIG", raising=False)
    monkeypatch.setattr(site, "PREFIXES", [])

    expected = [
        "/etc/mypkg",
        os.path.join(sys.prefix, "etc", "mypkg"),
        os.path.join(os.path.expanduser("~"), ".config", "mypkg"),
    ]
    config = Config("mypkg")
    assert config.paths == expected
    assert len(config.paths) == len(set(config.paths))  # No duplicate paths

    with monkeypatch.context() as m:
        m.setenv("MYPKG_CONFIG", "foo-bar")
        config = Config("mypkg")
        paths = config.paths
        assert paths == expected + ["foo-bar"]
        assert len(paths) == len(set(paths))

    with monkeypatch.context() as m:
        m.setenv("MYPKG_ROOT_CONFIG", "foo-bar")
        config = Config("mypkg")
        paths = config.paths
        assert paths == ["foo-bar"] + expected[1:]
        assert len(paths) == len(set(paths))

    with monkeypatch.context() as m:
        prefix = os.path.join("include", "this", "path")
        m.setattr(site, "PREFIXES", site.PREFIXES + [prefix])
        config = Config("mypkg")
        paths = config.paths
        assert os.path.join(prefix, "etc", "mypkg") in paths
        assert len(paths) == len(set(paths))


def test_serialization():
    config = Config(CONFIG_NAME)
    config.set(one_key="one_value")
    new_config = cloudpickle.loads(cloudpickle.dumps(config))
    assert new_config.get("one_key") == "one_value"


def test_deprecations_rename():
    config = Config(CONFIG_NAME, deprecations={"fuse_ave_width": "optimization.fuse.ave-width"})
    with pytest.warns(Warning) as info, config.set(fuse_ave_width=123):
        assert config.get("optimization.fuse.ave-width") == 123

    assert "optimization.fuse.ave-width" in str(info[0].message)


def test_deprecations_removed():
    config = Config(CONFIG_NAME, deprecations={"fuse_ave_width": None})
    with pytest.raises(ValueError):
        config.set(fuse_ave_width=123)


def test_config_serialization_functions():
    serialized = serialize({"array": {"svg": {"size": 150}}})
    config_dict = deserialize(serialized)
    assert config_dict["array"]["svg"]["size"] == 150


def test_config_object_serialization():
    config = Config(CONFIG_NAME)
    config.set({"array.svg.size": 150})
    ser_config = config.serialize()
    deser_dict = deserialize(ser_config)
    deser_config = Config(CONFIG_NAME)
    deser_config.update(deser_dict)
    assert deser_config.get("array.svg.size") == 150


def test_config_inheritance(monkeypatch):
    ser_dict = serialize({"array": {"svg": {"size": 150}}})
    monkeypatch.setenv(f"{ENV_PREFIX}_INTERNAL_INHERIT_CONFIG", ser_dict)
    config = Config(CONFIG_NAME)
    assert config.get("array.svg.size") == 150
