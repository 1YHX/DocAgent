import os
import pytest
from docagent.config import _float_env, _int_env


def test_int_env_returns_default_when_unset():
    os.environ.pop("_DOCAGENT_TEST_INT", None)
    assert _int_env("_DOCAGENT_TEST_INT", 42) == 42


def test_int_env_returns_default_when_empty(monkeypatch):
    monkeypatch.setenv("_DOCAGENT_TEST_INT", "")
    assert _int_env("_DOCAGENT_TEST_INT", 7) == 7


def test_int_env_parses_valid_integer(monkeypatch):
    monkeypatch.setenv("_DOCAGENT_TEST_INT", "8")
    assert _int_env("_DOCAGENT_TEST_INT", 4) == 8


def test_int_env_raises_on_non_integer(monkeypatch):
    monkeypatch.setenv("_DOCAGENT_TEST_INT", "abc")
    with pytest.raises(ValueError, match="_DOCAGENT_TEST_INT='abc' is not a valid integer"):
        _int_env("_DOCAGENT_TEST_INT", 4)


def test_int_env_raises_on_float_string(monkeypatch):
    monkeypatch.setenv("_DOCAGENT_TEST_INT", "3.14")
    with pytest.raises(ValueError):
        _int_env("_DOCAGENT_TEST_INT", 4)


def test_float_env_returns_default_when_unset():
    os.environ.pop("_DOCAGENT_TEST_FLOAT", None)
    assert _float_env("_DOCAGENT_TEST_FLOAT", 0.55) == 0.55


def test_float_env_parses_valid_float(monkeypatch):
    monkeypatch.setenv("_DOCAGENT_TEST_FLOAT", "0.72")
    assert _float_env("_DOCAGENT_TEST_FLOAT", 0.55) == 0.72


def test_float_env_raises_on_non_float(monkeypatch):
    monkeypatch.setenv("_DOCAGENT_TEST_FLOAT", "abc")
    with pytest.raises(ValueError, match="_DOCAGENT_TEST_FLOAT='abc' is not a valid float"):
        _float_env("_DOCAGENT_TEST_FLOAT", 0.55)
