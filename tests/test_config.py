import os
import pytest
from docagent.config import _int_env


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
