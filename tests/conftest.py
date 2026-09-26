"""Every unit test runs without external network access."""

import socket

import pytest

from f1_quali.config import load_config
from f1_quali.demo import synthetic_dataset


@pytest.fixture(autouse=True)
def forbid_network(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("Tests must run offline")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)


@pytest.fixture
def data():
    return synthetic_dataset(history_events=4, target_events=3)


@pytest.fixture
def config():
    return load_config()
