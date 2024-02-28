from __future__ import annotations

import importlib.metadata

import harmonwig as m


def test_version():
    assert importlib.metadata.version("harmonwig") == m.__version__
