"""Smoke tests for the CommuLab package foundation."""

import importlib
import sys


def test_python_version() -> None:
    assert sys.version_info >= (3, 12)


def test_package_import() -> None:
    assert importlib.import_module("commulab") is not None


def test_scientific_dependencies() -> None:
    import matplotlib
    import numpy
    import scipy

    assert numpy.__version__
    assert scipy.__version__
    assert matplotlib.__version__


def test_package_submodules() -> None:
    for name in ("numerical", "modulation", "channels", "analysis"):
        assert importlib.import_module(f"commulab.{name}") is not None
