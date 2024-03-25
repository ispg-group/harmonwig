"""
Copyright (c) 2024 Daniel Hollas. All rights reserved.

harmonwig: Harmonic Wigner sampling from QM calculations.
"""
from __future__ import annotations

from .harmonwig import HarmonicWigner

__version__ = "0.1.0a0"

# TODO: Expose more interface?
__all__ = ["__version__", "HarmonicWigner"]
