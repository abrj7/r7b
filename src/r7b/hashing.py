"""Deterministic pseudo-randomness keyed on strings, stable across machines."""

from __future__ import annotations

import hashlib
import math


def h01(*parts: object) -> float:
    """A uniform number in (0, 1) derived from the parts."""
    digest = hashlib.sha256("\x1f".join(map(str, parts)).encode()).digest()
    return (int.from_bytes(digest[:8], "big") + 0.5) / 2**64


def hnorm(*parts: object) -> float:
    """A standard normal number derived from the parts (Box-Muller)."""
    u1, u2 = h01(*parts, "a"), h01(*parts, "b")
    return math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
