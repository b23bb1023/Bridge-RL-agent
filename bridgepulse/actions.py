"""Action indexing for a network of N bridges: 0 = defer, then inspect / minor / major per bridge."""

from __future__ import annotations

DEFER, INSPECT, MINOR, MAJOR = 0, 1, 2, 3
KIND_NAMES = ("defer", "inspect", "minor", "major")


def num_actions(n: int) -> int:
    return 1 + 3 * n


def encode(kind: int, bridge: int, n: int) -> int:
    if kind == DEFER:
        return 0
    return 1 + (kind - 1) * n + bridge


def decode(action: int, n: int) -> tuple[int, int]:
    """Return (kind, bridge index); bridge is -1 for defer."""
    if action == 0:
        return DEFER, -1
    kind, bridge = divmod(action - 1, n)
    return kind + 1, bridge
