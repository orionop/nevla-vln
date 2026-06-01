#!/usr/bin/env python3
"""Geometric spatial-relation predicates over 3D object instances.

Pure standard-library geometry (no numpy), so it is unit-testable off-robot
with synthetic boxes AND imposes zero third-party dependencies at runtime in
the ROS container. Both the numerical counting filter (reasoning.counting) and
the object-reference candidate ranking call these to adjudicate relations like
"under the window", "on the table", "between the two columns", "farthest from
the columns".

Box convention matches the Marker fields in dummyVLM.cpp and eval_harness.scoring:
a box is a dict with center (cx, cy, cz) and full extents (l, w, h) in meters,
l along x, w along y, h along z. Heading is ignored (axis-aligned approximation),
consistent with the scoring proxy.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class Instance:
    """One detected object instance from the semantic map."""
    label: str
    bbox: dict                                   # cx,cy,cz,l,w,h[,heading]
    id: int = -1
    attributes: list[str] = field(default_factory=list)
    confidence: float = 1.0
    image_path: str = ""                         # best crop (.npy) for VLM verify


# --------------------------------------------------------------------------- #
# Low-level geometry on box dicts
# --------------------------------------------------------------------------- #
def center3(b: dict) -> tuple[float, float, float]:
    return (float(b["cx"]), float(b["cy"]), float(b["cz"]))


def center2(b: dict) -> tuple[float, float]:
    return (float(b["cx"]), float(b["cy"]))


def bounds(b: dict):
    """(min_xyz, max_xyz) corners of the axis-aligned box."""
    cx, cy, cz = center3(b)
    hl, hw, hh = b["l"] / 2.0, b["w"] / 2.0, b["h"] / 2.0
    return (cx - hl, cy - hw, cz - hh), (cx + hl, cy + hw, cz + hh)


def distance(a: dict, b: dict, planar: bool = True) -> float:
    """Center-to-center distance; planar (xy) by default."""
    ax, ay, az = center3(a)
    bx, by, bz = center3(b)
    if planar:
        return math.hypot(ax - bx, ay - by)
    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2 + (az - bz) ** 2)


def xy_overlap(a: dict, b: dict) -> bool:
    """Do the two boxes overlap in the horizontal (xy) footprint?"""
    a_min, a_max = bounds(a)
    b_min, b_max = bounds(b)
    return (a_min[0] <= b_max[0] and a_max[0] >= b_min[0]
            and a_min[1] <= b_max[1] and a_max[1] >= b_min[1])


# --------------------------------------------------------------------------- #
# Binary relations: relation(target, anchor) -> bool
# --------------------------------------------------------------------------- #
NEAR_THRESH_M = 1.5          # default planar radius for "near"
ON_GAP_M = 0.25              # vertical tolerance for "resting on top of"


def near(t: dict, a: dict, thresh: float = NEAR_THRESH_M) -> bool:
    return distance(t, a, planar=True) <= thresh


def below(t: dict, a: dict) -> bool:
    """target is below anchor: target center under anchor's center and the
    footprints overlap (so "below the window", "under the picture")."""
    return center3(t)[2] < center3(a)[2] and xy_overlap(t, a)


def above(t: dict, a: dict) -> bool:
    return center3(t)[2] > center3(a)[2] and xy_overlap(t, a)


# "under" is the same relation as "below" for our purposes
under = below


def on(t: dict, a: dict, gap: float = ON_GAP_M) -> bool:
    """target rests on top of anchor: target bottom ~ anchor top, footprints
    overlap (e.g. "potted plant on the table")."""
    t_min, _ = bounds(t)
    _, a_max = bounds(a)
    return xy_overlap(t, a) and abs(t_min[2] - a_max[2]) <= gap


def between(t: dict, a: dict, b: dict, tol: float = 1.0) -> bool:
    """target lies between anchors a and b in the xy plane: its perpendicular
    distance to the segment a-b is within tol AND its projection falls inside
    the segment (e.g. "between the two columns")."""
    px, py = center2(a)
    qx, qy = center2(b)
    xx, xy = center2(t)
    sx, sy = qx - px, qy - py
    seg_len2 = sx * sx + sy * sy
    if seg_len2 == 0.0:
        return distance(t, a, planar=True) <= tol
    s = ((xx - px) * sx + (xy - py) * sy) / seg_len2     # projection parameter
    if not (0.0 <= s <= 1.0):
        return False
    proj_x, proj_y = px + s * sx, py + s * sy
    return math.hypot(xx - proj_x, xy - proj_y) <= tol


# --------------------------------------------------------------------------- #
# Superlatives: pick one target from candidates relative to anchor(s)
# --------------------------------------------------------------------------- #
def closest_to(candidates: list[Instance], anchor: dict) -> Instance | None:
    if not candidates:
        return None
    return min(candidates, key=lambda c: distance(c.bbox, anchor, planar=True))


def farthest_from(candidates: list[Instance], anchor: dict) -> Instance | None:
    if not candidates:
        return None
    return max(candidates, key=lambda c: distance(c.bbox, anchor, planar=True))


# --------------------------------------------------------------------------- #
# Relation phrase dispatch
# --------------------------------------------------------------------------- #
# Maps a normalized relation phrase to a binary predicate (target, anchor)->bool.
_BINARY = {
    "near": near,
    "next to": near,
    "by": near,
    "beside": near,
    "below": below,
    "under": under,
    "underneath": under,
    "beneath": under,
    "above": above,
    "over": above,
    "on": on,
    "on top of": on,
}

# superlative phrases -> selector(candidates, anchor)->Instance
_SUPERLATIVE = {
    "closest to": closest_to,
    "nearest to": closest_to,
    "nearest": closest_to,
    "closest": closest_to,
    "farthest from": farthest_from,
    "furthest from": farthest_from,
    "farthest": farthest_from,
    "furthest": farthest_from,
}


def normalize_relation(phrase: str) -> str:
    return (phrase or "").strip().lower().rstrip(",")


def is_binary_relation(phrase: str) -> bool:
    return normalize_relation(phrase) in _BINARY


def is_superlative_relation(phrase: str) -> bool:
    return normalize_relation(phrase) in _SUPERLATIVE


def is_between_relation(phrase: str) -> bool:
    return normalize_relation(phrase) == "between"


def binary_predicate(phrase: str):
    return _BINARY.get(normalize_relation(phrase))


def superlative_selector(phrase: str):
    return _SUPERLATIVE.get(normalize_relation(phrase))
