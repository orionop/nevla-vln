#!/usr/bin/env python3
"""Tests for perception.semantic_map_adapter — ObjectNode -> Instance conversion
and SemanticMap queries. Uses synthetic ObjectNode-like objects (duck-typed), so
no ROS / tare_planner build is required.

Run: python3 ai_module/src/vln_orchestrator/test/test_semantic_map_adapter.py
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG))

from vln_orchestrator.perception.semantic_map_adapter import (  # noqa: E402
    SemanticMap,
    bbox_from_corners,
    object_node_to_instance,
)
from vln_orchestrator.reasoning.decomposition import heuristic_decompose  # noqa: E402


# --- synthetic ObjectNode stand-ins ---------------------------------------- #
@dataclass
class P:
    x: float
    y: float
    z: float


@dataclass
class FakeNode:
    object_id: list
    label: str
    bbox3d: list           # 8 P corners
    position: P = field(default_factory=lambda: P(0, 0, 0))
    img_path: str = ""


@dataclass
class FakeMsg:
    nodes: list


def corners(cx, cy, cz, l=0.5, w=0.5, h=0.5):
    """8 corners (heading 0): bottom 0-3, top 4-7, matching get_bbox_3d_oriented."""
    hl, hw, hh = l / 2, w / 2, h / 2
    bottom = [
        P(cx - hl, cy - hw, cz - hh),
        P(cx + hl, cy - hw, cz - hh),
        P(cx + hl, cy + hw, cz - hh),
        P(cx - hl, cy + hw, cz - hh),
    ]
    top = [P(p.x, p.y, p.z + h) for p in bottom]
    return bottom + top


def node(label, cx, cy, cz, l=0.5, w=0.5, h=0.5, oid=0, img=""):
    return FakeNode([oid], label, corners(cx, cy, cz, l, w, h), P(cx, cy, cz), img)


def test_corners_to_bbox():
    b = bbox_from_corners(corners(1, 2, 3, l=2, w=1, h=0.5))
    assert abs(b["cx"] - 1) < 1e-6 and abs(b["cy"] - 2) < 1e-6 and abs(b["cz"] - 3) < 1e-6
    assert abs(b["l"] - 2) < 1e-6 and abs(b["w"] - 1) < 1e-6 and abs(b["h"] - 0.5) < 1e-6
    assert abs(b["heading"]) < 1e-6
    print("✓ corners -> oriented bbox (center/extent/heading)")


def test_node_to_instance():
    inst = object_node_to_instance(node("Sofa", 0, 0, 0.3, oid=7, img="/o/7.npy"))
    assert inst.label == "sofa"          # lowercased
    assert inst.id == 7
    assert inst.image_path == "/o/7.npy"
    print("✓ ObjectNode -> Instance (label/id/image_path)")


def test_map_queries():
    sm = SemanticMap()
    sm.update_from_msg(FakeMsg([
        node("sofa", 0, 0, 0.3, l=2, w=1, h=0.6, oid=1),
        node("sofa", 8, 0, 0.3, l=2, w=1, h=0.6, oid=2),
        node("window", 0, 0, 2.0, l=1, w=0.2, h=1, oid=3),
        node("table", 3, 3, 0.5, oid=4),
    ]))
    assert len(sm) == 4
    assert len(sm.instances_of("sofas")) == 2          # plural-tolerant
    assert len(sm.instances_of("window")) == 1
    print("✓ SemanticMap update + instances_of")


def test_resolve_binary_relation():
    # "sofa below the window" -> the sofa under the window (id 1), not the far one.
    sm = SemanticMap()
    sm.update_from_msg(FakeMsg([
        node("sofa", 0, 0, 0.3, l=2, w=1, h=0.6, oid=1),
        node("sofa", 8, 0, 0.3, l=2, w=1, h=0.6, oid=2),
        node("window", 0, 0, 2.0, l=1, w=0.2, h=1, oid=3),
    ]))
    decomp = heuristic_decompose("Find the sofa below the window")
    best = sm.resolve(decomp)
    assert best is not None and best.id == 1
    print("✓ resolve binary relation (sofa below the window -> correct one)")


def test_resolve_superlative_and_locate():
    sm = SemanticMap()
    sm.update_from_msg(FakeMsg([
        node("hookah", 0, 0, 0.5, oid=10),
        node("plant", 1, 0, 0.5, oid=11),
        node("plant", 9, 0, 0.5, oid=12),
    ]))
    # superlative via resolve
    d = heuristic_decompose("Find the plant farthest from the hookah")
    best = sm.resolve(d)
    assert best is not None and best.id == 12
    # locate() should resolve the same phrase to a point
    inst = sm.locate("plant farthest from the hookah")
    assert inst is not None and inst.id == 12
    print("✓ resolve superlative + locate(phrase)")


if __name__ == "__main__":
    test_corners_to_bbox()
    test_node_to_instance()
    test_map_queries()
    test_resolve_binary_relation()
    test_resolve_superlative_and_locate()
    print("\nAll semantic-map adapter tests passed.")
