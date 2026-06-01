#!/usr/bin/env python3
"""Unit tests for reasoning.spatial predicates and reasoning.counting.

Pure geometry on synthetic instances — no ROS / VLM / perception needed.

Run: python3 ai_module/src/vln_orchestrator/test/test_spatial_counting.py
"""
from __future__ import annotations

import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG))

from vln_orchestrator.reasoning.decomposition import Decomposition  # noqa: E402
from vln_orchestrator.reasoning.counting import (  # noqa: E402
    count_matching,
    label_matches,
)
from vln_orchestrator.reasoning import spatial  # noqa: E402
from vln_orchestrator.reasoning.spatial import Instance  # noqa: E402


def box(cx, cy, cz, l=0.5, w=0.5, h=0.5):
    return {"cx": cx, "cy": cy, "cz": cz, "l": l, "w": w, "h": h, "heading": 0.0}


def test_label_matches():
    assert label_matches("sofas", "sofa")
    assert label_matches("sofa", "sofas")
    assert label_matches("potted plant", "plant")
    assert label_matches("office chair", "chair")
    assert not label_matches("window", "sofa")
    assert not label_matches("table", "")
    print("✓ label matching (plurals + compounds)")


def test_binary_predicates():
    table = box(0, 0, 0.5, l=2, w=2, h=1)          # top at z=1.0
    plant = box(0, 0, 1.2, l=0.3, w=0.3, h=0.4)     # bottom at z=1.0, on table
    window = box(0, 0, 2.0, l=1, w=0.2, h=1)        # high up
    sofa = box(0, 0, 0.3, l=1, w=1, h=0.6)          # below the window
    far = box(10, 10, 0.3)

    assert spatial.on(plant, table)
    assert not spatial.on(far, table)
    assert spatial.below(sofa, window)
    assert spatial.above(window, sofa)
    assert spatial.near(sofa, table)
    assert not spatial.near(far, table)
    print("✓ binary predicates (on / below / above / near)")


def test_between():
    col_a = box(-2, 0, 1.0)
    col_b = box(2, 0, 1.0)
    mid = box(0, 0, 0.5)
    off = box(0, 5, 0.5)
    outside = box(5, 0, 0.5)                         # beyond the segment
    assert spatial.between(mid, col_a, col_b)
    assert not spatial.between(off, col_a, col_b)
    assert not spatial.between(outside, col_a, col_b)
    print("✓ between (segment proximity + projection bounds)")


def test_superlatives():
    anchor = box(0, 0, 1.0)
    cands = [
        Instance("plant", box(1, 0, 0.5), id=1),
        Instance("plant", box(5, 0, 0.5), id=2),
        Instance("plant", box(9, 0, 0.5), id=3),
    ]
    assert spatial.closest_to(cands, anchor).id == 1
    assert spatial.farthest_from(cands, anchor).id == 3
    print("✓ superlatives (closest_to / farthest_from)")


def test_count_unconstrained():
    insts = [
        Instance("sofa", box(0, 0, 0.3)),
        Instance("sofa", box(3, 0, 0.3)),
        Instance("table", box(6, 0, 0.5)),
    ]
    d = Decomposition(target_object="sofas")
    assert count_matching(d, insts) == 2
    print("✓ count unconstrained (class only)")


def test_count_binary_relation():
    # "How many sofas are below a window?" — two sofas, one below a window.
    window = Instance("window", box(0, 0, 2.0, l=1, w=0.2, h=1))
    sofa_under = Instance("sofa", box(0, 0, 0.3, l=1, w=1, h=0.6))
    sofa_else = Instance("sofa", box(8, 0, 0.3, l=1, w=1, h=0.6))
    insts = [window, sofa_under, sofa_else]
    d = Decomposition(
        target_object="sofas", spatial_relation="below", anchor_object="window"
    )
    assert count_matching(d, insts) == 1
    print("✓ count with binary relation (sofas below a window)")


def test_count_missing_anchor_falls_back():
    # anchor class never detected -> fall back to class count (don't assert 0).
    insts = [Instance("sofa", box(0, 0, 0.3)), Instance("sofa", box(3, 0, 0.3))]
    d = Decomposition(
        target_object="sofa", spatial_relation="below", anchor_object="window"
    )
    assert count_matching(d, insts) == 2
    print("✓ count falls back to class count when anchor absent")


def test_count_superlative_collapses():
    anchor = Instance("hookah", box(0, 0, 0.5))
    plants = [
        Instance("plant", box(1, 0, 0.5)),
        Instance("plant", box(9, 0, 0.5)),
    ]
    d = Decomposition(
        target_object="plant",
        spatial_relation="farthest from",
        anchor_object="hookah",
    )
    # superlative picks a single object
    assert count_matching(d, [anchor] + plants) == 1
    print("✓ count superlative collapses to single match")


if __name__ == "__main__":
    test_label_matches()
    test_binary_predicates()
    test_between()
    test_superlatives()
    test_count_unconstrained()
    test_count_binary_relation()
    test_count_missing_anchor_falls_back()
    test_count_superlative_collapses()
    print("\nAll spatial + counting tests passed.")
