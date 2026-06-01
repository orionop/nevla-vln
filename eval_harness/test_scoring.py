#!/usr/bin/env python3
"""Self-contained sanity tests for eval_harness/scoring.py.

Run: python3 eval_harness/test_scoring.py
Uses the real training-set .ply trajectories as fixtures.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scoring import (
    aabb_iou_3d,
    discrete_frechet,
    dtw_distance,
    instruction_similarity,
    load_ply_trajectory,
    score_numerical,
    score_object_reference,
)

REPO = Path(__file__).resolve().parent.parent
GT = json.loads((Path(__file__).resolve().parent / "ground_truth.json").read_text())


def approx(a, b, tol=1e-6):
    return abs(a - b) <= tol


def test_numerical():
    assert score_numerical(2, 2) == 1.0
    assert score_numerical(3, 2) == 0.0
    assert score_numerical(None, 2) == 0.0
    print("✓ numerical exact-match")


def test_iou():
    box = {"cx": 0, "cy": 0, "cz": 0, "l": 2, "w": 2, "h": 2}
    assert approx(aabb_iou_3d(box, box), 1.0)                  # identical -> 1
    assert score_object_reference(box, box) == 2.0            # -> /2 full marks
    far = {"cx": 10, "cy": 0, "cz": 0, "l": 2, "w": 2, "h": 2}
    assert aabb_iou_3d(box, far) == 0.0                       # disjoint -> 0
    # half-overlap along x: shift by 1 (extent 2) -> inter 1*2*2=4, union 8+8-4=12
    half = {"cx": 1, "cy": 0, "cz": 0, "l": 2, "w": 2, "h": 2}
    assert approx(aabb_iou_3d(box, half), 4 / 12)
    print("✓ 3D AABB IoU + object-reference mapping")


def test_trajectory_self_consistency():
    # self-distance must be ~0; cross-distance to a different scene must be > 0
    ply_a = REPO / "questions/arabic_room/trajectory_q4.ply"
    ply_b = REPO / "questions/office_1/trajectory_q4.ply"
    a = load_ply_trajectory(str(ply_a))
    b = load_ply_trajectory(str(ply_b))
    assert a.shape[1] == 2 and len(a) > 10
    self_dtw = dtw_distance(a, a)
    self_fre = discrete_frechet(a, a)
    cross_dtw = dtw_distance(a, b)
    assert self_dtw < 1e-6, f"self DTW not ~0: {self_dtw}"
    assert self_fre < 1e-6, f"self Frechet not ~0: {self_fre}"
    assert cross_dtw > self_dtw, "cross-scene distance should exceed self distance"
    print(f"✓ trajectory self/cross consistency (self_dtw={self_dtw:.2e}, "
          f"cross_dtw={cross_dtw:.3f} m)")


def test_instruction_similarity_perfect_path():
    # feeding the GT path back in as the prediction -> near-zero distances
    ply = REPO / "questions/studio/trajectory_q5.ply"
    gt = load_ply_trajectory(str(ply))
    res = instruction_similarity(gt, str(ply))
    assert res["dtw_m"] < 1e-6 and res["frechet_m"] < 1e-6
    assert res["endpoint_err_m"] < 1e-6
    print(f"✓ instruction_similarity perfect-path (gt_points={res['gt_points']})")


def test_ground_truth_coverage():
    assert len(GT) == 15
    assert all(s["numerical"]["answer"] is not None for s in GT.values())
    n_traj = sum(1 for s in GT.values() for q in s["instruction_following"]
                 if q["trajectory_ply"])
    assert n_traj == 30
    print("✓ ground_truth.json coverage (15 scenes, 15 numerical, 30 trajectories)")


if __name__ == "__main__":
    test_numerical()
    test_iou()
    test_trajectory_self_consistency()
    test_instruction_similarity_perfect_path()
    test_ground_truth_coverage()
    print("\nAll scoring tests passed.")
