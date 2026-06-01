#!/usr/bin/env python3
"""Scoring functions for the CMU VLN Challenge 2026 internal eval harness.

The official scoring is done by a hidden `challenge_evaluation_node`. These are
our *proxy* scorers used during development to track progress on the 15 training
scenes. They mirror the official scheme as closely as the public spec allows:

  Numerical            (/1): exact integer match -> 0 or 1.
  Object Reference     (/2): 3D bounding-box overlap (IoU) with the GT box,
                             linearly mapped to [0, 2]. (GT box pending VLA-3D.)
  Instruction Following(/6): the official node scores constraint order + avoid
                             regions along the *driven* trajectory. We can't see
                             that logic, so we use trajectory similarity to the
                             reference .ply (DTW + discrete Frechet) as a tuning
                             proxy, plus hooks for explicit constraint checks.

Everything here is numpy-only (no scipy dependency at runtime is required for the
core distances, though scipy is available).
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
# Numerical
# --------------------------------------------------------------------------- #
def score_numerical(predicted: int | None, gt: int) -> float:
    """0 or 1, exact match (official scheme)."""
    if predicted is None:
        return 0.0
    return 1.0 if int(predicted) == int(gt) else 0.0


# --------------------------------------------------------------------------- #
# Object Reference  (3D axis-aligned bounding-box IoU -> [0, 2])
# --------------------------------------------------------------------------- #
def aabb_iou_3d(box_a: dict, box_b: dict) -> float:
    """3D IoU of two axis-aligned boxes.

    Each box is a dict with center (cx, cy, cz) and extents (l, w, h) in meters,
    matching the Marker fields used in dummyVLM.cpp (pose.position + scale).
    Heading is ignored here (axis-aligned approximation); good enough for a
    development proxy and exact when both boxes are axis-aligned.
    """
    def bounds(b):
        return (
            np.array([b["cx"] - b["l"] / 2, b["cy"] - b["w"] / 2, b["cz"] - b["h"] / 2]),
            np.array([b["cx"] + b["l"] / 2, b["cy"] + b["w"] / 2, b["cz"] + b["h"] / 2]),
        )

    a_min, a_max = bounds(box_a)
    b_min, b_max = bounds(box_b)
    inter = np.clip(np.minimum(a_max, b_max) - np.maximum(a_min, b_min), 0, None)
    inter_vol = float(np.prod(inter))
    vol_a = float(np.prod(a_max - a_min))
    vol_b = float(np.prod(b_max - b_min))
    union = vol_a + vol_b - inter_vol
    return inter_vol / union if union > 0 else 0.0


def score_object_reference(pred_box: dict | None, gt_box: dict | None) -> float:
    """IoU mapped linearly to [0, 2]. Returns 0 if either box is missing."""
    if pred_box is None or gt_box is None:
        return 0.0
    return 2.0 * aabb_iou_3d(pred_box, gt_box)


# --------------------------------------------------------------------------- #
# Instruction Following  (trajectory similarity proxy)
# --------------------------------------------------------------------------- #
def load_ply_trajectory(path: str, drop_z: bool = True) -> np.ndarray:
    """Load an ASCII PLY trajectory as an (N, 2) or (N, 3) float array."""
    with open(path, "r") as f:
        lines = f.read().splitlines()
    # find end_header
    start = next(i for i, ln in enumerate(lines) if ln.strip() == "end_header") + 1
    pts = []
    for ln in lines[start:]:
        ln = ln.strip()
        if not ln:
            continue
        x, y, z = (float(v) for v in ln.split()[:3])
        pts.append((x, y) if drop_z else (x, y, z))
    return np.asarray(pts, dtype=float)


def _resample(path: np.ndarray, n: int) -> np.ndarray:
    """Arc-length resample a polyline to n evenly spaced points."""
    if len(path) <= 1:
        return np.repeat(path, n, axis=0)[:n]
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    if cum[-1] == 0:
        return np.repeat(path[:1], n, axis=0)
    targets = np.linspace(0, cum[-1], n)
    out = np.empty((n, path.shape[1]))
    for d in range(path.shape[1]):
        out[:, d] = np.interp(targets, cum, path[:, d])
    return out


def dtw_distance(a: np.ndarray, b: np.ndarray, resample_n: int = 200) -> float:
    """Mean per-step DTW distance between two trajectories (resampled)."""
    a = _resample(a, resample_n)
    b = _resample(b, resample_n)
    na, nb = len(a), len(b)
    D = np.full((na + 1, nb + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, na + 1):
        ai = a[i - 1]
        for j in range(1, nb + 1):
            cost = np.linalg.norm(ai - b[j - 1])
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return float(D[na, nb] / (na + nb))


def discrete_frechet(a: np.ndarray, b: np.ndarray, resample_n: int = 200) -> float:
    """Discrete Frechet distance between two trajectories (resampled)."""
    a = _resample(a, resample_n)
    b = _resample(b, resample_n)
    na, nb = len(a), len(b)
    ca = np.full((na, nb), -1.0)

    def c(i, j):
        if ca[i, j] > -1:
            return ca[i, j]
        d = np.linalg.norm(a[i] - b[j])
        if i == 0 and j == 0:
            ca[i, j] = d
        elif i == 0:
            ca[i, j] = max(c(0, j - 1), d)
        elif j == 0:
            ca[i, j] = max(c(i - 1, 0), d)
        else:
            ca[i, j] = max(min(c(i - 1, j), c(i - 1, j - 1), c(i, j - 1)), d)
        return ca[i, j]

    # iterative fill to avoid recursion limits on long paths
    for i in range(na):
        for j in range(nb):
            d = np.linalg.norm(a[i] - b[j])
            if i == 0 and j == 0:
                ca[i, j] = d
            elif i == 0:
                ca[i, j] = max(ca[i, j - 1], d)
            elif j == 0:
                ca[i, j] = max(ca[i - 1, j], d)
            else:
                ca[i, j] = max(min(ca[i - 1, j], ca[i - 1, j - 1], ca[i, j - 1]), d)
    return float(ca[na - 1, nb - 1])


def instruction_similarity(pred_path: np.ndarray, gt_ply_path: str) -> dict:
    """Proxy similarity metrics between a predicted path and the GT .ply.

    Returns raw distances (meters). Lower is better. We deliberately do NOT
    collapse these into a /6 score yet, because the official score depends on
    constraint order / avoid-regions we cannot see — tune against these instead.
    """
    gt = load_ply_trajectory(gt_ply_path, drop_z=True)
    pred = np.asarray(pred_path, dtype=float)[:, :2]
    return {
        "dtw_m": dtw_distance(pred, gt),
        "frechet_m": discrete_frechet(pred, gt),
        "endpoint_err_m": float(np.linalg.norm(pred[-1] - gt[-1])),
        "gt_points": len(gt),
    }
