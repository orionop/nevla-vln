#!/usr/bin/env python3
"""Offline upper-bound eval: run OUR reasoning on the VLA-3D ground-truth map and
score every dev scene. No sim, no API (heuristic decomposition only).

This feeds the orchestrator's reasoning a PERFECT semantic map (every VLA-3D object
with true box + label) and asks each challenge question, then scores the answer:
  numerical            -> count_matching        vs gt int      (exact, 0/1)
  object_reference     -> SemanticMap.resolve   vs gt_bbox      (3D IoU, 0-2)
  instruction_following-> parse + locate path   vs gt trajectory(DTW/Frechet proxy)

It isolates REASONING quality from perception/exploration: a low score here is a
logic bug we can fix offline; a high score here but low live = perception/explore.

Run: VLA3D_DIR=/path/to/vla3d/Unity python3 eval_harness/run_eval.py [scene]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))                                   # eval_harness
sys.path.insert(0, str(ROOT / "ai_module/src/vln_orchestrator"))  # our package

from vla3d import load_scene_objects, vla3d_dir  # noqa: E402
import scoring  # noqa: E402

from vln_orchestrator.reasoning.spatial import Instance  # noqa: E402
from vln_orchestrator.reasoning.decomposition import heuristic_decompose  # noqa: E402
from vln_orchestrator.reasoning.counting import count_matching  # noqa: E402
from vln_orchestrator.reasoning.instruction_parser import parse_instruction  # noqa: E402
from vln_orchestrator.perception.semantic_map_adapter import SemanticMap  # noqa: E402


def scene_map(scene: str) -> SemanticMap:
    """VLA-3D objects -> a SemanticMap of perfect Instances."""
    sm = SemanticMap()
    # attributes=[] mirrors the live ObjectNode adapter (color/size are deferred
    # to VLM verification, not geometric filtering) so the offline number reflects
    # the real pipeline rather than zeroing on color-vocab mismatch (maroon!=red).
    sm._instances = [
        Instance(label=o.label, bbox=o.bbox, id=o.id, attributes=[])
        for o in load_scene_objects(scene)
    ]
    return sm


def eval_numerical(sm, q, gt):
    pred = count_matching(heuristic_decompose(q), sm.all_instances())
    return pred, gt, scoring.score_numerical(pred, gt)


def eval_objref(sm, q, gt_bbox):
    inst = sm.resolve(heuristic_decompose(q))
    pred = inst.bbox if inst else None
    return (inst.id if inst else None), scoring.score_object_reference(pred, gt_bbox)


def eval_instruction(sm, q, ply):
    goals = parse_instruction(q).ordered_waypoints
    path = []
    for g in goals:
        inst = sm.locate(g.landmark)
        if inst:
            path.append((inst.bbox["cx"], inst.bbox["cy"]))
    if not path or not (ROOT / ply).is_file():
        return len(path), None
    return len(path), scoring.instruction_similarity(path, str(ROOT / ply))


def _scene_set(arg: str | None) -> tuple[set[str] | None, str]:
    """Resolve which scenes to eval. Default = DEV split (anti-overfit: never tune
    on holdout). `--holdout` runs the private test set; a scene name runs just it."""
    split = json.load(open(HERE / "split.json"))
    if arg == "--holdout":
        return set(split["holdout"]), "HOLDOUT (private test — do NOT tune on this)"
    if arg == "--all":
        return None, "ALL 15 (reporting only — do NOT tune)"
    if arg:
        return {arg}, arg
    return set(split["dev"]), "DEV (8)"


def main(arg: str | None = None):
    scenes, label = _scene_set(arg)
    gt = json.load(open(HERE / "ground_truth.json"))
    root = vla3d_dir()
    num_hit = num_tot = 0
    obj_sum = obj_tot = 0.0
    print(f"# split: {label}")
    print(f"{'scene':<16}{'type':<6}{'pred':>6} {'gt/score'}")
    for scene, s in gt.items():
        if scenes is not None and scene not in scenes:
            continue
        if not (root / scene).is_dir():
            continue
        sm = scene_map(scene)
        # numerical
        n = s.get("numerical") or {}
        if n.get("answer") is not None:
            pred, g, sc = eval_numerical(sm, n["question"], n["answer"])
            num_hit += sc; num_tot += 1
            print(f"{scene:<16}{'num':<6}{pred:>6} gt={g} score={sc:.0f}")
        # object reference
        for r in s.get("object_reference", []):
            if not r.get("gt_bbox"):
                continue
            pid, sc = eval_objref(sm, r["question"], r["gt_bbox"])
            obj_sum += sc; obj_tot += 1
            print(f"{scene:<16}{'obj':<6}{str(pid):>6} score={sc:.2f}/2")
        # instruction following (proxy)
        for r in s.get("instruction_following", []):
            npts, sim = eval_instruction(sm, r["question"], r.get("trajectory_ply", ""))
            d = f"dtw={sim['dtw_m']:.2f}m end={sim['endpoint_err_m']:.2f}m" if sim else "no-path"
            print(f"{scene:<16}{'instr':<6}{npts:>6} {d}")
    print("\n=== summary ===")
    if num_tot:
        print(f"numerical:  {num_hit}/{num_tot} exact ({100*num_hit/num_tot:.0f}%)")
    if obj_tot:
        print(f"objref:     {obj_sum:.1f}/{2*obj_tot:.0f} pts (mean IoU {obj_sum/(2*obj_tot):.2f})")
    print("instruction: DTW/Frechet proxy per row (lower = closer to GT path)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
