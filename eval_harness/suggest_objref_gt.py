#!/usr/bin/env python3
"""Geometric GT *suggestions* for object-reference targets (confirm vs PDFs).

Loads VLA-3D objects as reasoning.Instance (perfect positions), decomposes each
question, and runs SemanticMap.resolve to suggest the target object_id by the
stated spatial relation. This is a fast accelerator for annotation, NOT final
ground truth — the authoritative answer is the orange-outlined object in each
scene's questions.pdf. Confirm/correct before trusting.

Writes the suggestion into objref_candidates.json as `suggested_object_id`
(leaving the human-owned `gt_object_id` untouched), plus a confidence flag:
  high  - relation resolved against a found anchor, single best
  low   - fell back (anchor not found / no relation) -> guess among candidates

Run: VLA3D_DIR=... python3 eval_harness/suggest_objref_gt.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ai_module" / "src" / "vln_orchestrator"))

from vln_orchestrator.perception.semantic_map_adapter import SemanticMap  # noqa: E402
from vln_orchestrator.reasoning.decomposition import heuristic_decompose  # noqa: E402
from vln_orchestrator.reasoning.spatial import Instance  # noqa: E402

from vla3d import load_scene_objects, vla3d_dir  # noqa: E402

CANDS = ROOT / "eval_harness" / "objref_candidates.json"


def scene_map(scene, root) -> SemanticMap:
    sm = SemanticMap()
    sm._instances = [
        Instance(label=o.label, bbox=o.bbox, id=o.id)   # attrs empty (color names noisy)
        for o in load_scene_objects(scene, root)
    ]
    return sm


def main() -> int:
    root = vla3d_dir()
    cands = json.loads(CANDS.read_text())
    maps: dict[str, SemanticMap] = {}

    high = low = 0
    for e in cands:
        scene = e["scene"]
        sm = maps.setdefault(scene, scene_map(scene, root))
        d = heuristic_decompose(e["question"])
        best = sm.resolve(d)
        # confidence: did we actually resolve via a found anchor + relation?
        resolved = bool(
            d.spatial_relation and d.anchor_object
            and sm.instances_of(d.anchor_object)
        )
        e["suggested_object_id"] = best.id if best else None
        e["suggest_confidence"] = "high" if (best and resolved) else "low"
        high += int(e["suggest_confidence"] == "high" and best is not None)
        low += int(e["suggest_confidence"] == "low")

    CANDS.write_text(json.dumps(cands, indent=2))
    print(f"suggestions written to {CANDS.relative_to(ROOT)}")
    print(f"  high-confidence (anchor+relation resolved): {high}")
    print(f"  low-confidence (fallback, needs PDF):       {low}")
    print("\nThese are SUGGESTIONS. Confirm each against the scene's questions.pdf,")
    print("then copy the correct id into gt_object_id and run apply_objref_gt.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
