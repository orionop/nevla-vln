#!/usr/bin/env python3
"""Assist manual annotation of object-reference GT targets.

For each object-reference question we decompose the target class and list the
class-matching objects from VLA-3D as candidates (id, label, colors, position).
The human picks the correct `gt_object_id` by comparing with the answer PDF.

Where a scene has exactly ONE object of the target class, the target is
unambiguous and we auto-fill `gt_object_id` (this is not a guess — there is only
one possible object).

Output: eval_harness/objref_candidates.json  (edit the null gt_object_id fields,
then run apply_objref_gt.py to write boxes into ground_truth.json).

Run: VLA3D_DIR=... python3 eval_harness/build_objref_candidates.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ai_module" / "src" / "vln_orchestrator"))

from vln_orchestrator.reasoning.counting import label_matches  # noqa: E402
from vln_orchestrator.reasoning.decomposition import heuristic_decompose  # noqa: E402

from vla3d import load_scene_objects, vla3d_dir  # noqa: E402

OUT = ROOT / "eval_harness" / "objref_candidates.json"
QUESTIONS = ROOT / "questions" / "questions.json"


def main() -> int:
    root = vla3d_dir()
    qd = json.loads(QUESTIONS.read_text())

    out = []
    n_q = n_auto = n_multi = n_none = 0
    for scene_entry in qd:
        scene = scene_entry["scene"]
        objs = load_scene_objects(scene, root)
        for q in scene_entry["questions"]["object_reference"]:
            n_q += 1
            d = heuristic_decompose(q)
            cands = [o for o in objs if label_matches(o.label, d.target_object)]
            cand_list = [
                {
                    "id": o.id,
                    "label": o.label,
                    "colors": o.colors,
                    "pos": [round(o.bbox["cx"], 2), round(o.bbox["cy"], 2),
                            round(o.bbox["cz"], 2)],
                }
                for o in cands
            ]
            gt = None
            if len(cand_list) == 1:          # unique class -> unambiguous
                gt = cand_list[0]["id"]
                n_auto += 1
            elif len(cand_list) == 0:
                n_none += 1
            else:
                n_multi += 1
            out.append({
                "scene": scene,
                "question": q,
                "target": d.target_object,
                "attributes": d.attributes,
                "relation": d.spatial_relation,
                "anchor": d.anchor_object,
                "candidates": cand_list,
                "gt_object_id": gt,         # auto for singletons; else fill by hand
            })

    OUT.write_text(json.dumps(out, indent=2))
    print(f"object-reference questions:        {n_q}")
    print(f"  auto-filled (unique class):      {n_auto}")
    print(f"  need manual pick (>1 candidate): {n_multi}")
    print(f"  no class match (broaden/manual):  {n_none}")
    print(f"wrote {OUT.relative_to(ROOT)}")
    print("\nNext: fill the null gt_object_id fields using the answer PDFs, then")
    print("run apply_objref_gt.py to populate ground_truth.json.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
