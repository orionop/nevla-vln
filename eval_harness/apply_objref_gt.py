#!/usr/bin/env python3
"""Write object-reference GT boxes into ground_truth.json from annotated ids.

Reads eval_harness/objref_candidates.json (the `gt_object_id` fields you filled),
looks up each target object's box in VLA-3D, and sets `gt_bbox` on the matching
object_reference entry in ground_truth.json. Idempotent and incremental: entries
whose gt_object_id is still null are skipped, so you can run it as you annotate.

Run: VLA3D_DIR=... python3 eval_harness/apply_objref_gt.py
"""
from __future__ import annotations

import json
from pathlib import Path

from vla3d import object_by_id, vla3d_dir

ROOT = Path(__file__).resolve().parents[1]
CANDS = ROOT / "eval_harness" / "objref_candidates.json"
GT = ROOT / "eval_harness" / "ground_truth.json"


def main() -> int:
    root = vla3d_dir()
    cands = json.loads(CANDS.read_text())
    gt = json.loads(GT.read_text())

    filled = skipped = missing = 0
    for entry in cands:
        oid = entry.get("gt_object_id")
        if oid is None:
            skipped += 1
            continue
        obj = object_by_id(entry["scene"], int(oid), root)
        if obj is None:
            print(f"  ! {entry['scene']} id={oid} not found in CSV")
            missing += 1
            continue
        # find the matching object_reference entry by scene + question text
        scene_gt = gt.get(entry["scene"], {})
        for ref in scene_gt.get("object_reference", []):
            if ref["question"] == entry["question"]:
                ref["gt_bbox"] = {**obj.bbox, "object_id": obj.id, "label": obj.label}
                filled += 1
                break

    GT.write_text(json.dumps(gt, indent=2))
    total = sum(len(s.get("object_reference", [])) for s in gt.values())
    have = sum(
        1 for s in gt.values()
        for r in s.get("object_reference", []) if r.get("gt_bbox")
    )
    print(f"filled this run: {filled} | still unannotated: {skipped} | missing id: {missing}")
    print(f"object-reference GT coverage: {have}/{total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
