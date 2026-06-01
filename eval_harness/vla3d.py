#!/usr/bin/env python3
"""Loader for VLA-3D Unity scene annotations (the challenge's 15 training scenes).

VLA-3D ships, per scene, `<scene>_object_result.csv` with every object's class,
oriented 3D bounding box, and dominant colors. This is the authoritative object
catalog we use for object-reference ground truth and as the color/attribute
source. (We do NOT auto-derive which object answers a given question — the
challenge questions are hand-authored and don't map cleanly onto VLA-3D's
generated statements; that target id is annotated by hand against the answer
PDFs, assisted by build_objref_candidates.py.)

Path comes from $VLA3D_DIR (see .env), e.g. /.../vla3d/Unity.

CSV columns used:
  object_id, raw_label, region_id,
  object_bbox_{cx,cy,cz,xlength,ylength,zlength,heading},
  object_color_{r,g,b,scheme}{1,2,3}
"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class VLAObject:
    id: int
    label: str
    region_id: int
    bbox: dict                         # cx,cy,cz,l,w,h,heading (our convention)
    colors: list[str] = field(default_factory=list)   # up to 3 dominant color names


def vla3d_dir() -> Path:
    d = os.environ.get("VLA3D_DIR", "")
    if not d:
        raise RuntimeError("VLA3D_DIR not set (add it to .env or the environment).")
    p = Path(d)
    if not p.is_dir():
        raise RuntimeError(f"VLA3D_DIR does not exist: {p}")
    return p


def _f(row: dict, key: str) -> float:
    v = row.get(key, "")
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def load_scene_objects(scene: str, root: Path | None = None) -> list[VLAObject]:
    """Parse <scene>/<scene>_object_result.csv into VLAObject list."""
    root = root or vla3d_dir()
    csv_path = root / scene / f"{scene}_object_result.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)
    objs: list[VLAObject] = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            bbox = {
                "cx": _f(row, "object_bbox_cx"),
                "cy": _f(row, "object_bbox_cy"),
                "cz": _f(row, "object_bbox_cz"),
                "l": _f(row, "object_bbox_xlength"),
                "w": _f(row, "object_bbox_ylength"),
                "h": _f(row, "object_bbox_zlength"),
                "heading": _f(row, "object_bbox_heading"),
            }
            colors = [
                row.get(f"object_color_scheme{i}", "_")
                for i in (1, 2, 3)
            ]
            colors = [c for c in colors if c and c != "_"]
            objs.append(VLAObject(
                id=int(_f(row, "object_id")),
                label=str(row.get("raw_label", "")).strip().lower(),
                region_id=int(_f(row, "region_id")),
                bbox=bbox,
                colors=colors,
            ))
    return objs


def object_by_id(scene: str, obj_id: int, root: Path | None = None) -> VLAObject | None:
    for o in load_scene_objects(scene, root):
        if o.id == obj_id:
            return o
    return None


if __name__ == "__main__":   # quick smoke check
    root = vla3d_dir()
    scenes = sorted(p.name for p in root.iterdir() if p.is_dir())
    print(f"VLA3D_DIR = {root}")
    print(f"scenes: {len(scenes)}")
    for s in scenes[:3]:
        objs = load_scene_objects(s, root)
        print(f"  {s}: {len(objs)} objects; sample:",
              [(o.id, o.label, o.colors[:1]) for o in objs[:4]])
