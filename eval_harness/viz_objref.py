#!/usr/bin/env python3
"""Top-down scene maps to disambiguate object-reference GT targets.

For each object-reference question, renders a bird's-eye (x-y) map of the scene:
  - all objects             faint gray dots + tiny labels (room context)
  - candidate target objects bold, labelled "id:label" (+ their VLA color)
  - anchor-class objects     green (helps read the spatial relation)
  - geometric suggestion     red star
Open the PNG next to the scene's questions.pdf to read off the correct id, put
it in objref_candidates.json -> gt_object_id, then run apply_objref_gt.py.

Run: VLA3D_DIR=... python3 eval_harness/viz_objref.py [scene]
Writes eval_harness/viz/<scene>_q<idx>.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ai_module" / "src" / "vln_orchestrator"))
from vln_orchestrator.reasoning.counting import label_matches  # noqa: E402

from vla3d import load_scene_objects, vla3d_dir  # noqa: E402

CANDS = ROOT / "eval_harness" / "objref_candidates.json"
OUTDIR = ROOT / "eval_harness" / "viz"

_COLOR = {  # VLA color name -> matplotlib color
    "red": "red", "maroon": "darkred", "crimson": "crimson", "pink": "pink",
    "orange": "orange", "yellow": "gold", "olive": "olive", "green": "green",
    "teal": "teal", "blue": "blue", "navy": "navy", "purple": "purple",
    "brown": "saddlebrown", "tan": "tan", "beige": "wheat",
    "gray": "gray", "grey": "gray", "black": "black", "white": "lightgray",
    "silver": "silver", "gold": "goldenrod",
}


def render(entry, objs, out_path):
    fig, ax = plt.subplots(figsize=(9, 9))
    # all objects: context
    for o in objs:
        ax.plot(o.bbox["cx"], o.bbox["cy"], ".", color="0.8", ms=4, zorder=1)
        ax.annotate(o.label, (o.bbox["cx"], o.bbox["cy"]), fontsize=5,
                    color="0.6", zorder=1)
    # anchor-class objects
    anchor = entry.get("anchor", "")
    for o in objs:
        if anchor and label_matches(o.label, anchor):
            ax.plot(o.bbox["cx"], o.bbox["cy"], "s", color="green", ms=9,
                    alpha=0.5, zorder=2)
    # candidates
    sug = entry.get("suggested_object_id")
    by_id = {o.id: o for o in objs}
    for c in entry["candidates"]:
        o = by_id.get(c["id"])
        if not o:
            continue
        col = _COLOR.get((c["colors"] or ["blue"])[0], "steelblue")
        ax.plot(o.bbox["cx"], o.bbox["cy"], "o", color=col, ms=12,
                markeredgecolor="black", zorder=3)
        ax.annotate(f"{c['id']}:{o.label}", (o.bbox["cx"], o.bbox["cy"]),
                    fontsize=8, fontweight="bold", zorder=4,
                    xytext=(4, 4), textcoords="offset points")
        if c["id"] == sug:
            ax.plot(o.bbox["cx"], o.bbox["cy"], "*", color="red", ms=22,
                    zorder=5, label=f"suggested id={sug}")
    ax.set_aspect("equal")
    ax.set_title(f"{entry['scene']}\nQ: {entry['question']}\n"
                 f"green=anchor('{anchor}')  red★=suggested",
                 fontsize=9)
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main() -> int:
    root = vla3d_dir()
    scene_filter = sys.argv[1] if len(sys.argv) > 1 else None
    cands = json.loads(CANDS.read_text())
    OUTDIR.mkdir(exist_ok=True)

    objs_cache: dict[str, list] = {}
    counters: dict[str, int] = {}
    n = 0
    for e in cands:
        scene = e["scene"]
        if scene_filter and scene != scene_filter:
            continue
        objs = objs_cache.setdefault(scene, load_scene_objects(scene, root))
        idx = counters.get(scene, 2)            # object-ref are Q2/Q3
        counters[scene] = idx + 1
        out = OUTDIR / f"{scene}_q{idx}.png"
        render(e, objs, out)
        n += 1
    print(f"wrote {n} scene maps to {OUTDIR.relative_to(ROOT)}/")
    print("Open each PNG beside the scene's questions.pdf to read off the id.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
