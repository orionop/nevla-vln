#!/usr/bin/env python3
"""Build a machine-readable ground-truth file for the CMU VLN Challenge 2026
training set.

Ground-truth sources (per scene, under questions/<scene>/):
  - questions.pdf          : numerical answer ("Print N in terminal") + annotated
                             images for object-reference targets (visual only).
  - trajectory_q4.ply      : reference trajectory for instruction-following Q4.
  - trajectory_q5.ply      : reference trajectory for instruction-following Q5.
  - ../questions.json      : the canonical question text for all 15 scenes.

What we can extract automatically:
  - numerical answers          -> integer, parsed from PDF text  (COMPLETE)
  - instruction-following GT    -> .ply reference paths           (COMPLETE)
  - object-reference GT bbox    -> NOT in repo. Only an orange outline in a 2D
                                   render. A numeric 3D GT bbox must come from the
                                   VLA-3D processed scene annotations (download).
                                   Left as null here; see object_reference[*].gt_bbox.

Usage:
  python3 eval_harness/build_ground_truth.py
  -> writes eval_harness/ground_truth.json
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
QUESTIONS_DIR = REPO / "questions"
OUT = Path(__file__).resolve().parent / "ground_truth.json"

PRINT_RE = re.compile(r"Print\s+(\d+)\s+in\s+terminal", re.IGNORECASE)


def numerical_answer_from_pdf(pdf_path: Path) -> int | None:
    """Extract the integer numerical answer from a scene's questions.pdf."""
    try:
        text = subprocess.check_output(
            ["pdftotext", str(pdf_path), "-"], stderr=subprocess.DEVNULL
        ).decode("utf-8", errors="ignore")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    m = PRINT_RE.search(text)
    return int(m.group(1)) if m else None


def main() -> None:
    questions = json.loads((QUESTIONS_DIR / "questions.json").read_text())
    ground_truth: dict[str, dict] = {}

    for entry in questions:
        scene = entry["scene"]
        q = entry["questions"]
        scene_dir = QUESTIONS_DIR / scene
        pdf = scene_dir / "questions.pdf"

        num_answer = numerical_answer_from_pdf(pdf) if pdf.exists() else None

        # instruction-following: pair each question with its trajectory .ply
        instr = []
        for i, text in enumerate(q.get("instruction_following", []), start=4):
            ply = scene_dir / f"trajectory_q{i}.ply"
            instr.append({
                "question": text,
                "trajectory_ply": str(ply.relative_to(REPO)) if ply.exists() else None,
            })

        ground_truth[scene] = {
            "numerical": {
                "question": q.get("numerical", [None])[0],
                "answer": num_answer,  # int, scored 0/1 by exact match
            },
            "object_reference": [
                {
                    "question": text,
                    "gt_bbox": None,  # TODO: fill from VLA-3D processed scene annotations
                }
                for text in q.get("object_reference", [])
            ],
            "instruction_following": instr,
        }

    OUT.write_text(json.dumps(ground_truth, indent=2))
    # quick coverage report
    n_num = sum(1 for s in ground_truth.values() if s["numerical"]["answer"] is not None)
    n_traj = sum(
        1 for s in ground_truth.values()
        for q in s["instruction_following"] if q["trajectory_ply"]
    )
    n_objref = sum(len(s["object_reference"]) for s in ground_truth.values())
    print(f"Wrote {OUT.relative_to(REPO)}")
    print(f"  scenes:                  {len(ground_truth)}")
    print(f"  numerical answers:       {n_num}/{len(ground_truth)}")
    print(f"  instruction trajectories:{n_traj}")
    print(f"  object-reference items:  {n_objref}  (gt_bbox pending VLA-3D)")


if __name__ == "__main__":
    main()
