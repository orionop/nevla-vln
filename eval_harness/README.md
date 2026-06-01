# eval_harness

Internal scoring harness for the CMU VLN Challenge 2026 (development only — kept
out of `ai_module/` so it is never part of the submission/runtime).

The official scorer is a hidden `challenge_evaluation_node`. These tools let us
track progress on the 15 training scenes with proxy scorers that mirror the public
scheme.

## Files
- `build_ground_truth.py` — parses `questions/<scene>/questions.pdf` and the
  `.ply` trajectories into `ground_truth.json`. Re-run after any data change.
- `ground_truth.json` — generated. 15 scenes:
  - `numerical.answer` — exact integer (from PDF text). **Complete.**
  - `instruction_following[*].trajectory_ply` — reference path. **Complete.**
  - `object_reference[*].gt_bbox` — **null, pending VLA-3D** (see below).
- `scoring.py` — scorers: `score_numerical` (0/1), `score_object_reference`
  (3D AABB IoU → 0-2), trajectory distances `dtw_distance` / `discrete_frechet`
  and `instruction_similarity` (proxy for the /6 type).
- `test_scoring.py` — sanity tests over the real `.ply` fixtures.
- `build_challenge_vocab.py` — mines object phrases from all 75 questions
  (spaCy `en_core_web_sm`) and writes the open-vocab detector vocabulary to
  `ai_module/src/vln_orchestrator/config/challenge_classes.yaml` (69 classes /
  86 prompts), plus a human-readable `challenge_vocab_report.txt`. Replaces
  SysNav's `objects.yaml`, which had the challenge's key classes (window, door,
  pillow, table, refrigerator…) commented out. **Jazzy integration:** point
  `detection_node`'s `object_file` parameter at the generated yaml.

## Run
```bash
python3 eval_harness/build_ground_truth.py    # regenerate ground_truth.json
python3 eval_harness/test_scoring.py          # run scorer tests
python3 eval_harness/build_challenge_vocab.py # regenerate detector vocabulary
```

## ⚠️ VLA-3D dependency
Object-reference GT 3D boxes are NOT in this repo — the PDFs only outline the
target in a 2D render. Numeric GT boxes (and the runtime object/region/attribute
data SysNav's semantic mapping consumes) come from the **VLA-3D** processed scene
annotations: https://github.com/HaochenZ11/VLA-3D . Download covering the 15
training scenes, then populate `object_reference[*].gt_bbox` to enable IoU scoring.
