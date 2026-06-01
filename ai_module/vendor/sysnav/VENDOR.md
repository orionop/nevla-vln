# Vendored SysNav modules

Source: https://github.com/zwandering/SysNav (commit cloned 2026-06-01)
License: **BSD 3-Clause** (© 2026 Haokun Zhu and contributors) — see `LICENSE`.
Paper: https://arxiv.org/abs/2603.06914 · Project: https://cmu-vln.github.io

## What's here
- `vlm_node/` — SysNav's VLM reasoning node (room typing, instruction
  decomposition, object verification). Pure Python + prompts.
- `semantic_mapping/` — detection + 3D semantic instance mapping. **SAM2 weights
  / `external/` were intentionally excluded** (63 MB) — install SAM2 separately
  on the Jazzy box (see SysNav `requirement.txt` + `set_yolo_*.py`).

## Why "vendor" and not run as-is
`vlm_node` is tightly coupled to SysNav's custom ROS messages (`tare_planner.msg`:
`RoomType`, `NavigationQuery`, `VlmAnswer`, `ObjectType`, …) and the exploration
planner architecture. It therefore **does not `colcon build` standalone** inside
`ai_module`. It lives here as integration reference.

Our actual runtime reasoning is a clean, ROS-free re-implementation in
`ai_module/src/vln_orchestrator/vln_orchestrator/reasoning/` (decomposition +
verification + VLM client), adapted from these prompts but unit-tested off-robot.

## Integration plan (Jazzy box)
- Reuse `semantic_mapping` for 3D object instances/bboxes; expose a
  `semantic_map.instances_of(class, attributes)` / `.locate(landmark)` API the
  handlers call (see hooks in `handlers/*.py`).
- Reuse `route_planner` (from the full SysNav clone) for via/avoid path planning.
- **Detection vocabulary must be expanded**: SysNav's `objects.yaml` is tuned to
  its benchmarks and comments out challenge-critical classes (pillow, window,
  refrigerator, door, …). The challenge leans on pillow/vase/picture/lamp/
  potted-plant/bowl/stool — rebuild the open-vocab class list for these scenes.

## License obligations (we comply)
Keep this `LICENSE`; retain copyright notices in any copied files; no endorsement
claims. (Weak-copyleft deps LGPL `gstthetauvc` / MPL2 Eigen live in the full
SysNav stack, not in this excerpt.)
