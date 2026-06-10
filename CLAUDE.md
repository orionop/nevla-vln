# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Submission for the **CMU Vision-Language-Navigation (VLN) Challenge 2026**. The
robot is dropped into an **unknown** Unity indoor scene and must answer one of
three question types about it, within a **10-min/question** budget (explore +
answer combined; the system is **relaunched per question**, so there is no
cross-question state):

| Type | Input on `/challenge_question` | Output |
|---|---|---|
| Numerical | "How many chairs are there" | `/numerical_response` `std_msgs/Int32` |
| Object reference | "Find the chair nearest the table" | `/selected_object_marker` `visualization_msgs/Marker` (CUBE) |
| Instruction following | "Go to the table then the sofa" | `/way_point_with_heading` `geometry_msgs/Pose2D` (streamed) |

Allowed inputs from the system: `/state_estimation` (Odometry), `/registered_scan`,
`/terrain_map`(`_ext`), 360° `/camera/image`. The base system **navigates** to
waypoints we publish (and snaps out-of-traversable ones onto traversable ground);
**exploration is the AI module's job.**

## The one hard rule: only touch `ai_module/`

The repo root is the organizers' base stack (BSD-3). **Never modify** the root
`README.md`/`LICENSE`, `autonomy_stack_mecanum_wheel_platform/` (git submodule),
or `docker/compose*.yml`. All our work + docs + attribution live **inside
`ai_module/`**. `_reference_SysNav/` is a read-only reference clone (not built,
not shipped). `eval_harness/` (offline scoring/vocab tooling) and `questions/`
are ours but run only on the authoring machine.

## Architecture (the big picture)

The whole system is the SysNav reference architecture rebuilt inside `ai_module`,
launched as **one coordinated startup** against the **plain** `system_simulation.sh`
(NOT the with-exploration variant — exploration lives here). All nodes use
`use_sim_time` so they share one clock.

```
/camera/image ─► detection_node ─► /detection_result ─► semantic_mapping_node ─► /object_nodes_list
  (YOLO-World, our challenge vocab)        (SAM2 + lidar fusion: 3D ObjectNodes)        │
                                                                                        ▼
  tare_planner (SysNav semantic TARE) + room_segmentation  ◄── /object_nodes_list ──────┘  drives /way_point
        ▲▼  RoomType / NavigationQuery / VlmAnswer / TargetObject(Instruction)
  vlm_node (coordinator): reads instruction on /keyboard_input → decompose → tell TARE the target,
                          decide which room to explore next / early-stop
  vln_orchestrator (OUR answerer): /challenge_question → wait for the map → 3 challenge outputs
```

Packages in `ai_module/src/` (+ `vendor/sysnav/semantic_mapping`):
- **`vln_orchestrator/`** — OUR code, the entry node + reasoning. `orchestrator_node.py`
  is a phase machine: on a question it classifies (`question_router.py`), forwards
  the instruction to `vlm_node` via `/keyboard_input`, **waits while TARE explores**
  (`external_exploration` param; convergence on map-stability/budget), then dispatches
  to `handlers/{numerical,object_reference,instruction_following}.py` which answer from
  the live map. `reasoning/` (decomposition, spatial predicates, counting, instruction
  parser, verification, `vlm_client.py`), `perception/semantic_map_adapter.py`
  (`ObjectNodeList` → `Instance`/`SemanticMap`), `exploration/explorer.py` (a
  lightweight fallback frontier explorer, superseded by TARE but kept).
- **`tare_planner/`** — SysNav's full C++ exploration planner (semantic, room-aware)
  + `room_segmentation` + the shared **`.msg`** interfaces (`ObjectNodeList`, etc.) +
  **vendored OR-Tools** `.so` (`or-tools/lib/`, linked by direct path; runtime needs it
  on `LD_LIBRARY_PATH`, set in the image's `.bashrc`).
- **`vlm_node/`** — SysNav's exploration coordinator (Python). Uses Gemini
  (`constants.py`: `GEMINI_API_KEY`, `gemini-2.5-flash`).
- **`semantic_mapping/`** (under `vendor/sysnav/`) — SysNav perception, adapted: the
  detector uses a YOLO-World `.pt` (not SysNav's TensorRT engine) + our
  `challenge_classes.yaml` vocabulary.
- **`dummy_vlm/`** — the organizers' placeholder we replace (kept for reference; it
  defines the exact topic/type contract).

## Build & run

**The Mac is authoring-only** (no GPU/sim). All builds + runs happen in Docker on a
GPU box (the project's is a 6 GB RTX 4050). Two containers via
`docker/compose_gpu.yml`: `iros2026_system` (sim + base autonomy) and
`iros2026_ai_module` (our image, built from `ai_module/docker/Dockerfile`). **There
is no volume mount** — source is COPY'd into the image, so changes need a rebuild OR
the hot-patch below.

```bash
# rebuild the ai_module image (heavy: torch/SAM2/open3d/PCL layers; cached after first)
cd ~/nevla-vln && git pull && cd docker && docker compose -f compose_gpu.yml up --build -d ai_module
```

The full bring-up (one coordinated launch vs terminal-by-terminal), troubleshooting,
and the GPU-host prerequisites are documented in **`ai_module/PC_SETUP.md`** (§6 is
the full SysNav exploration run). The eval-faithful run:
```bash
# system container: PLAIN system
/home/docker/autonomy_stack_mecanum_wheel_platform/system_simulation.sh
# ai_module container: all nodes, one launch (perception warms up, then TARE)
export GEMINI_API_KEY=<key>
ros2 launch vln_orchestrator full_system.launch.py scenario:=indoor
# ask:
ros2 topic pub --once /challenge_question std_msgs/msg/String "{data: 'How many chairs are there'}"
```

**Fast dev loop (no rebuild):** the packages build with `--symlink-install`, so
`docker cp` a changed source file into `…/ai_module/src/<pkg>/…` in the running
container and (for Python) re-run, or `colcon build --packages-select <pkg>` for a
quick in-container rebuild.

## Tests (offline, run on the Mac)

Pure-Python unit suites, no ROS needed:
```bash
python3 ai_module/src/vln_orchestrator/test/test_question_router.py   # also: _decomposition,
#   _instruction_parser, _spatial_counting, _semantic_map_adapter, _explorer
python3 eval_harness/test_scoring.py
```
After editing the node, `python3 -m py_compile <file>` to catch syntax errors before
a Docker rebuild.

## Conventions & gotchas (project-specific)

- **Commits:** one line, ≤10 words, no em dashes, **no `Co-Authored-By`/AI
  attribution**. Work happens on a shared private repo (`orionop/nevla-vln`); branch
  before PRs if on `main`.
- **No fabricated data:** use authoritative/verified ground truth or none — never
  silently fill guessed values (object-reference GT is deferred to the live sim).
- **Anti-overfitting:** 15 released scenes are dev-only; 3 hidden test scenes are the
  metric. No scene-specific hacks; broaden detection vocab; prefer VLM over heuristics.
- **numpy is pinned to 1.26.4** everywhere — the debian-built `cv2`/`cv_bridge` break
  on numpy 2.x. Any in-container `pip install` MUST pin `numpy==1.26.4`. Base-image
  debian packages have no pip RECORD, so pip needs `--ignore-installed` /
  `--break-system-packages`.
- **Docker build cache eviction** under disk pressure is the chronic enemy on the
  6 GB box — it silently re-triggers the multi-GB torch/SAM2 downloads. Keep ample
  free disk; `docker save` the built image to external storage as a restore point.
- The challenge **relaunches per question** → the orchestrator answers one question
  per launch (`one_shot` param, default true) and keeps no cross-question state.
