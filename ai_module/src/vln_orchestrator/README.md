# vln_orchestrator

CMU VLN Challenge 2026 AI module. Replaces `dummy_vlm`: it receives the challenge
question, classifies it, and dispatches to the pipeline for that question type.

## Architecture

```
/challenge_question ─► orchestrator_node ─► question_router.classify()
                                            │
                 ┌──────────────────────────┼───────────────────────────┐
                 ▼                           ▼                           ▼
        NumericalHandler          ObjectReferenceHandler       InstructionFollowingHandler
        (/numerical_response)     (/selected_object_marker)    (/way_point_with_heading)
```

- `question_router.py` — robust classification of the three question types.
  Validated to 100% on all 75 training questions (`test/test_question_router.py`).
  Handles phrasings the base `dummyVLM.cpp` mis-routes ("Count ...", "The ...").
- `orchestrator_node.py` — ROS2 node implementing the exact topic contract from
  `ai_module/src/dummy_vlm/src/dummyVLM.cpp` (subscribes `/challenge_question`,
  `/state_estimation`; publishes `/numerical_response`, `/selected_object_marker`,
  `/way_point_with_heading`).
- `handlers/` — one per question type. Currently scaffolded: each runs a safe
  fallback that always publishes a valid response, with documented TODO hooks for
  the real perception/reasoning pipeline (vendored from SysNav: `semantic_mapping`,
  `vlm_node`, `route_planner`, `exploration_planner`).

## Build & run (on the ROS2 Jazzy box)

```bash
cd ai_module           # colcon workspace root
colcon build --packages-select vln_orchestrator
source install/setup.bash
ros2 launch vln_orchestrator vln_orchestrator.launch.py
```

To swap it in for the dummy in the system startup script, replace the
`ros2 launch dummy_vlm dummy_vlm.launch` line with the launch command above.

## Status

| Piece | State |
|---|---|
| Question router | ✅ done, 100% on 75 training questions |
| ROS2 node + topic contract | ✅ matches dummyVLM.cpp |
| Reasoning: decomposition (`reasoning/`) | ✅ heuristic 30/30 targets + VLM path |
| Reasoning: verification | ✅ interface + prompt (runtime: VLM+image) |
| Reasoning: instruction parser | ✅ ordered sub-goals + via/avoid, 30/30 |
| Reasoning: spatial predicates | ✅ near/under/on/between/closest/farthest |
| Reasoning: counting | ✅ class+attr+relation filter over instances |
| Object-reference handler | 🟡 decomposition wired; perception hook pending |
| Instruction-following handler | 🟡 parser wired; planning hook pending |
| Numerical handler | 🟡 decomposition + counting wired; instance source pending |

Remaining 🟡 work is the perception/planning wiring (semantic map + route planner),
which lands on the Jazzy box where the full SysNav stack + SAM2/YOLO are present.
Reasoning is ROS-free and unit-tested on the Mac (`test/`).
