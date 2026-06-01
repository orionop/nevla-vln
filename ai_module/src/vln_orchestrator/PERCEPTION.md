# Perception integration (Track B) — GPU box checklist

The orchestrator already consumes a semantic map; this is what to stand up on the
GPU box (3090) to feed it real detections. The adapter + wiring are done and
unit-tested off-robot — this is purely the runtime bring-up.

## How it connects

```
SysNav detection_node (YOLO-World/YOLOe + SAM2)         [GPU]
        │  detections + masks
        ▼
SysNav semantic_mapping_node (3D instance mapping)      [GPU]
        │  publishes tare_planner/ObjectNodeList
        ▼  topic: /object_nodes_list
vln_orchestrator  ──► SemanticMap.update_from_msg()
                        ├─ all_instances()  → NumericalHandler.count
                        ├─ resolve(decomp)  → ObjectReferenceHandler marker
                        └─ locate(phrase)   → InstructionFollowingHandler waypoints
```

The subscription is **conditional**: `orchestrator_node._wire_perception()` only
subscribes if `tare_planner.msg.ObjectNodeList` imports. Until SysNav is built,
the node logs "fallback mode" and the handlers emit safe stubs — no code change
needed to flip between the two.

## Checklist

1. **Build the messages.** SysNav's `tare_planner` package must be built with the
   `ObjectNode` / `ObjectNodeList` messages (see
   `_reference_SysNav/src/exploration_planner/tare_planner/msg/`). The stock
   challenge `tare_planner` may not include `ObjectNode` — confirm and add it.
2. **Build + run SysNav perception** in the ai_module container:
   - `detection_node` (needs the open-vocab detector weights + SAM2 weights)
   - `semantic_mapping_node` (point `object_file` param at our
     `config/challenge_classes.yaml`, not SysNav's `objects.yaml`)
   - runtime deps: `torch`, SAM2, the YOLO-World/YOLOe weights, `open3d`, `scipy`.
3. **Confirm the topic.** `ros2 topic echo /object_nodes_list` should show
   populated `nodes[]` once exploration sees objects.
4. **Launch our node** as usual — it auto-detects the message and subscribes;
   the log flips to `perception: subscribed to /object_nodes_list.`
5. **Verify per type:**
   - numerical → count from `count_matching` over real instances
   - object-ref → marker on the resolved instance's box
   - instruction → ordered landmark waypoints streamed on `/way_point_with_heading`

## Known gaps to close on the box

- **Attribute verification:** `Instance.attributes` is empty (ObjectNode carries
  no color/material). For attribute-qualified queries ("red pillow"), wire
  `reasoning.verification.verify_candidate` using `Instance.image_path` (the best
  crop SysNav saved) to confirm/rank candidates with the VLM.
- **Route planning:** instruction-following currently streams straight-line
  landmark waypoints in order. Swap in SysNav's `route_planner` to plan around
  obstacles and honour via/avoid regions.
- **cv2** for image verification is assumed present in the perception image; add
  to the Dockerfile if not.
