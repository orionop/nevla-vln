#!/usr/bin/env python3
"""Adapter: SysNav semantic map -> our reasoning `Instance` type.

SysNav's semantic_mapping_node (detection + 3D instance mapping, GPU) publishes
`tare_planner/ObjectNodeList` on `/object_nodes_list`. Each `ObjectNode` has:
    object_id : int32[]            -> we take object_id[0] as the instance id
    label     : string             -> dominant class label
    position  : geometry_msgs/Point-> centroid (unused; we use the box center)
    bbox3d    : geometry_msgs/Point[8] -> ORIENTED 3D box corners
                 (corners 0-3 = bottom rectangle, 4-7 = top; see
                  semantic_mapping.single_object_new.get_bbox_3d_oriented)
    img_path  : string             -> best crop (.npy) for VLM verification
    is_asked_vlm, status, cloud, viewpoint_id : (unused here)

This module is ROS-free and duck-typed: `object_node_to_instance` reads attribute
fields off any object that looks like an ObjectNode, so it is unit-testable on the
Mac with synthetic objects (no rclpy / tare_planner build required). The node
imports `SemanticMap` and feeds it the received message; the message type itself
is imported lazily in the node, only when SysNav is built (on the GPU box).

Attributes (color/size/material) are NOT in ObjectNode -> Instance.attributes is
left empty; the VLM verification step (reasoning.verification, using img_path)
adjudicates attributes at runtime.
"""
from __future__ import annotations

import math

from vln_orchestrator.reasoning import spatial
from vln_orchestrator.reasoning.counting import filter_by_attributes, filter_by_label
from vln_orchestrator.reasoning.spatial import Instance


def _xyz(p) -> tuple[float, float, float]:
    return (float(p.x), float(p.y), float(p.z))


def bbox_from_corners(corners) -> dict:
    """Reconstruct SysNav's oriented 3D box (center + extents + heading) from its
    8 corners. corners[0..3] are the bottom rectangle in order, [4..7] the top.

    Returns a box dict {cx,cy,cz,l,w,h,heading} matching the Marker / spatial
    convention. l is the corner0->corner1 edge, w the corner1->corner2 edge,
    h the bottom->top height; heading is the yaw of the l-edge.
    """
    pts = [_xyz(c) for c in corners]
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)
    cz = sum(p[2] for p in pts) / len(pts)
    (x0, y0, z0), (x1, y1, _), (x2, y2, _) = pts[0], pts[1], pts[2]
    l = math.hypot(x1 - x0, y1 - y0)
    w = math.hypot(x2 - x1, y2 - y1)
    h = abs(pts[4][2] - pts[0][2])
    heading = math.atan2(y1 - y0, x1 - x0)
    return {"cx": cx, "cy": cy, "cz": cz, "l": l, "w": w, "h": h, "heading": heading}


def object_node_to_instance(node) -> Instance:
    """Convert one ObjectNode (or any object with the same fields) to an Instance."""
    obj_id = int(node.object_id[0]) if len(node.object_id) else -1
    return Instance(
        label=str(node.label).lower(),
        bbox=bbox_from_corners(node.bbox3d),
        id=obj_id,
        attributes=[],
        confidence=1.0,
        image_path=getattr(node, "img_path", "") or "",
    )


class SemanticMap:
    """Holds the latest set of `Instance`s from /object_nodes_list and answers the
    queries the handlers need. The node calls `update_from_msg` in the topic
    callback; handlers call `all_instances` / `instances_of` / `resolve` / `locate`.
    """

    def __init__(self) -> None:
        self._instances: list[Instance] = []

    # -- ingestion -------------------------------------------------------- #
    def update_from_msg(self, msg) -> None:
        """msg is a tare_planner/ObjectNodeList (duck-typed: has `.nodes`)."""
        self._instances = [object_node_to_instance(n) for n in msg.nodes]

    def __len__(self) -> int:
        return len(self._instances)

    # -- queries ---------------------------------------------------------- #
    def all_instances(self) -> list[Instance]:
        return list(self._instances)

    def instances_of(self, target: str, attributes=None) -> list[Instance]:
        cands = filter_by_label(self._instances, target)
        return filter_by_attributes(cands, attributes or [])

    def resolve(self, decomp) -> Instance | None:
        """Select the single best instance matching a Decomposition (target +
        attributes + optional spatial relation to an anchor). Used for
        object-reference grounding and instruction-following landmark lookup."""
        cands = self.instances_of(decomp.target_object, decomp.attributes)
        if not cands:
            return None

        rel = decomp.spatial_relation
        if rel and decomp.anchor_object:
            anchors = self.instances_of(decomp.anchor_object)
            if anchors:
                if spatial.is_superlative_relation(rel):
                    selector = spatial.superlative_selector(rel)
                    return selector(cands, anchors[0].bbox)
                if spatial.is_between_relation(rel):
                    for c in cands:
                        if any(
                            spatial.between(c.bbox, anchors[i].bbox, anchors[j].bbox)
                            for i in range(len(anchors))
                            for j in range(i + 1, len(anchors))
                        ):
                            return c
                if spatial.is_binary_relation(rel):
                    pred = spatial.binary_predicate(rel)
                    matches = [
                        c for c in cands
                        if any(pred(c.bbox, a.bbox) for a in anchors)
                    ]
                    if matches:
                        # prefer the match closest to its nearest anchor
                        return min(
                            matches,
                            key=lambda c: min(
                                spatial.distance(c.bbox, a.bbox) for a in anchors
                            ),
                        )
        # no usable relation -> highest-confidence candidate
        return max(cands, key=lambda c: c.confidence)

    def locate(self, phrase: str, decompose=None) -> Instance | None:
        """Resolve a free-text landmark phrase (e.g. an instruction sub-goal
        landmark) to a single instance. Decomposes the phrase first; defaults to
        the heuristic decomposer (no VLM/API key needed)."""
        if decompose is None:
            from vln_orchestrator.reasoning.decomposition import heuristic_decompose
            decompose = heuristic_decompose
        return self.resolve(decompose(phrase))
