#!/usr/bin/env python3
"""Instruction-following handler (/6) — drive an ordered, constraint-respecting path.

Wired so far:
  1. Parse the command into ORDERED sub-goals + via/avoid constraints
     (reasoning.instruction_parser — pure logic, validated on all 30 training cmds).

Pending perception/planning on the Jazzy box (documented hooks below):
  2. Localise each landmark in the semantic map (SysNav semantic_mapping).
  3. Plan a path through the ordered GOTO/STOP/VIA landmarks honoring avoid
     regions (SysNav route_planner visibility graph).
  4. Stream the Pose2D sequence to /way_point_with_heading, advancing on arrival
     (reach distance ~1.0 m; cf. pubPathWaypoints in dummyVLM.cpp).

Scored 0-6 with partial credit; penalties for wrong order / missed / forbidden.
"""
from __future__ import annotations

from vln_orchestrator.handlers.base import BaseHandler
from vln_orchestrator.reasoning.instruction_parser import GoalKind, parse_instruction


class InstructionFollowingHandler(BaseHandler):
    #: distance (m) at which a waypoint is considered reached before advancing
    REACH_DIST = 1.0

    def handle(self, question: str) -> None:
        parsed = parse_instruction(question)
        order = " -> ".join(
            f"{g.kind.value}:{g.landmark}" for g in parsed.subgoals
        )
        self.log.info(f"parsed instruction: {order}")
        if parsed.avoid_regions:
            self.log.info(
                "avoid: " + "; ".join(g.landmark for g in parsed.avoid_regions)
            )

        # --- PERCEPTION + PLANNING HOOK (Jazzy box) --------------------------
        # waypoints = []
        # for g in parsed.ordered_waypoints:           # GOTO/VIA/STOP in order
        #     pos = self.node.semantic_map.locate(g.landmark)   # (x, y) in map
        #     waypoints.append(pos)
        # path = self.node.route_planner.plan(
        #     waypoints, avoid=[self.node.semantic_map.region(a.landmark)
        #                       for a in parsed.avoid_regions])
        # self._stream_waypoints(path)                  # advance on arrival
        # return
        # ---------------------------------------------------------------------
        self.log.warn(
            "InstructionFollowingHandler: perception/planning not wired; "
            "using fallback."
        )
        self.fallback(question)

    def fallback(self, question: str) -> None:
        # Hold position (publish current pose) so the topic is exercised and the
        # robot does not wander. Real pipeline emits the planned ordered sequence.
        self.node.publish_waypoint(self.node.vehicle_x, self.node.vehicle_y, 0.0)
