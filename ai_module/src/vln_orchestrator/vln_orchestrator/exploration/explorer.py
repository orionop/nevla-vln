#!/usr/bin/env python3
"""Lightweight frontier-style exploration controller (pure Python, ROS-free).

The challenge requires the AI module to explore the unknown scene by sending
waypoints; the base system navigates to them and snaps out-of-traversable
waypoints onto traversable ground. The robot's camera is a 360-degree panorama,
so a single pose sees all directions — coverage just needs the robot to VISIT
enough of the traversable area (no in-place rotation required).

This controller picks the next exploration goal from the latest traversable
terrain points (from /terrain_map) and the robot's visited poses: the nearest
terrain point still far from everywhere we've been (an unexplored frontier),
clamped to a max step so we drive incrementally and reveal more terrain en
route. When no such frontier remains, the reachable area is covered.

Pure geometry on (x, y) tuples -> unit-testable off-robot.
"""
from __future__ import annotations

import math


class ExplorationController:
    def __init__(self, frontier_clearance: float = 2.0, max_step: float = 4.0,
                 visited_spacing: float = 0.5) -> None:
        # a terrain point is an unexplored frontier if it is at least this far
        # from every visited pose
        self.frontier_clearance = frontier_clearance
        # cap a single goal's distance from the current pose (drive incrementally)
        self.max_step = max_step
        # dedupe visited poses closer than this to keep the set small
        self.visited_spacing = visited_spacing
        self.visited: list[tuple[float, float]] = []
        self.terrain: list[tuple[float, float]] = []

    def set_terrain(self, points) -> None:
        """Latest traversable (x, y) points (from /terrain_map)."""
        self.terrain = [(float(x), float(y)) for x, y in points]

    def mark_visited(self, x: float, y: float) -> None:
        x, y = float(x), float(y)
        if not self.visited or math.hypot(
            x - self.visited[-1][0], y - self.visited[-1][1]
        ) > self.visited_spacing:
            self.visited.append((x, y))

    def _min_visited_dist(self, p) -> float:
        if not self.visited:
            return float("inf")
        return min(math.hypot(p[0] - vx, p[1] - vy) for vx, vy in self.visited)

    def next_goal(self, cur_x: float, cur_y: float):
        """Nearest unexplored frontier to the current pose, clamped to max_step;
        None if the reachable area looks covered (no frontier left)."""
        frontiers = [
            p for p in self.terrain
            if self._min_visited_dist(p) >= self.frontier_clearance
        ]
        if not frontiers:
            return None
        gx, gy = min(frontiers, key=lambda p: math.hypot(p[0] - cur_x, p[1] - cur_y))
        d = math.hypot(gx - cur_x, gy - cur_y)
        if d > self.max_step:
            s = self.max_step / d
            gx, gy = cur_x + (gx - cur_x) * s, cur_y + (gy - cur_y) * s
        return (gx, gy)

    def is_covered(self, cur_x: float, cur_y: float) -> bool:
        """Covered = we have terrain data and no unexplored frontier remains.
        Returns False when no terrain has been received yet (can't conclude)."""
        if not self.terrain:
            return False
        return self.next_goal(cur_x, cur_y) is None
