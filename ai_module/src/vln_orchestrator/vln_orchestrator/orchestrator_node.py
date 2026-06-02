#!/usr/bin/env python3
"""VLN Orchestrator node — replaces dummy_vlm.

Implements the challenge I/O contract (see ai_module/src/dummy_vlm/src/dummyVLM.cpp):

  Subscribes:
    /challenge_question   std_msgs/String     - the question to answer (1 Hz)
    /state_estimation     nav_msgs/Odometry   - vehicle pose

  Publishes:
    /numerical_response       std_msgs/Int32                  (Numerical /1)
    /selected_object_marker   visualization_msgs/Marker CUBE  (Object Ref /2)
    /way_point_with_heading   geometry_msgs/Pose2D            (Instruction /6)

The node classifies each incoming question with question_router.classify() and
dispatches to one of three handlers. Handlers run the perception/reasoning
pipeline (semantic mapping + VLM, adapted from SysNav) and call back into the
node's publish_* helpers. Per challenge rules the system is relaunched per
question, so there is no cross-question state to manage here.

NOTE: handler internals are scaffolded (safe fallback answers + TODO hooks).
Perception wiring lands once the SysNav semantic_mapping / vlm_node modules are
vendored into ai_module on the Jazzy box.
"""
from __future__ import annotations

import math

import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException

from std_msgs.msg import String, Int32
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose2D
from visualization_msgs.msg import Marker

from vln_orchestrator.question_router import QType, classify
from vln_orchestrator.handlers.numerical import NumericalHandler
from vln_orchestrator.handlers.object_reference import ObjectReferenceHandler
from vln_orchestrator.handlers.instruction_following import InstructionFollowingHandler


class VLNOrchestrator(Node):
    def __init__(self) -> None:
        super().__init__("vln_orchestrator")

        # --- subscriptions ---
        self.create_subscription(String, "/challenge_question", self._on_question, 5)
        self.create_subscription(Odometry, "/state_estimation", self._on_pose, 5)

        # --- publishers (exact topics/types from dummyVLM.cpp) ---
        self._numerical_pub = self.create_publisher(Int32, "/numerical_response", 5)
        self._marker_pub = self.create_publisher(Marker, "/selected_object_marker", 5)
        self._waypoint_pub = self.create_publisher(Pose2D, "/way_point_with_heading", 5)

        # --- vehicle state ---
        self.vehicle_x = 0.0
        self.vehicle_y = 0.0

        # --- instruction-following waypoint streaming ---
        self.REACH_DIST = 1.0                 # m; advance to next waypoint within this
        self._wp_queue: list[tuple[float, float]] = []

        # --- perception (semantic map), wired only if SysNav is built ---
        self.semantic_map = None
        self._wire_perception()

        # --- handlers ---
        self._handlers = {
            QType.NUMERICAL: NumericalHandler(self),
            QType.OBJECT_REFERENCE: ObjectReferenceHandler(self),
            QType.INSTRUCTION_FOLLOWING: InstructionFollowingHandler(self),
        }

        # one_shot=True (eval-faithful): answer one question per launch, then
        # ignore the rest (the eval relaunches per question and re-publishes the
        # same question at 1 Hz). Set one_shot:=false for dev convenience to
        # answer each NEW question in a single session without relaunching.
        self.one_shot = bool(self.declare_parameter("one_shot", True).value)
        self._answered = False
        self._last_question = None  # dedupe 1 Hz repeats of the same question
        self.get_logger().info(
            f"VLN Orchestrator ready (one_shot={self.one_shot}); "
            "awaiting /challenge_question..."
        )

    # ------------------------------------------------------------------ #
    # Perception wiring
    # ------------------------------------------------------------------ #
    def _wire_perception(self) -> None:
        """Subscribe to SysNav's semantic map IFF its message type is available.

        The ObjectNodeList message ships with SysNav's tare_planner build (on the
        GPU box). When it isn't present (e.g. the current minimal container), we
        skip the subscription and the handlers run their fallbacks — so the node
        works in both worlds without code changes.
        """
        try:
            from tare_planner.msg import ObjectNodeList
        except Exception as e:
            self.get_logger().warn(
                f"perception: tare_planner/ObjectNodeList unavailable ({e}); "
                "running in fallback mode (no semantic map)."
            )
            return
        from vln_orchestrator.perception.semantic_map_adapter import SemanticMap
        self.semantic_map = SemanticMap()
        self.create_subscription(
            ObjectNodeList, "/object_nodes_list", self._on_semantic_map, 50
        )
        self.get_logger().info("perception: subscribed to /object_nodes_list.")

    def _on_semantic_map(self, msg) -> None:
        if self.semantic_map is not None:
            self.semantic_map.update_from_msg(msg)

    # ------------------------------------------------------------------ #
    # Callbacks
    # ------------------------------------------------------------------ #
    def _on_pose(self, msg: Odometry) -> None:
        self.vehicle_x = msg.pose.pose.position.x
        self.vehicle_y = msg.pose.pose.position.y
        self._advance_waypoints()

    def _on_question(self, msg: String) -> None:
        question = msg.data.strip()
        if not question:
            return
        if question == self._last_question:   # ignore 1 Hz repeats of the same Q
            return
        if self.one_shot and self._answered:   # eval-faithful: one Q per launch
            return
        self._last_question = question
        qtype = classify(question)
        self.get_logger().info(f"Question [{qtype.value}]: {question}")
        try:
            self._handlers[qtype].handle(question)
        except Exception as e:  # never crash; always try to emit something
            self.get_logger().error(f"Handler {qtype.value} failed: {e}")
            self._handlers[qtype].fallback(question)
        self._answered = True

    # ------------------------------------------------------------------ #
    # Publish helpers (used by handlers)
    # ------------------------------------------------------------------ #
    def publish_numerical(self, value: int) -> None:
        self._numerical_pub.publish(Int32(data=int(value)))
        self.get_logger().info(f"-> /numerical_response {int(value)}")

    def publish_object_marker(self, bbox: dict, label: str, obj_id: int = 0) -> None:
        """bbox: {cx,cy,cz,l,w,h,heading} in the map frame (see dummyVLM CUBE)."""
        m = Marker()
        m.header.frame_id = "map"
        m.header.stamp = self.now_msg()
        m.ns = label
        m.id = int(obj_id)
        m.action = Marker.ADD
        m.type = Marker.CUBE
        m.pose.position.x = float(bbox["cx"])
        m.pose.position.y = float(bbox["cy"])
        m.pose.position.z = float(bbox["cz"])
        heading = float(bbox.get("heading", 0.0))
        m.pose.orientation.z = math.sin(heading / 2.0)
        m.pose.orientation.w = math.cos(heading / 2.0)
        m.scale.x = float(bbox["l"])
        m.scale.y = float(bbox["w"])
        m.scale.z = float(bbox["h"])
        m.color.a = 0.5
        m.color.b = 1.0
        self._marker_pub.publish(m)
        self.get_logger().info(
            f"-> /selected_object_marker '{label}' @ "
            f"({bbox['cx']:.2f},{bbox['cy']:.2f},{bbox['cz']:.2f})"
        )

    def publish_waypoint(self, x: float, y: float, heading: float = 0.0) -> None:
        self._waypoint_pub.publish(Pose2D(x=float(x), y=float(y), theta=float(heading)))

    def stream_waypoints(self, points: list) -> None:
        """Queue an ordered list of (x, y) waypoints and publish the first; the
        rest are advanced in _on_pose as the vehicle reaches each (cf. the
        sequential pubPathWaypoints behaviour of dummyVLM)."""
        self._wp_queue = [(float(x), float(y)) for x, y in points]
        if self._wp_queue:
            x, y = self._wp_queue[0]
            self.publish_waypoint(x, y, self._heading_towards(x, y))
            self.get_logger().info(f"streaming {len(self._wp_queue)} waypoint(s)")

    def _advance_waypoints(self) -> None:
        if not self._wp_queue:
            return
        tx, ty = self._wp_queue[0]
        if math.hypot(tx - self.vehicle_x, ty - self.vehicle_y) <= self.REACH_DIST:
            self._wp_queue.pop(0)
            if self._wp_queue:
                nx, ny = self._wp_queue[0]
                self.publish_waypoint(nx, ny, self._heading_towards(nx, ny))
            else:
                self.get_logger().info("waypoint sequence complete.")

    def _heading_towards(self, x: float, y: float) -> float:
        return math.atan2(y - self.vehicle_y, x - self.vehicle_x)

    def now_msg(self):
        return self.get_clock().now().to_msg()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = VLNOrchestrator()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except Exception as e:
        # On SIGINT, rclpy's executor can raise mid-spin while taking a message
        # (a known shutdown race -> RuntimeError "Unable to convert call
        # argument"). Per-question handler errors are already caught in
        # _on_question, so spin-level exceptions are shutdown/transport races:
        # log and exit cleanly rather than dying with a traceback (exit 1).
        node.get_logger().info(f"shutting down ({type(e).__name__})")
    finally:
        node.destroy_node()
        # The signal handler may have already shut the context down; guard so a
        # second shutdown() doesn't raise. Exit clean (0).
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
