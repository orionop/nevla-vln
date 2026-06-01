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

        # --- handlers ---
        self._handlers = {
            QType.NUMERICAL: NumericalHandler(self),
            QType.OBJECT_REFERENCE: ObjectReferenceHandler(self),
            QType.INSTRUCTION_FOLLOWING: InstructionFollowingHandler(self),
        }

        self._answered = False  # one question per launch
        self.get_logger().info("VLN Orchestrator ready; awaiting /challenge_question...")

    # ------------------------------------------------------------------ #
    # Callbacks
    # ------------------------------------------------------------------ #
    def _on_pose(self, msg: Odometry) -> None:
        self.vehicle_x = msg.pose.pose.position.x
        self.vehicle_y = msg.pose.pose.position.y

    def _on_question(self, msg: String) -> None:
        question = msg.data.strip()
        if not question or self._answered:
            return
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

    def now_msg(self):
        return self.get_clock().now().to_msg()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = VLNOrchestrator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
