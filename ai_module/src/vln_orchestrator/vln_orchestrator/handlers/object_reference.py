#!/usr/bin/env python3
"""Object-reference handler (/2) — locate one unique object, publish its 3D bbox.

Wired so far:
  1. Decompose the question into target/attributes/spatial/anchor/region
     (reasoning.decomposition — heuristic now, VLM when a key is present).

Pending perception on the Jazzy box (documented hooks below):
  2. Explore; build semantic map of candidate instances (SysNav semantic_mapping).
  3. Verify candidates with reasoning.verification.verify_candidate (VLM + image),
     select the single best match.
  4. publish_object_marker(bbox, label) — CUBE; its center doubles as a waypoint.

Scored by 3D bbox IoU with GT (0-2).
"""
from __future__ import annotations

import os

from vln_orchestrator.handlers.base import BaseHandler
from vln_orchestrator.reasoning.decomposition import (
    Decomposition,
    heuristic_decompose,
    vlm_decompose,
)


class ObjectReferenceHandler(BaseHandler):
    def __init__(self, node) -> None:
        super().__init__(node)
        self._client = None  # lazy VLM client; only built if a key is present

    def _decompose(self, question: str) -> Decomposition:
        """VLM decomposition when an API key is available, else heuristic."""
        if os.environ.get("GEMINI_API_KEY") or os.environ.get("DASHSCOPE_API_KEY"):
            try:
                if self._client is None:
                    from vln_orchestrator.reasoning.vlm_client import VLMClient
                    self._client = VLMClient()
                return vlm_decompose(question, self._client)
            except Exception as e:
                self.log.warn(f"VLM decompose failed ({e}); using heuristic.")
        return heuristic_decompose(question)

    def handle(self, question: str) -> None:
        decomp = self._decompose(question)
        self.log.info(
            f"decomposed -> target={decomp.target_object!r} "
            f"attrs={decomp.attributes} rel={decomp.spatial_relation!r} "
            f"anchor={decomp.anchor_object!r}"
        )

        sm = getattr(self.node, "semantic_map", None)
        if sm is not None and len(sm):
            best = sm.resolve(decomp)
            if best is not None:
                # TODO: VLM-verify attributes via best.image_path when several
                # candidates pass the geometry, to pick the right one.
                self.node.publish_object_marker(
                    best.bbox,
                    label=decomp.target_object or best.label,
                    obj_id=best.id,
                )
                return
            self.log.warn("ObjectReferenceHandler: no candidate matched; fallback.")
        else:
            self.log.warn("ObjectReferenceHandler: no semantic map; fallback box.")
        self.fallback(question)

    def fallback(self, question: str) -> None:
        # Always emit a marker (non-empty answer). Box at current vehicle pose.
        bbox = {
            "cx": self.node.vehicle_x, "cy": self.node.vehicle_y, "cz": 0.5,
            "l": 0.3, "w": 0.3, "h": 0.3, "heading": 0.0,
        }
        label = heuristic_decompose(question).target_object or "unknown_object"
        self.node.publish_object_marker(bbox, label=label)
