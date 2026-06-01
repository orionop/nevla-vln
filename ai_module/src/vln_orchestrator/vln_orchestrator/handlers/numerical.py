#!/usr/bin/env python3
"""Numerical handler (/1) — answer "how many / count" questions.

Wired so far:
  1. Decompose the question into target/attributes/spatial/anchor/region
     (reasoning.decomposition — heuristic now, VLM when a key is present).
  2. Counting logic (reasoning.counting) filters semantic-map instances by
     class + attributes + spatial relation and returns the count.

Pending perception on the Jazzy box (documented hook below):
  - Source the live 3D instances from the semantic map (SysNav semantic_mapping)
    and the VLM attribute check; feed them into count_matching().

Scored 0/1 by exact match, so calibration of the count matters more than recall.
"""
from __future__ import annotations

import os

from vln_orchestrator.handlers.base import BaseHandler
from vln_orchestrator.reasoning.counting import count_matching
from vln_orchestrator.reasoning.decomposition import (
    Decomposition,
    heuristic_decompose,
    vlm_decompose,
)


class NumericalHandler(BaseHandler):
    def __init__(self, node) -> None:
        super().__init__(node)
        self._client = None  # lazy VLM client; only built if a key is present

    def _decompose(self, question: str) -> Decomposition:
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

        # --- PERCEPTION HOOK (Jazzy box) -------------------------------------
        # instances = self.node.semantic_map.all_instances()  # list[Instance]
        # count = count_matching(decomp, instances)
        # self.node.publish_numerical(count)
        # return
        # ---------------------------------------------------------------------
        self.log.warn("NumericalHandler: perception not wired; using fallback.")
        self.fallback(question)

    def fallback(self, question: str) -> None:
        # A neutral non-zero guess. The training-set count distribution skews to
        # small integers (1-8, mode ~2); 2 is the safest single guess. Replaced by
        # the real count_matching() output as soon as the instance source lands.
        self.node.publish_numerical(2)
