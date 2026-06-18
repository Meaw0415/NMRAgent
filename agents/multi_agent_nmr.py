"""Backward-compatible import surface for the stable v2 multi-agent framework."""

from __future__ import annotations

from .multi_agent_nmr_v2 import MultiAgentNMR, MultiAgentNMRV2, MultiAgentState

__all__ = ["MultiAgentNMR", "MultiAgentNMRV2", "MultiAgentState"]
