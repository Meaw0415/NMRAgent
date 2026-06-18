"""NMRAgent package exports with lazy imports for heavy optional agents."""

from __future__ import annotations

__all__ = ["NMRAgent", "AgentState", "MultiAgentNMRV2", "MultiAgentNMR", "MultiAgentState"]


def __getattr__(name: str):
    if name in {"MultiAgentNMRV2", "MultiAgentNMR", "MultiAgentState"}:
        from .multi_agent_nmr_v2 import MultiAgentNMRV2, MultiAgentNMR, MultiAgentState
        return {"MultiAgentNMRV2": MultiAgentNMRV2, "MultiAgentNMR": MultiAgentNMR, "MultiAgentState": MultiAgentState}[name]
    if name in {"NMRAgent", "AgentState"}:
        from .nmr_agent import NMRAgent, AgentState
        return {"NMRAgent": NMRAgent, "AgentState": AgentState}[name]
    raise AttributeError(name)
