"""Marathon Agent - Autonomous hypothesis exploration."""

from cwe.agents.marathon import MarathonAgent, ExplorationStrategy, ExplorationResult
from cwe.agents.hypothesis import HypothesisGenerator, Hypothesis

__all__ = [
    "MarathonAgent",
    "ExplorationStrategy",
    "ExplorationResult",
    "HypothesisGenerator",
    "Hypothesis",
]
