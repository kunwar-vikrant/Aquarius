"""
Counterfactual World Engine

An agentic system for reconstructing and simulating alternate realities
from real-world incident artifacts using Vision Language Models.
"""

from cwe.engine import CounterfactualEngine
from cwe.models.incident import Incident
from cwe.models.timeline import Timeline, Event, CausalLink
from cwe.models.counterfactual import Counterfactual, Intervention
from cwe.physics.validator import PhysicsValidator
from cwe.alignment.synchronizer import TemporalSynchronizer

__version__ = "0.1.0"

__all__ = [
    # Core
    "CounterfactualEngine",
    # Models
    "Incident",
    "Timeline",
    "Event",
    "CausalLink",
    "Counterfactual",
    "Intervention",
    # Physics
    "PhysicsValidator",
    # Alignment
    "TemporalSynchronizer",
]
