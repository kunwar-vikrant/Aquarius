"""
Physics simulation module for kinematic validation.

Provides trajectory projection, collision detection, and
physics-based validation of VLM-generated timelines.
"""

from cwe.physics.kinematics import KinematicSimulator, Trajectory, SimulationResult
from cwe.physics.collision import CollisionDetector, CollisionPoint
from cwe.physics.validator import PhysicsValidator, ValidationResult

__all__ = [
    "KinematicSimulator",
    "Trajectory",
    "SimulationResult",
    "CollisionDetector",
    "CollisionPoint",
    "PhysicsValidator",
    "ValidationResult",
]
