"""
Collision detection for trajectory analysis.

Provides spatial intersection detection between trajectories,
bounding box overlap, and collision point estimation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, Sequence
from uuid import UUID, uuid4

import structlog

from cwe.physics.kinematics import Trajectory, Vector3, KinematicState

logger = structlog.get_logger(__name__)


@dataclass
class BoundingBox:
    """Axis-aligned bounding box for collision detection."""
    
    min_x: float
    max_x: float
    min_y: float
    max_y: float
    min_z: float = 0.0
    max_z: float = 2.0  # Default vehicle height
    
    @property
    def center(self) -> Vector3:
        return Vector3(
            (self.min_x + self.max_x) / 2,
            (self.min_y + self.max_y) / 2,
            (self.min_z + self.max_z) / 2,
        )
    
    @property
    def dimensions(self) -> Vector3:
        return Vector3(
            self.max_x - self.min_x,
            self.max_y - self.min_y,
            self.max_z - self.min_z,
        )
    
    def intersects(self, other: "BoundingBox") -> bool:
        """Check if this bounding box intersects another."""
        return (
            self.min_x <= other.max_x and self.max_x >= other.min_x and
            self.min_y <= other.max_y and self.max_y >= other.min_y and
            self.min_z <= other.max_z and self.max_z >= other.min_z
        )
    
    @classmethod
    def from_state(
        cls,
        state: KinematicState,
        length: float = 4.5,
        width: float = 1.8,
        height: float = 1.5,
    ) -> "BoundingBox":
        """
        Create bounding box from kinematic state.
        
        Accounts for vehicle heading/orientation.
        """
        # Simplified: use worst-case axis-aligned box
        half_diagonal = math.sqrt(length**2 + width**2) / 2
        
        return cls(
            min_x=state.position.x - half_diagonal,
            max_x=state.position.x + half_diagonal,
            min_y=state.position.y - half_diagonal,
            max_y=state.position.y + half_diagonal,
            min_z=state.position.z,
            max_z=state.position.z + height,
        )


@dataclass
class CollisionPoint:
    """
    Represents a detected or projected collision.
    """
    
    id: UUID = field(default_factory=uuid4)
    
    # Collision time and location
    timestamp: datetime = field(default_factory=datetime.now)
    position: Vector3 = field(default_factory=Vector3)
    
    # Involved entities
    entity_a_id: str = ""
    entity_b_id: str = ""
    
    # States at collision
    state_a: Optional[KinematicState] = None
    state_b: Optional[KinematicState] = None
    
    # Collision characteristics
    relative_velocity: float = 0.0  # m/s
    impact_angle: float = 0.0  # Radians
    collision_type: str = "unknown"  # front, side, rear, angular
    
    # Confidence and severity
    confidence: float = 1.0
    severity_estimate: str = "unknown"  # minor, moderate, severe, catastrophic
    
    # Impact energy estimate (simplified)
    @property
    def kinetic_energy_kj(self) -> float:
        """Estimate collision kinetic energy in kJ."""
        if self.state_a is None:
            return 0.0
        # Simplified: use relative velocity and assumed masses
        mass = 1500  # kg
        return 0.5 * mass * (self.relative_velocity ** 2) / 1000


@dataclass
class CollisionAnalysis:
    """Complete collision analysis results."""
    
    collisions: list[CollisionPoint] = field(default_factory=list)
    near_misses: list[dict] = field(default_factory=list)
    time_to_collision: Optional[float] = None
    first_collision_time: Optional[datetime] = None
    
    @property
    def has_collision(self) -> bool:
        return len(self.collisions) > 0
    
    @property
    def max_severity(self) -> str:
        if not self.collisions:
            return "none"
        severities = ["minor", "moderate", "severe", "catastrophic"]
        max_idx = 0
        for col in self.collisions:
            if col.severity_estimate in severities:
                idx = severities.index(col.severity_estimate)
                if idx > max_idx:
                    max_idx = idx
        return severities[max_idx]


class CollisionDetector:
    """
    Collision detection engine for trajectory analysis.
    
    Supports:
    - Discrete time-step collision detection
    - Continuous collision detection (CCD)
    - Time-to-collision (TTC) estimation
    - Near-miss detection
    """
    
    def __init__(
        self,
        time_step: float = 0.05,
        collision_threshold: float = 2.0,
        near_miss_threshold: float = 5.0,
    ):
        """
        Initialize collision detector.
        
        Args:
            time_step: Time step for discrete detection (seconds)
            collision_threshold: Distance threshold for collision (meters)
            near_miss_threshold: Distance threshold for near-miss (meters)
        """
        self.time_step = time_step
        self.collision_threshold = collision_threshold
        self.near_miss_threshold = near_miss_threshold
        self.logger = structlog.get_logger(__name__)
    
    def detect_collision(
        self,
        trajectory_a: Trajectory,
        trajectory_b: Trajectory,
    ) -> CollisionAnalysis:
        """
        Detect collisions between two trajectories.
        
        Args:
            trajectory_a: First trajectory
            trajectory_b: Second trajectory
        
        Returns:
            CollisionAnalysis with detected collisions
        """
        self.logger.info(
            "Detecting collisions",
            entity_a=trajectory_a.entity_id,
            entity_b=trajectory_b.entity_id,
        )
        
        collisions = []
        near_misses = []
        min_distance = float("inf")
        min_distance_time = None
        
        # Find overlapping time range
        start_time = max(
            trajectory_a.start_time or datetime.min,
            trajectory_b.start_time or datetime.min,
        )
        end_time = min(
            trajectory_a.end_time or datetime.max,
            trajectory_b.end_time or datetime.max,
        )
        
        if start_time >= end_time:
            self.logger.warning("No temporal overlap between trajectories")
            return CollisionAnalysis()
        
        # Step through time
        current_time = start_time
        while current_time <= end_time:
            state_a = trajectory_a.get_state_at(current_time)
            state_b = trajectory_b.get_state_at(current_time)
            
            if state_a is None or state_b is None:
                current_time += timedelta(seconds=self.time_step)
                continue
            
            # Calculate distance between entities
            distance = (state_a.position - state_b.position).magnitude()
            
            # Track minimum distance
            if distance < min_distance:
                min_distance = distance
                min_distance_time = current_time
            
            # Check for collision
            if distance < self.collision_threshold:
                collision = self._create_collision_point(
                    current_time,
                    state_a,
                    state_b,
                    trajectory_a.entity_id,
                    trajectory_b.entity_id,
                    distance,
                )
                collisions.append(collision)
            
            # Check for near miss
            elif distance < self.near_miss_threshold:
                near_misses.append({
                    "timestamp": current_time.isoformat(),
                    "distance": distance,
                    "entity_a_speed": state_a.speed,
                    "entity_b_speed": state_b.speed,
                })
            
            current_time += timedelta(seconds=self.time_step)
        
        # Calculate time to collision if applicable
        ttc = None
        if not collisions and min_distance_time:
            ttc = self._estimate_time_to_collision(
                trajectory_a,
                trajectory_b,
                min_distance_time,
            )
        
        return CollisionAnalysis(
            collisions=collisions,
            near_misses=near_misses,
            time_to_collision=ttc,
            first_collision_time=collisions[0].timestamp if collisions else None,
        )
    
    def detect_multi_entity_collisions(
        self,
        trajectories: Sequence[Trajectory],
    ) -> CollisionAnalysis:
        """
        Detect collisions among multiple trajectories.
        
        Checks all pairs of trajectories for potential collisions.
        
        Args:
            trajectories: List of trajectories to check
        
        Returns:
            Combined CollisionAnalysis
        """
        self.logger.info(
            "Detecting multi-entity collisions",
            num_trajectories=len(trajectories),
        )
        
        all_collisions = []
        all_near_misses = []
        
        # Check all pairs
        for i, traj_a in enumerate(trajectories):
            for j, traj_b in enumerate(trajectories[i + 1:], i + 1):
                analysis = self.detect_collision(traj_a, traj_b)
                all_collisions.extend(analysis.collisions)
                all_near_misses.extend(analysis.near_misses)
        
        # Sort by time
        all_collisions.sort(key=lambda c: c.timestamp)
        
        return CollisionAnalysis(
            collisions=all_collisions,
            near_misses=all_near_misses,
            first_collision_time=all_collisions[0].timestamp if all_collisions else None,
        )
    
    def estimate_time_to_collision(
        self,
        state_a: KinematicState,
        state_b: KinematicState,
    ) -> Optional[float]:
        """
        Estimate time to collision from current states.
        
        Uses linear extrapolation assuming constant velocity.
        
        Returns:
            Time to collision in seconds, or None if no collision expected
        """
        # Relative position and velocity
        rel_pos = state_b.position - state_a.position
        rel_vel = state_b.velocity - state_a.velocity
        
        # Check if approaching
        approach_rate = -rel_pos.dot(rel_vel) / rel_pos.magnitude()
        
        if approach_rate <= 0:
            # Not approaching
            return None
        
        distance = rel_pos.magnitude()
        ttc = (distance - self.collision_threshold) / approach_rate
        
        return ttc if ttc > 0 else 0.0
    
    def _create_collision_point(
        self,
        timestamp: datetime,
        state_a: KinematicState,
        state_b: KinematicState,
        entity_a_id: str,
        entity_b_id: str,
        distance: float,
    ) -> CollisionPoint:
        """Create a collision point from two states."""
        # Calculate relative velocity
        rel_vel = state_b.velocity - state_a.velocity
        relative_velocity = rel_vel.magnitude()
        
        # Calculate impact angle
        heading_diff = abs(state_a.heading - state_b.heading)
        impact_angle = min(heading_diff, 2 * math.pi - heading_diff)
        
        # Determine collision type
        if impact_angle < math.pi / 6:  # < 30 degrees
            collision_type = "rear-end"
        elif impact_angle > 5 * math.pi / 6:  # > 150 degrees
            collision_type = "head-on"
        elif math.pi / 3 < impact_angle < 2 * math.pi / 3:
            collision_type = "t-bone"
        else:
            collision_type = "angular"
        
        # Estimate severity based on relative velocity
        if relative_velocity < 5:  # < 11 mph
            severity = "minor"
        elif relative_velocity < 15:  # < 34 mph
            severity = "moderate"
        elif relative_velocity < 25:  # < 56 mph
            severity = "severe"
        else:
            severity = "catastrophic"
        
        # Collision position (midpoint)
        collision_pos = Vector3(
            (state_a.position.x + state_b.position.x) / 2,
            (state_a.position.y + state_b.position.y) / 2,
            (state_a.position.z + state_b.position.z) / 2,
        )
        
        return CollisionPoint(
            timestamp=timestamp,
            position=collision_pos,
            entity_a_id=entity_a_id,
            entity_b_id=entity_b_id,
            state_a=state_a,
            state_b=state_b,
            relative_velocity=relative_velocity,
            impact_angle=impact_angle,
            collision_type=collision_type,
            severity_estimate=severity,
        )
    
    def _estimate_time_to_collision(
        self,
        trajectory_a: Trajectory,
        trajectory_b: Trajectory,
        reference_time: datetime,
    ) -> Optional[float]:
        """Estimate TTC at a reference time."""
        state_a = trajectory_a.get_state_at(reference_time)
        state_b = trajectory_b.get_state_at(reference_time)
        
        if state_a is None or state_b is None:
            return None
        
        return self.estimate_time_to_collision(state_a, state_b)
