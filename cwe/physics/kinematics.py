"""
Kinematic simulation for trajectory projection and motion analysis.

This module provides 2D/3D trajectory simulation with support for:
- Vehicle motion modeling
- Acceleration/deceleration profiles
- Steering dynamics
- Friction and environmental factors
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, Sequence
from uuid import UUID, uuid4

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


class MotionModel(str, Enum):
    """Motion model types for trajectory simulation."""
    
    CONSTANT_VELOCITY = "constant_velocity"
    CONSTANT_ACCELERATION = "constant_acceleration"
    BICYCLE = "bicycle"  # Simplified vehicle dynamics
    POINT_MASS = "point_mass"
    FULL_DYNAMICS = "full_dynamics"


@dataclass
class Vector3:
    """3D vector for position, velocity, acceleration."""
    
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    
    def __add__(self, other: "Vector3") -> "Vector3":
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __sub__(self, other: "Vector3") -> "Vector3":
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)
    
    def __mul__(self, scalar: float) -> "Vector3":
        return Vector3(self.x * scalar, self.y * scalar, self.z * scalar)
    
    def magnitude(self) -> float:
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)
    
    def normalized(self) -> "Vector3":
        mag = self.magnitude()
        if mag == 0:
            return Vector3()
        return Vector3(self.x / mag, self.y / mag, self.z / mag)
    
    def dot(self, other: "Vector3") -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z
    
    def to_2d(self) -> tuple[float, float]:
        """Project to 2D (x, y)."""
        return (self.x, self.y)


@dataclass
class KinematicState:
    """Complete kinematic state at a point in time."""
    
    timestamp: datetime
    position: Vector3
    velocity: Vector3
    acceleration: Vector3
    heading: float = 0.0  # Radians from +X axis
    angular_velocity: float = 0.0  # Rad/s
    
    @property
    def speed(self) -> float:
        """Speed magnitude in m/s."""
        return self.velocity.magnitude()
    
    @property
    def speed_mph(self) -> float:
        """Speed in miles per hour."""
        return self.speed * 2.237
    
    @property
    def speed_kmh(self) -> float:
        """Speed in km/h."""
        return self.speed * 3.6


@dataclass
class TrajectoryPoint:
    """Single point along a trajectory with full state."""
    
    state: KinematicState
    confidence: float = 1.0
    source: str = "computed"  # computed, observed, interpolated


class Trajectory(BaseModel):
    """
    Complete trajectory representation with temporal bounds.
    
    A trajectory is a sequence of kinematic states over time,
    representing the path of an entity through space.
    """
    
    id: UUID = Field(default_factory=uuid4)
    entity_id: str
    entity_type: str = "vehicle"
    
    # Trajectory points (stored as dicts for Pydantic compatibility)
    points: list[dict] = Field(default_factory=list)
    
    # Temporal bounds
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    # Motion characteristics
    motion_model: MotionModel = MotionModel.POINT_MASS
    
    # Physical parameters
    mass_kg: float = 1500.0  # Default car mass
    length_m: float = 4.5
    width_m: float = 1.8
    
    # Metadata
    source: str = "computed"
    confidence: float = 1.0
    
    def add_point(self, point: TrajectoryPoint) -> None:
        """Add a trajectory point."""
        self.points.append({
            "timestamp": point.state.timestamp.isoformat(),
            "position": (point.state.position.x, point.state.position.y, point.state.position.z),
            "velocity": (point.state.velocity.x, point.state.velocity.y, point.state.velocity.z),
            "acceleration": (point.state.acceleration.x, point.state.acceleration.y, point.state.acceleration.z),
            "heading": point.state.heading,
            "angular_velocity": point.state.angular_velocity,
            "confidence": point.confidence,
            "source": point.source,
        })
        
        # Update temporal bounds
        if self.start_time is None or point.state.timestamp < self.start_time:
            self.start_time = point.state.timestamp
        if self.end_time is None or point.state.timestamp > self.end_time:
            self.end_time = point.state.timestamp
    
    def get_state_at(self, timestamp: datetime) -> Optional[KinematicState]:
        """
        Get interpolated kinematic state at a specific time.
        
        Uses linear interpolation between nearest points.
        """
        if not self.points:
            return None
        
        # Find bracketing points
        points_sorted = sorted(self.points, key=lambda p: p["timestamp"])
        
        before = None
        after = None
        
        for point in points_sorted:
            pt_time = datetime.fromisoformat(point["timestamp"])
            if pt_time <= timestamp:
                before = point
            elif after is None:
                after = point
                break
        
        if before is None:
            return None
        
        if after is None or before == after:
            # Extrapolate from last known point
            pos = before["position"]
            vel = before["velocity"]
            acc = before["acceleration"]
            return KinematicState(
                timestamp=timestamp,
                position=Vector3(*pos),
                velocity=Vector3(*vel),
                acceleration=Vector3(*acc),
                heading=before["heading"],
                angular_velocity=before["angular_velocity"],
            )
        
        # Interpolate between before and after
        before_time = datetime.fromisoformat(before["timestamp"])
        after_time = datetime.fromisoformat(after["timestamp"])
        
        total_dt = (after_time - before_time).total_seconds()
        dt = (timestamp - before_time).total_seconds()
        t = dt / total_dt if total_dt > 0 else 0
        
        # Linear interpolation
        def lerp(a: tuple, b: tuple) -> Vector3:
            return Vector3(
                a[0] + t * (b[0] - a[0]),
                a[1] + t * (b[1] - a[1]),
                a[2] + t * (b[2] - a[2]),
            )
        
        return KinematicState(
            timestamp=timestamp,
            position=lerp(before["position"], after["position"]),
            velocity=lerp(before["velocity"], after["velocity"]),
            acceleration=lerp(before["acceleration"], after["acceleration"]),
            heading=before["heading"] + t * (after["heading"] - before["heading"]),
            angular_velocity=before["angular_velocity"] + t * (after["angular_velocity"] - before["angular_velocity"]),
        )


@dataclass
class SimulationConfig:
    """Configuration for kinematic simulation."""
    
    dt: float = 0.1  # Time step in seconds
    max_acceleration: float = 10.0  # m/s^2
    max_deceleration: float = 12.0  # m/s^2 (braking)
    max_lateral_acceleration: float = 8.0  # m/s^2
    friction_coefficient: float = 0.7  # Dry asphalt
    gravity: float = 9.81  # m/s^2
    motion_model: MotionModel = MotionModel.POINT_MASS


@dataclass
class SimulationResult:
    """Result from a kinematic simulation run."""
    
    trajectory: Trajectory
    converged: bool = True
    error_message: Optional[str] = None
    physics_violations: list[dict] = field(default_factory=list)
    
    # Statistics
    max_speed: float = 0.0
    max_acceleration: float = 0.0
    total_distance: float = 0.0
    duration_seconds: float = 0.0


class KinematicSimulator:
    """
    Kinematic simulation engine for trajectory projection.
    
    Supports multiple motion models from simple constant velocity
    to full vehicle dynamics simulation.
    """
    
    def __init__(self, config: Optional[SimulationConfig] = None):
        self.config = config or SimulationConfig()
        self.logger = structlog.get_logger(__name__)
    
    def project_trajectory(
        self,
        initial_state: KinematicState,
        duration: timedelta,
        control_inputs: Optional[list[dict]] = None,
    ) -> SimulationResult:
        """
        Project a trajectory forward from initial state.
        
        Args:
            initial_state: Starting kinematic state
            duration: How far to project
            control_inputs: Optional sequence of control inputs
                           (steering, throttle, brake)
        
        Returns:
            SimulationResult with projected trajectory
        """
        self.logger.info(
            "Projecting trajectory",
            initial_speed=initial_state.speed,
            duration=duration.total_seconds(),
        )
        
        trajectory = Trajectory(
            entity_id="projected",
            entity_type="vehicle",
            motion_model=self.config.motion_model,
        )
        
        # Add initial point
        trajectory.add_point(TrajectoryPoint(
            state=initial_state,
            confidence=1.0,
            source="initial",
        ))
        
        current_state = initial_state
        current_time = initial_state.timestamp
        end_time = current_time + duration
        
        violations = []
        max_speed = initial_state.speed
        max_accel = initial_state.acceleration.magnitude()
        total_distance = 0.0
        
        step = 0
        while current_time < end_time:
            # Get control input for this step
            control = self._get_control_input(step, control_inputs)
            
            # Compute next state based on motion model
            next_state = self._step(current_state, control)
            
            # Check physics constraints
            violation = self._check_physics(current_state, next_state)
            if violation:
                violations.append(violation)
            
            # Update trajectory
            trajectory.add_point(TrajectoryPoint(
                state=next_state,
                confidence=0.95 ** step,  # Confidence decreases over time
                source="computed",
            ))
            
            # Update statistics
            if next_state.speed > max_speed:
                max_speed = next_state.speed
            accel_mag = next_state.acceleration.magnitude()
            if accel_mag > max_accel:
                max_accel = accel_mag
            
            displacement = (next_state.position - current_state.position).magnitude()
            total_distance += displacement
            
            current_state = next_state
            current_time += timedelta(seconds=self.config.dt)
            step += 1
        
        return SimulationResult(
            trajectory=trajectory,
            converged=True,
            physics_violations=violations,
            max_speed=max_speed,
            max_acceleration=max_accel,
            total_distance=total_distance,
            duration_seconds=duration.total_seconds(),
        )
    
    def reconstruct_from_observations(
        self,
        observations: list[dict],
        entity_id: str,
    ) -> SimulationResult:
        """
        Reconstruct a trajectory from sparse observations.
        
        Interpolates between known points using physics-based
        motion models to ensure physical plausibility.
        
        Args:
            observations: List of observed states with timestamps
            entity_id: Entity identifier
        
        Returns:
            SimulationResult with reconstructed trajectory
        """
        self.logger.info(
            "Reconstructing trajectory from observations",
            num_observations=len(observations),
            entity_id=entity_id,
        )
        
        if len(observations) < 2:
            self.logger.warning("Insufficient observations for reconstruction")
            return SimulationResult(
                trajectory=Trajectory(entity_id=entity_id),
                converged=False,
                error_message="Need at least 2 observations",
            )
        
        # Sort observations by time
        sorted_obs = sorted(observations, key=lambda o: o["timestamp"])
        
        trajectory = Trajectory(
            entity_id=entity_id,
            entity_type="vehicle",
            motion_model=self.config.motion_model,
        )
        
        violations = []
        total_distance = 0.0
        
        for i, obs in enumerate(sorted_obs):
            # Parse observation
            timestamp = obs["timestamp"]
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp)
            
            position = Vector3(*obs.get("position", (0, 0, 0)))
            velocity = Vector3(*obs.get("velocity", (0, 0, 0)))
            acceleration = Vector3(*obs.get("acceleration", (0, 0, 0)))
            
            state = KinematicState(
                timestamp=timestamp,
                position=position,
                velocity=velocity,
                acceleration=acceleration,
                heading=obs.get("heading", 0.0),
            )
            
            # Add observed point
            trajectory.add_point(TrajectoryPoint(
                state=state,
                confidence=obs.get("confidence", 1.0),
                source="observed",
            ))
            
            # If not first observation, check physics between points
            if i > 0:
                prev_obs = sorted_obs[i - 1]
                prev_timestamp = prev_obs["timestamp"]
                if isinstance(prev_timestamp, str):
                    prev_timestamp = datetime.fromisoformat(prev_timestamp)
                
                prev_pos = Vector3(*prev_obs.get("position", (0, 0, 0)))
                
                dt = (timestamp - prev_timestamp).total_seconds()
                displacement = (position - prev_pos).magnitude()
                total_distance += displacement
                
                if dt > 0:
                    implied_speed = displacement / dt
                    # Check if speed is physically plausible
                    if implied_speed > 100:  # ~225 mph, unrealistic for most scenarios
                        violations.append({
                            "type": "implausible_speed",
                            "timestamp": timestamp.isoformat(),
                            "implied_speed_mps": implied_speed,
                        })
        
        # Get duration
        first_time = trajectory.start_time
        last_time = trajectory.end_time
        duration = (last_time - first_time).total_seconds() if first_time and last_time else 0
        
        return SimulationResult(
            trajectory=trajectory,
            converged=True,
            physics_violations=violations,
            total_distance=total_distance,
            duration_seconds=duration,
        )
    
    def _step(self, state: KinematicState, control: dict) -> KinematicState:
        """
        Compute next state from current state and control input.
        
        Uses the configured motion model.
        """
        dt = self.config.dt
        
        if self.config.motion_model == MotionModel.CONSTANT_VELOCITY:
            # Simple constant velocity motion
            new_position = state.position + state.velocity * dt
            return KinematicState(
                timestamp=state.timestamp + timedelta(seconds=dt),
                position=new_position,
                velocity=state.velocity,
                acceleration=Vector3(),
                heading=state.heading,
            )
        
        elif self.config.motion_model == MotionModel.CONSTANT_ACCELERATION:
            # Constant acceleration motion
            new_velocity = state.velocity + state.acceleration * dt
            new_position = state.position + state.velocity * dt + state.acceleration * (0.5 * dt * dt)
            return KinematicState(
                timestamp=state.timestamp + timedelta(seconds=dt),
                position=new_position,
                velocity=new_velocity,
                acceleration=state.acceleration,
                heading=state.heading,
            )
        
        else:  # POINT_MASS or other
            # Apply control inputs
            throttle = control.get("throttle", 0.0)
            brake = control.get("brake", 0.0)
            steering = control.get("steering", 0.0)
            
            # Compute longitudinal acceleration
            if brake > 0:
                long_accel = -self.config.max_deceleration * brake
            else:
                long_accel = self.config.max_acceleration * throttle
            
            # Update heading based on steering
            new_heading = state.heading + state.angular_velocity * dt + steering * dt
            
            # Compute velocity direction
            direction = Vector3(
                math.cos(new_heading),
                math.sin(new_heading),
                0.0,
            )
            
            # Update speed
            current_speed = state.speed
            new_speed = max(0, current_speed + long_accel * dt)
            
            # Compute new velocity and acceleration
            new_velocity = direction * new_speed
            new_acceleration = direction * long_accel
            
            # Update position
            new_position = state.position + new_velocity * dt
            
            return KinematicState(
                timestamp=state.timestamp + timedelta(seconds=dt),
                position=new_position,
                velocity=new_velocity,
                acceleration=new_acceleration,
                heading=new_heading,
                angular_velocity=steering / dt if dt > 0 else 0,
            )
    
    def _get_control_input(
        self,
        step: int,
        control_inputs: Optional[list[dict]],
    ) -> dict:
        """Get control input for a given simulation step."""
        if control_inputs is None or step >= len(control_inputs):
            return {"throttle": 0.0, "brake": 0.0, "steering": 0.0}
        return control_inputs[step]
    
    def _check_physics(
        self,
        prev_state: KinematicState,
        next_state: KinematicState,
    ) -> Optional[dict]:
        """Check if the state transition violates physics constraints."""
        dt = self.config.dt
        
        # Check acceleration limits
        accel_mag = next_state.acceleration.magnitude()
        if accel_mag > self.config.max_acceleration:
            return {
                "type": "acceleration_exceeded",
                "timestamp": next_state.timestamp.isoformat(),
                "acceleration": accel_mag,
                "limit": self.config.max_acceleration,
            }
        
        # Check lateral acceleration (simplified)
        if next_state.angular_velocity != 0:
            lateral_accel = next_state.speed * abs(next_state.angular_velocity)
            if lateral_accel > self.config.max_lateral_acceleration:
                return {
                    "type": "lateral_acceleration_exceeded",
                    "timestamp": next_state.timestamp.isoformat(),
                    "lateral_acceleration": lateral_accel,
                    "limit": self.config.max_lateral_acceleration,
                }
        
        # Check friction limit (simplified tire model)
        max_friction_accel = self.config.friction_coefficient * self.config.gravity
        if accel_mag > max_friction_accel:
            return {
                "type": "friction_limit_exceeded",
                "timestamp": next_state.timestamp.isoformat(),
                "acceleration": accel_mag,
                "friction_limit": max_friction_accel,
            }
        
        return None
