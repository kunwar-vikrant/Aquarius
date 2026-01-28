"""
Physics validation for VLM-generated timelines.

Validates that events and trajectories in a timeline are physically
plausible, flagging violations and providing confidence adjustments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, Sequence
from uuid import UUID, uuid4

import structlog

from cwe.models.timeline import Timeline, Event, EventType
from cwe.physics.kinematics import (
    KinematicSimulator,
    Trajectory,
    Vector3,
    KinematicState,
    SimulationConfig,
)
from cwe.physics.collision import CollisionDetector, CollisionAnalysis

logger = structlog.get_logger(__name__)


@dataclass
class PhysicsViolation:
    """A detected physics violation in the timeline."""
    
    id: UUID = field(default_factory=uuid4)
    timestamp: datetime = field(default_factory=datetime.now)
    
    # What was violated
    violation_type: str = ""  # speed, acceleration, trajectory, collision, causality
    description: str = ""
    
    # Related events/entities
    event_ids: list[str] = field(default_factory=list)
    entity_ids: list[str] = field(default_factory=list)
    
    # Severity
    severity: str = "warning"  # info, warning, error, critical
    confidence_penalty: float = 0.1  # How much to reduce timeline confidence
    
    # Suggested fix
    suggestion: Optional[str] = None


@dataclass
class ValidationResult:
    """Result of physics validation."""
    
    is_valid: bool = True
    violations: list[PhysicsViolation] = field(default_factory=list)
    
    # Confidence adjustment
    original_confidence: float = 1.0
    adjusted_confidence: float = 1.0
    
    # Collision analysis
    collision_analysis: Optional[CollisionAnalysis] = None
    
    # Trajectory reconstructions
    reconstructed_trajectories: dict[str, Trajectory] = field(default_factory=dict)
    
    # Summary statistics
    num_events_validated: int = 0
    num_entities_validated: int = 0
    
    @property
    def num_violations(self) -> int:
        return len(self.violations)
    
    @property
    def has_critical_violations(self) -> bool:
        return any(v.severity == "critical" for v in self.violations)


class PhysicsValidator:
    """
    Validates timelines against physical constraints.
    
    Checks:
    - Speed limits (entity-specific and absolute)
    - Acceleration/deceleration limits
    - Trajectory plausibility
    - Collision consistency
    - Causal timing (effect doesn't precede cause)
    """
    
    def __init__(
        self,
        max_speed_mps: float = 50.0,  # ~112 mph
        max_acceleration: float = 15.0,  # m/s^2
        max_deceleration: float = 15.0,  # m/s^2 (hard braking)
        min_reaction_time: float = 0.2,  # seconds
    ):
        """
        Initialize physics validator.
        
        Args:
            max_speed_mps: Maximum plausible speed in m/s
            max_acceleration: Maximum acceleration in m/s^2
            max_deceleration: Maximum deceleration in m/s^2
            min_reaction_time: Minimum human reaction time
        """
        self.max_speed_mps = max_speed_mps
        self.max_acceleration = max_acceleration
        self.max_deceleration = max_deceleration
        self.min_reaction_time = min_reaction_time
        
        self.simulator = KinematicSimulator(SimulationConfig(
            max_acceleration=max_acceleration,
            max_deceleration=max_deceleration,
        ))
        self.collision_detector = CollisionDetector()
        self.logger = structlog.get_logger(__name__)
    
    def validate_timeline(
        self,
        timeline: Timeline,
        reconstruct_trajectories: bool = True,
    ) -> ValidationResult:
        """
        Validate a complete timeline for physics plausibility.
        
        Args:
            timeline: Timeline to validate
            reconstruct_trajectories: Whether to build and validate trajectories
        
        Returns:
            ValidationResult with violations and adjusted confidence
        """
        self.logger.info(
            "Validating timeline physics",
            timeline_id=str(timeline.id),
            num_events=len(timeline.events),
        )
        
        violations = []
        trajectories = {}
        
        # Validate individual events
        event_violations = self._validate_events(timeline.events)
        violations.extend(event_violations)
        
        # Validate causal links
        causal_violations = self._validate_causal_timing(timeline)
        violations.extend(causal_violations)
        
        # Reconstruct and validate trajectories
        if reconstruct_trajectories:
            trajectories = self._reconstruct_entity_trajectories(timeline)
            trajectory_violations = self._validate_trajectories(trajectories)
            violations.extend(trajectory_violations)
        
        # Detect collisions
        collision_analysis = None
        if len(trajectories) >= 2:
            collision_analysis = self.collision_detector.detect_multi_entity_collisions(
                list(trajectories.values())
            )
            collision_violations = self._validate_collision_consistency(
                timeline,
                collision_analysis,
            )
            violations.extend(collision_violations)
        
        # Calculate adjusted confidence
        original_confidence = timeline.confidence
        penalty = sum(v.confidence_penalty for v in violations)
        adjusted_confidence = max(0.0, original_confidence - penalty)
        
        return ValidationResult(
            is_valid=len([v for v in violations if v.severity in ("error", "critical")]) == 0,
            violations=violations,
            original_confidence=original_confidence,
            adjusted_confidence=adjusted_confidence,
            collision_analysis=collision_analysis,
            reconstructed_trajectories=trajectories,
            num_events_validated=len(timeline.events),
            num_entities_validated=len(trajectories),
        )
    
    def validate_counterfactual(
        self,
        original_timeline: Timeline,
        counterfactual_timeline: Timeline,
    ) -> ValidationResult:
        """
        Validate a counterfactual timeline against the original.
        
        Ensures the counterfactual diverges only at specified intervention
        points and remains physically plausible afterward.
        """
        self.logger.info(
            "Validating counterfactual",
            original_id=str(original_timeline.id),
            counterfactual_id=str(counterfactual_timeline.id),
        )
        
        # First validate the counterfactual itself
        result = self.validate_timeline(counterfactual_timeline)
        
        # Then check divergence consistency
        # (Future: implement intervention point validation)
        
        return result
    
    def _validate_events(self, events: Sequence[Event]) -> list[PhysicsViolation]:
        """Validate individual events for physics plausibility."""
        violations = []
        
        for event in events:
            # Skip non-physical events
            if event.event_type in (EventType.OBSERVATION, EventType.INFERENCE):
                continue
            
            # Check for speed in metadata
            if "speed" in event.metadata:
                speed = event.metadata["speed"]
                if isinstance(speed, (int, float)):
                    # Convert if needed (assume mph if > 50)
                    speed_mps = speed * 0.447 if speed > 50 else speed
                    
                    if speed_mps > self.max_speed_mps:
                        violations.append(PhysicsViolation(
                            timestamp=event.timestamp,
                            violation_type="speed",
                            description=f"Speed {speed_mps:.1f} m/s exceeds maximum {self.max_speed_mps:.1f} m/s",
                            event_ids=[str(event.id)],
                            severity="warning",
                            confidence_penalty=0.05,
                            suggestion="Verify speed measurement units and sensor accuracy",
                        ))
            
            # Check for acceleration in metadata
            if "acceleration" in event.metadata:
                accel = event.metadata["acceleration"]
                if isinstance(accel, (int, float)) and abs(accel) > self.max_acceleration:
                    violations.append(PhysicsViolation(
                        timestamp=event.timestamp,
                        violation_type="acceleration",
                        description=f"Acceleration {abs(accel):.1f} m/s² exceeds limit",
                        event_ids=[str(event.id)],
                        severity="warning",
                        confidence_penalty=0.05,
                    ))
        
        return violations
    
    def _validate_causal_timing(self, timeline: Timeline) -> list[PhysicsViolation]:
        """Validate that causal links respect temporal ordering."""
        violations = []
        
        # Build event lookup
        event_lookup = {str(e.id): e for e in timeline.events}
        
        for link in timeline.causal_links:
            cause_event = event_lookup.get(str(link.cause_id))
            effect_event = event_lookup.get(str(link.effect_id))
            
            if cause_event is None or effect_event is None:
                continue
            
            # Check temporal order
            if effect_event.timestamp <= cause_event.timestamp:
                violations.append(PhysicsViolation(
                    timestamp=effect_event.timestamp,
                    violation_type="causality",
                    description=f"Effect '{effect_event.description}' occurs before or at same time as cause '{cause_event.description}'",
                    event_ids=[str(cause_event.id), str(effect_event.id)],
                    severity="error",
                    confidence_penalty=0.2,
                    suggestion="Review temporal ordering of causal relationship",
                ))
            
            # Check minimum reaction time for human actions
            dt = (effect_event.timestamp - cause_event.timestamp).total_seconds()
            if dt > 0 and dt < self.min_reaction_time:
                if "human" in link.mechanism.lower() or "driver" in link.mechanism.lower():
                    violations.append(PhysicsViolation(
                        timestamp=effect_event.timestamp,
                        violation_type="reaction_time",
                        description=f"Human reaction time {dt:.3f}s below minimum {self.min_reaction_time:.3f}s",
                        event_ids=[str(cause_event.id), str(effect_event.id)],
                        severity="warning",
                        confidence_penalty=0.05,
                        suggestion="Consider if this is a reflex action or automated system",
                    ))
        
        return violations
    
    def _reconstruct_entity_trajectories(
        self,
        timeline: Timeline,
    ) -> dict[str, Trajectory]:
        """Reconstruct trajectories for entities from timeline events."""
        trajectories = {}
        
        # Group position updates by entity
        entity_observations: dict[str, list[dict]] = {}
        
        for event in timeline.events:
            if event.event_type != EventType.STATE_CHANGE:
                continue
            
            for entity_id in event.entity_ids:
                if entity_id not in entity_observations:
                    entity_observations[entity_id] = []
                
                # Extract position from event
                obs = {
                    "timestamp": event.timestamp,
                    "position": event.metadata.get("position", (0, 0, 0)),
                    "velocity": event.metadata.get("velocity", (0, 0, 0)),
                    "confidence": event.confidence,
                }
                entity_observations[entity_id].append(obs)
        
        # Reconstruct trajectory for each entity
        for entity_id, observations in entity_observations.items():
            if len(observations) >= 2:
                result = self.simulator.reconstruct_from_observations(
                    observations,
                    entity_id,
                )
                trajectories[entity_id] = result.trajectory
        
        return trajectories
    
    def _validate_trajectories(
        self,
        trajectories: dict[str, Trajectory],
    ) -> list[PhysicsViolation]:
        """Validate reconstructed trajectories for physics plausibility."""
        violations = []
        
        for entity_id, trajectory in trajectories.items():
            if not trajectory.points or len(trajectory.points) < 2:
                continue
            
            # Check velocity changes between points
            sorted_points = sorted(trajectory.points, key=lambda p: p["timestamp"])
            
            for i in range(1, len(sorted_points)):
                prev = sorted_points[i - 1]
                curr = sorted_points[i]
                
                prev_time = datetime.fromisoformat(prev["timestamp"])
                curr_time = datetime.fromisoformat(curr["timestamp"])
                dt = (curr_time - prev_time).total_seconds()
                
                if dt <= 0:
                    continue
                
                # Calculate implied acceleration
                prev_vel = Vector3(*prev["velocity"])
                curr_vel = Vector3(*curr["velocity"])
                accel = (curr_vel - prev_vel).magnitude() / dt
                
                if accel > self.max_acceleration * 1.5:  # Allow some tolerance
                    violations.append(PhysicsViolation(
                        timestamp=curr_time,
                        violation_type="trajectory_acceleration",
                        description=f"Entity {entity_id} acceleration {accel:.1f} m/s² exceeds physical limits",
                        entity_ids=[entity_id],
                        severity="warning",
                        confidence_penalty=0.05,
                    ))
        
        return violations
    
    def _validate_collision_consistency(
        self,
        timeline: Timeline,
        collision_analysis: CollisionAnalysis,
    ) -> list[PhysicsViolation]:
        """
        Validate that detected collisions are consistent with timeline events.
        
        Checks:
        - If simulation detects collision, timeline should have collision event
        - If timeline has collision event, simulation should confirm it
        """
        violations = []
        
        # Find collision events in timeline
        timeline_collisions = [
            e for e in timeline.events
            if e.event_type == EventType.COLLISION or "collision" in e.description.lower()
        ]
        
        # Check if simulation detected collisions that aren't in timeline
        for sim_collision in collision_analysis.collisions:
            # Look for matching timeline event (within 2 seconds)
            matched = False
            for tl_collision in timeline_collisions:
                dt = abs((sim_collision.timestamp - tl_collision.timestamp).total_seconds())
                if dt < 2.0:
                    matched = True
                    break
            
            if not matched:
                violations.append(PhysicsViolation(
                    timestamp=sim_collision.timestamp,
                    violation_type="missing_collision_event",
                    description=f"Physics simulation detected collision at {sim_collision.timestamp} not in timeline",
                    entity_ids=[sim_collision.entity_a_id, sim_collision.entity_b_id],
                    severity="warning",
                    confidence_penalty=0.1,
                    suggestion="Add collision event to timeline or verify trajectory data",
                ))
        
        # Check if timeline collisions are confirmed by simulation
        for tl_collision in timeline_collisions:
            matched = False
            for sim_collision in collision_analysis.collisions:
                dt = abs((sim_collision.timestamp - tl_collision.timestamp).total_seconds())
                if dt < 2.0:
                    matched = True
                    break
            
            if not matched and collision_analysis.collisions:
                violations.append(PhysicsViolation(
                    timestamp=tl_collision.timestamp,
                    violation_type="unconfirmed_collision",
                    description=f"Timeline collision event not confirmed by physics simulation",
                    event_ids=[str(tl_collision.id)],
                    severity="info",
                    confidence_penalty=0.05,
                    suggestion="Review entity trajectories around collision time",
                ))
        
        return violations
