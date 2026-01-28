"""
Tests for physics simulation module.
"""

import pytest
from datetime import datetime, timedelta

from cwe.physics.kinematics import (
    KinematicSimulator,
    KinematicState,
    Vector3,
    Trajectory,
    TrajectoryPoint,
    SimulationConfig,
    MotionModel,
)
from cwe.physics.collision import CollisionDetector, BoundingBox, CollisionPoint
from cwe.physics.validator import PhysicsValidator


class TestVector3:
    """Tests for Vector3 class."""
    
    def test_vector_creation(self):
        """Test vector creation."""
        v = Vector3(1.0, 2.0, 3.0)
        assert v.x == 1.0
        assert v.y == 2.0
        assert v.z == 3.0
    
    def test_vector_addition(self):
        """Test vector addition."""
        v1 = Vector3(1.0, 2.0, 3.0)
        v2 = Vector3(4.0, 5.0, 6.0)
        result = v1 + v2
        assert result.x == 5.0
        assert result.y == 7.0
        assert result.z == 9.0
    
    def test_vector_magnitude(self):
        """Test vector magnitude."""
        v = Vector3(3.0, 4.0, 0.0)
        assert v.magnitude() == 5.0
    
    def test_vector_normalized(self):
        """Test vector normalization."""
        v = Vector3(3.0, 4.0, 0.0)
        n = v.normalized()
        assert abs(n.magnitude() - 1.0) < 1e-6


class TestKinematicState:
    """Tests for KinematicState class."""
    
    def test_state_creation(self):
        """Test state creation."""
        state = KinematicState(
            timestamp=datetime.now(),
            position=Vector3(0, 0, 0),
            velocity=Vector3(10, 0, 0),
            acceleration=Vector3(0, 0, 0),
        )
        assert state.speed == 10.0
    
    def test_speed_conversion(self):
        """Test speed unit conversions."""
        state = KinematicState(
            timestamp=datetime.now(),
            position=Vector3(),
            velocity=Vector3(10, 0, 0),  # 10 m/s
            acceleration=Vector3(),
        )
        
        # 10 m/s ≈ 22.37 mph
        assert 22 < state.speed_mph < 23


class TestTrajectory:
    """Tests for Trajectory class."""
    
    def test_trajectory_creation(self):
        """Test trajectory creation."""
        traj = Trajectory(entity_id="vehicle_a")
        assert traj.entity_id == "vehicle_a"
        assert len(traj.points) == 0
    
    def test_add_points(self):
        """Test adding points to trajectory."""
        traj = Trajectory(entity_id="vehicle_a")
        
        now = datetime.now()
        point = TrajectoryPoint(
            state=KinematicState(
                timestamp=now,
                position=Vector3(0, 0, 0),
                velocity=Vector3(10, 0, 0),
                acceleration=Vector3(),
            )
        )
        
        traj.add_point(point)
        assert len(traj.points) == 1
        assert traj.start_time == now
        assert traj.end_time == now


class TestKinematicSimulator:
    """Tests for KinematicSimulator class."""
    
    def test_simulator_creation(self):
        """Test simulator creation."""
        sim = KinematicSimulator()
        assert sim.config.dt == 0.1
    
    def test_constant_velocity_projection(self):
        """Test trajectory projection with constant velocity."""
        config = SimulationConfig(
            motion_model=MotionModel.CONSTANT_VELOCITY
        )
        sim = KinematicSimulator(config)
        
        initial = KinematicState(
            timestamp=datetime.now(),
            position=Vector3(0, 0, 0),
            velocity=Vector3(10, 0, 0),  # 10 m/s in X direction
            acceleration=Vector3(),
        )
        
        result = sim.project_trajectory(initial, timedelta(seconds=10))
        
        assert result.converged
        assert result.total_distance > 0
        assert len(result.trajectory.points) > 0
    
    def test_reconstruct_from_observations(self):
        """Test trajectory reconstruction."""
        sim = KinematicSimulator()
        
        observations = [
            {
                "timestamp": datetime(2024, 1, 15, 14, 30, 0),
                "position": (0, 0, 0),
                "velocity": (10, 0, 0),
            },
            {
                "timestamp": datetime(2024, 1, 15, 14, 30, 5),
                "position": (50, 0, 0),
                "velocity": (10, 0, 0),
            },
        ]
        
        result = sim.reconstruct_from_observations(observations, "vehicle_a")
        
        assert result.converged
        assert result.trajectory.entity_id == "vehicle_a"


class TestBoundingBox:
    """Tests for BoundingBox class."""
    
    def test_bounding_box_creation(self):
        """Test bounding box creation."""
        box = BoundingBox(
            min_x=0, max_x=10,
            min_y=0, max_y=5,
        )
        assert box.center.x == 5.0
        assert box.center.y == 2.5
    
    def test_bounding_box_intersection(self):
        """Test bounding box intersection."""
        box1 = BoundingBox(min_x=0, max_x=10, min_y=0, max_y=10)
        box2 = BoundingBox(min_x=5, max_x=15, min_y=5, max_y=15)
        box3 = BoundingBox(min_x=20, max_x=30, min_y=20, max_y=30)
        
        assert box1.intersects(box2)  # Overlapping
        assert not box1.intersects(box3)  # Not overlapping


class TestCollisionDetector:
    """Tests for CollisionDetector class."""
    
    def test_detector_creation(self):
        """Test detector creation."""
        detector = CollisionDetector()
        assert detector.collision_threshold == 2.0
    
    def test_detect_collision(self):
        """Test collision detection between two trajectories."""
        detector = CollisionDetector()
        
        # Create two trajectories that intersect
        traj_a = Trajectory(entity_id="vehicle_a")
        traj_b = Trajectory(entity_id="vehicle_b")
        
        now = datetime.now()
        
        # Vehicle A moving right
        traj_a.add_point(TrajectoryPoint(
            state=KinematicState(
                timestamp=now,
                position=Vector3(0, 0, 0),
                velocity=Vector3(10, 0, 0),
                acceleration=Vector3(),
            )
        ))
        traj_a.add_point(TrajectoryPoint(
            state=KinematicState(
                timestamp=now + timedelta(seconds=1),
                position=Vector3(10, 0, 0),
                velocity=Vector3(10, 0, 0),
                acceleration=Vector3(),
            )
        ))
        
        # Vehicle B also at similar position
        traj_b.add_point(TrajectoryPoint(
            state=KinematicState(
                timestamp=now,
                position=Vector3(0, 1, 0),  # Close to A
                velocity=Vector3(10, 0, 0),
                acceleration=Vector3(),
            )
        ))
        traj_b.add_point(TrajectoryPoint(
            state=KinematicState(
                timestamp=now + timedelta(seconds=1),
                position=Vector3(10, 1, 0),
                velocity=Vector3(10, 0, 0),
                acceleration=Vector3(),
            )
        ))
        
        analysis = detector.detect_collision(traj_a, traj_b)
        
        # Distance is 1m, which is less than 2m threshold
        assert analysis.has_collision
    
    def test_time_to_collision(self):
        """Test time-to-collision estimation."""
        detector = CollisionDetector()
        
        state_a = KinematicState(
            timestamp=datetime.now(),
            position=Vector3(0, 0, 0),
            velocity=Vector3(10, 0, 0),
            acceleration=Vector3(),
        )
        
        state_b = KinematicState(
            timestamp=datetime.now(),
            position=Vector3(50, 0, 0),
            velocity=Vector3(-10, 0, 0),  # Approaching
            acceleration=Vector3(),
        )
        
        ttc = detector.estimate_time_to_collision(state_a, state_b)
        
        assert ttc is not None
        # Relative velocity is 20 m/s, distance is 50m
        # TTC should be around 2.4 seconds (accounting for threshold)
        assert 2 < ttc < 3


class TestPhysicsValidator:
    """Tests for PhysicsValidator class."""
    
    def test_validator_creation(self):
        """Test validator creation."""
        validator = PhysicsValidator()
        assert validator.max_speed_mps == 50.0
    
    def test_validate_empty_timeline(self):
        """Test validation of empty timeline."""
        from cwe.models.timeline import Timeline
        from uuid import uuid4
        from datetime import datetime, timedelta
        
        validator = PhysicsValidator()
        now = datetime.now()
        timeline = Timeline(
            incident_id=uuid4(),
            start_time=now,
            end_time=now + timedelta(minutes=5),
        )
        
        result = validator.validate_timeline(timeline, reconstruct_trajectories=False)
        
        assert result.is_valid
        assert len(result.violations) == 0
