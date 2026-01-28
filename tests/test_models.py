"""
Tests for data models - aligned with actual model definitions.
"""

import pytest
from datetime import datetime, timedelta
from uuid import uuid4

from cwe.models.timeline import (
    Event,
    EventType,
    Timeline,
    CausalLink,
    CausalRelation,
    Entity,
    EntityState,
)
from cwe.models.incident import Incident, IncidentStatus
from cwe.models.counterfactual import (
    Counterfactual,
    Intervention,
    InterventionType,
    OutcomeComparison,
    OutcomeSeverity,
)
from cwe.models.artifact import VideoArtifact, LogArtifact, ReportArtifact, ArtifactType


class TestEvent:
    """Tests for Event model."""
    
    def test_event_creation(self):
        """Test basic event creation."""
        event = Event(
            timestamp=datetime.now(),
            event_type=EventType.ACTION_INITIATED,
            description="Vehicle begins braking",
            entities=[uuid4()],
        )
        
        assert event.id is not None
        assert event.event_type == EventType.ACTION_INITIATED
        assert len(event.entities) == 1
    
    def test_event_with_metadata(self):
        """Test event with metadata."""
        event = Event(
            timestamp=datetime.now(),
            event_type=EventType.STATE_CHANGE,
            description="Speed changed",
            metadata={"speed": 45.0, "unit": "mph"},
        )
        
        assert event.metadata["speed"] == 45.0
    
    def test_event_serialization(self):
        """Test event JSON serialization."""
        event = Event(
            timestamp=datetime(2024, 1, 15, 14, 30, 0),
            event_type=EventType.COLLISION,
            description="Collision occurred",
        )
        
        data = event.model_dump()
        assert "timestamp" in data
        assert data["event_type"] == "collision"


class TestTimeline:
    """Tests for Timeline model."""
    
    def test_timeline_creation(self):
        """Test basic timeline creation."""
        now = datetime.now()
        timeline = Timeline(
            incident_id=uuid4(),
            name="Main timeline",
            start_time=now,
            end_time=now + timedelta(minutes=5),
        )
        
        assert timeline.id is not None
        assert len(timeline.events) == 0
        assert timeline.is_canonical is True
    
    def test_add_events(self):
        """Test adding events to timeline."""
        now = datetime.now()
        timeline = Timeline(
            incident_id=uuid4(),
            start_time=now,
            end_time=now + timedelta(minutes=5),
        )
        
        event1 = Event(
            timestamp=now,
            event_type=EventType.OBSERVATION,
            description="First event",
        )
        event2 = Event(
            timestamp=now + timedelta(seconds=5),
            event_type=EventType.ACTION_INITIATED,
            description="Second event",
        )
        
        timeline.events.append(event1)
        timeline.events.append(event2)
        
        assert len(timeline.events) == 2
    
    def test_timeline_get_event(self):
        """Test getting event by ID."""
        now = datetime.now()
        timeline = Timeline(
            incident_id=uuid4(),
            start_time=now,
            end_time=now + timedelta(minutes=5),
        )
        
        event = Event(
            timestamp=now,
            event_type=EventType.OBSERVATION,
            description="Test event",
        )
        timeline.events.append(event)
        
        found = timeline.get_event(event.id)
        assert found is not None
        assert found.description == "Test event"


class TestCausalLink:
    """Tests for CausalLink model."""
    
    def test_causal_link_creation(self):
        """Test causal link creation."""
        link = CausalLink(
            source_event_id=uuid4(),
            target_event_id=uuid4(),
            relation=CausalRelation.CAUSES,
            mechanism="Driver reaction to traffic light",
            confidence=0.9,
        )
        
        assert link.relation == CausalRelation.CAUSES
        assert link.confidence == 0.9
    
    def test_causal_link_with_delay(self):
        """Test causal link with time delay."""
        link = CausalLink(
            source_event_id=uuid4(),
            target_event_id=uuid4(),
            relation=CausalRelation.ENABLES,
            mechanism="Brake light triggers reaction",
            delay_seconds=0.5,
        )
        
        assert link.delay_seconds == 0.5


class TestIncident:
    """Tests for Incident model."""
    
    def test_incident_creation(self):
        """Test incident creation."""
        incident = Incident(
            name="Traffic collision at Main & Oak",
            description="Two-vehicle collision at intersection",
        )
        
        assert incident.id is not None
        assert incident.status == IncidentStatus.CREATED
        assert len(incident.artifact_ids) == 0
    
    def test_incident_status_transition(self):
        """Test incident status changes."""
        incident = Incident(name="Test incident")
        
        incident.status = IncidentStatus.ANALYZING
        assert incident.status == IncidentStatus.ANALYZING
        
        incident.status = IncidentStatus.COMPLETED
        assert incident.status == IncidentStatus.COMPLETED


class TestCounterfactual:
    """Tests for Counterfactual model."""
    
    def test_counterfactual_creation(self):
        """Test counterfactual creation."""
        intervention = Intervention(
            intervention_type=InterventionType.MODIFY_EVENT,
            description="Driver brakes 2 seconds earlier",
        )
        
        cf = Counterfactual(
            incident_id=uuid4(),
            canonical_timeline_id=uuid4(),
            intervention=intervention,
        )
        
        assert cf.id is not None
        assert cf.intervention.intervention_type == InterventionType.MODIFY_EVENT
    
    def test_counterfactual_with_outcome(self):
        """Test counterfactual with outcome comparison."""
        intervention = Intervention(
            intervention_type=InterventionType.ADVANCE_EVENT,
            description="Driver brakes 2 seconds earlier",
            parameters={"delay_seconds": -2.0},
        )
        
        outcome = OutcomeComparison(
            canonical_outcome="Collision at intersection",
            canonical_severity=OutcomeSeverity.SEVERE,
            canonical_score=0.8,
            counterfactual_outcome="Near miss, no collision",
            counterfactual_severity=OutcomeSeverity.MINOR,
            counterfactual_score=0.2,
            outcome_improved=True,
            improvement_magnitude=0.6,
        )
        
        cf = Counterfactual(
            incident_id=uuid4(),
            canonical_timeline_id=uuid4(),
            intervention=intervention,
            outcome=outcome,
        )
        
        assert cf.outcome is not None
        assert cf.outcome.outcome_improved


class TestArtifacts:
    """Tests for artifact models."""
    
    def test_video_artifact(self):
        """Test video artifact creation."""
        artifact = VideoArtifact(
            filename="dashcam.mp4",
            incident_id=uuid4(),
            duration_seconds=120.0,
            fps=30.0,
            resolution=(1920, 1080),
        )
        
        assert artifact.artifact_type == ArtifactType.VIDEO
        assert artifact.fps == 30.0
    
    def test_log_artifact(self):
        """Test log artifact creation."""
        artifact = LogArtifact(
            filename="system.log",
            incident_id=uuid4(),
            log_format="jsonl",
            entry_count=1500,
        )
        
        assert artifact.artifact_type == ArtifactType.LOG
        assert artifact.entry_count == 1500
    
    def test_report_artifact(self):
        """Test report artifact creation."""
        artifact = ReportArtifact(
            filename="incident_report.pdf",
            incident_id=uuid4(),
            document_type="pdf",
            full_text="Collision occurred at 14:30...",
        )
        
        assert artifact.artifact_type == ArtifactType.REPORT
        assert "14:30" in artifact.full_text


class TestEntity:
    """Tests for Entity model."""
    
    def test_entity_creation(self):
        """Test entity creation."""
        entity = Entity(
            name="Vehicle A",
            entity_type="vehicle",
            properties={"make": "Toyota", "model": "Camry"},
        )
        
        assert entity.id is not None
        assert entity.properties["make"] == "Toyota"
    
    def test_entity_state(self):
        """Test entity state."""
        entity_id = uuid4()
        state = EntityState(
            entity_id=entity_id,
            timestamp=datetime.now(),
            properties={"speed": 45.0, "position": (100, 50)},
        )
        
        assert state.properties["speed"] == 45.0
