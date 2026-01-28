"""Timeline and causal graph models."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class EventType(str, Enum):
    """Classification of events in the timeline."""
    
    # State changes
    STATE_CHANGE = "state_change"
    POSITION_CHANGE = "position_change"
    VELOCITY_CHANGE = "velocity_change"
    
    # Actions
    ACTION_INITIATED = "action_initiated"
    ACTION_COMPLETED = "action_completed"
    ACTION_FAILED = "action_failed"
    
    # Observations
    OBSERVATION = "observation"
    DETECTION = "detection"
    
    # Interactions
    INTERACTION = "interaction"
    COLLISION = "collision"
    COMMUNICATION = "communication"
    
    # System events
    SYSTEM_EVENT = "system_event"
    ERROR = "error"
    ALERT = "alert"
    
    # Meta events
    TIMELINE_START = "timeline_start"
    TIMELINE_END = "timeline_end"
    BRANCH_POINT = "branch_point"


class CausalRelation(str, Enum):
    """Types of causal relationships between events."""
    
    CAUSES = "causes"          # A directly causes B
    ENABLES = "enables"        # A makes B possible (necessary but not sufficient)
    PREVENTS = "prevents"      # A stops B from happening
    DELAYS = "delays"          # A postpones B
    ACCELERATES = "accelerates"  # A makes B happen sooner
    CORRELATES = "correlates"  # A and B co-occur (no direct causation)


class EntityState(BaseModel):
    """State of an entity at a point in time."""
    
    timestamp: datetime
    properties: dict[str, Any] = Field(default_factory=dict)
    position: tuple[float, float, float] | None = None  # x, y, z
    velocity: tuple[float, float, float] | None = None  # vx, vy, vz
    orientation: tuple[float, float, float] | None = None  # roll, pitch, yaw
    

class Entity(BaseModel):
    """An entity tracked through the timeline."""
    
    id: UUID = Field(default_factory=uuid4)
    name: str
    entity_type: str  # vehicle, person, system, component, etc.
    properties: dict[str, Any] = Field(default_factory=dict)
    states: list[EntityState] = Field(default_factory=list)
    
    def get_state_at(self, timestamp: datetime) -> EntityState | None:
        """Get entity state at or just before the given timestamp."""
        relevant_states = [s for s in self.states if s.timestamp <= timestamp]
        if not relevant_states:
            return None
        return max(relevant_states, key=lambda s: s.timestamp)


class StateChange(BaseModel):
    """A change in an entity's state."""
    
    entity_id: UUID
    property_name: str
    old_value: Any
    new_value: Any
    

class EvidenceRef(BaseModel):
    """Reference to evidence supporting an event or link."""
    
    artifact_id: UUID
    artifact_type: str
    location: str  # Frame number, log line, page number, etc.
    excerpt: str | None = None  # Relevant excerpt
    confidence: float = 1.0


class Event(BaseModel):
    """An event in the timeline."""
    
    id: UUID = Field(default_factory=uuid4)
    timestamp: datetime
    event_type: EventType
    description: str
    
    # Entities involved
    entities: list[UUID] = Field(default_factory=list)
    
    # State changes caused by this event
    state_changes: list[StateChange] = Field(default_factory=list)
    
    # Evidence supporting this event
    evidence: list[EvidenceRef] = Field(default_factory=list)
    
    # Confidence in this event (0.0 - 1.0)
    confidence: float = 1.0
    
    # VLM reasoning that identified this event
    reasoning: str | None = None
    
    # Metadata
    metadata: dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        json_encoders = {
            UUID: str,
            datetime: lambda v: v.isoformat(),
        }


class CausalLink(BaseModel):
    """A causal relationship between two events."""
    
    id: UUID = Field(default_factory=uuid4)
    source_event_id: UUID
    target_event_id: UUID
    
    relation: CausalRelation
    
    # Natural language explanation of the causal mechanism
    mechanism: str
    
    # Confidence in this causal link (0.0 - 1.0)
    confidence: float = 1.0
    
    # Evidence supporting this link
    evidence: list[EvidenceRef] = Field(default_factory=list)
    
    # Time delay between cause and effect (if applicable)
    delay_seconds: float | None = None
    
    # VLM reasoning that identified this link
    reasoning: str | None = None


class Timeline(BaseModel):
    """A timeline of events with causal relationships."""
    
    id: UUID = Field(default_factory=uuid4)
    incident_id: UUID
    
    # Whether this is the canonical (actual) timeline or a counterfactual
    is_canonical: bool = True
    
    # If counterfactual, the intervention that created it
    intervention_id: UUID | None = None
    
    # Timeline bounds
    start_time: datetime
    end_time: datetime
    
    # Events in chronological order
    events: list[Event] = Field(default_factory=list)
    
    # Causal relationships
    causal_links: list[CausalLink] = Field(default_factory=list)
    
    # Entities tracked in this timeline
    entities: list[Entity] = Field(default_factory=list)
    
    # Point where this timeline diverges from canonical (if counterfactual)
    divergence_point: datetime | None = None
    
    # Overall confidence in the timeline
    confidence: float = 1.0
    
    # VLM session info
    vlm_provider: str | None = None
    vlm_model: str | None = None
    vlm_session_id: str | None = None
    
    # Metadata
    metadata: dict[str, Any] = Field(default_factory=dict)
    
    def get_event(self, event_id: UUID) -> Event | None:
        """Get an event by ID."""
        for event in self.events:
            if event.id == event_id:
                return event
        return None
    
    def get_events_in_range(
        self, 
        start: datetime, 
        end: datetime
    ) -> list[Event]:
        """Get all events within a time range."""
        return [
            e for e in self.events 
            if start <= e.timestamp <= end
        ]
    
    def get_causes(self, event_id: UUID) -> list[Event]:
        """Get all events that cause the given event."""
        cause_ids = [
            link.source_event_id 
            for link in self.causal_links 
            if link.target_event_id == event_id
            and link.relation in (CausalRelation.CAUSES, CausalRelation.ENABLES)
        ]
        return [e for e in self.events if e.id in cause_ids]
    
    def get_effects(self, event_id: UUID) -> list[Event]:
        """Get all events caused by the given event."""
        effect_ids = [
            link.target_event_id 
            for link in self.causal_links 
            if link.source_event_id == event_id
            and link.relation in (CausalRelation.CAUSES, CausalRelation.ENABLES)
        ]
        return [e for e in self.events if e.id in effect_ids]
    
    def get_root_causes(self) -> list[Event]:
        """Get events that have no causes (root causes)."""
        events_with_causes = {
            link.target_event_id 
            for link in self.causal_links
            if link.relation == CausalRelation.CAUSES
        }
        return [e for e in self.events if e.id not in events_with_causes]
    
    def get_terminal_events(self) -> list[Event]:
        """Get events that have no effects (outcomes)."""
        events_with_effects = {
            link.source_event_id 
            for link in self.causal_links
            if link.relation == CausalRelation.CAUSES
        }
        return [e for e in self.events if e.id not in events_with_effects]
    
    def to_networkx(self):
        """Convert to NetworkX DiGraph for graph analysis."""
        import networkx as nx
        
        G = nx.DiGraph()
        
        # Add events as nodes
        for event in self.events:
            G.add_node(
                str(event.id),
                timestamp=event.timestamp,
                event_type=event.event_type.value,
                description=event.description,
                confidence=event.confidence,
            )
        
        # Add causal links as edges
        for link in self.causal_links:
            G.add_edge(
                str(link.source_event_id),
                str(link.target_event_id),
                relation=link.relation.value,
                mechanism=link.mechanism,
                confidence=link.confidence,
            )
        
        return G
