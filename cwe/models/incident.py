"""Incident model - container for all artifacts and analysis."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from cwe.models.artifact import Artifact
    from cwe.models.timeline import Timeline
    from cwe.models.counterfactual import Counterfactual


class IncidentStatus(str, Enum):
    """Status of incident analysis."""
    
    CREATED = "created"
    INGESTING = "ingesting"
    ALIGNING = "aligning"
    ANALYZING = "analyzing"
    READY = "ready"
    EXPLORING = "exploring"
    COMPLETED = "completed"
    ERROR = "error"


class IncidentMetadata(BaseModel):
    """Metadata about the incident."""
    
    # Domain classification
    domain: str | None = None  # traffic, cyber, industrial, etc.
    severity: str | None = None  # critical, high, medium, low
    
    # Location info
    location: str | None = None
    coordinates: tuple[float, float] | None = None  # lat, lon
    
    # Time info
    incident_time: datetime | None = None
    reported_time: datetime | None = None
    
    # Source info
    source: str | None = None
    case_number: str | None = None
    
    # Custom fields
    custom: dict[str, str] = Field(default_factory=dict)


class Incident(BaseModel):
    """
    An incident being analyzed by the Counterfactual World Engine.
    
    This is the top-level container that holds all artifacts, timelines,
    and counterfactual analyses for a single incident.
    """
    
    id: UUID = Field(default_factory=uuid4)
    name: str
    description: str | None = None
    
    # Status tracking
    status: IncidentStatus = IncidentStatus.CREATED
    status_message: str | None = None
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Metadata
    metadata: IncidentMetadata = Field(default_factory=IncidentMetadata)
    
    # References to related data (populated when loaded)
    artifact_ids: list[UUID] = Field(default_factory=list)
    canonical_timeline_id: UUID | None = None
    counterfactual_ids: list[UUID] = Field(default_factory=list)
    
    # Analysis configuration
    config: dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        json_encoders = {
            UUID: str,
            datetime: lambda v: v.isoformat(),
        }


class IncidentSummary(BaseModel):
    """Summary view of an incident for listing."""
    
    id: UUID
    name: str
    status: IncidentStatus
    domain: str | None
    artifact_count: int
    counterfactual_count: int
    created_at: datetime
    updated_at: datetime
