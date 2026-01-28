"""Evidence models for linking claims to source artifacts."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class EvidenceType(str, Enum):
    """Types of evidence."""
    
    # Video evidence
    VIDEO_FRAME = "video_frame"
    VIDEO_SEGMENT = "video_segment"
    
    # Log evidence
    LOG_ENTRY = "log_entry"
    LOG_PATTERN = "log_pattern"
    
    # Document evidence
    REPORT_EXCERPT = "report_excerpt"
    REPORT_TABLE = "report_table"
    
    # Sensor evidence
    SENSOR_READING = "sensor_reading"
    SENSOR_TRACE = "sensor_trace"
    
    # Derived evidence
    VLM_OBSERVATION = "vlm_observation"
    COMPUTED = "computed"


class Evidence(BaseModel):
    """
    Evidence supporting an event, causal link, or counterfactual claim.
    
    Links VLM-generated claims back to source artifacts for traceability.
    """
    
    id: UUID = Field(default_factory=uuid4)
    
    # What this evidence supports
    supports_type: str  # "event", "causal_link", "counterfactual"
    supports_id: UUID
    
    # Evidence type and source
    evidence_type: EvidenceType
    artifact_id: UUID
    
    # Location within artifact
    # For video: frame numbers
    # For logs: line numbers
    # For documents: page/paragraph
    # For sensors: timestamp range
    location: dict[str, Any] = Field(default_factory=dict)
    
    # Extracted content
    excerpt: str | None = None  # Text excerpt
    frame_path: str | None = None  # Path to extracted frame
    
    # Confidence that this evidence supports the claim
    relevance_score: float = 1.0
    
    # VLM's reasoning about this evidence
    vlm_interpretation: str | None = None
    
    # Timestamps
    artifact_timestamp: datetime | None = None
    extracted_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_encoders = {
            UUID: str,
            datetime: lambda v: v.isoformat(),
        }


class EvidenceChain(BaseModel):
    """A chain of evidence supporting a complex claim."""
    
    id: UUID = Field(default_factory=uuid4)
    
    # The claim being supported
    claim: str
    claim_type: str
    claim_id: UUID
    
    # Ordered list of evidence
    evidence_ids: list[UUID] = Field(default_factory=list)
    
    # Narrative explanation of the chain
    narrative: str | None = None
    
    # Overall chain strength
    chain_strength: float = 0.0
    
    # Weakest link in the chain
    weakest_link_id: UUID | None = None
    weakest_link_score: float | None = None
