"""Artifact models for ingested data."""

from __future__ import annotations

from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class ArtifactType(str, Enum):
    """Types of artifacts that can be ingested."""
    
    VIDEO = "video"
    LOG = "log"
    REPORT = "report"
    SENSOR = "sensor"
    IMAGE = "image"
    AUDIO = "audio"
    UNKNOWN = "unknown"


class ArtifactStatus(str, Enum):
    """Processing status of an artifact."""
    
    PENDING = "pending"
    PROCESSING = "processing"
    READY = "ready"
    ERROR = "error"


class Artifact(BaseModel):
    """Base class for all artifact types."""
    
    id: UUID = Field(default_factory=uuid4)
    incident_id: UUID
    
    # File info
    filename: str
    file_path: str | None = None  # Path in object storage
    file_size: int | None = None
    mime_type: str | None = None
    
    # Type and status
    artifact_type: ArtifactType
    status: ArtifactStatus = ArtifactStatus.PENDING
    
    # Temporal info
    start_time: datetime | None = None
    end_time: datetime | None = None
    duration: timedelta | None = None
    
    # Timezone info for alignment
    timezone: str | None = None
    clock_offset_seconds: float = 0.0  # Adjustment for clock drift
    
    # Processing metadata
    processed_at: datetime | None = None
    processing_error: str | None = None
    
    # Custom metadata
    metadata: dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        json_encoders = {
            UUID: str,
            datetime: lambda v: v.isoformat(),
            timedelta: lambda v: v.total_seconds(),
        }


class VideoFrame(BaseModel):
    """A single frame extracted from video."""
    
    frame_number: int
    timestamp: datetime
    timestamp_offset: float  # Seconds from video start
    file_path: str  # Path to extracted frame image
    is_keyframe: bool = False
    scene_id: str | None = None  # Scene segmentation label
    

class VideoArtifact(Artifact):
    """Video artifact with video-specific metadata."""
    
    artifact_type: ArtifactType = ArtifactType.VIDEO
    
    # Video properties
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    frame_count: int | None = None
    codec: str | None = None
    
    # Extracted frames
    keyframes: list[VideoFrame] = Field(default_factory=list)
    extracted_frames_dir: str | None = None
    
    # Scene segmentation
    scenes: list[dict[str, Any]] = Field(default_factory=list)
    
    # OCR'd timestamps from video (for alignment)
    ocr_timestamps: list[dict[str, Any]] = Field(default_factory=list)


class LogEntry(BaseModel):
    """A single log entry."""
    
    line_number: int
    timestamp: datetime | None = None
    level: str | None = None  # DEBUG, INFO, WARN, ERROR, etc.
    source: str | None = None  # Component/service that generated the log
    message: str
    raw_line: str
    parsed_fields: dict[str, Any] = Field(default_factory=dict)


class LogArtifact(Artifact):
    """Log file artifact with parsed entries."""
    
    artifact_type: ArtifactType = ArtifactType.LOG
    
    # Log format
    log_format: str | None = None  # syslog, json, csv, custom
    timestamp_format: str | None = None
    
    # Raw content for small logs
    raw_content: str | None = None

    # Parsed entries (may be stored separately for large logs)
    entry_count: int = 0
    entries_file: str | None = None  # Path to parsed entries file
    
    # Sample entries for context
    sample_entries: list[LogEntry] = Field(default_factory=list)
    
    # Detected patterns/anomalies
    detected_patterns: list[dict[str, Any]] = Field(default_factory=list)


class ReportSection(BaseModel):
    """A section from a document/report."""
    
    section_id: str
    title: str | None = None
    page_numbers: list[int] = Field(default_factory=list)
    content: str
    content_type: str = "text"  # text, table, image_description


class ReportArtifact(Artifact):
    """Document/report artifact with extracted content."""
    
    artifact_type: ArtifactType = ArtifactType.REPORT
    
    # Document properties
    page_count: int | None = None
    document_type: str | None = None  # pdf, docx, html, etc.
    
    # Extracted content
    full_text: str | None = None
    sections: list[ReportSection] = Field(default_factory=list)
    
    # Extracted entities/dates
    extracted_dates: list[dict[str, Any]] = Field(default_factory=list)
    extracted_entities: list[dict[str, Any]] = Field(default_factory=list)


class SensorReading(BaseModel):
    """A single sensor reading."""
    
    timestamp: datetime
    sensor_id: str
    sensor_type: str
    value: float | dict[str, float]
    unit: str | None = None


class SensorArtifact(Artifact):
    """Sensor/telemetry data artifact."""
    
    artifact_type: ArtifactType = ArtifactType.SENSOR
    
    # Sensor info
    sensor_type: str | None = None  # gps, accelerometer, gyroscope, etc.
    sensor_id: str | None = None
    sampling_rate_hz: float | None = None
    
    # Data
    reading_count: int = 0
    readings_file: str | None = None  # Path to readings file
    
    # Sample readings
    sample_readings: list[SensorReading] = Field(default_factory=list)
    
    # Derived metrics
    derived_metrics: dict[str, Any] = Field(default_factory=dict)
