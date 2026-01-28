"""
Temporal synchronization for multimodal data streams.

Handles timestamp normalization, anchor point detection, and
cross-modal alignment for accurate event correlation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Optional, Sequence, Callable
from uuid import UUID, uuid4

import structlog

logger = structlog.get_logger(__name__)


class TimestampFormat(str, Enum):
    """Common timestamp formats."""
    
    ISO8601 = "iso8601"
    UNIX_SECONDS = "unix_seconds"
    UNIX_MILLIS = "unix_millis"
    UNIX_MICROS = "unix_micros"
    VIDEO_FRAME = "video_frame"  # Frame number + FPS
    RELATIVE = "relative"  # Offset from start
    CUSTOM = "custom"


class AnchorType(str, Enum):
    """Types of temporal anchor points."""
    
    EXPLICIT = "explicit"  # Directly specified
    CROSS_MODAL = "cross_modal"  # Same event in multiple streams
    METADATA = "metadata"  # From file/stream metadata
    INFERRED = "inferred"  # VLM or heuristic-based
    USER = "user"  # Manually specified


@dataclass
class AnchorPoint:
    """
    A temporal anchor point linking multiple streams.
    
    Anchors are known-good synchronization points where we can
    confidently align timestamps across different data sources.
    """
    
    id: UUID = field(default_factory=uuid4)
    anchor_type: AnchorType = AnchorType.EXPLICIT
    
    # Description of the anchor event
    description: str = ""
    
    # Timestamps in each source
    # source_id -> (timestamp, format, confidence)
    timestamps: dict[str, tuple[datetime, TimestampFormat, float]] = field(default_factory=dict)
    
    # Confidence in this anchor
    confidence: float = 1.0
    
    # How this anchor was determined
    determination_method: str = ""
    
    def add_source_timestamp(
        self,
        source_id: str,
        timestamp: datetime,
        format: TimestampFormat = TimestampFormat.ISO8601,
        confidence: float = 1.0,
    ) -> None:
        """Add a timestamp for a specific source."""
        self.timestamps[source_id] = (timestamp, format, confidence)


@dataclass
class TimestampMapping:
    """
    Mapping between timestamps in different sources.
    
    Allows conversion from one source's time base to another's.
    """
    
    source_id: str
    target_id: str
    
    # Offset: target_time = source_time + offset
    offset: timedelta = timedelta()
    
    # Scale factor (for drift correction): target_time = source_time * scale + offset
    scale: float = 1.0
    
    # Confidence in this mapping
    confidence: float = 1.0
    
    # Anchor points used to derive this mapping
    anchor_ids: list[UUID] = field(default_factory=list)
    
    def map_timestamp(self, source_time: datetime) -> datetime:
        """Map a timestamp from source to target time base."""
        if self.scale == 1.0:
            return source_time + self.offset
        
        # Apply scale (drift correction)
        # Convert to seconds since epoch for scaling
        epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
        source_seconds = (source_time.replace(tzinfo=timezone.utc) - epoch).total_seconds()
        scaled_seconds = source_seconds * self.scale
        scaled_time = epoch + timedelta(seconds=scaled_seconds)
        
        return scaled_time + self.offset
    
    def inverse(self) -> "TimestampMapping":
        """Create inverse mapping (target -> source)."""
        if self.scale == 1.0:
            return TimestampMapping(
                source_id=self.target_id,
                target_id=self.source_id,
                offset=-self.offset,
                scale=1.0,
                confidence=self.confidence,
                anchor_ids=self.anchor_ids.copy(),
            )
        else:
            # For scaled mappings, inverse is more complex
            return TimestampMapping(
                source_id=self.target_id,
                target_id=self.source_id,
                offset=timedelta(seconds=-self.offset.total_seconds() / self.scale),
                scale=1.0 / self.scale,
                confidence=self.confidence * 0.95,  # Slight confidence penalty
                anchor_ids=self.anchor_ids.copy(),
            )


@dataclass
class AlignmentResult:
    """Result of temporal alignment operation."""
    
    success: bool = True
    
    # Mappings between all source pairs
    mappings: dict[tuple[str, str], TimestampMapping] = field(default_factory=dict)
    
    # Anchor points used
    anchor_points: list[AnchorPoint] = field(default_factory=list)
    
    # Reference time base (all other sources aligned to this)
    reference_source: Optional[str] = None
    
    # Quality metrics
    alignment_confidence: float = 1.0
    max_uncertainty_seconds: float = 0.0
    
    # Warnings and errors
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    
    def get_mapping(self, source_id: str, target_id: str) -> Optional[TimestampMapping]:
        """Get mapping between two sources."""
        if (source_id, target_id) in self.mappings:
            return self.mappings[(source_id, target_id)]
        if (target_id, source_id) in self.mappings:
            return self.mappings[(target_id, source_id)].inverse()
        return None
    
    def to_reference(self, source_id: str, timestamp: datetime) -> Optional[datetime]:
        """Convert timestamp from any source to reference time base."""
        if self.reference_source is None:
            return None
        if source_id == self.reference_source:
            return timestamp
        
        mapping = self.get_mapping(source_id, self.reference_source)
        if mapping:
            return mapping.map_timestamp(timestamp)
        return None


class TemporalSynchronizer:
    """
    Synchronizes timestamps across multiple data sources.
    
    Strategies:
    1. Anchor-based: Use known sync points across sources
    2. Metadata-based: Use file/stream metadata for initial alignment
    3. Content-based: Use VLM to detect same events in different streams
    4. Correlation-based: Find statistical correlations between streams
    """
    
    def __init__(
        self,
        max_drift_seconds: float = 1.0,
        anchor_min_confidence: float = 0.8,
    ):
        """
        Initialize synchronizer.
        
        Args:
            max_drift_seconds: Maximum expected clock drift
            anchor_min_confidence: Minimum confidence for anchor acceptance
        """
        self.max_drift_seconds = max_drift_seconds
        self.anchor_min_confidence = anchor_min_confidence
        self.logger = structlog.get_logger(__name__)
    
    def synchronize(
        self,
        sources: dict[str, list[dict]],
        anchor_points: Optional[list[AnchorPoint]] = None,
        reference_source: Optional[str] = None,
    ) -> AlignmentResult:
        """
        Synchronize multiple data sources.
        
        Args:
            sources: Dictionary of source_id -> list of timestamped records
            anchor_points: Known synchronization points
            reference_source: Source to use as time base reference
        
        Returns:
            AlignmentResult with mappings and quality metrics
        """
        self.logger.info(
            "Synchronizing sources",
            num_sources=len(sources),
            num_anchors=len(anchor_points) if anchor_points else 0,
        )
        
        if not sources:
            return AlignmentResult(success=False, errors=["No sources provided"])
        
        source_ids = list(sources.keys())
        
        # Select reference source
        if reference_source is None:
            reference_source = self._select_reference_source(sources)
        
        if reference_source not in sources:
            return AlignmentResult(
                success=False,
                errors=[f"Reference source '{reference_source}' not found"],
            )
        
        # Find or create anchor points
        anchors = list(anchor_points) if anchor_points else []
        if not anchors:
            inferred_anchors = self._infer_anchors(sources)
            anchors.extend(inferred_anchors)
        
        if not anchors:
            self.logger.warning("No anchor points found, using default offset")
            anchors = self._create_default_anchors(sources)
        
        # Compute mappings from anchors
        mappings = {}
        warnings = []
        max_uncertainty = 0.0
        
        for source_id in source_ids:
            if source_id == reference_source:
                continue
            
            mapping = self._compute_mapping(
                source_id,
                reference_source,
                anchors,
            )
            
            if mapping:
                mappings[(source_id, reference_source)] = mapping
                
                # Estimate uncertainty
                uncertainty = self._estimate_uncertainty(mapping, anchors)
                if uncertainty > max_uncertainty:
                    max_uncertainty = uncertainty
                
                if uncertainty > self.max_drift_seconds:
                    warnings.append(
                        f"High uncertainty ({uncertainty:.3f}s) for source '{source_id}'"
                    )
            else:
                warnings.append(
                    f"Could not compute mapping for source '{source_id}'"
                )
        
        # Calculate overall confidence
        if mappings:
            confidence = sum(m.confidence for m in mappings.values()) / len(mappings)
        else:
            confidence = 0.0
        
        return AlignmentResult(
            success=len(mappings) > 0,
            mappings=mappings,
            anchor_points=anchors,
            reference_source=reference_source,
            alignment_confidence=confidence,
            max_uncertainty_seconds=max_uncertainty,
            warnings=warnings,
        )
    
    def add_anchor(
        self,
        result: AlignmentResult,
        anchor: AnchorPoint,
    ) -> AlignmentResult:
        """
        Add a new anchor point and recompute alignments.
        
        Useful for iterative refinement as more information becomes available.
        """
        updated_anchors = result.anchor_points + [anchor]
        
        # Recompute all mappings with new anchor
        sources = {src: [] for src in self._get_sources_from_result(result)}
        
        return self.synchronize(
            sources,
            anchor_points=updated_anchors,
            reference_source=result.reference_source,
        )
    
    def align_event(
        self,
        event_timestamp: datetime,
        source_id: str,
        result: AlignmentResult,
    ) -> Optional[datetime]:
        """
        Align a single event timestamp to reference time base.
        
        Args:
            event_timestamp: Original timestamp from source
            source_id: Source the timestamp came from
            result: Alignment result with mappings
        
        Returns:
            Aligned timestamp in reference time base
        """
        return result.to_reference(source_id, event_timestamp)
    
    def _select_reference_source(self, sources: dict[str, list]) -> str:
        """
        Select the best source to use as time reference.
        
        Prefers sources with:
        - More data points
        - Higher timestamp precision
        - Known-good time synchronization (e.g., system logs)
        """
        # Simple heuristic: choose source with most data points
        # In practice, might prefer system logs or other authoritative sources
        best_source = max(sources.keys(), key=lambda s: len(sources[s]))
        
        # Override if we have system logs
        for source_id in sources:
            if "log" in source_id.lower() or "system" in source_id.lower():
                best_source = source_id
                break
        
        return best_source
    
    def _infer_anchors(
        self,
        sources: dict[str, list[dict]],
    ) -> list[AnchorPoint]:
        """
        Infer anchor points from data content.
        
        Looks for:
        - Events with matching descriptions
        - Timestamps that are suspiciously close
        - Known anchor event types (startup, collision, etc.)
        """
        anchors = []
        
        # Look for events with similar descriptions or timestamps
        # This is a simplified version; full implementation would use VLM
        
        source_ids = list(sources.keys())
        if len(source_ids) < 2:
            return anchors
        
        # Look for explicit anchor keywords
        anchor_keywords = ["start", "collision", "impact", "begin", "end", "alert"]
        
        for source_id, records in sources.items():
            for record in records:
                description = str(record.get("description", "")).lower()
                
                for keyword in anchor_keywords:
                    if keyword in description:
                        # Found potential anchor
                        anchor = AnchorPoint(
                            anchor_type=AnchorType.INFERRED,
                            description=f"Inferred anchor: {keyword}",
                            confidence=0.7,
                            determination_method=f"keyword match: {keyword}",
                        )
                        
                        timestamp = record.get("timestamp")
                        if timestamp:
                            if isinstance(timestamp, str):
                                try:
                                    timestamp = datetime.fromisoformat(timestamp)
                                except ValueError:
                                    continue
                            anchor.add_source_timestamp(source_id, timestamp)
                        
                        anchors.append(anchor)
                        break
        
        return anchors
    
    def _create_default_anchors(
        self,
        sources: dict[str, list[dict]],
    ) -> list[AnchorPoint]:
        """
        Create default anchors when none are found.
        
        Uses first timestamp in each source as a rough alignment.
        """
        anchor = AnchorPoint(
            anchor_type=AnchorType.METADATA,
            description="First timestamp alignment",
            confidence=0.5,
            determination_method="first timestamp in each source",
        )
        
        for source_id, records in sources.items():
            if records:
                timestamp = records[0].get("timestamp")
                if timestamp:
                    if isinstance(timestamp, str):
                        try:
                            timestamp = datetime.fromisoformat(timestamp)
                        except ValueError:
                            continue
                    anchor.add_source_timestamp(source_id, timestamp)
        
        return [anchor] if anchor.timestamps else []
    
    def _compute_mapping(
        self,
        source_id: str,
        target_id: str,
        anchors: list[AnchorPoint],
    ) -> Optional[TimestampMapping]:
        """
        Compute timestamp mapping between two sources using anchors.
        """
        # Find anchors that have both sources
        relevant_anchors = []
        for anchor in anchors:
            if source_id in anchor.timestamps and target_id in anchor.timestamps:
                relevant_anchors.append(anchor)
        
        if not relevant_anchors:
            self.logger.warning(
                "No common anchors",
                source=source_id,
                target=target_id,
            )
            return None
        
        # Calculate average offset
        offsets = []
        for anchor in relevant_anchors:
            source_ts, _, source_conf = anchor.timestamps[source_id]
            target_ts, _, target_conf = anchor.timestamps[target_id]
            
            offset = target_ts - source_ts
            weight = anchor.confidence * source_conf * target_conf
            offsets.append((offset.total_seconds(), weight))
        
        # Weighted average
        total_weight = sum(w for _, w in offsets)
        if total_weight == 0:
            return None
        
        avg_offset = sum(o * w for o, w in offsets) / total_weight
        avg_confidence = total_weight / len(offsets)
        
        return TimestampMapping(
            source_id=source_id,
            target_id=target_id,
            offset=timedelta(seconds=avg_offset),
            confidence=min(1.0, avg_confidence),
            anchor_ids=[a.id for a in relevant_anchors],
        )
    
    def _estimate_uncertainty(
        self,
        mapping: TimestampMapping,
        anchors: list[AnchorPoint],
    ) -> float:
        """
        Estimate uncertainty in a mapping based on anchor variance.
        """
        relevant_anchors = [
            a for a in anchors
            if a.id in mapping.anchor_ids
        ]
        
        if len(relevant_anchors) < 2:
            return self.max_drift_seconds  # Unknown uncertainty
        
        # Calculate variance in offsets
        offsets = []
        for anchor in relevant_anchors:
            if mapping.source_id in anchor.timestamps and mapping.target_id in anchor.timestamps:
                source_ts, _, _ = anchor.timestamps[mapping.source_id]
                target_ts, _, _ = anchor.timestamps[mapping.target_id]
                offsets.append((target_ts - source_ts).total_seconds())
        
        if len(offsets) < 2:
            return self.max_drift_seconds
        
        mean_offset = sum(offsets) / len(offsets)
        variance = sum((o - mean_offset) ** 2 for o in offsets) / len(offsets)
        
        return variance ** 0.5  # Standard deviation
    
    def _get_sources_from_result(self, result: AlignmentResult) -> list[str]:
        """Extract source IDs from alignment result."""
        sources = set()
        for src, tgt in result.mappings.keys():
            sources.add(src)
            sources.add(tgt)
        if result.reference_source:
            sources.add(result.reference_source)
        return list(sources)
