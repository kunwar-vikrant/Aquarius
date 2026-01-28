"""
Clock drift detection and correction.

Handles scenarios where different data sources have clocks
that run at slightly different rates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, Sequence
from uuid import UUID, uuid4

import structlog

from cwe.alignment.synchronizer import AnchorPoint, TimestampMapping

logger = structlog.get_logger(__name__)


@dataclass
class DriftProfile:
    """
    Profile of clock drift for a data source.
    
    Captures both constant offset and linear drift rate.
    """
    
    source_id: str
    
    # Reference time base this profile is relative to
    reference_id: str
    
    # Constant offset at reference time
    offset_at_reference: timedelta = timedelta()
    
    # Reference timestamp
    reference_timestamp: datetime = field(default_factory=datetime.now)
    
    # Drift rate: seconds of drift per second of elapsed time
    # drift_rate = 0 means no drift
    # drift_rate = 0.001 means source gains 1ms per second relative to reference
    # drift_rate = -0.001 means source loses 1ms per second relative to reference
    drift_rate: float = 0.0
    
    # Confidence in drift estimation
    confidence: float = 1.0
    
    # How drift was determined
    determination_method: str = ""
    
    # Number of anchor points used
    anchor_count: int = 0
    
    def get_corrected_offset(self, at_time: datetime) -> timedelta:
        """
        Get the corrected offset at a specific time.
        
        Accounts for linear drift from reference timestamp.
        """
        elapsed = (at_time - self.reference_timestamp).total_seconds()
        drift_correction = self.drift_rate * elapsed
        
        return self.offset_at_reference + timedelta(seconds=drift_correction)
    
    def correct_timestamp(self, source_timestamp: datetime) -> datetime:
        """
        Correct a timestamp from this source to reference time.
        """
        offset = self.get_corrected_offset(source_timestamp)
        return source_timestamp + offset
    
    @property
    def is_significant(self) -> bool:
        """Check if drift is significant enough to matter."""
        # More than 1ms per minute is significant
        return abs(self.drift_rate) > 1e-5


@dataclass
class DriftAnalysis:
    """Result of drift analysis."""
    
    profiles: dict[str, DriftProfile] = field(default_factory=dict)
    
    # Quality metrics
    overall_confidence: float = 1.0
    max_drift_detected: float = 0.0  # Max drift rate found
    
    # Recommendations
    requires_correction: bool = False
    warnings: list[str] = field(default_factory=list)
    
    def get_profile(self, source_id: str) -> Optional[DriftProfile]:
        """Get drift profile for a source."""
        return self.profiles.get(source_id)
    
    def correct_timestamp(
        self,
        source_id: str,
        timestamp: datetime,
    ) -> Optional[datetime]:
        """Correct a timestamp using its source's drift profile."""
        profile = self.profiles.get(source_id)
        if profile:
            return profile.correct_timestamp(timestamp)
        return None


class DriftCorrector:
    """
    Detects and corrects clock drift between data sources.
    
    Uses multiple anchor points over time to detect linear drift
    and applies corrections to align all sources to a common
    time base.
    """
    
    def __init__(
        self,
        min_anchors_for_drift: int = 3,
        min_time_span_seconds: float = 60.0,
        drift_significance_threshold: float = 1e-5,
    ):
        """
        Initialize drift corrector.
        
        Args:
            min_anchors_for_drift: Minimum anchors needed to estimate drift
            min_time_span_seconds: Minimum time span for reliable drift estimation
            drift_significance_threshold: Minimum drift rate to consider significant
        """
        self.min_anchors_for_drift = min_anchors_for_drift
        self.min_time_span_seconds = min_time_span_seconds
        self.drift_significance_threshold = drift_significance_threshold
        self.logger = structlog.get_logger(__name__)
    
    def analyze_drift(
        self,
        anchors: Sequence[AnchorPoint],
        reference_id: str,
    ) -> DriftAnalysis:
        """
        Analyze clock drift across all sources.
        
        Args:
            anchors: Anchor points with synchronized timestamps
            reference_id: Source to use as reference (assumed drift-free)
        
        Returns:
            DriftAnalysis with profiles for all sources
        """
        self.logger.info(
            "Analyzing clock drift",
            num_anchors=len(anchors),
            reference=reference_id,
        )
        
        # Group anchors by source pairs
        source_pairs: dict[str, list[tuple[datetime, datetime]]] = {}
        
        for anchor in anchors:
            if reference_id not in anchor.timestamps:
                continue
            
            ref_ts, _, ref_conf = anchor.timestamps[reference_id]
            
            for source_id, (source_ts, _, source_conf) in anchor.timestamps.items():
                if source_id == reference_id:
                    continue
                
                if source_id not in source_pairs:
                    source_pairs[source_id] = []
                
                source_pairs[source_id].append((source_ts, ref_ts))
        
        # Analyze each source
        profiles = {}
        max_drift = 0.0
        warnings = []
        requires_correction = False
        
        for source_id, pairs in source_pairs.items():
            profile = self._compute_drift_profile(
                source_id,
                reference_id,
                pairs,
            )
            
            if profile:
                profiles[source_id] = profile
                
                if abs(profile.drift_rate) > max_drift:
                    max_drift = abs(profile.drift_rate)
                
                if profile.is_significant:
                    requires_correction = True
                    self.logger.warning(
                        "Significant drift detected",
                        source=source_id,
                        drift_rate=profile.drift_rate,
                    )
                    warnings.append(
                        f"Source '{source_id}' has significant drift: "
                        f"{profile.drift_rate * 1000:.3f} ms/s"
                    )
            else:
                warnings.append(
                    f"Could not compute drift profile for '{source_id}'"
                )
        
        # Calculate overall confidence
        if profiles:
            confidence = sum(p.confidence for p in profiles.values()) / len(profiles)
        else:
            confidence = 0.0
        
        return DriftAnalysis(
            profiles=profiles,
            overall_confidence=confidence,
            max_drift_detected=max_drift,
            requires_correction=requires_correction,
            warnings=warnings,
        )
    
    def apply_correction(
        self,
        analysis: DriftAnalysis,
        records: dict[str, list[dict]],
    ) -> dict[str, list[dict]]:
        """
        Apply drift correction to data records.
        
        Args:
            analysis: Drift analysis with profiles
            records: Source ID -> list of records with timestamps
        
        Returns:
            Corrected records with adjusted timestamps
        """
        self.logger.info(
            "Applying drift correction",
            num_sources=len(records),
        )
        
        corrected = {}
        
        for source_id, source_records in records.items():
            profile = analysis.get_profile(source_id)
            
            if profile is None or not profile.is_significant:
                # No correction needed
                corrected[source_id] = source_records
                continue
            
            # Apply correction to each record
            corrected_records = []
            for record in source_records:
                corrected_record = record.copy()
                
                if "timestamp" in record:
                    ts = record["timestamp"]
                    if isinstance(ts, str):
                        try:
                            ts = datetime.fromisoformat(ts)
                        except ValueError:
                            corrected_records.append(corrected_record)
                            continue
                    
                    corrected_ts = profile.correct_timestamp(ts)
                    corrected_record["timestamp"] = corrected_ts.isoformat()
                    corrected_record["_original_timestamp"] = ts.isoformat()
                    corrected_record["_drift_corrected"] = True
                
                corrected_records.append(corrected_record)
            
            corrected[source_id] = corrected_records
        
        return corrected
    
    def create_mapping_with_drift(
        self,
        profile: DriftProfile,
    ) -> TimestampMapping:
        """
        Create a TimestampMapping that incorporates drift correction.
        """
        return TimestampMapping(
            source_id=profile.source_id,
            target_id=profile.reference_id,
            offset=profile.offset_at_reference,
            scale=1.0 + profile.drift_rate,  # Incorporate drift into scale
            confidence=profile.confidence,
        )
    
    def _compute_drift_profile(
        self,
        source_id: str,
        reference_id: str,
        pairs: list[tuple[datetime, datetime]],
    ) -> Optional[DriftProfile]:
        """
        Compute drift profile from timestamp pairs.
        
        Uses linear regression to find drift rate.
        """
        if len(pairs) < self.min_anchors_for_drift:
            self.logger.debug(
                "Insufficient anchors for drift estimation",
                source=source_id,
                anchor_count=len(pairs),
            )
            # Return profile with zero drift
            if pairs:
                first_source, first_ref = pairs[0]
                return DriftProfile(
                    source_id=source_id,
                    reference_id=reference_id,
                    offset_at_reference=first_ref - first_source,
                    reference_timestamp=first_ref,
                    drift_rate=0.0,
                    confidence=0.5,
                    determination_method="single point (no drift estimation)",
                    anchor_count=len(pairs),
                )
            return None
        
        # Sort by reference time
        pairs = sorted(pairs, key=lambda p: p[1])
        
        # Check time span
        first_ref = pairs[0][1]
        last_ref = pairs[-1][1]
        time_span = (last_ref - first_ref).total_seconds()
        
        if time_span < self.min_time_span_seconds:
            self.logger.debug(
                "Insufficient time span for drift estimation",
                source=source_id,
                time_span=time_span,
            )
            # Use average offset
            offsets = [(ref - src).total_seconds() for src, ref in pairs]
            avg_offset = sum(offsets) / len(offsets)
            
            return DriftProfile(
                source_id=source_id,
                reference_id=reference_id,
                offset_at_reference=timedelta(seconds=avg_offset),
                reference_timestamp=first_ref,
                drift_rate=0.0,
                confidence=0.7,
                determination_method="average offset (insufficient time span)",
                anchor_count=len(pairs),
            )
        
        # Linear regression: offset = a + b * elapsed_time
        # where b is the drift rate
        
        # Convert to offset vs elapsed time
        data_points = []
        for source_ts, ref_ts in pairs:
            elapsed = (ref_ts - first_ref).total_seconds()
            offset = (ref_ts - source_ts).total_seconds()
            data_points.append((elapsed, offset))
        
        # Simple linear regression
        n = len(data_points)
        sum_x = sum(p[0] for p in data_points)
        sum_y = sum(p[1] for p in data_points)
        sum_xy = sum(p[0] * p[1] for p in data_points)
        sum_xx = sum(p[0] ** 2 for p in data_points)
        
        # Avoid division by zero
        denom = n * sum_xx - sum_x ** 2
        if abs(denom) < 1e-10:
            drift_rate = 0.0
            base_offset = sum_y / n
        else:
            drift_rate = (n * sum_xy - sum_x * sum_y) / denom
            base_offset = (sum_y - drift_rate * sum_x) / n
        
        # Calculate R-squared for confidence
        mean_y = sum_y / n
        ss_tot = sum((p[1] - mean_y) ** 2 for p in data_points)
        ss_res = sum((p[1] - (base_offset + drift_rate * p[0])) ** 2 for p in data_points)
        
        if ss_tot > 0:
            r_squared = 1 - (ss_res / ss_tot)
        else:
            r_squared = 1.0
        
        # Confidence based on R-squared and number of points
        confidence = max(0.0, min(1.0, r_squared * (1 - 1 / len(pairs))))
        
        return DriftProfile(
            source_id=source_id,
            reference_id=reference_id,
            offset_at_reference=timedelta(seconds=base_offset),
            reference_timestamp=first_ref,
            drift_rate=drift_rate,
            confidence=confidence,
            determination_method=f"linear regression (R²={r_squared:.4f})",
            anchor_count=len(pairs),
        )
