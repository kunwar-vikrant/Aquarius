"""
Tests for temporal alignment module.
"""

import pytest
from datetime import datetime, timedelta

from cwe.alignment.synchronizer import (
    TemporalSynchronizer,
    AnchorPoint,
    AnchorType,
    TimestampMapping,
    TimestampFormat,
)
from cwe.alignment.drift import DriftCorrector, DriftProfile


class TestAnchorPoint:
    """Tests for AnchorPoint class."""
    
    def test_anchor_creation(self):
        """Test anchor point creation."""
        anchor = AnchorPoint(
            anchor_type=AnchorType.EXPLICIT,
            description="Collision event",
        )
        
        assert anchor.id is not None
        assert anchor.confidence == 1.0
    
    def test_add_timestamps(self):
        """Test adding timestamps to anchor."""
        anchor = AnchorPoint(description="Test anchor")
        
        now = datetime.now()
        anchor.add_source_timestamp("video", now)
        anchor.add_source_timestamp("logs", now + timedelta(milliseconds=50))
        
        assert len(anchor.timestamps) == 2


class TestTimestampMapping:
    """Tests for TimestampMapping class."""
    
    def test_mapping_creation(self):
        """Test mapping creation."""
        mapping = TimestampMapping(
            source_id="video",
            target_id="logs",
            offset=timedelta(milliseconds=100),
        )
        
        assert mapping.scale == 1.0
    
    def test_map_timestamp(self):
        """Test timestamp mapping."""
        mapping = TimestampMapping(
            source_id="video",
            target_id="logs",
            offset=timedelta(seconds=1),
        )
        
        source_time = datetime(2024, 1, 15, 14, 30, 0)
        mapped_time = mapping.map_timestamp(source_time)
        
        expected = datetime(2024, 1, 15, 14, 30, 1)
        assert mapped_time == expected
    
    def test_inverse_mapping(self):
        """Test inverse mapping creation."""
        mapping = TimestampMapping(
            source_id="video",
            target_id="logs",
            offset=timedelta(seconds=1),
        )
        
        inverse = mapping.inverse()
        
        assert inverse.source_id == "logs"
        assert inverse.target_id == "video"
        assert inverse.offset == timedelta(seconds=-1)


class TestTemporalSynchronizer:
    """Tests for TemporalSynchronizer class."""
    
    def test_synchronizer_creation(self):
        """Test synchronizer creation."""
        sync = TemporalSynchronizer()
        assert sync.max_drift_seconds == 1.0
    
    def test_synchronize_with_anchors(self):
        """Test synchronization with explicit anchors."""
        sync = TemporalSynchronizer()
        
        # Create test data
        now = datetime.now()
        sources = {
            "video": [{"timestamp": now, "description": "collision"}],
            "logs": [{"timestamp": now + timedelta(milliseconds=50), "description": "collision"}],
        }
        
        # Create anchor
        anchor = AnchorPoint(
            anchor_type=AnchorType.EXPLICIT,
            description="Collision event",
        )
        anchor.add_source_timestamp("video", now)
        anchor.add_source_timestamp("logs", now + timedelta(milliseconds=50))
        
        result = sync.synchronize(sources, anchor_points=[anchor], reference_source="video")
        
        assert result.success
        assert result.reference_source == "video"
    
    def test_align_event(self):
        """Test single event alignment."""
        sync = TemporalSynchronizer()
        
        now = datetime.now()
        sources = {
            "video": [{"timestamp": now, "description": "event"}],
            "logs": [{"timestamp": now + timedelta(seconds=1), "description": "event"}],
        }
        
        anchor = AnchorPoint(description="Test")
        anchor.add_source_timestamp("video", now)
        anchor.add_source_timestamp("logs", now + timedelta(seconds=1))
        
        result = sync.synchronize(sources, anchor_points=[anchor], reference_source="video")
        
        # Align a logs timestamp to video time
        log_time = now + timedelta(seconds=2)
        aligned = sync.align_event(log_time, "logs", result)
        
        # Should be 1 second earlier in video time
        assert aligned is not None
        expected = now + timedelta(seconds=1)
        assert abs((aligned - expected).total_seconds()) < 0.01


class TestDriftProfile:
    """Tests for DriftProfile class."""
    
    def test_profile_creation(self):
        """Test drift profile creation."""
        profile = DriftProfile(
            source_id="sensor",
            reference_id="system",
            drift_rate=0.001,  # 1ms per second
        )
        
        assert profile.is_significant
    
    def test_correct_timestamp(self):
        """Test timestamp correction."""
        now = datetime.now()
        profile = DriftProfile(
            source_id="sensor",
            reference_id="system",
            offset_at_reference=timedelta(milliseconds=100),
            reference_timestamp=now,
            drift_rate=0.0,  # No drift
        )
        
        corrected = profile.correct_timestamp(now)
        expected = now + timedelta(milliseconds=100)
        
        assert corrected == expected


class TestDriftCorrector:
    """Tests for DriftCorrector class."""
    
    def test_corrector_creation(self):
        """Test corrector creation."""
        corrector = DriftCorrector()
        assert corrector.min_anchors_for_drift == 3
    
    def test_analyze_no_drift(self):
        """Test analysis with no significant drift."""
        corrector = DriftCorrector()
        
        # Create anchors with no drift
        now = datetime.now()
        anchors = []
        
        for i in range(5):
            anchor = AnchorPoint(description=f"Anchor {i}")
            ts = now + timedelta(seconds=i * 60)  # Every minute
            anchor.add_source_timestamp("reference", ts)
            anchor.add_source_timestamp("sensor", ts + timedelta(milliseconds=100))  # Constant offset
            anchors.append(anchor)
        
        analysis = corrector.analyze_drift(anchors, "reference")
        
        assert "sensor" in analysis.profiles
        # Should have minimal drift rate
        assert abs(analysis.profiles["sensor"].drift_rate) < 0.0001
    
    def test_analyze_with_drift(self):
        """Test analysis with significant drift."""
        corrector = DriftCorrector(min_time_span_seconds=60)
        
        # Create anchors with increasing offset (simulating drift)
        now = datetime.now()
        anchors = []
        
        for i in range(5):
            anchor = AnchorPoint(description=f"Anchor {i}")
            ts = now + timedelta(seconds=i * 60)  # Every minute
            # Sensor gains 100ms per minute
            drift_offset = timedelta(milliseconds=100 * (i + 1))
            anchor.add_source_timestamp("reference", ts)
            anchor.add_source_timestamp("sensor", ts + drift_offset)
            anchors.append(anchor)
        
        analysis = corrector.analyze_drift(anchors, "reference")
        
        assert "sensor" in analysis.profiles
        # Should detect drift
        profile = analysis.profiles["sensor"]
        assert abs(profile.drift_rate) > 0.0001
