"""
Temporal alignment module for multimodal stream synchronization.

Aligns timestamps across different data sources:
- Video frames
- Log entries
- Sensor readings
- External reports
"""

from cwe.alignment.synchronizer import (
    TemporalSynchronizer,
    AlignmentResult,
    TimestampMapping,
    AnchorPoint,
)
from cwe.alignment.drift import DriftCorrector, DriftProfile

__all__ = [
    "TemporalSynchronizer",
    "AlignmentResult",
    "TimestampMapping",
    "AnchorPoint",
    "DriftCorrector",
    "DriftProfile",
]
