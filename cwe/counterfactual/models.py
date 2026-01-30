"""
Data models for counterfactual analysis.

Counterfactuals answer "what if?" questions by simulating alternative scenarios.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import uuid4


class InterventionType(Enum):
    """Types of counterfactual interventions."""
    
    # Modify a parameter value
    PARAMETER_CHANGE = "parameter_change"
    
    # Remove an event from the timeline
    EVENT_REMOVAL = "event_removal"
    
    # Add a new event that didn't happen
    EVENT_ADDITION = "event_addition"
    
    # Change when an event occurred
    TIMING_SHIFT = "timing_shift"
    
    # Replace one entity's behavior with another's
    BEHAVIOR_SUBSTITUTION = "behavior_substitution"
    
    # Change environmental conditions
    ENVIRONMENT_CHANGE = "environment_change"
    
    # Modify system/technology capabilities
    SYSTEM_CAPABILITY = "system_capability"


class OutcomeSeverity(Enum):
    """Severity levels for outcomes."""
    NONE = "none"
    MINOR = "minor"
    MODERATE = "moderate"
    SEVERE = "severe"
    CATASTROPHIC = "catastrophic"
    FATAL = "fatal"


@dataclass
class Intervention:
    """
    A counterfactual intervention - a change we hypothetically make to the timeline.
    
    Examples:
    - "What if the driver had braked 2 seconds earlier?"
    - "What if the truck wasn't there?"
    - "What if lane-keep assist was enabled?"
    """
    
    id: str = field(default_factory=lambda: str(uuid4()))
    
    # Type of intervention
    intervention_type: InterventionType = InterventionType.PARAMETER_CHANGE
    
    # Human-readable description
    description: str = ""
    
    # Target entity ID (if applicable)
    target_entity_id: Optional[str] = None
    
    # Target event ID (if applicable)  
    target_event_id: Optional[str] = None
    
    # The specific change being made
    parameter_name: Optional[str] = None
    original_value: Optional[str] = None
    counterfactual_value: Optional[str] = None
    
    # For timing shifts
    time_delta_seconds: Optional[float] = None
    
    # For event additions
    new_event_description: Optional[str] = None
    new_event_timestamp: Optional[datetime] = None
    
    # Reasoning for why this intervention is interesting
    hypothesis: str = ""
    
    # Feasibility: How realistic is this intervention? (0.0 = impossible, 1.0 = trivial)
    # Low: requires technology that doesn't exist, or impossible human behavior
    # Medium: requires policy changes, training, or system upgrades
    # High: could be implemented with existing technology/processes
    feasibility: float = 0.5
    
    # Expected effect if intervention succeeds
    expected_effect: str = ""
    
    # Rationale for choosing this feasibility score
    feasibility_rationale: str = ""
    
    def to_prompt(self) -> str:
        """Convert intervention to a prompt-friendly description."""
        if self.intervention_type == InterventionType.PARAMETER_CHANGE:
            return f"Change {self.parameter_name} from {self.original_value} to {self.counterfactual_value}"
        elif self.intervention_type == InterventionType.EVENT_REMOVAL:
            return f"Remove the event: {self.description}"
        elif self.intervention_type == InterventionType.EVENT_ADDITION:
            return f"Add new event at {self.new_event_timestamp}: {self.new_event_description}"
        elif self.intervention_type == InterventionType.TIMING_SHIFT:
            direction = "earlier" if self.time_delta_seconds < 0 else "later"
            return f"Shift event timing {abs(self.time_delta_seconds)}s {direction}: {self.description}"
        elif self.intervention_type == InterventionType.SYSTEM_CAPABILITY:
            return f"Enable/modify system capability: {self.description}"
        else:
            return self.description


@dataclass
class CounterfactualOutcome:
    """The outcome of a counterfactual simulation."""
    
    # What happened in the alternative timeline
    description: str = ""
    
    # Did the primary negative outcome (crash, outage, etc.) still occur?
    primary_outcome_occurred: bool = True
    
    # Severity comparison
    original_severity: OutcomeSeverity = OutcomeSeverity.SEVERE
    counterfactual_severity: OutcomeSeverity = OutcomeSeverity.SEVERE
    
    # Quantitative impact estimates
    injury_reduction_percent: Optional[float] = None
    damage_reduction_percent: Optional[float] = None
    time_saved_seconds: Optional[float] = None
    
    # Key differences from original timeline
    prevented_events: list[str] = field(default_factory=list)
    new_events: list[str] = field(default_factory=list)
    modified_events: list[str] = field(default_factory=list)
    
    # Confidence in this simulation
    confidence: float = 0.0
    
    # Chain of reasoning
    reasoning: str = ""


@dataclass
class DivergencePoint:
    """A point where the counterfactual timeline diverges from the original."""
    
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Original event (if any)
    original_event_id: Optional[str] = None
    original_event_description: str = ""
    
    # Counterfactual event (if any)
    counterfactual_event_description: str = ""
    
    # Why the divergence occurred
    cause: str = ""
    
    # Downstream effects
    cascade_effects: list[str] = field(default_factory=list)


@dataclass 
class CounterfactualScenario:
    """
    A complete counterfactual scenario with intervention and simulated outcome.
    """
    
    id: str = field(default_factory=lambda: str(uuid4()))
    
    # Reference to original incident
    incident_id: str = ""
    
    # The intervention(s) being tested
    interventions: list[Intervention] = field(default_factory=list)
    
    # Human-readable scenario name
    name: str = ""
    
    # The question being answered
    question: str = ""
    
    # Simulated outcome
    outcome: Optional[CounterfactualOutcome] = None
    
    # Points where timelines diverge
    divergence_points: list[DivergencePoint] = field(default_factory=list)
    
    # Alternative timeline events (subset that changed)
    alternative_events: list[dict] = field(default_factory=list)
    
    # Analysis summary
    summary: str = ""
    
    # Recommendations based on this counterfactual
    recommendations: list[str] = field(default_factory=list)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    simulation_duration_seconds: float = 0.0


@dataclass
class CounterfactualAnalysis:
    """
    Complete counterfactual analysis with multiple scenarios.
    """
    
    id: str = field(default_factory=lambda: str(uuid4()))
    incident_id: str = ""
    
    # Original timeline summary
    original_timeline_summary: str = ""
    original_outcome_description: str = ""
    original_severity: OutcomeSeverity = OutcomeSeverity.SEVERE
    
    # All scenarios analyzed
    scenarios: list[CounterfactualScenario] = field(default_factory=list)
    
    # Key findings across all scenarios
    key_findings: list[str] = field(default_factory=list)
    
    # Ranked interventions by effectiveness
    intervention_ranking: list[dict] = field(default_factory=list)
    
    # Overall recommendations
    recommendations: list[str] = field(default_factory=list)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    total_simulation_time_seconds: float = 0.0
