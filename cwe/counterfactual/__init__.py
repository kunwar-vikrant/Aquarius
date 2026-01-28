"""
Counterfactual simulation engine for the CWE.

This module enables "what-if" analysis by simulating alternative scenarios
and comparing outcomes to the original timeline.
"""

from .models import (
    Intervention,
    InterventionType,
    CounterfactualOutcome,
    CounterfactualScenario,
    CounterfactualAnalysis,
    DivergencePoint,
    OutcomeSeverity,
)
from .simulator import CounterfactualSimulator
from .generator import InterventionGenerator
from .report import format_counterfactual_report, save_counterfactual_report

__all__ = [
    "Intervention",
    "InterventionType", 
    "CounterfactualOutcome",
    "CounterfactualScenario",
    "CounterfactualAnalysis",
    "DivergencePoint",
    "OutcomeSeverity",
    "CounterfactualSimulator",
    "InterventionGenerator",
    "format_counterfactual_report",
    "save_counterfactual_report",
]
