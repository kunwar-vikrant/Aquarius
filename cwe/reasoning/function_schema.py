"""Function schemas for VLM function calling."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class VLMFunction:
    """Definition of a function the VLM can call."""
    
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., Any] | None = None
    
    def to_schema(self) -> dict[str, Any]:
        """Convert to JSON Schema format for VLM."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


class FunctionRegistry:
    """Registry of functions available to the VLM."""
    
    def __init__(self):
        self._functions: dict[str, VLMFunction] = {}
    
    def register(self, func: VLMFunction) -> None:
        """Register a function."""
        self._functions[func.name] = func
    
    def get(self, name: str) -> VLMFunction | None:
        """Get a function by name."""
        return self._functions.get(name)
    
    def get_all(self) -> list[VLMFunction]:
        """Get all registered functions."""
        return list(self._functions.values())
    
    def get_schemas(self) -> list[dict[str, Any]]:
        """Get all function schemas for VLM."""
        return [f.to_schema() for f in self._functions.values()]
    
    async def execute(self, name: str, arguments: dict[str, Any]) -> Any:
        """Execute a function by name."""
        func = self._functions.get(name)
        if not func:
            raise ValueError(f"Unknown function: {name}")
        if not func.handler:
            raise ValueError(f"Function {name} has no handler")
        
        result = func.handler(**arguments)
        if hasattr(result, "__await__"):
            result = await result
        return result


def get_timeline_functions() -> list[VLMFunction]:
    """
    Get the standard set of functions for timeline construction.
    
    These functions are called by the VLM to emit structured data
    during timeline analysis.
    """
    
    return [
        VLMFunction(
            name="emit_event",
            description=(
                "Register a detected event in the timeline. Call this whenever you "
                "identify a significant event, state change, action, or observation. "
                "Events should be as granular as possible."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "timestamp": {
                        "type": "string",
                        "description": "ISO 8601 timestamp of the event (e.g., '2024-01-15T14:30:00Z')",
                    },
                    "event_type": {
                        "type": "string",
                        "enum": [
                            "state_change", "position_change", "velocity_change",
                            "action_initiated", "action_completed", "action_failed",
                            "observation", "detection", "interaction", "collision",
                            "communication", "system_event", "error", "alert",
                            "timeline_start", "timeline_end", "branch_point",
                        ],
                        "description": "Category of the event",
                    },
                    "description": {
                        "type": "string",
                        "description": "Detailed description of what happened",
                    },
                    "entities": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "name": {"type": "string"},
                                "type": {"type": "string"},
                            },
                        },
                        "description": "Entities involved in this event",
                    },
                    "confidence": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "description": "Confidence in this event (0.0 to 1.0)",
                    },
                    "evidence_refs": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "artifact_type": {"type": "string"},
                                "location": {"type": "string"},
                                "excerpt": {"type": "string"},
                            },
                        },
                        "description": "References to evidence supporting this event",
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Your reasoning for identifying this event",
                    },
                },
                "required": ["timestamp", "event_type", "description", "confidence"],
            },
        ),
        
        VLMFunction(
            name="add_causal_link",
            description=(
                "Establish a causal relationship between two events. Call this when "
                "you identify that one event causes, enables, prevents, or delays another."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "source_event_id": {
                        "type": "string",
                        "description": "ID of the causing event",
                    },
                    "target_event_id": {
                        "type": "string",
                        "description": "ID of the affected event",
                    },
                    "relation": {
                        "type": "string",
                        "enum": ["causes", "enables", "prevents", "delays", "accelerates", "correlates"],
                        "description": "Type of causal relationship",
                    },
                    "mechanism": {
                        "type": "string",
                        "description": "Natural language explanation of how/why A affects B",
                    },
                    "confidence": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "description": "Confidence in this causal link (0.0 to 1.0)",
                    },
                    "delay_seconds": {
                        "type": "number",
                        "description": "Time delay between cause and effect (optional)",
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Your reasoning for establishing this causal link",
                    },
                },
                "required": ["source_event_id", "target_event_id", "relation", "mechanism", "confidence"],
            },
        ),
        
        VLMFunction(
            name="register_entity",
            description=(
                "Register a new entity (vehicle, person, system, etc.) that will be "
                "tracked through the timeline."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "entity_id": {
                        "type": "string",
                        "description": "Unique identifier for this entity",
                    },
                    "name": {
                        "type": "string",
                        "description": "Human-readable name for the entity",
                    },
                    "entity_type": {
                        "type": "string",
                        "description": "Type of entity (vehicle, person, system, component, etc.)",
                    },
                    "properties": {
                        "type": "object",
                        "description": "Initial properties of the entity",
                    },
                },
                "required": ["entity_id", "name", "entity_type"],
            },
        ),
        
        VLMFunction(
            name="update_entity_state",
            description=(
                "Track a change in an entity's state (position, velocity, properties, etc.)."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "entity_id": {
                        "type": "string",
                        "description": "ID of the entity",
                    },
                    "timestamp": {
                        "type": "string",
                        "description": "ISO 8601 timestamp of the state change",
                    },
                    "property_name": {
                        "type": "string",
                        "description": "Name of the property that changed",
                    },
                    "old_value": {
                        "description": "Previous value (if known)",
                    },
                    "new_value": {
                        "description": "New value",
                    },
                    "position": {
                        "type": "object",
                        "properties": {
                            "x": {"type": "number"},
                            "y": {"type": "number"},
                            "z": {"type": "number"},
                        },
                        "description": "Position in 3D space (optional)",
                    },
                    "velocity": {
                        "type": "object",
                        "properties": {
                            "vx": {"type": "number"},
                            "vy": {"type": "number"},
                            "vz": {"type": "number"},
                        },
                        "description": "Velocity vector (optional)",
                    },
                },
                "required": ["entity_id", "timestamp"],
            },
        ),
        
        VLMFunction(
            name="flag_uncertainty",
            description=(
                "Flag an area of uncertainty that may require human review or "
                "additional data. Use this when you encounter missing information, "
                "ambiguity, or conflicting evidence."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "context": {
                        "type": "string",
                        "description": "Description of what you were analyzing",
                    },
                    "uncertainty_type": {
                        "type": "string",
                        "enum": ["missing_data", "ambiguous", "conflicting", "low_confidence", "assumption"],
                        "description": "Type of uncertainty",
                    },
                    "description": {
                        "type": "string",
                        "description": "Detailed description of the uncertainty",
                    },
                    "possible_interpretations": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of possible interpretations or scenarios",
                    },
                    "data_needed": {
                        "type": "string",
                        "description": "What additional data would resolve this uncertainty",
                    },
                    "impact": {
                        "type": "string",
                        "enum": ["low", "medium", "high", "critical"],
                        "description": "How much this uncertainty affects the overall analysis",
                    },
                },
                "required": ["context", "uncertainty_type", "description"],
            },
        ),
        
        VLMFunction(
            name="request_frame_analysis",
            description=(
                "Request detailed analysis of a specific video segment. Use this when "
                "you need to examine a particular time range more closely."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "video_id": {
                        "type": "string",
                        "description": "ID of the video artifact",
                    },
                    "start_timestamp": {
                        "type": "string",
                        "description": "Start of the segment to analyze (ISO 8601)",
                    },
                    "end_timestamp": {
                        "type": "string",
                        "description": "End of the segment to analyze (ISO 8601)",
                    },
                    "query": {
                        "type": "string",
                        "description": "What you're looking for in this segment",
                    },
                    "frame_interval": {
                        "type": "number",
                        "description": "Desired interval between frames in seconds",
                    },
                },
                "required": ["video_id", "start_timestamp", "end_timestamp", "query"],
            },
        ),
        
        VLMFunction(
            name="set_timeline_bounds",
            description=(
                "Set the start and end time of the timeline being analyzed."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "start_time": {
                        "type": "string",
                        "description": "Start of the timeline (ISO 8601)",
                    },
                    "end_time": {
                        "type": "string",
                        "description": "End of the timeline (ISO 8601)",
                    },
                    "description": {
                        "type": "string",
                        "description": "Brief description of what this timeline covers",
                    },
                },
                "required": ["start_time", "end_time"],
            },
        ),
    ]


def get_counterfactual_functions() -> list[VLMFunction]:
    """
    Get the standard set of functions for counterfactual reasoning.
    
    These functions are called by the VLM during counterfactual simulation.
    """
    
    return [
        VLMFunction(
            name="evaluate_intervention_validity",
            description=(
                "Evaluate whether a proposed intervention is logically and physically valid."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "intervention_description": {
                        "type": "string",
                        "description": "The proposed intervention",
                    },
                    "is_valid": {
                        "type": "boolean",
                        "description": "Whether the intervention is valid",
                    },
                    "validity_issues": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of issues if invalid",
                    },
                    "physical_constraints_violated": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Physical laws or constraints violated",
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Explanation of validity assessment",
                    },
                },
                "required": ["intervention_description", "is_valid"],
            },
        ),
        
        VLMFunction(
            name="emit_alternate_event",
            description=(
                "Emit an event in the alternate (counterfactual) timeline. "
                "Use this for events that differ from the canonical timeline."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "timestamp": {
                        "type": "string",
                        "description": "ISO 8601 timestamp",
                    },
                    "event_type": {
                        "type": "string",
                        "description": "Category of the event",
                    },
                    "description": {
                        "type": "string",
                        "description": "What happens in this alternate reality",
                    },
                    "diverges_from": {
                        "type": "string",
                        "description": "ID of canonical event this diverges from (if applicable)",
                    },
                    "confidence": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "description": "Confidence in this alternate event",
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Why this event occurs given the intervention",
                    },
                },
                "required": ["timestamp", "event_type", "description", "confidence", "reasoning"],
            },
        ),
        
        VLMFunction(
            name="compare_outcomes",
            description=(
                "Compare the outcome of the counterfactual timeline with the canonical timeline."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "canonical_outcome": {
                        "type": "string",
                        "description": "Description of what actually happened",
                    },
                    "canonical_severity": {
                        "type": "string",
                        "enum": ["none", "minor", "moderate", "severe", "catastrophic"],
                    },
                    "counterfactual_outcome": {
                        "type": "string",
                        "description": "Description of what would have happened",
                    },
                    "counterfactual_severity": {
                        "type": "string",
                        "enum": ["none", "minor", "moderate", "severe", "catastrophic"],
                    },
                    "outcome_improved": {
                        "type": "boolean",
                        "description": "Whether the counterfactual outcome is better",
                    },
                    "key_differences": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Key differences between timelines",
                    },
                    "confidence": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                    },
                    "explanation": {
                        "type": "string",
                        "description": "Detailed explanation of the comparison",
                    },
                },
                "required": [
                    "canonical_outcome", "canonical_severity",
                    "counterfactual_outcome", "counterfactual_severity",
                    "outcome_improved", "confidence", "explanation"
                ],
            },
        ),
        
        VLMFunction(
            name="suggest_intervention",
            description=(
                "Suggest an intervention (counterfactual) to explore. "
                "Use this during autonomous exploration to propose hypotheses."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": "Natural language description of the intervention",
                    },
                    "intervention_type": {
                        "type": "string",
                        "enum": [
                            "remove_event", "add_event", "modify_event",
                            "delay_event", "advance_event",
                            "modify_entity", "add_entity", "remove_entity",
                        ],
                    },
                    "target_event_id": {
                        "type": "string",
                        "description": "ID of the event to modify (if applicable)",
                    },
                    "expected_impact": {
                        "type": "string",
                        "enum": ["low", "medium", "high"],
                        "description": "Expected impact on outcome",
                    },
                    "priority": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "description": "How important it is to explore this intervention",
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Why this intervention is worth exploring",
                    },
                },
                "required": ["description", "intervention_type", "expected_impact", "priority", "reasoning"],
            },
        ),
    ]
