"""Hypothesis generation for counterfactual exploration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import UUID, uuid4

import structlog

from cwe.models.timeline import Timeline, Event
from cwe.models.counterfactual import Intervention, InterventionType
from cwe.reasoning.providers.base import VLMProvider, Message

logger = structlog.get_logger()


@dataclass
class Hypothesis:
    """A hypothesis about a potential intervention."""
    
    id: str = field(default_factory=lambda: str(uuid4()))
    description: str = ""
    
    intervention_type: InterventionType = InterventionType.NATURAL_LANGUAGE
    target_event_id: UUID | None = None
    target_entity_id: UUID | None = None
    parameters: dict[str, Any] = field(default_factory=dict)
    
    # Priority (0.0 - 1.0) for exploration ordering
    priority: float = 0.5
    
    # Expected impact
    expected_impact: str = "medium"  # low, medium, high
    
    # Source of this hypothesis
    source: str = "llm"  # llm, user, exhaustive, follow_up
    
    # Reasoning for this hypothesis
    reasoning: str = ""


HYPOTHESIS_PROMPT = """You are analyzing an incident timeline to identify promising counterfactual interventions.

# Timeline Events
{events}

# Causal Relationships
{causal_links}

# Task
Generate {max_hypotheses} hypotheses about interventions that could have changed the outcome.

For each hypothesis, consider:
1. **Timing interventions**: What if something happened earlier or later?
2. **Removal interventions**: What if something didn't happen?
3. **Modification interventions**: What if something happened differently?
4. **Entity interventions**: What if an entity had different properties?

Focus on:
- High-leverage intervention points (events with many downstream effects)
- Realistic and actionable changes
- Interventions with clear causal pathways to different outcomes

Call `suggest_intervention` for each hypothesis, providing:
- Clear description of the intervention
- Intervention type
- Expected impact (low/medium/high)
- Priority (0-1) for how important it is to explore
- Reasoning for why this intervention is worth exploring
"""


class HypothesisGenerator:
    """
    Generates hypotheses for counterfactual exploration.
    
    Uses VLM to identify promising intervention points in a timeline.
    """
    
    def __init__(self, provider: VLMProvider):
        self.provider = provider
    
    async def generate(
        self,
        timeline: Timeline,
        max_hypotheses: int = 10,
    ) -> list[Hypothesis]:
        """
        Generate hypotheses for the given timeline.
        
        Args:
            timeline: The timeline to analyze
            max_hypotheses: Maximum number of hypotheses to generate
            
        Returns:
            List of Hypothesis objects ordered by priority
        """
        logger.info("Generating hypotheses", max_hypotheses=max_hypotheses)
        
        # Format timeline for prompt
        events_text = self._format_events(timeline.events)
        causal_text = self._format_causal_links(timeline)
        
        prompt = HYPOTHESIS_PROMPT.format(
            events=events_text,
            causal_links=causal_text,
            max_hypotheses=max_hypotheses,
        )
        
        messages = [
            Message.system(
                "You are an expert incident analyst generating counterfactual hypotheses. "
                "Think carefully about what interventions could have changed the outcome."
            ),
            Message.user(prompt),
        ]
        
        # Get function schema
        from cwe.reasoning.function_schema import get_counterfactual_functions
        functions = [f.to_schema() for f in get_counterfactual_functions()]
        
        # Generate hypotheses
        response = await self.provider.generate(
            messages=messages,
            functions=functions,
            enable_thinking=True,
        )
        
        # Parse function calls into hypotheses
        hypotheses = []
        for fc in response.function_calls:
            if fc.name == "suggest_intervention":
                hypothesis = Hypothesis(
                    description=fc.arguments.get("description", ""),
                    intervention_type=InterventionType(
                        fc.arguments.get("intervention_type", "natural_language")
                    ),
                    target_event_id=UUID(fc.arguments["target_event_id"]) 
                        if fc.arguments.get("target_event_id") else None,
                    priority=fc.arguments.get("priority", 0.5),
                    expected_impact=fc.arguments.get("expected_impact", "medium"),
                    reasoning=fc.arguments.get("reasoning", ""),
                    source="llm",
                )
                hypotheses.append(hypothesis)
        
        # Sort by priority
        hypotheses.sort(key=lambda h: h.priority, reverse=True)
        
        logger.info("Generated hypotheses", count=len(hypotheses))
        
        return hypotheses[:max_hypotheses]
    
    async def generate_variations(
        self,
        intervention: Intervention,
        timeline: Timeline,
        num_variations: int = 3,
    ) -> list[Hypothesis]:
        """
        Generate variations of a successful intervention.
        
        Used to explore similar interventions when one shows promise.
        """
        logger.debug("Generating variations", intervention=intervention.description[:50])
        
        prompt = f"""An intervention was found to improve the outcome of an incident:

Intervention: {intervention.description}
Type: {intervention.intervention_type.value}

Generate {num_variations} variations of this intervention that might also be effective:
- Try different timing (earlier/later)
- Try different magnitude
- Try combining with related changes

Call `suggest_intervention` for each variation."""
        
        messages = [
            Message.system("Generate variations of a successful counterfactual intervention."),
            Message.user(prompt),
        ]
        
        from cwe.reasoning.function_schema import get_counterfactual_functions
        functions = [f.to_schema() for f in get_counterfactual_functions()]
        
        response = await self.provider.generate(
            messages=messages,
            functions=functions,
        )
        
        variations = []
        for fc in response.function_calls:
            if fc.name == "suggest_intervention":
                hypothesis = Hypothesis(
                    description=fc.arguments.get("description", ""),
                    intervention_type=InterventionType(
                        fc.arguments.get("intervention_type", "natural_language")
                    ),
                    priority=fc.arguments.get("priority", 0.5) * 0.8,  # Slightly lower priority
                    expected_impact=fc.arguments.get("expected_impact", "medium"),
                    reasoning=fc.arguments.get("reasoning", ""),
                    source="follow_up",
                )
                variations.append(hypothesis)
        
        return variations[:num_variations]
    
    def _format_events(self, events: list[Event]) -> str:
        """Format events for the prompt."""
        sorted_events = sorted(events, key=lambda e: e.timestamp)
        
        lines = []
        for event in sorted_events:
            lines.append(
                f"- [{event.id}] {event.timestamp.isoformat()}: "
                f"{event.event_type.value} - {event.description} "
                f"(confidence: {event.confidence:.2f})"
            )
        
        return "\n".join(lines)
    
    def _format_causal_links(self, timeline: Timeline) -> str:
        """Format causal links for the prompt."""
        if not timeline.causal_links:
            return "No causal links established yet."
        
        lines = []
        for link in timeline.causal_links:
            lines.append(
                f"- {link.source_event_id} {link.relation.value} {link.target_event_id}: "
                f"{link.mechanism} (confidence: {link.confidence:.2f})"
            )
        
        return "\n".join(lines)
