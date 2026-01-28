"""
Intervention generator - automatically identifies meaningful counterfactuals.
"""

import json
import structlog
from datetime import datetime
from typing import Optional

from ..models.timeline import Timeline
from ..reasoning.providers.base import VLMProvider
from .models import (
    Intervention,
    InterventionType,
)
from .prompts import INTERVENTION_GENERATION_PROMPT

logger = structlog.get_logger()


class InterventionGenerator:
    """
    Generates meaningful counterfactual interventions from a timeline.
    
    Uses the VLM to identify:
    - Critical decision points
    - Missing safeguards
    - Timing-sensitive events
    - Alternative behaviors
    """
    
    def __init__(self, provider: VLMProvider):
        self.provider = provider
    
    async def generate_interventions(
        self,
        timeline: Timeline,
        num_interventions: int = 5,
        focus_areas: Optional[list[str]] = None,
    ) -> list[Intervention]:
        """
        Generate counterfactual interventions for a timeline.
        
        Args:
            timeline: The original incident timeline
            num_interventions: Number of interventions to generate
            focus_areas: Optional areas to focus on (e.g., "human factors", "technology")
            
        Returns:
            List of Intervention objects
        """
        logger.info(
            "Generating interventions",
            num_interventions=num_interventions,
            focus_areas=focus_areas,
        )
        
        # Prepare timeline summary
        incident_summary = self._summarize_incident(timeline)
        timeline_events = self._format_events(timeline)
        causal_chain = self._format_causal_chain(timeline)
        original_outcome = self._describe_outcome(timeline)
        
        # Build the prompt
        prompt = INTERVENTION_GENERATION_PROMPT.format(
            incident_summary=incident_summary,
            timeline_events=timeline_events,
            causal_chain=causal_chain,
            original_outcome=original_outcome,
            num_interventions=num_interventions,
        )
        
        if focus_areas:
            prompt += f"\n\nFocus particularly on these areas: {', '.join(focus_areas)}"
        
        # Define the function for structured output
        functions = [
            {
                "name": "propose_intervention",
                "description": "Propose a counterfactual intervention",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Short name for this intervention"
                        },
                        "intervention_type": {
                            "type": "string",
                            "enum": [t.value for t in InterventionType],
                            "description": "Type of intervention"
                        },
                        "description": {
                            "type": "string",
                            "description": "Detailed description of the intervention"
                        },
                        "target_entity_id": {
                            "type": "string",
                            "description": "ID of the entity being modified (if applicable)"
                        },
                        "target_event_id": {
                            "type": "string",
                            "description": "ID of the event being modified (if applicable)"
                        },
                        "parameter_name": {
                            "type": "string",
                            "description": "Name of parameter being changed (for parameter_change type)"
                        },
                        "original_value": {
                            "type": "string",
                            "description": "Original value of the parameter"
                        },
                        "counterfactual_value": {
                            "type": "string",
                            "description": "New value in the counterfactual scenario"
                        },
                        "time_delta_seconds": {
                            "type": "number",
                            "description": "Time shift in seconds (negative = earlier)"
                        },
                        "hypothesis": {
                            "type": "string",
                            "description": "Why this intervention would change the outcome"
                        },
                        "expected_impact": {
                            "type": "string",
                            "enum": ["low", "medium", "high", "critical"],
                            "description": "Expected impact on outcome"
                        }
                    },
                    "required": ["name", "intervention_type", "description", "hypothesis"]
                }
            },
            {
                "name": "complete_generation",
                "description": "Signal that intervention generation is complete",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "summary": {
                            "type": "string",
                            "description": "Summary of generated interventions"
                        }
                    }
                }
            }
        ]
        
        # Call the VLM
        interventions = []
        messages = [{"role": "user", "content": prompt}]
        
        while len(interventions) < num_interventions:
            response = await self.provider.generate(
                messages=messages,
                functions=functions,
                temperature=0.7,  # Some creativity for diverse interventions
            )
            
            if response.function_calls:
                for call in response.function_calls:
                    if call.name == "propose_intervention":
                        args = call.arguments
                        intervention = Intervention(
                            intervention_type=InterventionType(args.get("intervention_type", "parameter_change")),
                            description=args.get("description", ""),
                            target_entity_id=args.get("target_entity_id"),
                            target_event_id=args.get("target_event_id"),
                            parameter_name=args.get("parameter_name"),
                            original_value=args.get("original_value"),
                            counterfactual_value=args.get("counterfactual_value"),
                            time_delta_seconds=args.get("time_delta_seconds"),
                            hypothesis=args.get("hypothesis", ""),
                        )
                        interventions.append(intervention)
                        logger.debug(
                            "Generated intervention",
                            name=args.get("name"),
                            type=intervention.intervention_type.value,
                        )
                    elif call.name == "complete_generation":
                        break
                
                # Continue the conversation
                messages.append({"role": "assistant", "content": None, "function_calls": response.function_calls})
                messages.append({
                    "role": "user",
                    "content": f"Generated {len(interventions)}/{num_interventions} interventions. Continue if more are needed."
                })
            else:
                break
        
        logger.info("Intervention generation complete", count=len(interventions))
        return interventions[:num_interventions]
    
    def generate_standard_interventions(
        self,
        timeline: Timeline,
        domain: str = "general",
    ) -> list[Intervention]:
        """
        Generate standard domain-specific interventions without VLM call.
        
        Useful for quick analysis with common what-if scenarios.
        """
        interventions = []
        
        if domain == "traffic":
            interventions.extend(self._traffic_interventions(timeline))
        elif domain == "devops":
            interventions.extend(self._devops_interventions(timeline))
        elif domain == "business":
            interventions.extend(self._business_interventions(timeline))
        else:
            interventions.extend(self._general_interventions(timeline))
        
        return interventions
    
    def _traffic_interventions(self, timeline: Timeline) -> list[Intervention]:
        """Standard traffic incident interventions."""
        return [
            Intervention(
                intervention_type=InterventionType.TIMING_SHIFT,
                description="Driver brakes 2 seconds earlier",
                parameter_name="brake_activation_time",
                time_delta_seconds=-2.0,
                hypothesis="Earlier braking would reduce speed at impact or prevent collision entirely",
            ),
            Intervention(
                intervention_type=InterventionType.SYSTEM_CAPABILITY,
                description="Automatic Emergency Braking (AEB) activates at earlier threshold",
                parameter_name="aeb_activation_threshold",
                original_value="1.2s TTC",
                counterfactual_value="2.5s TTC",
                hypothesis="More aggressive AEB would intervene before human reaction",
            ),
            Intervention(
                intervention_type=InterventionType.SYSTEM_CAPABILITY,
                description="Lane Keep Assist actively corrects drift",
                parameter_name="lane_keep_assist",
                original_value="warning_only",
                counterfactual_value="active_steering",
                hypothesis="Active lane keeping would prevent unintentional lane departure",
            ),
            Intervention(
                intervention_type=InterventionType.PARAMETER_CHANGE,
                description="Following distance increased by 50%",
                parameter_name="following_distance",
                original_value="current",
                counterfactual_value="1.5x current",
                hypothesis="Greater following distance provides more reaction time",
            ),
            Intervention(
                intervention_type=InterventionType.BEHAVIOR_SUBSTITUTION,
                description="Driver attention score maintained above 70/100",
                parameter_name="driver_attention_score",
                original_value="28/100",
                counterfactual_value="70/100",
                hypothesis="Alert driver would have noticed hazard and responded appropriately",
            ),
        ]
    
    def _devops_interventions(self, timeline: Timeline) -> list[Intervention]:
        """Standard DevOps incident interventions."""
        return [
            Intervention(
                intervention_type=InterventionType.SYSTEM_CAPABILITY,
                description="Circuit breaker trips earlier (50% error rate threshold)",
                parameter_name="circuit_breaker_threshold",
                original_value="80%",
                counterfactual_value="50%",
                hypothesis="Earlier circuit breaker would isolate failure faster",
            ),
            Intervention(
                intervention_type=InterventionType.PARAMETER_CHANGE,
                description="Memory limits set 50% lower to trigger OOM protection earlier",
                parameter_name="memory_limit",
                original_value="current",
                counterfactual_value="0.5x current",
                hypothesis="Lower limits would trigger graceful degradation before cascade",
            ),
            Intervention(
                intervention_type=InterventionType.SYSTEM_CAPABILITY,
                description="Automated rollback on error rate spike",
                parameter_name="auto_rollback",
                original_value="disabled",
                counterfactual_value="enabled at 5% error rate",
                hypothesis="Automatic rollback would revert problematic deployment",
            ),
            Intervention(
                intervention_type=InterventionType.TIMING_SHIFT,
                description="Deployment moved to low-traffic window",
                parameter_name="deployment_time",
                time_delta_seconds=-43200,  # 12 hours earlier
                hypothesis="Lower traffic would reduce blast radius of any issues",
            ),
            Intervention(
                intervention_type=InterventionType.EVENT_ADDITION,
                description="Canary deployment catches issue with 1% traffic",
                new_event_description="Canary deployment detects memory leak with 1% traffic sample",
                hypothesis="Gradual rollout would catch issue before full deployment",
            ),
        ]
    
    def _business_interventions(self, timeline: Timeline) -> list[Intervention]:
        """Standard business decision interventions."""
        return [
            Intervention(
                intervention_type=InterventionType.TIMING_SHIFT,
                description="Decision delayed by 2 weeks for more data",
                time_delta_seconds=1209600,  # 2 weeks
                hypothesis="Additional time would reveal risks not apparent initially",
            ),
            Intervention(
                intervention_type=InterventionType.EVENT_ADDITION,
                description="External review/audit conducted before decision",
                new_event_description="Independent review identifies key risks",
                hypothesis="Outside perspective would catch blind spots",
            ),
            Intervention(
                intervention_type=InterventionType.BEHAVIOR_SUBSTITUTION,
                description="Dissenting opinion given more weight in decision",
                parameter_name="decision_weight_dissent",
                original_value="low",
                counterfactual_value="high",
                hypothesis="Addressing concerns would lead to different choice",
            ),
        ]
    
    def _general_interventions(self, timeline: Timeline) -> list[Intervention]:
        """General-purpose interventions."""
        return [
            Intervention(
                intervention_type=InterventionType.TIMING_SHIFT,
                description="Key action taken 1 minute earlier",
                time_delta_seconds=-60,
                hypothesis="Earlier action provides more margin for error",
            ),
            Intervention(
                intervention_type=InterventionType.EVENT_REMOVAL,
                description="Remove the triggering event",
                hypothesis="Understand if outcome was inevitable or dependent on trigger",
            ),
        ]
    
    def _summarize_incident(self, timeline: Timeline) -> str:
        """Create a summary of the incident."""
        entities = [f"- {e.name} ({e.entity_type})" for e in timeline.entities[:10]]
        return f"""
Incident with {len(timeline.entities)} entities and {len(timeline.events)} events.
Confidence: {timeline.confidence:.1%}

Key Entities:
{chr(10).join(entities)}

Timeline spans: {timeline.start_time} to {timeline.end_time}
"""
    
    def _format_events(self, timeline: Timeline) -> str:
        """Format events for the prompt."""
        lines = []
        for e in sorted(timeline.events, key=lambda x: x.timestamp)[:30]:
            lines.append(f"[{e.timestamp.strftime('%H:%M:%S.%f')[:-3]}] ({e.event_type}) {e.description[:100]}")
        return "\n".join(lines)
    
    def _format_causal_chain(self, timeline: Timeline) -> str:
        """Format causal links."""
        lines = []
        for link in timeline.causal_links[:20]:
            source = next((e for e in timeline.events if e.id == link.source_event_id), None)
            target = next((e for e in timeline.events if e.id == link.target_event_id), None)
            if source and target:
                lines.append(f"- {source.description[:50]}... → [{link.relation}] → {target.description[:50]}...")
        return "\n".join(lines) if lines else "No causal links identified"
    
    def _describe_outcome(self, timeline: Timeline) -> str:
        """Describe the final outcome."""
        # Find collision/failure/critical events
        critical_events = [
            e for e in timeline.events 
            if e.event_type in ("collision", "failure", "critical", "alert")
        ]
        
        if critical_events:
            return "\n".join([f"- {e.description}" for e in critical_events[:5]])
        
        # Fall back to last few events
        last_events = sorted(timeline.events, key=lambda x: x.timestamp)[-3:]
        return "\n".join([f"- {e.description}" for e in last_events])
