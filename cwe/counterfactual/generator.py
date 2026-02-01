"""
Intervention generator - automatically identifies meaningful counterfactuals.
"""

import json
import structlog
from datetime import datetime
from typing import Optional

from ..models.timeline import Timeline
from ..reasoning.providers.base import VLMProvider, Message
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
                "description": "Propose a specific, context-aware counterfactual intervention based on this incident",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Short, descriptive name (e.g., 'Earlier AEB Activation', 'Red Light Compliance')"
                        },
                        "intervention_type": {
                            "type": "string",
                            "enum": [t.value for t in InterventionType],
                            "description": "Type of intervention: timing_shift, parameter_change, behavior_substitution, system_capability, event_removal, event_addition"
                        },
                        "description": {
                            "type": "string",
                            "description": "SPECIFIC description - exactly what changes and by how much? Include numbers where possible."
                        },
                        "target_entity_id": {
                            "type": "string",
                            "description": "ID of the entity being modified (from the timeline)"
                        },
                        "target_event_id": {
                            "type": "string",
                            "description": "ID of the specific event being modified (from the timeline)"
                        },
                        "parameter_name": {
                            "type": "string",
                            "description": "Name of parameter being changed (e.g., 'brake_activation_time', 'error_threshold')"
                        },
                        "original_value": {
                            "type": "string",
                            "description": "Original value from the incident"
                        },
                        "counterfactual_value": {
                            "type": "string",
                            "description": "New value in the counterfactual scenario"
                        },
                        "time_delta_seconds": {
                            "type": "number",
                            "description": "Time shift in seconds (negative = earlier, positive = later)"
                        },
                        "hypothesis": {
                            "type": "string",
                            "description": "MECHANISTIC explanation of WHY this changes the outcome. Include physics/logic."
                        },
                        "expected_impact": {
                            "type": "string",
                            "enum": ["low", "medium", "high", "critical"],
                            "description": "Expected impact on preventing/mitigating the outcome"
                        },
                        "feasibility": {
                            "type": "number",
                            "description": "How feasible is this intervention? 0.0 (impossible) to 1.0 (trivial). Consider: Is this a config change (0.9)? Behavior change (0.5)? New system (0.3)? Magic (0.0)?"
                        },
                        "feasibility_rationale": {
                            "type": "string",
                            "description": "Brief explanation of feasibility score"
                        }
                    },
                    "required": ["name", "intervention_type", "description", "hypothesis", "feasibility"]
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
        messages = [Message.user(prompt)]
        
        max_iterations = 10  # Safety limit
        iteration = 0
        
        while len(interventions) < num_interventions and iteration < max_iterations:
            iteration += 1
            response = await self.provider.generate(
                messages=messages,
                functions=functions,
                temperature=0.8,  # Higher creativity for diverse, specific interventions
            )
            
            if response.function_calls:
                for call in response.function_calls:
                    if call.name == "propose_intervention":
                        args = call.arguments
                        # Handle case where VLM returns comma-separated types
                        intervention_type_str = args.get("intervention_type", "parameter_change")
                        if "," in intervention_type_str:
                            intervention_type_str = intervention_type_str.split(",")[0].strip()
                        try:
                            intervention_type = InterventionType(intervention_type_str)
                        except ValueError:
                            logger.warning("Invalid intervention type, using default", type=intervention_type_str)
                            intervention_type = InterventionType.PARAMETER_CHANGE
                        
                        intervention = Intervention(
                            intervention_type=intervention_type,
                            description=args.get("description", ""),
                            target_entity_id=args.get("target_entity_id"),
                            target_event_id=args.get("target_event_id"),
                            parameter_name=args.get("parameter_name"),
                            original_value=args.get("original_value"),
                            counterfactual_value=args.get("counterfactual_value"),
                            time_delta_seconds=args.get("time_delta_seconds"),
                            hypothesis=args.get("hypothesis", ""),
                            feasibility=args.get("feasibility", 0.5),
                            feasibility_rationale=args.get("feasibility_rationale", ""),
                            expected_effect=args.get("description", ""),  # Use description as expected effect
                        )
                        interventions.append(intervention)
                        logger.info(
                            "Generated intervention",
                            name=args.get("name"),
                            type=intervention.intervention_type.value,
                            feasibility=intervention.feasibility,
                        )
                    elif call.name == "complete_generation":
                        break
                
                # Continue the conversation with Message objects
                messages.append(Message.assistant(response.text or ""))
                messages.append(Message.user(
                    f"Generated {len(interventions)}/{num_interventions} interventions. Continue if more are needed."
                ))
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
        elif domain == "financial":
            interventions.extend(self._financial_interventions(timeline))
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
                feasibility=0.6,
                feasibility_rationale="Requires driver attention and recognition; realistic but not guaranteed",
                expected_effect="Collision avoided or significantly reduced impact speed",
            ),
            Intervention(
                intervention_type=InterventionType.SYSTEM_CAPABILITY,
                description="Automatic Emergency Braking (AEB) activates at earlier threshold",
                parameter_name="aeb_activation_threshold",
                original_value="1.2s TTC",
                counterfactual_value="2.5s TTC",
                hypothesis="More aggressive AEB would intervene before human reaction",
                feasibility=0.85,
                feasibility_rationale="OTA software update; technology exists in current vehicles",
                expected_effect="System intervenes earlier, preventing or mitigating collision",
            ),
            Intervention(
                intervention_type=InterventionType.SYSTEM_CAPABILITY,
                description="Lane Keep Assist actively corrects drift",
                parameter_name="lane_keep_assist",
                original_value="warning_only",
                counterfactual_value="active_steering",
                hypothesis="Active lane keeping would prevent unintentional lane departure",
                feasibility=0.8,
                feasibility_rationale="Feature exists in many vehicles; requires enabling or upgrade",
                expected_effect="Vehicle automatically corrects lane drift before departure",
            ),
            Intervention(
                intervention_type=InterventionType.PARAMETER_CHANGE,
                description="Following distance increased by 50%",
                parameter_name="following_distance",
                original_value="current",
                counterfactual_value="1.5x current",
                hypothesis="Greater following distance provides more reaction time",
                feasibility=0.5,
                feasibility_rationale="Requires driver behavior change; difficult to enforce",
                expected_effect="Additional 1-2 seconds of reaction time available",
            ),
            Intervention(
                intervention_type=InterventionType.BEHAVIOR_SUBSTITUTION,
                description="Driver attention score maintained above 70/100",
                parameter_name="driver_attention_score",
                original_value="28/100",
                counterfactual_value="70/100",
                hypothesis="Alert driver would have noticed hazard and responded appropriately",
                feasibility=0.7,
                feasibility_rationale="Driver monitoring systems can enforce; requires attention or break",
                expected_effect="Driver detects hazard and reacts within normal reaction time",
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
                feasibility=0.9,
                feasibility_rationale="Configuration change; can be deployed immediately",
                expected_effect="Failure isolated within seconds, preventing cascade",
            ),
            Intervention(
                intervention_type=InterventionType.PARAMETER_CHANGE,
                description="Memory limits set 50% lower to trigger OOM protection earlier",
                parameter_name="memory_limit",
                original_value="current",
                counterfactual_value="0.5x current",
                hypothesis="Lower limits would trigger graceful degradation before cascade",
                feasibility=0.75,
                feasibility_rationale="Requires capacity planning; may impact normal operations",
                expected_effect="OOM killer triggers graceful restart before memory exhaustion",
            ),
            Intervention(
                intervention_type=InterventionType.SYSTEM_CAPABILITY,
                description="Automated rollback on error rate spike",
                parameter_name="auto_rollback",
                original_value="disabled",
                counterfactual_value="enabled at 5% error rate",
                hypothesis="Automatic rollback would revert problematic deployment",
                feasibility=0.85,
                feasibility_rationale="Standard CI/CD feature; requires pipeline update",
                expected_effect="Bad deployment reverted within 2-5 minutes of detection",
            ),
            Intervention(
                intervention_type=InterventionType.TIMING_SHIFT,
                description="Deployment moved to low-traffic window",
                parameter_name="deployment_time",
                time_delta_seconds=-43200,  # 12 hours earlier
                hypothesis="Lower traffic would reduce blast radius of any issues",
                feasibility=0.7,
                feasibility_rationale="Requires scheduling changes and team coordination",
                expected_effect="Blast radius reduced by 80%+ during low-traffic period",
            ),
            Intervention(
                intervention_type=InterventionType.EVENT_ADDITION,
                description="Canary deployment catches issue with 1% traffic",
                new_event_description="Canary deployment detects memory leak with 1% traffic sample",
                hypothesis="Gradual rollout would catch issue before full deployment",
                feasibility=0.8,
                feasibility_rationale="Requires canary infrastructure; common best practice",
                expected_effect="Issue detected with minimal user impact during canary phase",
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
                feasibility=0.6,
                feasibility_rationale="Depends on deadline flexibility and stakeholder pressure",
                expected_effect="Additional data reveals key risk factors",
            ),
            Intervention(
                intervention_type=InterventionType.EVENT_ADDITION,
                description="External review/audit conducted before decision",
                new_event_description="Independent review identifies key risks",
                hypothesis="Outside perspective would catch blind spots",
                feasibility=0.7,
                feasibility_rationale="Requires budget and timeline for external engagement",
                expected_effect="Independent reviewer identifies overlooked risks",
            ),
            Intervention(
                intervention_type=InterventionType.BEHAVIOR_SUBSTITUTION,
                description="Dissenting opinion given more weight in decision",
                parameter_name="decision_weight_dissent",
                original_value="low",
                counterfactual_value="high",
                hypothesis="Addressing concerns would lead to different choice",
                feasibility=0.5,
                feasibility_rationale="Requires cultural change in decision-making process",
                expected_effect="Dissenting concerns addressed, risks mitigated",
            ),
        ]
    
    def _financial_interventions(self, timeline: Timeline) -> list[Intervention]:
        """Standard financial/trading incident interventions."""
        return [
            Intervention(
                intervention_type=InterventionType.SYSTEM_CAPABILITY,
                description="Price gap filter pauses trading on feed recovery",
                parameter_name="price_gap_filter",
                original_value="disabled",
                counterfactual_value="enabled (0.1% threshold)",
                hypothesis="Gap filter would detect stale-to-fresh price discontinuity and pause for validation",
                feasibility=0.9,
                feasibility_rationale="Standard risk control; configuration change only",
                expected_effect="Trading paused for 30s on price discontinuity for validation",
            ),
            Intervention(
                intervention_type=InterventionType.PARAMETER_CHANGE,
                description="Kill switch threshold reduced to $1M loss",
                parameter_name="kill_switch_threshold",
                original_value="$2.5M",
                counterfactual_value="$1M",
                hypothesis="Lower threshold would halt algorithm earlier, limiting total loss",
                feasibility=0.85,
                feasibility_rationale="Parameter change; may require risk committee approval",
                expected_effect="Algorithm halted at $1M loss instead of $2.5M",
            ),
            Intervention(
                intervention_type=InterventionType.SYSTEM_CAPABILITY,
                description="Human confirmation required for extreme signals",
                parameter_name="human_confirmation",
                original_value="not_required",
                counterfactual_value="required for |momentum| > 4.0",
                hypothesis="Human review would catch erroneous extreme signals before execution",
                feasibility=0.65,
                feasibility_rationale="Adds latency; may miss opportunities but catches anomalies",
                expected_effect="Extreme signals reviewed by human before execution",
            ),
            Intervention(
                intervention_type=InterventionType.PARAMETER_CHANGE,
                description="Soft rate limit cannot be bypassed without approval",
                parameter_name="soft_limit_bypass",
                original_value="auto_bypass_on_critical",
                counterfactual_value="requires_manual_approval",
                hypothesis="Preventing auto-bypass would throttle order flow during anomalies",
                feasibility=0.8,
                feasibility_rationale="Policy change with configuration update",
                expected_effect="Order rate capped during anomalies, preventing cascade",
            ),
            Intervention(
                intervention_type=InterventionType.BEHAVIOR_SUBSTITUTION,
                description="Use limit orders with IOC instead of market orders",
                parameter_name="order_type",
                original_value="MARKET",
                counterfactual_value="LIMIT with IOC",
                hypothesis="Limit orders would cap slippage and prevent trading through thin liquidity",
                feasibility=0.75,
                feasibility_rationale="Code change; may reduce fill rates in normal conditions",
                expected_effect="Slippage capped at limit price; no trading through book",
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
                feasibility=0.5,
                feasibility_rationale="Depends on situational awareness and decision speed",
                expected_effect="Additional 60 seconds of reaction time available",
            ),
            Intervention(
                intervention_type=InterventionType.EVENT_REMOVAL,
                description="Remove the triggering event",
                hypothesis="Understand if outcome was inevitable or dependent on trigger",
                feasibility=0.1,
                feasibility_rationale="Typically unrealistic; useful for causal analysis only",
                expected_effect="Counterfactual baseline without triggering event",
            ),
        ]
    
    def _summarize_incident(self, timeline: Timeline) -> str:
        """Create a rich summary of the incident for intervention generation."""
        # Group entities by type
        entity_types = {}
        for e in timeline.entities:
            entity_types.setdefault(e.entity_type, []).append(e)
        
        entities_summary = []
        for etype, entities in entity_types.items():
            names = [f"{e.name} (id: {str(e.id)[:8]})" for e in entities[:5]]
            entities_summary.append(f"  {etype}: {', '.join(names)}")
        
        # Detect incident type from events
        event_types = [e.event_type.lower() for e in timeline.events]
        event_text = " ".join([e.description.lower() for e in timeline.events])
        
        if any(t in event_types for t in ["collision", "crash", "impact"]) or "collision" in event_text:
            incident_type = "TRAFFIC/VEHICLE COLLISION"
            domain_context = """
Domain Context (Traffic):
- Human reaction time: 1.0-1.5s perception + 0.2s action
- Braking: ~0.7g (7 m/s²) for cars, ~0.4g for trucks
- Stopping distance = v²/(2×a) + reaction_distance
- Impact energy scales with v² (double speed = 4x energy)
- Key factors: speed, following distance, attention, visibility, road conditions
"""
        elif any(t in event_types for t in ["failure", "outage", "error", "timeout"]) or "deploy" in event_text:
            incident_type = "DEVOPS/SYSTEM OUTAGE"
            domain_context = """
Domain Context (DevOps):
- Cascade failures propagate in milliseconds to seconds
- Circuit breakers: typically 50-80% error threshold
- Memory issues: ~30s to OOM from first warning
- Rollback time: 2-10 minutes depending on pipeline
- Key factors: monitoring, circuit breakers, rate limits, rollback automation
"""
        elif any(t in event_types for t in ["trade", "order", "price"]) or "trading" in event_text:
            incident_type = "FINANCIAL/TRADING INCIDENT"
            domain_context = """
Domain Context (Trading):
- Market orders execute immediately at any price
- Limit orders cap slippage but may not fill
- Kill switches: typically $1-10M loss thresholds
- Flash crashes: can move markets 5%+ in seconds
- Key factors: order types, position limits, circuit breakers, human oversight
"""
        else:
            incident_type = "GENERAL INCIDENT"
            domain_context = ""
        
        # Find the climax event
        critical_events = [
            e for e in timeline.events 
            if e.event_type in ("collision", "failure", "critical", "alert", "crash", "error")
            or any(word in e.description.lower() for word in ["collision", "crash", "failure", "error", "critical"])
        ]
        climax = critical_events[0].description if critical_events else "Unknown outcome"
        
        return f"""
## Incident Type: {incident_type}

## Summary
- Entities involved: {len(timeline.entities)}
- Events in timeline: {len(timeline.events)}
- Causal links identified: {len(timeline.causal_links)}
- Timeline confidence: {timeline.confidence:.1%}
- Duration: {timeline.start_time} to {timeline.end_time}

## Entities by Type
{chr(10).join(entities_summary)}

## Critical Outcome
{climax}
{domain_context}
"""
    
    def _format_events(self, timeline: Timeline) -> str:
        """Format events with IDs for the prompt so VLM can reference them."""
        lines = []
        for e in sorted(timeline.events, key=lambda x: x.timestamp)[:40]:
            confidence_marker = "✓" if e.confidence > 0.9 else "?" if e.confidence < 0.7 else ""
            # Convert UUID to string before slicing
            event_id = str(e.id)[:8] if e.id else "unknown"
            lines.append(
                f"[{e.timestamp.strftime('%H:%M:%S.%f')[:-3]}] "
                f"(id:{event_id}) "
                f"({e.event_type}) "
                f"{e.description[:120]} {confidence_marker}"
            )
        return "\n".join(lines)
    
    def _format_causal_chain(self, timeline: Timeline) -> str:
        """Format causal links with reasoning."""
        lines = []
        for link in timeline.causal_links[:20]:
            source = next((e for e in timeline.events if e.id == link.source_event_id), None)
            target = next((e for e in timeline.events if e.id == link.target_event_id), None)
            if source and target:
                lines.append(
                    f"- [{source.timestamp.strftime('%H:%M:%S')}] {source.description[:40]}...\n"
                    f"    ──[{link.relation}]──▶\n"
                    f"  [{target.timestamp.strftime('%H:%M:%S')}] {target.description[:40]}..."
                )
        return "\n".join(lines) if lines else "No explicit causal links identified - analyze event sequence for implicit causation."
    
    def _describe_outcome(self, timeline: Timeline) -> str:
        """Describe the final outcome in detail for counterfactual analysis."""
        # Find critical/outcome events
        critical_keywords = ["collision", "crash", "failure", "error", "outage", "loss", "injury", "damage", "critical", "fatal"]
        
        critical_events = []
        for e in timeline.events:
            if e.event_type in ("collision", "failure", "critical", "alert", "crash", "error", "outcome"):
                critical_events.append(e)
            elif any(kw in e.description.lower() for kw in critical_keywords):
                critical_events.append(e)
        
        # Sort by timestamp to get chronological order
        critical_events = sorted(critical_events, key=lambda x: x.timestamp)
        
        if critical_events:
            outcome_lines = []
            for e in critical_events[:5]:
                outcome_lines.append(f"- [{e.timestamp.strftime('%H:%M:%S')}] {e.description}")
            
            # Try to identify severity indicators
            all_text = " ".join([e.description.lower() for e in critical_events])
            severity_hints = []
            if "fatal" in all_text or "death" in all_text:
                severity_hints.append("SEVERITY: Potentially fatal")
            elif "injury" in all_text or "injuries" in all_text:
                severity_hints.append("SEVERITY: Injuries involved")
            elif "total loss" in all_text or "destroyed" in all_text:
                severity_hints.append("SEVERITY: Major property damage")
            
            return f"""
## Critical Events Leading to Outcome
{chr(10).join(outcome_lines)}

{chr(10).join(severity_hints) if severity_hints else ""}

## What We Want to Prevent/Mitigate
The goal of counterfactual analysis is to find interventions that would have:
1. Prevented the outcome entirely, OR
2. Significantly reduced the severity
"""
        
        # Fall back to last few events
        last_events = sorted(timeline.events, key=lambda x: x.timestamp)[-5:]
        return f"""
## Final Events (outcome unclear - analyze carefully)
{chr(10).join([f"- [{e.timestamp.strftime('%H:%M:%S')}] {e.description}" for e in last_events])}

Note: No explicit critical event identified. Look for implicit harm in the sequence.
"""
