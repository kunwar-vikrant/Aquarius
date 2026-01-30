"""
Counterfactual simulation engine.

Simulates alternative timelines based on interventions.
"""

import json
import structlog
from datetime import datetime
from typing import Optional
import time

from ..models.timeline import Timeline, Event
from ..reasoning.providers.base import VLMProvider, Message, FunctionCall
from .models import (
    Intervention,
    InterventionType,
    CounterfactualOutcome,
    CounterfactualScenario,
    CounterfactualAnalysis,
    DivergencePoint,
    OutcomeSeverity,
)
from .prompts import COUNTERFACTUAL_SIMULATION_PROMPT, OUTCOME_COMPARISON_PROMPT, RECOMMENDATIONS_PROMPT
from .generator import InterventionGenerator

logger = structlog.get_logger()


class CounterfactualSimulator:
    """
    Simulates counterfactual scenarios by applying interventions to timelines.
    
    The simulation process:
    1. Take the original timeline
    2. Apply an intervention (parameter change, event removal, etc.)
    3. Ask the VLM to reason about causal propagation
    4. Generate the alternative timeline
    5. Compare outcomes
    """
    
    def __init__(self, provider: VLMProvider):
        self.provider = provider
        self.generator = InterventionGenerator(provider)
    
    async def simulate_scenario(
        self,
        timeline: Timeline,
        intervention: Intervention,
    ) -> CounterfactualScenario:
        """
        Simulate a single counterfactual scenario.
        
        Args:
            timeline: Original incident timeline
            intervention: The intervention to apply
            
        Returns:
            CounterfactualScenario with simulated outcome
        """
        start_time = time.time()
        
        logger.info(
            "Simulating counterfactual",
            intervention_type=intervention.intervention_type.value,
            description=intervention.description[:50],
        )
        
        # Format the original timeline
        original_timeline_text = self._format_timeline(timeline)
        original_outcome_text = self._describe_original_outcome(timeline)
        
        # Build simulation prompt
        prompt = COUNTERFACTUAL_SIMULATION_PROMPT.format(
            original_timeline=original_timeline_text,
            original_outcome=original_outcome_text,
            intervention_description=intervention.to_prompt(),
            intervention_hypothesis=intervention.hypothesis,
        )
        
        # Define functions for structured simulation output
        functions = [
            {
                "name": "set_divergence_point",
                "description": "Identify where the timeline diverges due to the intervention",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "timestamp": {"type": "string", "description": "ISO timestamp of divergence"},
                        "original_event_description": {"type": "string"},
                        "counterfactual_event_description": {"type": "string"},
                        "cause": {"type": "string", "description": "Why the divergence occurs"},
                        "cascade_effects": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Downstream effects of this divergence"
                        }
                    },
                    "required": ["timestamp", "cause", "counterfactual_event_description"]
                }
            },
            {
                "name": "emit_alternative_event",
                "description": "Emit an event in the alternative timeline",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "timestamp": {"type": "string"},
                        "event_type": {"type": "string"},
                        "description": {"type": "string"},
                        "differs_from_original": {"type": "boolean"},
                        "original_event_id": {"type": "string", "description": "ID of original event this replaces (if any)"},
                    },
                    "required": ["timestamp", "event_type", "description", "differs_from_original"]
                }
            },
            {
                "name": "assess_outcome",
                "description": "Assess the outcome of the counterfactual scenario",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "description": {"type": "string", "description": "What happens in this scenario"},
                        "primary_outcome_occurred": {
                            "type": "boolean",
                            "description": "Did the main negative outcome (crash, outage, etc.) still happen?"
                        },
                        "severity": {
                            "type": "string",
                            "enum": ["none", "minor", "moderate", "severe", "catastrophic", "fatal"]
                        },
                        "injury_reduction_percent": {"type": "number"},
                        "damage_reduction_percent": {"type": "number"},
                        "prevented_events": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Events from original timeline that were prevented"
                        },
                        "new_events": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "New events that occur only in counterfactual"
                        },
                        "reasoning": {"type": "string", "description": "Detailed reasoning for this assessment"},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1}
                    },
                    "required": ["description", "primary_outcome_occurred", "severity", "reasoning", "confidence"]
                }
            },
            {
                "name": "complete_simulation",
                "description": "Signal simulation is complete",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "summary": {"type": "string"},
                        "key_insight": {"type": "string"},
                        "recommendation": {"type": "string"}
                    },
                    "required": ["summary"]
                }
            }
        ]
        
        # Run simulation
        messages = [Message.user(prompt)]
        
        scenario = CounterfactualScenario(
            incident_id=timeline.incident_id,
            interventions=[intervention],
            name=intervention.description[:50],
            question=f"What if {intervention.description.lower()}?",
        )
        
        alternative_events = []
        simulation_complete = False
        max_iterations = 10
        iteration = 0
        
        while not simulation_complete and iteration < max_iterations:
            iteration += 1
            
            response = await self.provider.generate(
                messages=messages,
                functions=functions,
                temperature=0.3,  # More deterministic for simulation
            )
            
            if response.function_calls:
                for call in response.function_calls:
                    if call.name == "set_divergence_point":
                        args = call.arguments
                        try:
                            timestamp = datetime.fromisoformat(args["timestamp"].replace("Z", "+00:00"))
                        except:
                            timestamp = datetime.now()
                        
                        divergence = DivergencePoint(
                            timestamp=timestamp,
                            original_event_description=args.get("original_event_description", ""),
                            counterfactual_event_description=args.get("counterfactual_event_description", ""),
                            cause=args.get("cause", ""),
                            cascade_effects=args.get("cascade_effects", []),
                        )
                        scenario.divergence_points.append(divergence)
                        logger.debug("Divergence point identified", timestamp=str(timestamp))
                    
                    elif call.name == "emit_alternative_event":
                        args = call.arguments
                        alternative_events.append(args)
                    
                    elif call.name == "assess_outcome":
                        args = call.arguments
                        scenario.outcome = CounterfactualOutcome(
                            description=args.get("description", ""),
                            primary_outcome_occurred=args.get("primary_outcome_occurred", True),
                            counterfactual_severity=OutcomeSeverity(args.get("severity", "moderate")),
                            injury_reduction_percent=args.get("injury_reduction_percent"),
                            damage_reduction_percent=args.get("damage_reduction_percent"),
                            prevented_events=args.get("prevented_events", []),
                            new_events=args.get("new_events", []),
                            reasoning=args.get("reasoning", ""),
                            confidence=args.get("confidence", 0.5),
                        )
                        # Set original severity from timeline
                        scenario.outcome.original_severity = self._assess_original_severity(timeline)
                    
                    elif call.name == "complete_simulation":
                        args = call.arguments
                        scenario.summary = args.get("summary", "")
                        if args.get("recommendation"):
                            scenario.recommendations.append(args["recommendation"])
                        simulation_complete = True
                        break
                
                # Continue conversation using proper Message objects
                # Create an assistant message with function calls info
                assistant_msg = Message.assistant(f"[Called functions: {', '.join(c.name for c in response.function_calls)}]")
                assistant_msg.function_calls = [
                    {"id": fc.id, "name": fc.name, "arguments": fc.arguments}
                    for fc in response.function_calls
                ]
                messages.append(assistant_msg)
                messages.append(Message.user("Continue the simulation. Trace all causal effects and assess the final outcome."))
            else:
                # Text response, try to extract insight
                if response.text:
                    scenario.summary = response.text[:500]
                simulation_complete = True
        
        scenario.alternative_events = alternative_events
        scenario.simulation_duration_seconds = time.time() - start_time
        
        logger.info(
            "Simulation complete",
            divergence_points=len(scenario.divergence_points),
            alternative_events=len(alternative_events),
            outcome_occurred=scenario.outcome.primary_outcome_occurred if scenario.outcome else None,
            duration=f"{scenario.simulation_duration_seconds:.1f}s",
        )
        
        return scenario
    
    async def run_full_analysis(
        self,
        timeline: Timeline,
        interventions: Optional[list[Intervention]] = None,
        num_auto_interventions: int = 5,
        domain: str = "general",
    ) -> CounterfactualAnalysis:
        """
        Run a complete counterfactual analysis with multiple scenarios.
        
        Args:
            timeline: Original incident timeline
            interventions: Specific interventions to test (optional)
            num_auto_interventions: Number of interventions to auto-generate if none provided
            domain: Domain hint for intervention generation
            
        Returns:
            Complete CounterfactualAnalysis
        """
        start_time = time.time()
        
        logger.info(
            "Starting full counterfactual analysis",
            incident_id=timeline.incident_id,
            provided_interventions=len(interventions) if interventions else 0,
        )
        
        # Initialize analysis
        analysis = CounterfactualAnalysis(
            incident_id=timeline.incident_id,
            original_timeline_summary=self._format_timeline(timeline),
            original_outcome_description=self._describe_original_outcome(timeline),
            original_severity=self._assess_original_severity(timeline),
        )
        
        # Get interventions
        if not interventions:
            logger.info("Generating interventions automatically")
            
            # PRIMARY: Use VLM to generate context-aware interventions
            # This analyzes the actual incident and proposes relevant what-ifs
            try:
                interventions = await self.generator.generate_interventions(
                    timeline,
                    num_interventions=num_auto_interventions,
                    focus_areas=None,  # Let VLM decide based on incident type
                )
                logger.info(f"VLM generated {len(interventions)} context-aware interventions")
            except Exception as e:
                logger.warning("VLM intervention generation failed", error=str(e))
                interventions = []
            
            # FALLBACK: If VLM didn't generate enough, supplement with domain standards
            if len(interventions) < num_auto_interventions:
                standard = self.generator.generate_standard_interventions(timeline, domain)
                # Only add standards not already covered by VLM-generated ones
                existing_types = {i.intervention_type for i in interventions}
                for std in standard:
                    if len(interventions) >= num_auto_interventions:
                        break
                    # Avoid duplicating similar intervention types
                    if std.intervention_type not in existing_types:
                        interventions.append(std)
                        existing_types.add(std.intervention_type)
                logger.info(f"Supplemented with {len(interventions) - len([i for i in interventions if i not in standard])} standard interventions")
        
        # Simulate each scenario
        for i, intervention in enumerate(interventions):
            logger.info(f"Simulating scenario {i+1}/{len(interventions)}", intervention=intervention.description[:50])
            
            try:
                scenario = await self.simulate_scenario(timeline, intervention)
                analysis.scenarios.append(scenario)
            except Exception as e:
                logger.error("Scenario simulation failed", error=str(e), intervention=intervention.description)
        
        # Rank interventions by effectiveness
        analysis.intervention_ranking = self._rank_interventions(analysis.scenarios)
        
        # Generate key findings
        analysis.key_findings = self._extract_findings(analysis)
        
        # Generate recommendations
        analysis.recommendations = await self._generate_recommendations(analysis)
        
        analysis.total_simulation_time_seconds = time.time() - start_time
        
        logger.info(
            "Counterfactual analysis complete",
            scenarios=len(analysis.scenarios),
            findings=len(analysis.key_findings),
            recommendations=len(analysis.recommendations),
            total_time=f"{analysis.total_simulation_time_seconds:.1f}s",
        )
        
        return analysis
    
    def _format_timeline(self, timeline: Timeline) -> str:
        """Format timeline for prompts."""
        lines = ["## Entities"]
        for e in timeline.entities[:10]:
            props = ", ".join(f"{k}={v}" for k, v in (e.properties or {}).items())
            lines.append(f"- **{e.name}** ({e.entity_type}): {props}")
        
        lines.append("\n## Events (chronological)")
        for event in sorted(timeline.events, key=lambda x: x.timestamp)[:40]:
            ts = event.timestamp.strftime("%H:%M:%S.%f")[:-3]
            lines.append(f"- [{ts}] ({event.event_type}) {event.description}")
        
        lines.append("\n## Causal Links")
        for link in timeline.causal_links[:15]:
            source = next((e for e in timeline.events if e.id == link.source_event_id), None)
            target = next((e for e in timeline.events if e.id == link.target_event_id), None)
            if source and target:
                lines.append(f"- {source.description[:40]}... →[{link.relation}]→ {target.description[:40]}...")
        
        return "\n".join(lines)
    
    def _describe_original_outcome(self, timeline: Timeline) -> str:
        """Describe the original incident outcome."""
        # Find critical events
        critical_types = {"collision", "failure", "critical", "alert", "error", "crash"}
        critical_events = [
            e for e in timeline.events 
            if e.event_type in critical_types or any(t in e.description.lower() for t in critical_types)
        ]
        
        if critical_events:
            descriptions = [f"- {e.description}" for e in critical_events[:5]]
            return "Critical events:\n" + "\n".join(descriptions)
        
        # Fall back to last events
        last = sorted(timeline.events, key=lambda x: x.timestamp)[-3:]
        return "Final events:\n" + "\n".join([f"- {e.description}" for e in last])
    
    def _assess_original_severity(self, timeline: Timeline) -> OutcomeSeverity:
        """Assess severity of original incident."""
        text = " ".join([e.description.lower() for e in timeline.events])
        
        if any(word in text for word in ["fatal", "death", "died"]):
            return OutcomeSeverity.FATAL
        elif any(word in text for word in ["catastrophic", "total loss", "destroyed"]):
            return OutcomeSeverity.CATASTROPHIC
        elif any(word in text for word in ["severe", "critical", "major", "hospitalized"]):
            return OutcomeSeverity.SEVERE
        elif any(word in text for word in ["moderate", "significant"]):
            return OutcomeSeverity.MODERATE
        elif any(word in text for word in ["minor", "slight"]):
            return OutcomeSeverity.MINOR
        else:
            return OutcomeSeverity.MODERATE  # Default assumption
    
    def _calculate_effectiveness(self, scenario: CounterfactualScenario) -> float:
        """
        Calculate normalized effectiveness score [0-1].
        
        Measures how much the intervention reduces harm:
        - Full prevention = 1.0
        - Severity reduction contributes proportionally
        - No improvement = 0.0
        """
        if not scenario.outcome:
            return 0.0
        
        outcome = scenario.outcome
        
        # Full prevention is maximum effectiveness
        if not outcome.primary_outcome_occurred:
            return 1.0
        
        # Otherwise, calculate based on severity reduction
        severity_order = list(OutcomeSeverity)
        orig_idx = severity_order.index(outcome.original_severity)
        new_idx = severity_order.index(outcome.counterfactual_severity)
        
        if orig_idx == 0:  # Original was already NONE
            return 0.0
        
        # Normalize severity reduction to [0, 1]
        # e.g., CATASTROPHIC (4) -> MINOR (1) = reduction of 3/4 = 0.75
        reduction = (orig_idx - new_idx) / orig_idx
        return max(0.0, min(1.0, reduction))
    
    def _rank_interventions(self, scenarios: list[CounterfactualScenario]) -> list[dict]:
        """
        Rank interventions using multiplicative scoring.
        
        Score = Effectiveness × Feasibility × Confidence
        
        This ensures:
        - Impossible interventions score 0 (feasibility = 0)
        - Ineffective interventions score 0 (effectiveness = 0)
        - Uncertain simulations are penalized (low confidence)
        """
        rankings = []
        
        for scenario in scenarios:
            if not scenario.outcome:
                continue
            
            intervention = scenario.interventions[0] if scenario.interventions else None
            if not intervention:
                continue
            
            outcome = scenario.outcome
            
            # E: Effectiveness [0-1] - how much harm is reduced?
            effectiveness = self._calculate_effectiveness(scenario)
            
            # F: Feasibility [0-1] - can we actually implement this?
            feasibility = getattr(intervention, 'feasibility', 0.5)
            
            # C: Confidence [0-1] - how sure is the simulation?
            confidence = outcome.confidence
            
            # Multiplicative combination: all factors must be good
            score = effectiveness * feasibility * confidence
            
            # Also store raw score scaled to 0-100 for display
            display_score = round(score * 100, 1)
            
            rankings.append({
                "intervention": intervention.description,
                "effectiveness_score": display_score,
                "raw_score": round(score, 4),
                "effectiveness": round(effectiveness, 3),
                "feasibility": round(feasibility, 3),
                "confidence": round(confidence, 3),
                "prevented_outcome": not outcome.primary_outcome_occurred,
                "severity_change": f"{outcome.original_severity.value} → {outcome.counterfactual_severity.value}",
            })
        
        return sorted(rankings, key=lambda x: x["raw_score"], reverse=True)
    
    def _extract_findings(self, analysis: CounterfactualAnalysis) -> list[str]:
        """Extract key findings from analysis."""
        findings = []
        
        # Check for preventable scenarios
        prevented = [s for s in analysis.scenarios if s.outcome and not s.outcome.primary_outcome_occurred]
        if prevented:
            findings.append(
                f"The incident could have been prevented in {len(prevented)}/{len(analysis.scenarios)} "
                f"simulated scenarios."
            )
            for s in prevented[:2]:
                findings.append(f"  • Prevention possible with: {s.interventions[0].description}")
        
        # Timing-sensitive interventions
        timing_scenarios = [
            s for s in analysis.scenarios 
            if s.interventions and s.interventions[0].intervention_type == InterventionType.TIMING_SHIFT
        ]
        for s in timing_scenarios:
            if s.outcome and not s.outcome.primary_outcome_occurred:
                findings.append(
                    f"Timing was critical: {s.interventions[0].description} would have prevented the outcome."
                )
        
        # System/technology findings
        tech_scenarios = [
            s for s in analysis.scenarios
            if s.interventions and s.interventions[0].intervention_type == InterventionType.SYSTEM_CAPABILITY
        ]
        effective_tech = [s for s in tech_scenarios if s.outcome and s.outcome.confidence > 0.7]
        if effective_tech:
            findings.append(
                f"Technology interventions showed promise: {len(effective_tech)} scenarios showed significant improvement."
            )
        
        return findings
    
    async def _generate_recommendations(self, analysis: CounterfactualAnalysis) -> list[str]:
        """Generate recommendations based on analysis."""
        # Start with scenario-specific recommendations
        recommendations = []
        
        for scenario in analysis.scenarios:
            if scenario.recommendations:
                recommendations.extend(scenario.recommendations)
        
        # Add findings-based recommendations
        if analysis.intervention_ranking:
            top = analysis.intervention_ranking[0]
            if top["effectiveness_score"] > 50:
                recommendations.append(
                    f"HIGH PRIORITY: Implement '{top['intervention']}' - "
                    f"effectiveness score {top['effectiveness_score']}, "
                    f"confidence {top['confidence']:.0%}"
                )
        
        # Deduplicate
        return list(dict.fromkeys(recommendations))[:10]
