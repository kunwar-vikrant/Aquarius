"""Marathon Agent - Autonomous counterfactual exploration."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

import structlog

from cwe.models.timeline import Timeline, Event
from cwe.models.counterfactual import (
    Counterfactual, Intervention, InterventionType, 
    OutcomeComparison, OutcomeSeverity, CounterfactualBatch
)
from cwe.reasoning.providers.base import VLMProvider
from cwe.agents.hypothesis import HypothesisGenerator, Hypothesis

logger = structlog.get_logger()


class ExplorationStrategy(str, Enum):
    """Strategies for exploring counterfactual space."""
    
    EXHAUSTIVE = "exhaustive"      # Try all single interventions
    GREEDY = "greedy"              # Follow most promising branches
    MCTS = "mcts"                  # Monte Carlo Tree Search
    LLM_GUIDED = "llm_guided"      # VLM proposes interventions
    HUMAN_GUIDED = "human_guided"  # Prioritize user-specified hypotheses


@dataclass
class ExplorationConfig:
    """Configuration for Marathon agent exploration."""
    
    strategy: ExplorationStrategy = ExplorationStrategy.LLM_GUIDED
    
    # Resource limits
    max_iterations: int = 100
    max_parallel_branches: int = 5
    max_depth: int = 3  # Max intervention chain depth
    
    # Quality thresholds
    min_confidence: float = 0.5
    min_improvement: float = 0.1  # Min outcome improvement to report
    
    # Pruning
    prune_low_impact: bool = True
    prune_threshold: float = 0.2
    
    # Time limits
    timeout_seconds: float | None = None


@dataclass
class ExplorationResult:
    """Result of Marathon agent exploration."""
    
    id: UUID = field(default_factory=uuid4)
    incident_id: UUID | None = None
    
    # Exploration metadata
    strategy: ExplorationStrategy = ExplorationStrategy.LLM_GUIDED
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: datetime | None = None
    
    # Results
    counterfactuals: list[Counterfactual] = field(default_factory=list)
    ranked_counterfactuals: list[UUID] = field(default_factory=list)
    
    # Best outcomes
    best_outcome: Counterfactual | None = None
    worst_avoided: Counterfactual | None = None
    
    # Statistics
    hypotheses_generated: int = 0
    hypotheses_explored: int = 0
    hypotheses_pruned: int = 0
    
    # Resource usage
    total_vlm_calls: int = 0
    total_tokens: int = 0
    
    # Insights
    high_leverage_interventions: list[dict[str, Any]] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)


class MarathonAgent:
    """
    Autonomous agent for exploring counterfactual scenarios.
    
    The Marathon agent:
    1. Generates hypotheses about potential interventions
    2. Simulates counterfactual timelines for each hypothesis
    3. Ranks outcomes by impact (improvement vs canonical)
    4. Reports actionable insights and recommendations
    """
    
    def __init__(
        self,
        provider: VLMProvider,
        config: ExplorationConfig | None = None,
    ):
        self.provider = provider
        self.config = config or ExplorationConfig()
        
        self.hypothesis_generator = HypothesisGenerator(provider)
        
        # Exploration state
        self._timeline: Timeline | None = None
        self._explored: set[str] = set()  # Intervention hashes
        self._results: ExplorationResult | None = None
    
    async def explore(
        self,
        timeline: Timeline,
        seed_hypotheses: list[str] | None = None,
    ) -> ExplorationResult:
        """
        Explore counterfactual space for the given timeline.
        
        Args:
            timeline: The canonical timeline to explore alternatives for
            seed_hypotheses: Optional user-provided hypotheses to prioritize
            
        Returns:
            ExplorationResult with ranked counterfactuals and insights
        """
        logger.info(
            "Starting Marathon exploration",
            timeline_id=str(timeline.id),
            strategy=self.config.strategy.value,
            max_iterations=self.config.max_iterations,
        )
        
        self._timeline = timeline
        self._explored = set()
        self._results = ExplorationResult(
            incident_id=timeline.incident_id,
            strategy=self.config.strategy,
        )
        
        # Generate initial hypotheses
        hypotheses = await self._generate_hypotheses(seed_hypotheses)
        self._results.hypotheses_generated = len(hypotheses)
        
        # Exploration loop
        iteration = 0
        while iteration < self.config.max_iterations and hypotheses:
            iteration += 1
            
            # Select hypotheses to explore this iteration
            batch = self._select_batch(hypotheses)
            
            # Explore in parallel
            results = await asyncio.gather(
                *[self._explore_hypothesis(h) for h in batch],
                return_exceptions=True,
            )
            
            # Process results
            for hypothesis, result in zip(batch, results):
                if isinstance(result, Exception):
                    logger.error("Hypothesis exploration failed", error=str(result))
                    continue
                
                if result:
                    self._results.counterfactuals.append(result)
                    self._results.hypotheses_explored += 1
                    
                    # Generate follow-up hypotheses if promising
                    if result.confidence >= self.config.min_confidence:
                        follow_ups = await self._generate_follow_ups(result)
                        hypotheses.extend(follow_ups)
            
            # Remove explored hypotheses
            hypotheses = [h for h in hypotheses if h.id not in self._explored]
            
            # Prune low-impact hypotheses if enabled
            if self.config.prune_low_impact:
                pre_prune = len(hypotheses)
                hypotheses = self._prune_hypotheses(hypotheses)
                self._results.hypotheses_pruned += pre_prune - len(hypotheses)
            
            logger.debug(
                "Exploration iteration complete",
                iteration=iteration,
                explored=self._results.hypotheses_explored,
                remaining=len(hypotheses),
            )
        
        # Finalize results
        self._finalize_results()
        
        logger.info(
            "Marathon exploration complete",
            counterfactuals=len(self._results.counterfactuals),
            best_improvement=self._results.best_outcome.outcome.improvement_magnitude if self._results.best_outcome and self._results.best_outcome.outcome else None,
        )
        
        return self._results
    
    async def _generate_hypotheses(
        self,
        seed_hypotheses: list[str] | None = None,
    ) -> list[Hypothesis]:
        """Generate initial hypotheses for exploration."""
        hypotheses = []
        
        # Add seed hypotheses if provided
        if seed_hypotheses:
            for seed in seed_hypotheses:
                hypotheses.append(Hypothesis(
                    description=seed,
                    intervention_type=InterventionType.NATURAL_LANGUAGE,
                    priority=1.0,  # User-provided = highest priority
                    source="user",
                ))
        
        # Generate VLM hypotheses based on strategy
        if self.config.strategy == ExplorationStrategy.LLM_GUIDED:
            vlm_hypotheses = await self.hypothesis_generator.generate(
                self._timeline,
                max_hypotheses=self.config.max_iterations // 2,
            )
            hypotheses.extend(vlm_hypotheses)
        
        elif self.config.strategy == ExplorationStrategy.EXHAUSTIVE:
            # Generate single-intervention hypotheses for each event
            for event in self._timeline.events:
                # Removal hypothesis
                hypotheses.append(Hypothesis(
                    description=f"What if '{event.description[:50]}' didn't happen?",
                    intervention_type=InterventionType.REMOVE_EVENT,
                    target_event_id=event.id,
                    priority=0.5,
                    source="exhaustive",
                ))
                
                # Timing hypotheses
                hypotheses.append(Hypothesis(
                    description=f"What if '{event.description[:50]}' happened 30 seconds earlier?",
                    intervention_type=InterventionType.ADVANCE_EVENT,
                    target_event_id=event.id,
                    parameters={"advance_seconds": 30},
                    priority=0.3,
                    source="exhaustive",
                ))
        
        return hypotheses
    
    def _select_batch(self, hypotheses: list[Hypothesis]) -> list[Hypothesis]:
        """Select hypotheses to explore in this iteration."""
        # Sort by priority
        sorted_hypotheses = sorted(hypotheses, key=lambda h: h.priority, reverse=True)
        
        # Take top N
        batch_size = min(self.config.max_parallel_branches, len(sorted_hypotheses))
        return sorted_hypotheses[:batch_size]
    
    async def _explore_hypothesis(self, hypothesis: Hypothesis) -> Counterfactual | None:
        """Explore a single hypothesis."""
        self._explored.add(hypothesis.id)
        
        logger.debug("Exploring hypothesis", description=hypothesis.description[:50])
        
        # Create intervention
        intervention = Intervention(
            intervention_type=hypothesis.intervention_type,
            description=hypothesis.description,
            parameters=hypothesis.parameters,
            target_event_ids=[hypothesis.target_event_id] if hypothesis.target_event_id else [],
        )
        
        # Simulate counterfactual (simplified - full implementation would use CounterfactualSimulator)
        counterfactual = await self._simulate_counterfactual(intervention)
        
        if counterfactual:
            counterfactual.auto_generated = True
            counterfactual.generation_strategy = self.config.strategy.value
        
        self._results.total_vlm_calls += 1
        
        return counterfactual
    
    async def _simulate_counterfactual(
        self, 
        intervention: Intervention
    ) -> Counterfactual | None:
        """Simulate a counterfactual timeline."""
        from cwe.reasoning.providers.base import Message
        from cwe.reasoning.function_schema import get_counterfactual_functions
        
        # Build prompt
        prompt = self._build_counterfactual_prompt(intervention)
        
        messages = [
            Message.system(
                "You are simulating an alternate timeline. Given the canonical events "
                "and an intervention (change), reason about what would have happened differently."
            ),
            Message.user(prompt),
        ]
        
        # Get VLM response
        response = await self.provider.generate(
            messages=messages,
            functions=[f.to_schema() for f in get_counterfactual_functions()],
            enable_thinking=True,
        )
        
        self._results.total_tokens += response.total_tokens
        
        # Parse response into Counterfactual
        counterfactual = Counterfactual(
            incident_id=self._timeline.incident_id,
            canonical_timeline_id=self._timeline.id,
            intervention=intervention,
            explanation=response.text,
            vlm_provider=self.provider.provider_type.value,
            vlm_model=self.provider.model,
            vlm_raw_response=response.text,
        )
        
        # Process function calls to build outcome
        for fc in response.function_calls:
            if fc.name == "compare_outcomes":
                counterfactual.outcome = OutcomeComparison(
                    canonical_outcome=fc.arguments.get("canonical_outcome", ""),
                    canonical_severity=OutcomeSeverity(fc.arguments.get("canonical_severity", "moderate")),
                    canonical_score=0.5,
                    counterfactual_outcome=fc.arguments.get("counterfactual_outcome", ""),
                    counterfactual_severity=OutcomeSeverity(fc.arguments.get("counterfactual_severity", "moderate")),
                    counterfactual_score=0.5,
                    outcome_improved=fc.arguments.get("outcome_improved", False),
                    improvement_magnitude=self._calculate_improvement(
                        fc.arguments.get("canonical_severity", "moderate"),
                        fc.arguments.get("counterfactual_severity", "moderate"),
                    ),
                    key_differences=fc.arguments.get("key_differences", []),
                )
                counterfactual.confidence = fc.arguments.get("confidence", 0.5)
        
        return counterfactual
    
    def _build_counterfactual_prompt(self, intervention: Intervention) -> str:
        """Build prompt for counterfactual simulation."""
        events_text = "\n".join(
            f"- [{e.timestamp.isoformat()}] {e.description}"
            for e in sorted(self._timeline.events, key=lambda e: e.timestamp)
        )
        
        return f"""# Canonical Timeline

{events_text}

# Intervention

{intervention.description}

# Task

Simulate what would have happened if this intervention had occurred.
Call `compare_outcomes` to summarize the comparison between the canonical and counterfactual timelines.

Think step by step about:
1. Which events would change or not occur
2. What new events might occur
3. How the final outcome differs
"""
    
    def _calculate_improvement(self, canonical: str, counterfactual: str) -> float:
        """Calculate improvement magnitude between severity levels."""
        severity_scores = {
            "none": 0.0,
            "minor": 0.25,
            "moderate": 0.5,
            "severe": 0.75,
            "catastrophic": 1.0,
        }
        
        canonical_score = severity_scores.get(canonical, 0.5)
        counterfactual_score = severity_scores.get(counterfactual, 0.5)
        
        return canonical_score - counterfactual_score  # Positive = improvement
    
    async def _generate_follow_ups(
        self, 
        counterfactual: Counterfactual
    ) -> list[Hypothesis]:
        """Generate follow-up hypotheses based on a promising counterfactual."""
        if counterfactual.outcome and counterfactual.outcome.outcome_improved:
            # Generate variations of successful interventions
            return await self.hypothesis_generator.generate_variations(
                counterfactual.intervention,
                self._timeline,
            )
        return []
    
    def _prune_hypotheses(self, hypotheses: list[Hypothesis]) -> list[Hypothesis]:
        """Prune low-priority hypotheses."""
        return [h for h in hypotheses if h.priority >= self.config.prune_threshold]
    
    def _finalize_results(self) -> None:
        """Finalize exploration results."""
        self._results.completed_at = datetime.utcnow()
        
        # Rank counterfactuals by improvement
        ranked = sorted(
            self._results.counterfactuals,
            key=lambda c: c.outcome.improvement_magnitude if c.outcome else 0,
            reverse=True,
        )
        
        self._results.ranked_counterfactuals = [c.id for c in ranked]
        
        # Identify best/worst outcomes
        if ranked:
            best_improved = [c for c in ranked if c.outcome and c.outcome.outcome_improved]
            if best_improved:
                self._results.best_outcome = best_improved[0]
            
            # Find worst avoided (intervention that would have made things worse)
            worst = [c for c in ranked if c.outcome and not c.outcome.outcome_improved]
            if worst:
                self._results.worst_avoided = worst[-1]
        
        # Extract high-leverage interventions
        high_leverage = [
            {
                "intervention": c.intervention.description,
                "improvement": c.outcome.improvement_magnitude,
                "confidence": c.confidence,
            }
            for c in ranked[:5]
            if c.outcome and c.outcome.improvement_magnitude > self.config.min_improvement
        ]
        self._results.high_leverage_interventions = high_leverage
        
        # Generate recommendations
        self._results.recommendations = self._generate_recommendations(ranked)
    
    def _generate_recommendations(
        self, 
        ranked: list[Counterfactual]
    ) -> list[str]:
        """Generate actionable recommendations from exploration."""
        recommendations = []
        
        # Top interventions that improve outcomes
        for cf in ranked[:3]:
            if cf.outcome and cf.outcome.outcome_improved and cf.confidence >= 0.6:
                recommendations.append(
                    f"Consider: {cf.intervention.description} "
                    f"(Estimated improvement: {cf.outcome.improvement_magnitude:.0%}, "
                    f"Confidence: {cf.confidence:.0%})"
                )
        
        # Warnings about interventions that make things worse
        for cf in reversed(ranked[-3:]):
            if cf.outcome and not cf.outcome.outcome_improved and cf.confidence >= 0.6:
                recommendations.append(
                    f"Avoid: {cf.intervention.description} "
                    f"(Would worsen outcome by {abs(cf.outcome.improvement_magnitude):.0%})"
                )
        
        return recommendations
