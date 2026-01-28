"""
Report formatter for counterfactual analysis.
"""

from datetime import datetime
from pathlib import Path
from typing import Optional

from .models import (
    CounterfactualAnalysis,
    CounterfactualScenario,
    OutcomeSeverity,
)


def format_counterfactual_report(
    analysis: CounterfactualAnalysis,
    original_incident_name: str = "Incident",
) -> str:
    """
    Format a counterfactual analysis as a Markdown report.
    """
    lines = [
        "# Counterfactual Analysis Report",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Incident:** {original_incident_name}",
        f"**Scenarios Analyzed:** {len(analysis.scenarios)}",
        f"**Total Analysis Time:** {analysis.total_simulation_time_seconds:.1f}s",
        "",
        "---",
        "",
        "## Executive Summary",
        "",
    ]
    
    # Executive summary
    prevented_count = sum(
        1 for s in analysis.scenarios 
        if s.outcome and not s.outcome.primary_outcome_occurred
    )
    
    if prevented_count > 0:
        lines.append(
            f"**{prevented_count} of {len(analysis.scenarios)} interventions could have prevented the incident.**"
        )
    else:
        lines.append(
            "No single intervention was found to definitively prevent the incident, "
            "but several could have reduced its severity."
        )
    
    lines.append("")
    
    # Key findings
    if analysis.key_findings:
        lines.append("### Key Findings")
        lines.append("")
        for finding in analysis.key_findings:
            lines.append(f"- {finding}")
        lines.append("")
    
    # Top recommendations
    if analysis.recommendations:
        lines.append("### Top Recommendations")
        lines.append("")
        for i, rec in enumerate(analysis.recommendations[:5], 1):
            lines.append(f"{i}. {rec}")
        lines.append("")
    
    lines.append("---")
    lines.append("")
    
    # Intervention effectiveness ranking
    lines.append("## Intervention Effectiveness Ranking")
    lines.append("")
    lines.append("| Rank | Intervention | Score | Prevented? | Severity Change | Confidence |")
    lines.append("|------|--------------|-------|------------|-----------------|------------|")
    
    for i, ranking in enumerate(analysis.intervention_ranking[:10], 1):
        prevented = "✅ YES" if ranking["prevented_outcome"] else "❌ No"
        lines.append(
            f"| {i} | {ranking['intervention'][:40]}... | {ranking['effectiveness_score']:.0f} | "
            f"{prevented} | {ranking['severity_change']} | {ranking['confidence']:.0%} |"
        )
    
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # Detailed scenario analysis
    lines.append("## Detailed Scenario Analysis")
    lines.append("")
    
    for i, scenario in enumerate(analysis.scenarios, 1):
        lines.extend(_format_scenario(scenario, i))
        lines.append("")
    
    lines.append("---")
    lines.append("")
    
    # Original incident context
    lines.append("## Original Incident Context")
    lines.append("")
    lines.append(f"**Original Severity:** {analysis.original_severity.value.upper()}")
    lines.append("")
    lines.append("### Original Outcome")
    lines.append("")
    lines.append(analysis.original_outcome_description)
    lines.append("")
    
    return "\n".join(lines)


def _format_scenario(scenario: CounterfactualScenario, index: int) -> list[str]:
    """Format a single scenario."""
    lines = [
        f"### Scenario {index}: {scenario.name}",
        "",
        f"**Question:** {scenario.question}",
        "",
    ]
    
    # Intervention details
    if scenario.interventions:
        intervention = scenario.interventions[0]
        lines.append("**Intervention:**")
        lines.append(f"- Type: `{intervention.intervention_type.value}`")
        lines.append(f"- Description: {intervention.description}")
        if intervention.hypothesis:
            lines.append(f"- Hypothesis: {intervention.hypothesis}")
        lines.append("")
    
    # Divergence points
    if scenario.divergence_points:
        lines.append("**Divergence Points:**")
        lines.append("")
        for dp in scenario.divergence_points:
            lines.append(f"⚡ **{dp.timestamp.strftime('%H:%M:%S')}** - {dp.cause}")
            if dp.original_event_description:
                lines.append(f"  - Original: {dp.original_event_description}")
            lines.append(f"  - Counterfactual: {dp.counterfactual_event_description}")
            if dp.cascade_effects:
                lines.append("  - Cascade effects:")
                for effect in dp.cascade_effects[:3]:
                    lines.append(f"    - {effect}")
        lines.append("")
    
    # Outcome
    if scenario.outcome:
        outcome = scenario.outcome
        
        # Outcome box
        if outcome.primary_outcome_occurred:
            status = "⚠️ INCIDENT STILL OCCURS"
            status_color = "yellow"
        else:
            status = "✅ INCIDENT PREVENTED"
            status_color = "green"
        
        lines.append(f"**Outcome:** {status}")
        lines.append("")
        lines.append(f"| Metric | Original | Counterfactual |")
        lines.append(f"|--------|----------|----------------|")
        lines.append(f"| Outcome Occurred | Yes | {'Yes' if outcome.primary_outcome_occurred else 'No'} |")
        lines.append(f"| Severity | {outcome.original_severity.value} | {outcome.counterfactual_severity.value} |")
        
        if outcome.injury_reduction_percent is not None:
            lines.append(f"| Injury Reduction | - | {outcome.injury_reduction_percent:.0f}% |")
        if outcome.damage_reduction_percent is not None:
            lines.append(f"| Damage Reduction | - | {outcome.damage_reduction_percent:.0f}% |")
        
        lines.append(f"| Confidence | - | {outcome.confidence:.0%} |")
        lines.append("")
        
        # Prevented events
        if outcome.prevented_events:
            lines.append("**Events Prevented:**")
            for event in outcome.prevented_events[:5]:
                lines.append(f"- ~~{event}~~")
            lines.append("")
        
        # New events
        if outcome.new_events:
            lines.append("**New Events (in counterfactual):**")
            for event in outcome.new_events[:5]:
                lines.append(f"- 🆕 {event}")
            lines.append("")
        
        # Reasoning
        if outcome.reasoning:
            lines.append("**Reasoning:**")
            lines.append("")
            lines.append(f"> {outcome.reasoning}")
            lines.append("")
    
    # Summary
    if scenario.summary:
        lines.append("**Summary:**")
        lines.append(scenario.summary)
        lines.append("")
    
    # Recommendations
    if scenario.recommendations:
        lines.append("**Recommendations from this scenario:**")
        for rec in scenario.recommendations:
            lines.append(f"- {rec}")
        lines.append("")
    
    return lines


def save_counterfactual_report(
    analysis: CounterfactualAnalysis,
    output_dir: Path,
    incident_name: str = "Incident",
) -> Path:
    """Save counterfactual report to file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = "".join(c if c.isalnum() or c in "- " else "_" for c in incident_name[:40])
    safe_name = safe_name.replace(" ", "_").lower()
    
    filename = f"{timestamp}_counterfactual_{safe_name}.md"
    filepath = output_dir / filename
    
    content = format_counterfactual_report(analysis, incident_name)
    filepath.write_text(content)
    
    return filepath
