"""
Prompts for counterfactual reasoning.
"""

INTERVENTION_GENERATION_PROMPT = """You are an expert incident analyst specializing in counterfactual reasoning.

Given the following incident timeline, generate meaningful "what-if" interventions that could have changed the outcome.

## Original Incident
{incident_summary}

## Timeline Events (chronological)
{timeline_events}

## Causal Chain
{causal_chain}

## Original Outcome
{original_outcome}

---

Generate {num_interventions} counterfactual interventions. For each, consider:

1. **Prevention Points**: Where could the causal chain have been broken?
2. **Timing Sensitivity**: Which events had critical timing that, if changed, would alter the outcome?
3. **Missing Safeguards**: What systems, behaviors, or policies could have prevented this?
4. **Human Factors**: What different decisions could key actors have made?
5. **Environmental Changes**: What external conditions could have changed the outcome?

Focus on interventions that are:
- Realistic and actionable (not "what if physics worked differently")
- Specific and testable
- Likely to have meaningful impact on the outcome
- Useful for generating recommendations

For each intervention, explain your hypothesis about why it would change the outcome.
"""

COUNTERFACTUAL_SIMULATION_PROMPT = """You are an expert in causal reasoning and incident simulation.

## Task
Simulate what would have happened if the following intervention had occurred during the incident.

## Original Incident Timeline
{original_timeline}

## Original Outcome
{original_outcome}

## Intervention to Apply
{intervention_description}

Hypothesis: {intervention_hypothesis}

---

## Instructions

1. **Identify the Divergence Point**: At what moment does the timeline change due to this intervention?

2. **Trace Causal Propagation**: How does this change ripple through the causal chain?
   - Which downstream events are prevented?
   - Which events still occur but are modified?
   - What new events might occur that didn't in the original timeline?

3. **Simulate the Alternative Timeline**: Describe what happens from the divergence point forward.

4. **Assess the Outcome**: 
   - Does the primary negative outcome (collision, outage, injury, etc.) still occur?
   - If yes, how is its severity changed?
   - If no, what is the new outcome?

5. **Quantify the Impact** (where possible):
   - Injury reduction (percentage estimate)
   - Damage reduction (percentage estimate)  
   - Time saved or lost

6. **Confidence Assessment**: How confident are you in this simulation? Consider:
   - Physical/technical plausibility
   - Behavioral realism
   - Completeness of available data

## Domain Knowledge to Apply

For traffic incidents:
- Human reaction time: typically 1.0-1.5 seconds
- Braking deceleration: ~0.7g for passenger cars, ~0.4g for loaded trucks
- Stopping distance = (speed²) / (2 × deceleration) + reaction_distance
- Impact severity scales roughly with velocity squared

For system outages:
- Circuit breaker patterns and recovery times
- Cascade failure propagation speeds
- Typical MTTR for various component types

Be rigorous and realistic. Don't assume interventions have magical effects.
"""

OUTCOME_COMPARISON_PROMPT = """You are comparing two incident timelines: the original and a counterfactual simulation.

## Original Timeline
{original_timeline}

## Original Outcome
{original_outcome}

## Counterfactual Scenario
Intervention: {intervention}
{counterfactual_timeline}

## Counterfactual Outcome
{counterfactual_outcome}

---

Provide a structured comparison:

1. **Divergence Analysis**
   - First point of divergence (timestamp and event)
   - Number of downstream events affected
   - Critical events that were prevented/changed

2. **Outcome Comparison**
   | Metric | Original | Counterfactual | Difference |
   |--------|----------|----------------|------------|
   | Primary outcome occurred | | | |
   | Severity level | | | |
   | Injuries | | | |
   | Property damage | | | |
   | Response time | | | |

3. **Key Insights**
   - What does this counterfactual reveal about the incident?
   - What recommendations emerge from this analysis?

4. **Confidence in Comparison**: [0-100%]
   - Are the differences significant or within uncertainty bounds?
"""

RECOMMENDATIONS_PROMPT = """Based on the counterfactual analysis of this incident, generate actionable recommendations.

## Incident Summary
{incident_summary}

## Counterfactual Scenarios Tested
{scenarios_summary}

## Effectiveness Ranking
{effectiveness_ranking}

---

Generate recommendations in these categories:

1. **Immediate Actions** (can be implemented now)
   - What changes would have the highest impact with lowest effort?

2. **System/Technology Improvements**
   - What technical safeguards should be added or enhanced?
   - What monitoring/alerting would help?

3. **Process/Policy Changes**  
   - What procedures should be modified?
   - What training is needed?

4. **Long-term Investments**
   - What infrastructure changes are warranted?
   - What research questions need answers?

For each recommendation:
- Cite which counterfactual scenario supports it
- Estimate effectiveness (% reduction in similar incidents)
- Note any tradeoffs or costs

Prioritize recommendations by: impact × feasibility / cost
"""
