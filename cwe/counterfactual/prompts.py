"""
Prompts for counterfactual reasoning.
"""

INTERVENTION_GENERATION_PROMPT = """You are a world-class incident analyst with expertise in causal inference, safety engineering, and counterfactual reasoning.

Your task: Generate highly specific, context-aware "what-if" interventions for this incident.

## Original Incident
{incident_summary}

## Timeline Events (chronological)
{timeline_events}

## Causal Chain
{causal_chain}

## Original Outcome
{original_outcome}

---

## Your Analysis Framework

### Step 1: Identify the Causal Structure
Before proposing interventions, mentally map:
- **Root causes**: What initiated the chain of events?
- **Proximate causes**: What directly caused the harm?
- **Necessary conditions**: What HAD to be true for this outcome?
- **Sufficient conditions**: What alone was enough to cause harm?
- **Amplifying factors**: What made things worse than they could have been?

### Step 2: Find Intervention Points
Look for:
1. **Breaking the chain**: Where could a single change prevent everything downstream?
2. **Latent failures** (Reason's Swiss Cheese): What defenses were missing or failed?
3. **Timing windows**: What if something happened seconds/minutes earlier or later?
4. **Information gaps**: Who didn't know what they needed to know?
5. **Decision points**: Where did someone choose A when B would have helped?
6. **System gaps**: What automated safeguard should have existed?

### Step 3: Generate SPECIFIC Interventions
BAD (too generic): "Driver should have been more careful"
GOOD (specific): "Driver checks mirrors before lane change at T-3.2s"

BAD: "System should have better monitoring"  
GOOD: "Memory usage alert triggers at 70% instead of 90%, giving 4min extra response time"

BAD: "Better communication"
GOOD: "Cross-traffic warning display shows approaching vehicles with 3s TTC"

---

## Generate {num_interventions} Counterfactual Interventions

For each intervention, you MUST provide:

1. **Name**: Short, descriptive (e.g., "Earlier AEB Activation", "Cross-traffic Alert System")

2. **Type**: Choose the most fitting:
   - `timing_shift`: Change WHEN something happened
   - `parameter_change`: Change a value/threshold/setting  
   - `behavior_substitution`: Replace an action with a different one
   - `system_capability`: Add or enhance a technical system
   - `event_removal`: Remove an event from the timeline (test necessity)
   - `event_addition`: Add a new event (e.g., warning, alert, check)

3. **Specific Description**: Exactly what changes and by how much?

4. **Target**: Which entity or event is being modified?

5. **Hypothesis**: WHY would this change the outcome? Be mechanistic:
   - "At 60 km/h, 2s earlier braking = 33m less traveled = avoids collision point"
   - "Circuit breaker at 50% errors = isolation in 200ms vs 2000ms = prevents cascade to DB tier"

6. **Expected Impact**: low / medium / high / critical

## Quality Criteria
- ✅ Physically/technically plausible
- ✅ Specific enough to simulate
- ✅ Targets actual causal factors from this incident
- ✅ Would generate actionable recommendations
- ❌ NOT generic advice that applies to any incident
- ❌ NOT impossible ("what if gravity didn't exist")
- ❌ NOT trivial ("what if the incident didn't happen")

Think deeply about THIS incident. What SPECIFICALLY could have been different?
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
