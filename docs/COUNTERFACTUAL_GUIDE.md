# Counterfactual Analysis Guide

Complete guide to the counterfactual simulation system in the Counterfactual World Engine.

## Overview

The counterfactual analysis system answers "what if" questions by:

1. Generating candidate interventions
2. Simulating alternate timelines
3. Assessing outcome differences
4. Ranking intervention effectiveness

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  COUNTERFACTUAL PIPELINE                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────┐      ┌─────────────────────────────┐  │
│  │ Canonical       │      │ Intervention Generator       │  │
│  │ Timeline        │─────►│ - Domain-specific templates  │  │
│  │ (from reasoner) │      │ - VLM-generated creative     │  │
│  └─────────────────┘      └─────────────────────────────┘  │
│                                      │                      │
│                                      ▼                      │
│                           ┌─────────────────────────────┐  │
│                           │ Counterfactual Simulator     │  │
│                           │ - Divergence identification  │  │
│                           │ - Alternate event generation │  │
│                           │ - Outcome assessment         │  │
│                           └─────────────────────────────┘  │
│                                      │                      │
│                                      ▼                      │
│                           ┌─────────────────────────────┐  │
│                           │ Effectiveness Ranker         │  │
│                           │ - Score calculation          │  │
│                           │ - Recommendation generation  │  │
│                           └─────────────────────────────┘  │
│                                      │                      │
│                                      ▼                      │
│                           ┌─────────────────────────────┐  │
│                           │ Report Generator             │  │
│                           │ - Markdown output            │  │
│                           │ - Executive summary          │  │
│                           └─────────────────────────────┘  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Intervention Generation

### Domain-Specific Standard Interventions

The system provides pre-defined interventions optimized for each domain:

#### Traffic/Vehicle Domain

| Intervention | Type | Description |
|--------------|------|-------------|
| Earlier braking | `timing` | Driver applies brakes 2 seconds earlier |
| Lower AEB threshold | `parameter_change` | AEB triggers at 3.0s TTC instead of 2.5s |
| Active lane keep assist | `system_capability` | System prevents lane departure |
| Increased following distance | `behavior_substitution` | 3-second gap maintained |
| Driver attention monitoring | `system_capability` | Distraction alert system active |

```python
# From cwe/counterfactual/generator.py
def _traffic_interventions(self) -> list[Intervention]:
    return [
        Intervention(
            description="Driver applies brakes 2 seconds earlier",
            intervention_type=InterventionType.TIMING,
            rationale="Earlier recognition could provide stopping distance",
            expected_effect="Collision avoided or severity reduced",
            feasibility=0.7,
        ),
        # ... more interventions
    ]
```

#### DevOps/Infrastructure Domain

| Intervention | Type | Description |
|--------------|------|-------------|
| Lower circuit breaker | `parameter_change` | Trips at 50% error rate vs 80% |
| Memory limits | `parameter_change` | 80% memory cap with auto-restart |
| Auto-rollback | `system_capability` | Automatic rollback on error spike |
| Staged deployment | `timing` | Canary deploy with 30-min bake |
| Enhanced monitoring | `system_capability` | Early warning on connection pool exhaustion |

#### Financial/Trading Domain

| Intervention | Type | Description |
|--------------|------|-------------|
| Price gap filter | `system_capability` | Pause trading on feed recovery discontinuity |
| Lower kill switch | `parameter_change` | Threshold reduced from $2.5M to $1M |
| Human confirmation | `system_capability` | Require approval for extreme signals |
| No soft limit bypass | `parameter_change` | Critical urgency cannot bypass throttle |
| Limit orders only | `behavior_substitution` | Use LIMIT IOC instead of MARKET orders |

### VLM-Generated Interventions

For creative exploration, the VLM can generate novel interventions:

```python
# Prompt template for intervention generation
INTERVENTION_GENERATION_PROMPT = """
Analyze this incident timeline and propose interventions that could have 
prevented or mitigated the outcome.

Consider:
1. Prevention Points - Events that if changed would break the causal chain
2. Timing Sensitivity - Where small timing changes could help
3. Missing Safeguards - What controls should have existed
4. Human Factors - How human actions could have differed
5. Environmental Changes - External factors that could help

For each intervention, assess:
- Feasibility (0-1): How realistic is this change?
- Domain relevance (0-1): How appropriate for this domain?
"""
```

---

## Simulation Algorithm

### Core Simulation Loop

```python
async def simulate_single_scenario(
    self,
    timeline: Timeline,
    intervention: Intervention
) -> CounterfactualScenario:
    """
    Simulate one intervention scenario.
    
    Algorithm:
    1. Build context with canonical timeline + intervention
    2. Initialize simulation state
    3. Loop (max 10 iterations):
       a. Send context to VLM
       b. Process function calls:
          - set_divergence_point → Mark where timeline changes
          - emit_alternative_event → Add alternate events
          - assess_outcome → Evaluate result
          - complete_simulation → End loop
       c. Update context with VLM response
    4. Build CounterfactualScenario from accumulated state
    """
```

### Simulation Functions

The VLM uses 4 specialized functions during simulation:

1. **`set_divergence_point`** - Identifies where the alternate timeline branches
2. **`emit_alternative_event`** - Adds events to the alternate timeline
3. **`assess_outcome`** - Evaluates whether the bad outcome still occurs
4. **`complete_simulation`** - Signals simulation is complete

### Simulation Context

The VLM receives:
- Full canonical timeline
- Causal graph
- Intervention description
- Domain context

```python
COUNTERFACTUAL_SIMULATION_PROMPT = """
You are simulating an alternate timeline where the following intervention 
is applied to the incident:

INTERVENTION: {intervention.description}
TYPE: {intervention.intervention_type}

Given the canonical timeline and causal relationships, determine:
1. The divergence point - where does the timeline first change?
2. What alternative events occur?
3. Does the catastrophic outcome still happen?
4. What is the severity in the alternate timeline?

Be realistic about cascading effects. Consider:
- Physical constraints (can't violate physics)
- System behavior (how would systems actually respond?)
- Human factors (realistic human reactions)
- Timing constraints (events take time)
"""
```

---

## Effectiveness Scoring

### Score Formula

Interventions are ranked using a **multiplicative scoring function** that ensures all factors must be favorable for a high score:

$$\text{Score}(I) = E(I) \times F(I) \times C(I)$$

| Factor | Symbol | Range | Description |
|--------|--------|-------|-------------|
| **Effectiveness** | $E(I)$ | [0, 1] | How much harm is prevented/reduced? |
| **Feasibility** | $F(I)$ | [0, 1] | Can we realistically implement this? |
| **Confidence** | $C(I)$ | [0, 1] | How certain is the simulation? |

This multiplicative structure ensures:
- **Impossible interventions score 0** — even perfect prevention is worthless if $F(I) = 0$
- **Ineffective interventions score 0** — high feasibility doesn't help if $E(I) = 0$
- **Uncertain simulations are penalized** — low confidence reduces score proportionally

```python
def _rank_interventions(self, scenarios: list[CounterfactualScenario]) -> list[dict]:
    """
    Score = Effectiveness × Feasibility × Confidence
    """
    for scenario in scenarios:
        # E: Effectiveness [0-1] - normalized harm reduction
        effectiveness = self._calculate_effectiveness(scenario)
        
        # F: Feasibility [0-1] - can we actually implement this?
        feasibility = intervention.feasibility
        
        # C: Confidence [0-1] - how sure is the simulation?
        confidence = outcome.confidence
        
        # Multiplicative: all factors must be good
        score = effectiveness * feasibility * confidence
```

### Effectiveness Calculation

Effectiveness measures normalized harm reduction:

- **Full prevention**: $E = 1.0$ (outcome completely avoided)
- **Partial reduction**: $E = \frac{\text{original severity} - \text{new severity}}{\text{original severity}}$
- **No improvement**: $E = 0.0$

Example: CATASTROPHIC (4) → MINOR (1) = $(4-1)/4 = 0.75$

### Feasibility Guidelines

| Score | Description | Example |
|-------|-------------|---------|
| 0.0 - 0.2 | Impossible or unrealistic | "Car should have teleported" |
| 0.3 - 0.5 | Requires major changes | Cultural/behavioral shifts |
| 0.6 - 0.7 | Moderate effort | Policy changes, training |
| 0.8 - 0.9 | Easy to implement | Configuration changes |
| 1.0 | Trivial | Already supported, just enable |

### Severity Levels

| Level | Value | Description |
|-------|-------|-------------|
| NONE | 0 | No negative outcome |
| MINOR | 1 | Small impact, easily recoverable |
| MODERATE | 2 | Significant but manageable |
| SEVERE | 3 | Major impact, difficult recovery |
| CATASTROPHIC | 4 | Extreme impact, long-term effects |
| FATAL | 5 | Loss of life |

### Ranking Output

```python
@dataclass
class RankedIntervention:
    intervention: Intervention
    scenario: CounterfactualScenario
    score: float
    rank: int
    prevented: bool
    severity_change: str  # e.g., "catastrophic → minor"
    confidence: float
```

---

## Intervention Types

### Timing Interventions

Change **when** something happens.

```python
InterventionType.TIMING
```

**Examples:**
- "Driver brakes 2 seconds earlier"
- "Deploy update 30 minutes later"
- "Kill switch triggers 500ms sooner"

**Simulation considerations:**
- Must respect physical constraints (human reaction time ~250ms)
- Consider cascading timing effects
- Account for system latencies

### Parameter Change Interventions

Modify system **thresholds or limits**.

```python
InterventionType.PARAMETER_CHANGE
```

**Examples:**
- "Circuit breaker at 50% instead of 80%"
- "Kill switch threshold reduced to $1M"
- "Memory limit set to 80%"

**Simulation considerations:**
- Would the change have affected the incident?
- Are there side effects (false positives)?
- Was the parameter value the actual cause?

### Behavior Substitution Interventions

Replace one action with another.

```python
InterventionType.BEHAVIOR_SUBSTITUTION
```

**Examples:**
- "Use LIMIT orders instead of MARKET orders"
- "Take highway exit instead of continuing"
- "Scale horizontally instead of vertically"

**Simulation considerations:**
- Is the substitute behavior realistic?
- Does it address the root cause?
- What are the tradeoffs?

### System Capability Interventions

Add features that didn't exist.

```python
InterventionType.SYSTEM_CAPABILITY
```

**Examples:**
- "Price gap filter pauses trading on feed recovery"
- "Automatic emergency braking system"
- "Circuit breaker on database connections"

**Simulation considerations:**
- Would the feature have detected the problem?
- Fast enough to prevent cascade?
- Any false positive concerns?

### Human Action Interventions

Modify human behavior or decisions.

```python
InterventionType.HUMAN_ACTION
```

**Examples:**
- "Operator acknowledges alert within 30 seconds"
- "Driver maintains attention on road"
- "Trader manually overrides algorithm"

**Simulation considerations:**
- Realistic human response times
- Cognitive load and attention
- Training and experience factors

---

## Report Generation

### Report Structure

```markdown
# Counterfactual Analysis Report

## Executive Summary
- Incident: [name]
- Scenarios analyzed: [N]
- Prevention rate: [X/N]

## Intervention Effectiveness Ranking
| Rank | Intervention | Score | Prevented? | Severity Change |
|------|-------------|-------|------------|-----------------|
| 1    | ...         | 239   | ✅ YES     | catastrophic→none |

## Key Findings
- Finding 1
- Finding 2

## Recommendations
1. Recommendation with rationale
2. ...

## Scenario Details
### Scenario 1: [Intervention Name]
- Divergence point
- Alternative events
- Outcome assessment
```

### Report Generation Code

```python
# From cwe/counterfactual/report.py
class CounterfactualReportGenerator:
    def generate_report(
        self,
        incident: Incident,
        timeline: Timeline,
        analysis: CounterfactualAnalysis
    ) -> str:
        """Generate markdown report from analysis results."""
        sections = [
            self._executive_summary(incident, analysis),
            self._ranking_table(analysis.ranked_interventions),
            self._key_findings(analysis.findings),
            self._recommendations(analysis.recommendations),
            self._scenario_details(analysis.scenarios)
        ]
        return "\n\n".join(sections)
```

---

## Full Analysis Pipeline

### `run_full_analysis()` Method

```python
async def run_full_analysis(
    self,
    timeline: Timeline,
    interventions: list[Intervention] | None = None,
    generate_interventions: bool = True,
    max_generated: int = 5
) -> CounterfactualAnalysis:
    """
    Run complete counterfactual analysis.
    
    Steps:
    1. Generate interventions (if not provided)
       - Standard domain interventions
       - VLM-generated creative interventions
    2. Simulate each intervention scenario
       - Run simulation loop
       - Collect alternate events
       - Assess outcomes
    3. Rank interventions by effectiveness
    4. Generate findings and recommendations
    5. Build analysis report
    """
```

### Execution Flow

```
                    ┌──────────────────┐
                    │ Timeline + Graph │
                    └────────┬─────────┘
                             │
              ┌──────────────┴──────────────┐
              ▼                              ▼
┌─────────────────────────┐    ┌─────────────────────────┐
│ Standard Interventions   │    │ VLM-Generated           │
│ (5 per domain)          │    │ (up to 5 creative)      │
└───────────┬─────────────┘    └───────────┬─────────────┘
            │                              │
            └──────────────┬───────────────┘
                           │
                           ▼
              ┌─────────────────────────┐
              │ For each intervention:  │
              │ 1. Build context        │
              │ 2. Run simulation loop  │
              │ 3. Collect results      │
              └───────────┬─────────────┘
                          │
                          ▼
              ┌─────────────────────────┐
              │ Calculate scores        │
              │ Sort by effectiveness   │
              │ Generate findings       │
              └───────────┬─────────────┘
                          │
                          ▼
              ┌─────────────────────────┐
              │ CounterfactualAnalysis  │
              │ - scenarios             │
              │ - ranked_interventions  │
              │ - findings              │
              │ - recommendations       │
              └─────────────────────────┘
```

---

## Configuration

### Simulation Parameters

```python
# In cwe/counterfactual/simulator.py
MAX_ITERATIONS = 10      # Per scenario
TEMPERATURE = 0.3        # Low for deterministic simulation
MAX_TOKENS = 4096        # Per VLM call
```

### Intervention Generation

```python
# In cwe/counterfactual/generator.py
MAX_VLM_INTERVENTIONS = 5    # Creative interventions
INTERVENTION_TEMPERATURE = 0.7  # Higher for creativity
```

---

## Example Usage

### Programmatic

```python
from cwe.counterfactual.simulator import CounterfactualSimulator
from cwe.counterfactual.generator import InterventionGenerator

# Initialize
simulator = CounterfactualSimulator(vlm_provider)
generator = InterventionGenerator(vlm_provider)

# Generate interventions
interventions = generator.generate_standard_interventions("financial")

# Run simulation
analysis = await simulator.run_full_analysis(
    timeline=canonical_timeline,
    interventions=interventions
)

# Access results
for ranked in analysis.ranked_interventions:
    print(f"{ranked.rank}. {ranked.intervention.description}")
    print(f"   Score: {ranked.score}, Prevented: {ranked.prevented}")
```

### CLI

```bash
# Run with standard interventions
cwe analyze --incident ./test_data/incident_001/ --counterfactual

# Run with VLM-generated interventions
cwe analyze --incident ./test_data/incident_001/ --counterfactual --generate-interventions

# Specify domain
cwe analyze --incident ./test_data/incident_001/ --counterfactual --domain financial
```

---

## Best Practices

### Intervention Design

1. **Target root causes** - Interventions addressing root cause (like missing gap filter) score highest
2. **Consider feasibility** - Unrealistic interventions provide less actionable insight
3. **Domain appropriateness** - Use domain-specific templates when available
4. **Multiple layers** - Test interventions at different points in the causal chain

### Interpretation

1. **Prevention vs Mitigation** - Some interventions prevent entirely, others reduce severity
2. **Confidence levels** - Lower confidence means more uncertainty in simulation
3. **Side effects** - Consider unintended consequences not captured in simulation
4. **Implementation cost** - Easiest-to-implement interventions may be most valuable

### Validation

1. **Compare against known outcomes** - Use incidents with known counterfactual outcomes
2. **Expert review** - Have domain experts validate recommendations
3. **Sensitivity analysis** - Test how results change with different parameters
