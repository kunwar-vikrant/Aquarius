# VLM Function Reference

Complete reference for all VLM function calls used by the Counterfactual World Engine.

## Overview

The CWE uses structured function calling to enable VLMs to build timelines, establish causality, and simulate counterfactuals. Functions are organized into three categories:

1. **Timeline Construction** - Build the canonical event sequence
2. **Counterfactual Simulation** - Generate alternate timelines
3. **Utility Functions** - Supporting operations

---

## Timeline Construction Functions

### `emit_event`

Register a detected event in the timeline.

```python
def emit_event(
    timestamp: str,           # ISO8601 format
    event_type: str,          # See EventType enum
    description: str,         # Natural language description
    entities: List[dict],     # Entities involved
    confidence: float,        # 0.0 to 1.0
    evidence_refs: List[str]  # References to artifacts
) -> str:
```

**Event Types:**
- `state_change` - Entity property changed
- `action` - Deliberate action taken
- `observation` - Passive observation
- `threshold_breach` - Limit exceeded
- `communication` - Message/signal sent
- `system_event` - Automated system action
- `environmental` - External condition change
- `failure` - Component/system failure
- `recovery` - Return to normal state

**Example:**
```json
{
  "name": "emit_event",
  "arguments": {
    "timestamp": "2024-01-15T10:14:30.446Z",
    "event_type": "threshold_breach",
    "description": "ALGO-7734 detects price discontinuity in SPY: -0.33% gap exceeds momentum threshold",
    "entities": [
      {"id": "ALGO-7734", "role": "detector"},
      {"id": "SPY", "role": "subject"}
    ],
    "confidence": 0.95,
    "evidence_refs": ["trading_algo_engine.log:L45"]
  }
}
```

---

### `add_causal_link`

Establish a causal relationship between two events.

```python
def add_causal_link(
    source_event_id: str,     # UUID of cause event
    target_event_id: str,     # UUID of effect event
    relation: str,            # causes|enables|prevents|delays
    mechanism: str,           # How causation occurs
    confidence: float,        # 0.0 to 1.0
    reasoning: str            # Why this link exists
) -> str:
```

**Relation Types:**
- `causes` - Direct causation (A → B)
- `enables` - Makes possible but doesn't guarantee
- `prevents` - Blocks from occurring
- `delays` - Slows or postpones

**Example:**
```json
{
  "name": "add_causal_link",
  "arguments": {
    "source_event_id": "443b950e-7ea7-44eb-97b8-987cc21f463e",
    "target_event_id": "b0d3aac3-adec-4903-a2ca-509502f7b446",
    "relation": "causes",
    "mechanism": "Feed recovery introduced price gap triggering algo's discontinuity detection logic 1ms later",
    "confidence": 1.0,
    "reasoning": "Precise timestamps in trading_algo_engine.log: 10:14:30.445 feed recovered, 10:14:30.446 discontinuity detected"
  }
}
```

---

### `register_entity`

Track an entity (person, system, object) through the timeline.

```python
def register_entity(
    entity_id: str,          # Unique identifier
    name: str,               # Display name
    entity_type: str,        # Classification
    properties: dict         # Initial properties
) -> str:
```

**Entity Types:**
- `vehicle` - Cars, trucks, motorcycles
- `person` - Human actors
- `system` - Software systems, algorithms
- `financial_asset` - Stocks, ETFs, currencies
- `infrastructure` - Servers, networks, exchanges
- `organization` - Companies, teams
- `sensor` - Monitoring devices

**Example:**
```json
{
  "name": "register_entity",
  "arguments": {
    "entity_id": "ALGO-7734",
    "name": "APEX Momentum Strategy",
    "entity_type": "system",
    "properties": {
      "strategy": "APEX-MOMENTUM",
      "status": "ACTIVE",
      "max_position": 500000
    }
  }
}
```

---

### `update_entity_state`

Track entity state changes over time.

```python
def update_entity_state(
    entity_id: str,          # Entity to update
    timestamp: str,          # When change occurred
    property_name: str,      # Which property
    old_value: Any,          # Previous value
    new_value: Any           # New value
) -> str:
```

**Example:**
```json
{
  "name": "update_entity_state",
  "arguments": {
    "entity_id": "ALGO-7734",
    "timestamp": "2024-01-15T10:14:33.001Z",
    "property_name": "status",
    "old_value": "ACTIVE",
    "new_value": "HALTED"
  }
}
```

---

### `flag_uncertainty`

Flag areas requiring human review or additional data.

```python
def flag_uncertainty(
    context: str,                     # What is uncertain
    uncertainty_type: str,            # missing_data|ambiguous|conflicting
    description: str,                 # Details
    possible_interpretations: List[str],  # Alternatives
    data_needed: str,                 # What would resolve it
    impact: str                       # low|medium|high
) -> str:
```

**Example:**
```json
{
  "name": "flag_uncertainty",
  "arguments": {
    "context": "Signal escalation terminology",
    "uncertainty_type": "ambiguous",
    "description": "Algo log references EXTREME_SELL at 10:14:31 while postmortem reports PANIC_SELL at 10:14:32",
    "possible_interpretations": [
      "EXTREME_SELL and PANIC_SELL are synonymous",
      "Separate thresholds: STRONG → EXTREME → PANIC",
      "Timestamp discrepancy due to log rounding"
    ],
    "data_needed": "Full algo_engine.log for signal definitions",
    "impact": "low"
  }
}
```

---

### `set_timeline_bounds`

Define the temporal boundaries of the analysis.

```python
def set_timeline_bounds(
    start_time: str,         # ISO8601 start
    end_time: str,           # ISO8601 end
    confidence: float        # Bounds confidence
) -> str:
```

---

### `request_frame_analysis`

Request detailed VLM analysis of a specific video segment.

```python
def request_frame_analysis(
    video_id: str,                    # Artifact ID
    timestamp_range: Tuple[str, str], # Start, end
    query: str                        # What to analyze
) -> str:
```

---

## Counterfactual Simulation Functions

These functions are used during counterfactual scenario simulation.

### `set_divergence_point`

Mark where the alternate timeline diverges from canonical.

```python
def set_divergence_point(
    event_id: str,           # Event where divergence begins
    reason: str              # Why timeline diverges here
) -> str:
```

**Example:**
```json
{
  "name": "set_divergence_point",
  "arguments": {
    "event_id": "443b950e-7ea7-44eb-97b8-987cc21f463e",
    "reason": "Price gap filter activates, pausing trading instead of generating STRONG_SELL signal"
  }
}
```

---

### `emit_alternative_event`

Add an event to the alternate timeline.

```python
def emit_alternative_event(
    timestamp: str,          # When event occurs
    description: str,        # What happens differently
    entities: List[str],     # Entities involved
    replaces_event: str,     # Original event ID (optional)
    confidence: float        # Likelihood
) -> str:
```

**Example:**
```json
{
  "name": "emit_alternative_event",
  "arguments": {
    "timestamp": "2024-01-15T10:14:30.446Z",
    "description": "Gap filter detects 0.33% discontinuity, initiates 30-second trading pause",
    "entities": ["ALGO-7734", "GAP_FILTER"],
    "replaces_event": "b0d3aac3-adec-4903-a2ca-509502f7b446",
    "confidence": 0.92
  }
}
```

---

### `assess_outcome`

Evaluate the outcome of the alternate timeline.

```python
def assess_outcome(
    outcome_occurred: bool,        # Did bad outcome happen?
    severity: str,                 # none|minor|moderate|severe|catastrophic|fatal
    description: str,              # Outcome description
    prevented_events: List[str],   # Events that didn't happen
    key_differences: List[str]     # Major changes
) -> str:
```

**Example:**
```json
{
  "name": "assess_outcome",
  "arguments": {
    "outcome_occurred": false,
    "severity": "none",
    "description": "Flash crash completely prevented. Algo paused during feed recovery, resumed with valid data.",
    "prevented_events": [
      "STRONG_SELL signal generation",
      "Order burst (12 orders)",
      "LULD halts",
      "Kill switch activation"
    ],
    "key_differences": [
      "Trading paused for 30 seconds during feed recovery",
      "No position changes during pause",
      "Normal operations resumed at 10:15:00"
    ]
  }
}
```

---

### `complete_simulation`

Signal that the simulation is complete.

```python
def complete_simulation(
    summary: str,            # Brief summary
    confidence: float        # Overall confidence
) -> str:
```

---

### `propose_intervention`

Used during intervention generation to suggest counterfactuals.

```python
def propose_intervention(
    description: str,              # What to change
    intervention_type: str,        # timing|parameter|behavior|system|human_action
    target_event_id: str,          # Event to modify (optional)
    target_entity_id: str,         # Entity to modify (optional)
    rationale: str,                # Why this might help
    expected_effect: str,          # Predicted outcome
    feasibility: float,            # 0.0 to 1.0
    domain_relevance: float        # Domain-appropriateness
) -> str:
```

**Intervention Types:**
- `timing` - "What if X happened earlier/later?"
- `parameter_change` - "What if threshold was different?"
- `behavior_substitution` - "What if they did Y instead?"
- `system_capability` - "What if system had feature Z?"
- `human_action` - "What if person intervened?"

---

## Function Call Flow

### Timeline Construction

```
┌─────────────────────────────────────────────────────────┐
│                    COARSE PASS                          │
│  register_entity() × N                                  │
│  emit_event() × M (major events only)                   │
│  flag_uncertainty() (ambiguous data)                    │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                   DETAILED PASS                         │
│  emit_event() (fill intermediate events)                │
│  update_entity_state() (track changes)                  │
│  request_frame_analysis() (video detail)                │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                    CAUSAL PASS                          │
│  add_causal_link() × K                                  │
│  flag_uncertainty() (conflicting evidence)              │
└─────────────────────────────────────────────────────────┘
```

### Counterfactual Simulation

```
┌─────────────────────────────────────────────────────────┐
│              Per Intervention Scenario                   │
│                                                         │
│  1. set_divergence_point()                              │
│  2. emit_alternative_event() × N                        │
│  3. assess_outcome()                                    │
│  4. complete_simulation()                               │
└─────────────────────────────────────────────────────────┘
```

---

## Schema Definitions

Full JSON Schema for function definitions is in `cwe/reasoning/function_schema.py`.

### Function Schema Structure

```python
{
    "name": "function_name",
    "description": "What the function does",
    "parameters": {
        "type": "object",
        "properties": {
            "param1": {"type": "string", "description": "..."},
            "param2": {"type": "number", "description": "..."}
        },
        "required": ["param1"]
    }
}
```

---

## Provider-Specific Notes

### Gemini
- Uses native function calling with `tools` parameter
- Supports parallel function calls
- Best for video frame analysis

### xAI (Grok)
- OpenAI-compatible API
- Requires `tool_choice: "auto"` for function calling
- Message content must not be null when role is "assistant"

### Claude
- Uses `tools` with `tool_use` blocks
- Returns tool calls in content array
- Best for reasoning-heavy operations

### OpenAI (GPT-4)
- Standard function calling interface
- Supports `tool_choice: "required"` to force calls
- Good balance of speed and quality
