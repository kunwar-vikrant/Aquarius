# Multi-Pass Timeline Construction

Detailed documentation of the three-pass timeline construction algorithm used by the Counterfactual World Engine.

## Overview

The CWE builds incident timelines through a **three-pass reasoning process**:

1. **Coarse Pass** - Identify major events and entities
2. **Detailed Pass** - Fill intermediate events, update entity states
3. **Causal Pass** - Establish cause-effect relationships

Each pass uses VLM function calling to incrementally build the timeline.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   TIMELINE CONSTRUCTION                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Artifacts ──► Coarse Pass ──► Detailed Pass ──► Causal Pass │
│                    │                │                │       │
│                    ▼                ▼                ▼       │
│              Major Events    All Events      Causal Links    │
│              Entity IDs      State Changes   Mechanisms      │
│              Time Bounds     Evidence Refs   Confidence      │
│                                                             │
│  Output: Complete Timeline with Causal Graph                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Pass 1: Coarse Pass

### Purpose

Establish the skeleton of the incident:
- Identify all entities involved
- Extract major events (not every detail)
- Set timeline boundaries
- Flag obvious uncertainties

### VLM Functions Used

| Function | Purpose |
|----------|---------|
| `register_entity` | Track entities through the timeline |
| `emit_event` | Log major events only |
| `set_timeline_bounds` | Define start/end times |
| `flag_uncertainty` | Mark ambiguous data |

### Prompt Structure

```python
COARSE_PASS_PROMPT = """
Perform a COARSE PASS analysis of this incident.

Your objectives:
1. REGISTER all entities mentioned (vehicles, people, systems, etc.)
2. EMIT major events only - the key moments that define the incident
3. SET timeline bounds based on earliest and latest timestamps
4. FLAG any uncertainties or ambiguities in the data

Do NOT try to capture every detail - focus on the skeleton.
Do NOT establish causal relationships yet - that comes later.

Available functions:
- register_entity(entity_id, name, entity_type, properties)
- emit_event(timestamp, event_type, description, entities, confidence, evidence_refs)
- set_timeline_bounds(start_time, end_time, confidence)
- flag_uncertainty(context, uncertainty_type, description, ...)
"""
```

### Iteration Loop

```python
async def _run_coarse_pass(self) -> None:
    """
    Run coarse pass with iterative refinement.
    
    - Max 20 iterations
    - Continue until VLM returns text (no more function calls)
    - Accumulate results in timeline builder
    """
    for iteration in range(MAX_ITERATIONS):
        response = await self.provider.generate(
            messages=self.context,
            functions=COARSE_FUNCTIONS
        )
        
        if response.function_calls:
            # Process each function call
            for call in response.function_calls:
                self._handle_function_call(call)
            # Add VLM response to context for next iteration
            self.context.append(response_message)
        else:
            # VLM returned text = pass complete
            break
```

### Output

After coarse pass, the timeline contains:
- Entity registry with IDs and types
- 10-20 major events (not comprehensive)
- Timeline bounds
- Uncertainty flags

---

## Pass 2: Detailed Pass

### Purpose

Fill in the gaps identified in the coarse pass:
- Add intermediate events between major events
- Track entity state changes over time
- Request detailed analysis of ambiguous areas
- Link events to specific evidence

### VLM Functions Used

| Function | Purpose |
|----------|---------|
| `emit_event` | Add intermediate events |
| `update_entity_state` | Track property changes |
| `request_frame_analysis` | Request video detail |
| `flag_uncertainty` | Mark remaining unknowns |

### Prompt Structure

```python
DETAILED_PASS_PROMPT = """
Perform a DETAILED PASS analysis.

You have the coarse timeline from Pass 1. Now:

1. FILL GAPS: Add intermediate events between the major events
2. TRACK STATE: Update entity states as they change
3. REQUEST DETAIL: Ask for video frame analysis if needed
4. EVIDENCE: Link events to specific log entries, frames, etc.

Current timeline:
{coarse_timeline_summary}

Focus on:
- Events that bridge the gaps between major events
- State changes that led to incidents
- Evidence that supports each event
"""
```

### Context Prioritization

During detailed pass, context is prioritized:

```python
class ContextPriority(Enum):
    CRITICAL = 1  # Must include (active window)
    HIGH = 2      # Include if space (±5 min of incident)
    MEDIUM = 3    # Include if space (related events)
    LOW = 4       # Summarize or omit
```

The context manager ensures:
1. Recent events always in context
2. Flagged uncertainties prioritized
3. Low-priority content summarized

### Output

After detailed pass:
- Comprehensive event sequence (30-50 events typical)
- Entity state tracking over time
- Evidence references for each event
- Remaining uncertainties flagged

---

## Pass 3: Causal Pass

### Purpose

Establish cause-effect relationships:
- Link events with causal relationships
- Identify root causes
- Determine causal mechanisms
- Assign confidence to each link

### VLM Functions Used

| Function | Purpose |
|----------|---------|
| `add_causal_link` | Establish cause→effect |
| `flag_uncertainty` | Mark uncertain causality |

### Causal Relations

```python
class CausalRelation(str, Enum):
    CAUSES = "causes"       # A directly causes B
    ENABLES = "enables"     # A makes B possible
    PREVENTS = "prevents"   # A stops B from happening
    DELAYS = "delays"       # A slows down B
```

### Prompt Structure

```python
CAUSAL_PASS_PROMPT = """
Perform a CAUSAL PASS analysis.

You have the complete event timeline. Now establish causal relationships.

For each relationship:
1. SOURCE: Which event is the cause?
2. TARGET: Which event is the effect?
3. RELATION: causes / enables / prevents / delays
4. MECHANISM: HOW does the cause lead to the effect?
5. CONFIDENCE: How certain is this relationship? (0-1)
6. REASONING: What evidence supports this?

Current timeline:
{detailed_timeline_summary}

Consider:
- Direct causation (A caused B)
- Enabling conditions (A made B possible)
- Prevention (A stopped B)
- Timing effects (A delayed B)

Identify the ROOT CAUSES - events with no incoming causal links.
"""
```

### Causal Link Example

```python
{
    "name": "add_causal_link",
    "arguments": {
        "source_event_id": "feed-latency-spike",
        "target_event_id": "price-gap-detection",
        "relation": "causes",
        "mechanism": "Feed latency of 1500ms caused stale data; on recovery, the price gap (-0.33%) was interpreted as momentum signal",
        "confidence": 1.0,
        "reasoning": "Timestamps in market_data_feed.log show 28.500 latency spike, 30.445 recovery with gap; algo_engine.log shows 30.446 gap detection"
    }
}
```

### Output

After causal pass:
- Complete causal graph
- Root cause identification
- Causal mechanisms documented
- Confidence scores for all links

---

## State Management

### Timeline Builder

```python
class TimelineBuilder:
    """Accumulates timeline data across passes."""
    
    def __init__(self, incident_id: UUID):
        self.incident_id = incident_id
        self.entities: dict[str, Entity] = {}
        self.events: list[Event] = []
        self.causal_links: list[CausalLink] = []
        self.uncertainties: list[Uncertainty] = []
        self.timeline_bounds: tuple[datetime, datetime] | None = None
    
    def register_entity(self, **kwargs) -> str:
        """Add entity to registry."""
        
    def emit_event(self, **kwargs) -> str:
        """Add event to timeline."""
        
    def add_causal_link(self, **kwargs) -> str:
        """Add causal relationship."""
        
    def build(self) -> Timeline:
        """Construct final Timeline object."""
```

### Function Call Handler

```python
def _handle_function_call(self, call: FunctionCall) -> str:
    """
    Route function call to appropriate handler.
    
    Returns result message to feed back to VLM.
    """
    handlers = {
        "register_entity": self.builder.register_entity,
        "emit_event": self.builder.emit_event,
        "add_causal_link": self.builder.add_causal_link,
        "update_entity_state": self.builder.update_entity_state,
        "flag_uncertainty": self.builder.flag_uncertainty,
        "set_timeline_bounds": self.builder.set_timeline_bounds,
        "request_frame_analysis": self._request_frame_analysis,
    }
    
    handler = handlers.get(call.name)
    if handler:
        return handler(**call.arguments)
    else:
        return f"Unknown function: {call.name}"
```

---

## Context Management

### Context Window

Each pass manages a context window:

```python
class ContextWindow:
    """Token-limited context management."""
    
    def __init__(self, max_tokens: int = 100000):
        self.max_tokens = max_tokens
        self.chunks: list[ContextChunk] = []
    
    def add(self, chunk: ContextChunk) -> None:
        """Add chunk, evicting low-priority if needed."""
        self._ensure_space(chunk.estimated_tokens)
        self.chunks.append(chunk)
    
    def _ensure_space(self, needed: int) -> None:
        """Evict low-priority chunks to make room."""
        while self._total_tokens() + needed > self.max_tokens:
            # Find lowest priority chunk
            lowest = min(self.chunks, key=lambda c: c.priority.value)
            if lowest.priority == ContextPriority.CRITICAL:
                raise ContextOverflowError()
            self.chunks.remove(lowest)
```

### Token Estimation

```python
def _estimate_tokens(self, text: str) -> int:
    """
    Estimate token count for text.
    
    Uses simple heuristic: ~4 characters per token.
    More accurate than word count for code/logs.
    """
    return len(text) // 4
```

### Summarization

When context overflows, low-priority content is summarized:

```python
async def summarize_chunk(self, chunk: ContextChunk) -> ContextChunk:
    """Use VLM to summarize a context chunk."""
    response = await self.provider.generate(
        messages=[
            Message.system("Summarize this incident data concisely."),
            Message.user(chunk.content)
        ]
    )
    return ContextChunk(
        content=response.content,
        priority=chunk.priority,
        is_summary=True,
        estimated_tokens=self._estimate_tokens(response.content)
    )
```

---

## Confidence Calculation

### Timeline Confidence

```python
def calculate_timeline_confidence(self, timeline: Timeline) -> float:
    """
    Calculate overall timeline confidence.
    
    Formula:
    confidence = (
        event_confidence * 0.4 +
        causal_confidence * 0.4 +
        completeness * 0.2
    )
    
    Where:
    - event_confidence = average of all event confidence scores
    - causal_confidence = average of all causal link scores
    - completeness = 1 - (uncertainty_count / event_count)
    """
    event_conf = mean([e.confidence for e in timeline.events])
    causal_conf = mean([l.confidence for l in timeline.causal_links])
    completeness = 1 - (len(timeline.uncertainties) / len(timeline.events))
    
    return event_conf * 0.4 + causal_conf * 0.4 + completeness * 0.2
```

---

## Example Execution

### Sample Log Output

```
2024-01-29 00:52:42 [info] Starting timeline construction incident_id=4c69457d
2024-01-29 00:52:42 [info] Running coarse pass
2024-01-29 00:52:55 [debug] VLM response function_calls=12 iteration=1
2024-01-29 00:52:55 [debug] Handling function call name=register_entity
2024-01-29 00:52:55 [debug] Handling function call name=register_entity
2024-01-29 00:52:55 [debug] Handling function call name=emit_event
... (more function calls)
2024-01-29 00:53:10 [debug] VLM response function_calls=8 iteration=2
2024-01-29 00:53:25 [debug] VLM response function_calls=0 iteration=3
2024-01-29 00:53:25 [info] Coarse pass complete entities=14 events=15

2024-01-29 00:53:25 [info] Running detailed pass
2024-01-29 00:53:45 [debug] VLM response function_calls=24 iteration=1
... (more iterations)
2024-01-29 00:54:04 [info] Detailed pass complete events=39

2024-01-29 00:54:04 [info] Running causal pass
2024-01-29 00:55:23 [debug] VLM response function_calls=16 iteration=1
2024-01-29 00:55:34 [debug] VLM response function_calls=0 iteration=2
2024-01-29 00:55:34 [info] Causal pass complete causal_links=33

2024-01-29 00:55:34 [info] Timeline construction complete
    entities=14 events=39 causal_links=33 confidence=0.995
```

### Metrics Summary

| Pass | Iterations | Function Calls | Duration |
|------|------------|----------------|----------|
| Coarse | 3 | ~25 | 45s |
| Detailed | 2 | ~30 | 40s |
| Causal | 2 | ~20 | 90s |
| **Total** | **7** | **~75** | **~3 min** |

---

## Tuning Parameters

### Pass Configuration

```python
# In cwe/alignment/temporal.py

MAX_ITERATIONS = 20        # Per pass, safety limit
COARSE_TEMPERATURE = 0.2   # Low for consistency
DETAILED_TEMPERATURE = 0.3 # Slightly higher for detail
CAUSAL_TEMPERATURE = 0.2   # Low for accurate causality
```

### Quality vs Speed Tradeoffs

| Parameter | Higher Value | Lower Value |
|-----------|--------------|-------------|
| MAX_ITERATIONS | More complete | Faster |
| TEMPERATURE | More creative | More consistent |
| max_tokens | More detail | Less cost |

---

## Error Handling

### Iteration Limits

```python
if iteration >= MAX_ITERATIONS:
    logger.warning(f"Pass {pass_name} hit iteration limit")
    # Continue with partial results
```

### Invalid Function Calls

```python
try:
    result = handler(**call.arguments)
except TypeError as e:
    logger.error(f"Invalid arguments for {call.name}: {e}")
    result = f"Error: {e}"
except Exception as e:
    logger.error(f"Function {call.name} failed: {e}")
    result = f"Error: {e}"
```

### Context Overflow

```python
try:
    context_window.add(new_chunk)
except ContextOverflowError:
    # Summarize existing content
    await context_window.summarize_lowest_priority()
    context_window.add(new_chunk)
```
