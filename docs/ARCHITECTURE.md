# Counterfactual World Engine - System Architecture

## Overview

The Counterfactual World Engine (CWE) is an agentic system that reconstructs and simulates alternate realities from real-world incident artifacts using Vision Language Models (VLMs) as the core reasoning engine.

## Related Documentation

| Document | Description |
|----------|-------------|
| [VLM Function Reference](VLM_FUNCTION_REFERENCE.md) | Complete API reference for all VLM function calls |
| [Counterfactual Guide](COUNTERFACTUAL_GUIDE.md) | Simulation algorithm, intervention types, scoring |
| [Timeline Construction](TIMELINE_CONSTRUCTION.md) | Multi-pass reasoning algorithm details |
| [Providers Guide](PROVIDERS.md) | VLM provider configuration (Gemini, Grok, Claude, GPT-4) |
| [Design Decisions](DECISIONS.md) | Architectural decision records |

## Core Capabilities

1. **Multimodal Ingestion** - Ingest video, logs, reports, sensor data
2. **Temporal Alignment** - Synchronize multimodal streams to unified timeline
3. **Causal Graph Construction** - Extract cause-effect relationships
4. **Counterfactual Simulation** - "What if" scenario exploration
5. **Visual Synthesis** - Overlay trajectories and alternate outcomes
6. **Autonomous Exploration** - Agent-driven hypothesis generation and ranking

---

## System Components

### 1. Ingestion Layer (`cwe/ingestion/`)

```
┌─────────────────────────────────────────────────────────────┐
│                    INGESTION LAYER                          │
├─────────────┬─────────────┬─────────────┬──────────────────┤
│ Video       │ Log Parser  │ Report      │ Sensor/Telemetry │
│ Processor   │             │ Extractor   │ Adapter          │
├─────────────┴─────────────┴─────────────┴──────────────────┤
│              Unified Artifact Schema                        │
└─────────────────────────────────────────────────────────────┘
```

**Components:**
- `video_processor.py` - Frame extraction, keyframe detection, scene segmentation
- `log_parser.py` - Multi-format log parsing (syslog, JSON, custom)
- `report_extractor.py` - PDF/document text extraction with structure preservation
- `sensor_adapter.py` - Telemetry/sensor data normalization
- `artifact_schema.py` - Unified data model for all artifact types

**Improvement Suggestion:** Add a **streaming ingestion mode** for real-time incident monitoring, not just post-hoc analysis.

---

### 2. Temporal Alignment Engine (`cwe/alignment/`)

```
┌─────────────────────────────────────────────────────────────┐
│              TEMPORAL ALIGNMENT ENGINE                       │
├─────────────────────────────────────────────────────────────┤
│  Video Frames ──┐                                           │
│  Log Entries ───┼──► Timestamp Normalizer ──► Unified       │
│  Report Events ─┤                            Timeline       │
│  Sensor Data ───┘                                           │
├─────────────────────────────────────────────────────────────┤
│  Cross-Modal Anchor Detection (VLM-assisted)                │
│  - Visual event detection ←→ Log correlation                │
│  - OCR timestamps from video ←→ System timestamps           │
└─────────────────────────────────────────────────────────────┘
```

**Key Challenge:** Clock drift, timezone mismatches, missing timestamps.

**Solution:** VLM-assisted anchor point detection - use visual cues (clocks in frame, state changes) correlated with log events to establish temporal anchors.

---

### 3. VLM Reasoning Core (`cwe/reasoning/`)

```
┌─────────────────────────────────────────────────────────────┐
│                   VLM REASONING CORE                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Context Window Manager                  │   │
│  │  - Chunking strategy for long incidents              │   │
│  │  - Priority-based context allocation                 │   │
│  │  - Rolling window for extended timelines             │   │
│  └─────────────────────────────────────────────────────┘   │
│                           │                                 │
│                           ▼                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              VLM Provider Abstraction                │   │
│  │  - Gemini 2.5 Pro / Flash                           │   │
│  │  - Grok (xAI)                                       │   │
│  │  - Claude (Anthropic)                               │   │
│  │  - GPT-4o (OpenAI)                                  │   │
│  └─────────────────────────────────────────────────────┘   │
│                           │                                 │
│                           ▼                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Function Call Schema                    │   │
│  │  - emit_event()                                     │   │
│  │  - add_causal_link()                                │   │
│  │  - update_entity_state()                            │   │
│  │  - flag_uncertainty()                               │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Improvement Suggestion:** Implement **VLM ensemble voting** - run the same analysis through multiple VLMs and use consensus for higher confidence assertions, flagging disagreements for human review.

---

### 4. Causal Graph Engine (`cwe/causality/`)

```
┌─────────────────────────────────────────────────────────────┐
│                  CAUSAL GRAPH ENGINE                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│    Events (nodes)          Causal Links (edges)             │
│    ┌──────────┐           ┌──────────────────┐             │
│    │ Event A  │──────────►│ causes/enables/  │             │
│    │ t=0:00   │           │ prevents/delays  │             │
│    └──────────┘           └──────────────────┘             │
│         │                                                   │
│         ▼                                                   │
│    ┌──────────┐                                            │
│    │ Event B  │  Confidence: 0.85                          │
│    │ t=0:05   │  Evidence: [frame_234, log_entry_89]       │
│    └──────────┘                                            │
│                                                             │
│    Graph Operations:                                        │
│    - Topological sort (execution order)                    │
│    - Counterfactual propagation                            │
│    - Minimal intervention sets                             │
│    - Root cause identification                             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Data Model:**
- **Event**: timestamp, type, entities involved, state changes, evidence
- **CausalLink**: source, target, relation_type, confidence, mechanism
- **Entity**: id, type, state_over_time, properties

---

### 5. Counterfactual Simulator (`cwe/counterfactual/`)

```
┌─────────────────────────────────────────────────────────────┐
│              COUNTERFACTUAL SIMULATOR                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Input: Canonical Timeline + Intervention                   │
│         "What if Event X didn't happen?"                    │
│         "What if Entity Y had property Z?"                  │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Intervention Compiler                   │   │
│  │  - Parse natural language interventions             │   │
│  │  - Map to graph modifications                       │   │
│  │  - Validate physical/logical consistency            │   │
│  └─────────────────────────────────────────────────────┘   │
│                           │                                 │
│                           ▼                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              VLM Re-reasoning Engine                 │   │
│  │  - Inject intervention into context                 │   │
│  │  - Re-run causal propagation                        │   │
│  │  - Generate alternate timeline                      │   │
│  └─────────────────────────────────────────────────────┘   │
│                           │                                 │
│                           ▼                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Divergence Analyzer                     │   │
│  │  - Compare canonical vs alternate                   │   │
│  │  - Identify branching points                        │   │
│  │  - Calculate outcome probabilities                  │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  Output: AlternateTimeline JSON + Confidence + Explanation  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Improvement Suggestion:** Add **intervention templates** for common counterfactuals:
- Timing interventions ("5 minutes earlier/later")
- Removal interventions ("X didn't happen")
- Substitution interventions ("Y instead of X")
- Property interventions ("X had property P")

---

### 6. Kinematic Simulator (`cwe/physics/`)

```
┌─────────────────────────────────────────────────────────────┐
│              KINEMATIC SIMULATOR                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Purpose: Deterministic physics for trajectory overlays     │
│           Validates VLM reasoning against physical laws     │
│                                                             │
│  Capabilities:                                              │
│  - 2D/3D trajectory projection                             │
│  - Collision detection                                      │
│  - Time-to-impact calculations                             │
│  - Velocity/acceleration estimation from video             │
│                                                             │
│  Integration with VLM:                                      │
│  - VLM proposes entity positions/velocities                │
│  - Simulator validates physical plausibility               │
│  - Discrepancies flagged for re-analysis                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Improvement Suggestion:** Make this pluggable - different domains need different simulators:
- Traffic/vehicle incidents → kinematic physics
- Network incidents → packet flow simulation
- Industrial processes → state machine simulation

---

### 7. Marathon Agent (`cwe/agents/`)

```
┌─────────────────────────────────────────────────────────────┐
│                    MARATHON AGENT                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Autonomous hypothesis exploration and ranking              │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Hypothesis Generator                    │   │
│  │  - Identify intervention points in timeline         │   │
│  │  - Generate candidate counterfactuals               │   │
│  │  - Prioritize by impact potential                   │   │
│  └─────────────────────────────────────────────────────┘   │
│                           │                                 │
│                           ▼                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Scenario Runner                         │   │
│  │  - Execute counterfactual simulations               │   │
│  │  - Manage parallel exploration                      │   │
│  │  - Track explored vs unexplored branches            │   │
│  └─────────────────────────────────────────────────────┘   │
│                           │                                 │
│                           ▼                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Outcome Ranker                          │   │
│  │  - Score outcomes by severity/desirability          │   │
│  │  - Identify high-leverage intervention points       │   │
│  │  - Generate actionable recommendations              │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  Agent Loop:                                                │
│  1. Observe canonical timeline                             │
│  2. Generate hypotheses                                    │
│  3. Simulate counterfactuals                               │
│  4. Rank outcomes                                          │
│  5. Report findings                                        │
│  6. (Optional) Request human guidance on ambiguities       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Improvement Suggestion:** Add **exploration strategies**:
- Breadth-first: Explore all single interventions first
- Depth-first: Follow most promising branches deeply
- Monte Carlo Tree Search: Balance exploration/exploitation
- Human-guided: Prioritize user-specified hypotheses

---

### 8. State Management (`cwe/state/`)

```
┌─────────────────────────────────────────────────────────────┐
│                  STATE MANAGEMENT                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Session State (in-memory + persistent)                     │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  - Canonical timeline                               │   │
│  │  - Causal graph                                     │   │
│  │  - VLM reasoning traces (thoughtSignature tokens)   │   │
│  │  - Explored counterfactuals                         │   │
│  │  - Entity states over time                          │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  Persistence Strategy:                                      │
│  - PostgreSQL: Structured data (events, entities, links)   │
│  - Redis: Session cache, VLM context management            │
│  - Object Storage (S3/GCS): Raw artifacts, video frames    │
│  - Vector DB (Pinecone/Weaviate): Semantic search over     │
│    incidents for similar case retrieval                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

### 9. API & Interface Layer (`cwe/api/`)

```
┌─────────────────────────────────────────────────────────────┐
│                   API & INTERFACE                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  REST API (FastAPI)                                         │
│  ├── POST /incidents              - Create new incident     │
│  ├── POST /incidents/{id}/ingest  - Add artifacts           │
│  ├── GET  /incidents/{id}/timeline - Get canonical timeline │
│  ├── GET  /incidents/{id}/graph   - Get causal graph        │
│  ├── POST /incidents/{id}/counterfactual - Run "what if"    │
│  ├── POST /incidents/{id}/explore - Start Marathon agent    │
│  └── GET  /incidents/{id}/results - Get exploration results │
│                                                             │
│  WebSocket API                                              │
│  └── /ws/incidents/{id}/stream    - Real-time updates       │
│                                                             │
│  CLI Interface                                              │
│  └── cwe analyze <incident_dir>   - Batch analysis mode     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Data Models

### Core Schemas

```python
# Simplified representation - full schemas in cwe/models/

class Event:
    id: str
    timestamp: datetime
    type: EventType
    description: str
    entities: List[EntityRef]
    state_changes: List[StateChange]
    evidence: List[EvidenceRef]
    confidence: float
    
class CausalLink:
    source_event: str
    target_event: str
    relation: CausalRelation  # CAUSES, ENABLES, PREVENTS, DELAYS
    mechanism: str  # Natural language explanation
    confidence: float
    evidence: List[EvidenceRef]

class Timeline:
    id: str
    incident_id: str
    is_canonical: bool
    intervention: Optional[Intervention]
    events: List[Event]
    causal_links: List[CausalLink]
    divergence_point: Optional[datetime]
    
class Counterfactual:
    id: str
    timeline_id: str
    intervention: Intervention
    alternate_timeline: Timeline
    outcome_score: float
    confidence: float
    explanation: str
    evidence_chain: List[str]
```

---

## VLM Integration Strategy

### Function Call Schema

```python
# Functions exposed to VLM for structured output

def emit_event(
    timestamp: str,
    event_type: str,
    description: str,
    entities: List[dict],
    confidence: float,
    evidence_refs: List[str]
) -> str:
    """Register a detected event in the timeline."""

def add_causal_link(
    source_event_id: str,
    target_event_id: str,
    relation: str,  # "causes" | "enables" | "prevents" | "delays"
    mechanism: str,
    confidence: float
) -> str:
    """Establish causal relationship between events."""

def update_entity_state(
    entity_id: str,
    timestamp: str,
    property_name: str,
    old_value: Any,
    new_value: Any
) -> str:
    """Track entity state changes over time."""

def flag_uncertainty(
    context: str,
    uncertainty_type: str,  # "missing_data" | "ambiguous" | "conflicting"
    possible_interpretations: List[str]
) -> str:
    """Flag areas requiring human review or additional data."""

def request_frame_analysis(
    video_id: str,
    timestamp_range: Tuple[str, str],
    query: str
) -> str:
    """Request detailed analysis of specific video segment."""
```

### Context Management Strategy

For long incidents that exceed context windows:

1. **Hierarchical Summarization**
   - Full detail for critical windows (±5 min of key events)
   - Summarized context for broader timeline
   - Reference pointers to raw artifacts

2. **Sliding Window with Anchors**
   - Maintain key "anchor events" in context
   - Slide detailed window as analysis progresses
   - Use VLM's own summaries to compress history

3. **Multi-Pass Analysis**
   - Pass 1: Coarse timeline extraction
   - Pass 2: Detailed analysis of flagged segments
   - Pass 3: Causal graph construction
   - Pass 4: Counterfactual reasoning

---

## Technology Stack

| Layer | Technology | Rationale |
|-------|------------|-----------|
| Language | Python 3.11+ | ML ecosystem, async support |
| API Framework | FastAPI | Async, auto-docs, WebSocket |
| Task Queue | Celery + Redis | Background VLM calls |
| Database | PostgreSQL | Structured data, JSONB |
| Cache | Redis | Session state, VLM context |
| Object Storage | S3/GCS/MinIO | Video, artifacts |
| Vector DB | Weaviate | Similar incident retrieval |
| VLM Clients | google-genai, openai, anthropic | Multi-provider |
| Video Processing | OpenCV, ffmpeg | Frame extraction |
| Visualization | Plotly, Three.js | Timeline & trajectory viz |
| Testing | pytest, hypothesis | Property-based testing |

---

## Development Phases

### Phase 1: Foundation (Weeks 1-3)
- [ ] Project scaffolding and CI/CD
- [ ] Core data models and schemas
- [ ] VLM provider abstraction layer
- [ ] Basic ingestion (video frames, logs)
- [ ] Simple timeline extraction

### Phase 2: Causal Reasoning (Weeks 4-6)
- [ ] Temporal alignment engine
- [ ] Causal graph construction
- [ ] Function call schema for VLM
- [ ] Basic counterfactual generation

### Phase 3: Simulation (Weeks 7-9)
- [ ] Kinematic simulator integration
- [ ] Visual overlay generation
- [ ] Counterfactual comparison UI

### Phase 4: Agentic Exploration (Weeks 10-12)
- [ ] Marathon agent implementation
- [ ] Hypothesis generation strategies
- [ ] Outcome ranking system
- [ ] Multi-VLM ensemble voting

### Phase 5: Production Hardening (Weeks 13-16)
- [ ] Horizontal scaling
- [ ] Monitoring and observability
- [ ] Human-in-the-loop workflows
- [ ] Documentation and examples

---

## Key Design Decisions to Make

1. **VLM Primary Provider**: Gemini 2.5 Pro (best multimodal) vs Grok (real-time) vs Claude (reasoning)?
2. **Deployment Model**: Cloud-native (GCP/AWS) vs self-hosted vs hybrid?
3. **Real-time vs Batch**: Support streaming analysis or batch-only initially?
4. **Domain Scope**: Start with specific domain (traffic, cyber, industrial) or generic?
5. **Human-in-the-loop**: Required for all counterfactuals or only low-confidence?

---

## Risk Mitigations

| Risk | Mitigation |
|------|------------|
| VLM hallucination | Ensemble voting, evidence linking, confidence thresholds |
| Context overflow | Hierarchical summarization, multi-pass analysis |
| Physics violations | Kinematic validator, domain-specific constraints |
| Cost explosion | Caching, tiered analysis (cheap→expensive), rate limiting |
| Latency | Async processing, background jobs, streaming responses |

---

## Next Steps

1. Review and refine this architecture
2. Make key design decisions
3. Set up project scaffolding
4. Implement Phase 1 components
5. Build first end-to-end demo with sample incident
