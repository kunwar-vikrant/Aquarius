# Design Decisions - Counterfactual World Engine

This document tracks key architectural decisions and their rationale.

## Decision Log

### DEC-001: Primary VLM Provider

**Status:** PENDING

**Options:**
1. **Gemini 2.5 Pro** - Best multimodal (video understanding), 2M token context, native function calling
2. **Grok 3** - Strong reasoning, real-time data access, competitive pricing
3. **Claude 3.5/4** - Best reasoning/analysis, excellent structured output
4. **GPT-4o** - Good multimodal, wide adoption, function calling

**Recommendation:** Start with Gemini 2.5 Pro as primary (best video understanding), Claude as secondary (best counterfactual reasoning). Design abstraction layer to swap easily.

**Factors:**
- Video understanding capability (Gemini leads)
- Context window size (Gemini 2M, Claude 200K, GPT-4o 128K)
- Function calling reliability
- Cost per token
- Latency

---

### DEC-002: Domain Focus

**Status:** PENDING

**Options:**
1. **Traffic/Vehicle Incidents** - Rich visual data, clear physics, public datasets
2. **Cybersecurity Incidents** - Log-heavy, enterprise demand, less visual
3. **Industrial/Manufacturing** - Sensor-heavy, safety critical, niche
4. **Generic Multi-domain** - Flexible but harder to optimize

**Recommendation:** Start with Traffic/Vehicle Incidents
- Rich multimodal data (dashcam, CCTV, GPS logs, accident reports)
- Clear physical constraints (kinematics, collisions)
- Available public datasets (NHTSA, dashcam compilations)
- High visual reasoning demands showcase VLM strengths

---

### DEC-003: Deployment Architecture

**Status:** PENDING

**Options:**
1. **Monolithic** - Single deployable, simpler ops, limited scale
2. **Microservices** - Component isolation, independent scaling, complex ops
3. **Modular Monolith** - Single deploy, clean boundaries, easy to split later

**Recommendation:** Modular Monolith initially
- Clear module boundaries (ingestion, reasoning, simulation, agents)
- Single deployment simplifies early development
- Can split into services when scaling demands

---

### DEC-004: State Persistence Strategy

**Status:** PENDING

**Options:**
1. **PostgreSQL + Redis** - Relational + cache, mature, well-understood
2. **MongoDB** - Document store, flexible schema, good for evolving models
3. **Event Sourcing** - Full audit trail, replay capability, complex
4. **Hybrid** - Different stores for different data types

**Recommendation:** Hybrid approach
- PostgreSQL: Core entities (incidents, events, links) with JSONB for flexibility
- Redis: Session state, VLM context cache, real-time coordination
- S3/GCS: Raw artifacts (video, documents)
- (Optional) Weaviate: Semantic search for similar incidents

---

### DEC-005: VLM Context Management

**Status:** PENDING

**Challenge:** Incidents may have hours of video + thousands of log lines, exceeding any context window.

**Options:**
1. **Chunked Processing** - Split into windows, merge results
2. **Hierarchical Summarization** - Summary layers, detail on demand
3. **RAG-based** - Embed everything, retrieve relevant context
4. **VLM-guided Navigation** - Let VLM request specific data

**Recommendation:** Hierarchical + VLM-guided Navigation
- Build timeline skeleton in first pass (low detail)
- VLM can "zoom in" on segments via function calls
- Maintain compressed context of full incident
- Cache VLM summaries for reuse

---

### DEC-006: Counterfactual Confidence Scoring

**Status:** PENDING

**Challenge:** How to quantify confidence in alternate timeline predictions?

**Approach:**
1. **Evidence Chain Completeness** - % of causal links with supporting evidence
2. **VLM Self-Reported Confidence** - Model's stated certainty
3. **Physical Plausibility** - Kinematic simulator validation
4. **Ensemble Agreement** - Cross-VLM consensus
5. **Intervention Distance** - How "far" from canonical reality

**Formula (draft):**
```
confidence = (
    0.25 * evidence_score +
    0.20 * vlm_confidence +
    0.25 * physics_validity +
    0.20 * ensemble_agreement +
    0.10 * (1 - intervention_distance)
)
```

---

### DEC-007: Agent Exploration Strategy

**Status:** PENDING

**Options:**
1. **Exhaustive** - Try all single interventions
2. **Greedy** - Follow most promising branches
3. **MCTS** - Monte Carlo Tree Search (balance explore/exploit)
4. **LLM-guided** - Use VLM to propose promising interventions

**Recommendation:** LLM-guided with bounded exploration
- Use VLM to generate ranked list of promising interventions
- Explore top-N in parallel
- Use outcome scores to guide deeper exploration
- Set compute budget, not iteration limit

---

### DEC-008: Human-in-the-Loop Integration

**Status:** PENDING

**Trigger Conditions:**
- Confidence below threshold (< 0.6)
- Conflicting evidence detected
- High-stakes counterfactual (severe outcome change)
- VLM explicitly flags uncertainty

**Interface Options:**
1. **Async Review Queue** - Batch review, non-blocking
2. **Interactive Session** - Real-time human guidance
3. **Hybrid** - Async default, escalate to interactive

**Recommendation:** Async Review Queue with escalation
- Non-blocking for throughput
- Dashboard for reviewing flagged items
- Escalation path for time-sensitive analysis

---

## Open Questions

1. **Licensing**: What license for the project? (Apache 2.0 recommended for adoption)
2. **Eval Framework**: How to benchmark counterfactual quality? (need ground truth incidents)
3. **Privacy**: How to handle PII in videos/logs? (anonymization pipeline needed)
4. **Multi-tenancy**: Single-tenant initially or design for multi-tenant?
5. **Versioning**: How to version timelines/graphs as analysis evolves?

---

## Decision Template

```markdown
### DEC-XXX: [Title]

**Status:** PENDING | DECIDED | SUPERSEDED

**Context:** [Why is this decision needed?]

**Options:**
1. [Option A] - [Pros/Cons]
2. [Option B] - [Pros/Cons]

**Decision:** [What was decided]

**Rationale:** [Why this option]

**Consequences:** [What changes as a result]
```
