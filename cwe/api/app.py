"""FastAPI application for the Counterfactual World Engine."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from cwe import __version__
from cwe.engine import CounterfactualEngine

app = FastAPI(
    title="Counterfactual World Engine",
    description="API for reconstructing and simulating alternate realities from incident artifacts",
    version=__version__,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global engine instance (would be dependency-injected in production)
_engine: CounterfactualEngine | None = None


def get_engine() -> CounterfactualEngine:
    """Get or create the engine instance."""
    global _engine
    if _engine is None:
        _engine = CounterfactualEngine()
    return _engine


# ----- Request/Response Models -----

class IncidentCreate(BaseModel):
    """Request to create a new incident."""
    name: str
    description: str | None = None
    domain: str | None = None
    severity: str | None = None
    location: str | None = None


class IncidentResponse(BaseModel):
    """Response with incident details."""
    id: UUID
    name: str
    description: str | None
    status: str
    artifact_count: int = 0
    created_at: datetime


class TimelineResponse(BaseModel):
    """Response with timeline details."""
    id: UUID
    incident_id: UUID
    events: int
    causal_links: int
    entities: int
    confidence: float
    start_time: datetime | None
    end_time: datetime | None


class CounterfactualRequest(BaseModel):
    """Request to run a counterfactual simulation."""
    intervention: str = Field(..., description="What-if intervention description")


class CounterfactualResponse(BaseModel):
    """Response with counterfactual results."""
    id: UUID
    intervention: str
    outcome_improved: bool | None
    improvement_magnitude: float | None
    confidence: float
    explanation: str | None


class ExploreRequest(BaseModel):
    """Request to start autonomous exploration."""
    strategy: str = "llm_guided"
    max_iterations: int = 10
    seed_hypotheses: list[str] | None = None


class ExploreResponse(BaseModel):
    """Response with exploration results."""
    id: UUID
    hypotheses_generated: int
    hypotheses_explored: int
    counterfactuals: int
    best_intervention: str | None
    best_improvement: float | None
    recommendations: list[str]


# ----- Endpoints -----

@app.get("/")
async def root():
    """Health check."""
    return {
        "service": "Counterfactual World Engine",
        "version": __version__,
        "status": "healthy",
    }


@app.post("/incidents", response_model=IncidentResponse)
async def create_incident(request: IncidentCreate):
    """Create a new incident for analysis."""
    engine = get_engine()
    
    handle = engine.create_incident(
        name=request.name,
        description=request.description,
        domain=request.domain,
        severity=request.severity,
        location=request.location,
    )
    
    return IncidentResponse(
        id=handle.id,
        name=handle.incident.name,
        description=handle.incident.description,
        status=handle.incident.status.value,
        artifact_count=len(handle.incident.artifact_ids),
        created_at=handle.incident.created_at,
    )


@app.get("/incidents", response_model=list[IncidentResponse])
async def list_incidents():
    """List all incidents."""
    engine = get_engine()
    incidents = engine.list_incidents()
    
    return [
        IncidentResponse(
            id=inc.id,
            name=inc.name,
            description=inc.description,
            status=inc.status.value,
            artifact_count=len(inc.artifact_ids),
            created_at=inc.created_at,
        )
        for inc in incidents
    ]


@app.get("/incidents/{incident_id}", response_model=IncidentResponse)
async def get_incident(incident_id: UUID):
    """Get incident details."""
    engine = get_engine()
    handle = engine.get_incident(incident_id)
    
    if not handle:
        raise HTTPException(status_code=404, detail="Incident not found")
    
    return IncidentResponse(
        id=handle.id,
        name=handle.incident.name,
        description=handle.incident.description,
        status=handle.incident.status.value,
        artifact_count=len(handle.incident.artifact_ids),
        created_at=handle.incident.created_at,
    )


@app.post("/incidents/{incident_id}/artifacts/video")
async def upload_video(
    incident_id: UUID,
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None,
):
    """Upload a video artifact."""
    engine = get_engine()
    handle = engine.get_incident(incident_id)
    
    if not handle:
        raise HTTPException(status_code=404, detail="Incident not found")
    
    # Save file temporarily and ingest
    import tempfile
    from pathlib import Path
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    handle.ingest_video(tmp_path)
    
    return {"message": "Video uploaded", "filename": file.filename}


@app.post("/incidents/{incident_id}/artifacts/logs")
async def upload_logs(
    incident_id: UUID,
    file: UploadFile = File(...),
):
    """Upload a log file artifact."""
    engine = get_engine()
    handle = engine.get_incident(incident_id)
    
    if not handle:
        raise HTTPException(status_code=404, detail="Incident not found")
    
    import tempfile
    from pathlib import Path
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    handle.ingest_logs(tmp_path)
    
    return {"message": "Logs uploaded", "filename": file.filename}


@app.post("/incidents/{incident_id}/timeline", response_model=TimelineResponse)
async def build_timeline(incident_id: UUID, background_tasks: BackgroundTasks):
    """Build the canonical timeline for an incident."""
    engine = get_engine()
    handle = engine.get_incident(incident_id)
    
    if not handle:
        raise HTTPException(status_code=404, detail="Incident not found")
    
    # This could be a long-running task - in production, run in background
    timeline = await handle.build_timeline()
    
    return TimelineResponse(
        id=timeline.id,
        incident_id=timeline.incident_id,
        events=len(timeline.events),
        causal_links=len(timeline.causal_links),
        entities=len(timeline.entities),
        confidence=timeline.confidence,
        start_time=timeline.start_time if timeline.start_time != datetime.min else None,
        end_time=timeline.end_time if timeline.end_time != datetime.max else None,
    )


@app.get("/incidents/{incident_id}/timeline", response_model=TimelineResponse)
async def get_timeline(incident_id: UUID):
    """Get the canonical timeline for an incident."""
    engine = get_engine()
    handle = engine.get_incident(incident_id)
    
    if not handle:
        raise HTTPException(status_code=404, detail="Incident not found")
    
    timeline = handle.get_timeline()
    
    if not timeline:
        raise HTTPException(status_code=404, detail="Timeline not built yet")
    
    return TimelineResponse(
        id=timeline.id,
        incident_id=timeline.incident_id,
        events=len(timeline.events),
        causal_links=len(timeline.causal_links),
        entities=len(timeline.entities),
        confidence=timeline.confidence,
        start_time=timeline.start_time if timeline.start_time != datetime.min else None,
        end_time=timeline.end_time if timeline.end_time != datetime.max else None,
    )


@app.get("/incidents/{incident_id}/timeline/events")
async def get_timeline_events(incident_id: UUID):
    """Get all events in the timeline."""
    engine = get_engine()
    handle = engine.get_incident(incident_id)
    
    if not handle:
        raise HTTPException(status_code=404, detail="Incident not found")
    
    timeline = handle.get_timeline()
    
    if not timeline:
        raise HTTPException(status_code=404, detail="Timeline not built yet")
    
    return [
        {
            "id": str(event.id),
            "timestamp": event.timestamp.isoformat(),
            "type": event.event_type.value,
            "description": event.description,
            "confidence": event.confidence,
            "entities": [str(e) for e in event.entities],
        }
        for event in sorted(timeline.events, key=lambda e: e.timestamp)
    ]


@app.get("/incidents/{incident_id}/timeline/graph")
async def get_causal_graph(incident_id: UUID):
    """Get the causal graph for the timeline."""
    engine = get_engine()
    handle = engine.get_incident(incident_id)
    
    if not handle:
        raise HTTPException(status_code=404, detail="Incident not found")
    
    timeline = handle.get_timeline()
    
    if not timeline:
        raise HTTPException(status_code=404, detail="Timeline not built yet")
    
    # Return in a format suitable for visualization
    nodes = [
        {
            "id": str(event.id),
            "label": event.description[:50],
            "timestamp": event.timestamp.isoformat(),
            "type": event.event_type.value,
        }
        for event in timeline.events
    ]
    
    edges = [
        {
            "source": str(link.source_event_id),
            "target": str(link.target_event_id),
            "relation": link.relation.value,
            "mechanism": link.mechanism,
            "confidence": link.confidence,
        }
        for link in timeline.causal_links
    ]
    
    return {"nodes": nodes, "edges": edges}


@app.post("/incidents/{incident_id}/counterfactual", response_model=CounterfactualResponse)
async def run_counterfactual(incident_id: UUID, request: CounterfactualRequest):
    """Run a counterfactual simulation."""
    engine = get_engine()
    handle = engine.get_incident(incident_id)
    
    if not handle:
        raise HTTPException(status_code=404, detail="Incident not found")
    
    if not handle.get_timeline():
        raise HTTPException(status_code=400, detail="Build timeline first")
    
    result = await handle.what_if(request.intervention)
    
    return CounterfactualResponse(
        id=result.id,
        intervention=result.intervention.description,
        outcome_improved=result.outcome.outcome_improved if result.outcome else None,
        improvement_magnitude=result.outcome.improvement_magnitude if result.outcome else None,
        confidence=result.confidence,
        explanation=result.explanation,
    )


@app.post("/incidents/{incident_id}/explore", response_model=ExploreResponse)
async def explore_counterfactuals(incident_id: UUID, request: ExploreRequest):
    """Start autonomous counterfactual exploration."""
    engine = get_engine()
    handle = engine.get_incident(incident_id)
    
    if not handle:
        raise HTTPException(status_code=404, detail="Incident not found")
    
    if not handle.get_timeline():
        raise HTTPException(status_code=400, detail="Build timeline first")
    
    results = await handle.explore(
        strategy=request.strategy,
        max_iterations=request.max_iterations,
        seed_hypotheses=request.seed_hypotheses,
    )
    
    return ExploreResponse(
        id=results.id,
        hypotheses_generated=results.hypotheses_generated,
        hypotheses_explored=results.hypotheses_explored,
        counterfactuals=len(results.counterfactuals),
        best_intervention=results.best_outcome.intervention.description if results.best_outcome else None,
        best_improvement=results.best_outcome.outcome.improvement_magnitude if results.best_outcome and results.best_outcome.outcome else None,
        recommendations=results.recommendations,
    )


@app.get("/incidents/{incident_id}/counterfactuals")
async def list_counterfactuals(incident_id: UUID):
    """List all counterfactuals for an incident."""
    engine = get_engine()
    handle = engine.get_incident(incident_id)
    
    if not handle:
        raise HTTPException(status_code=404, detail="Incident not found")
    
    counterfactuals = handle.get_counterfactuals()
    
    return [
        {
            "id": str(cf.id),
            "intervention": cf.intervention.description,
            "outcome_improved": cf.outcome.outcome_improved if cf.outcome else None,
            "improvement": cf.outcome.improvement_magnitude if cf.outcome else None,
            "confidence": cf.confidence,
            "created_at": cf.created_at.isoformat(),
        }
        for cf in counterfactuals
    ]
