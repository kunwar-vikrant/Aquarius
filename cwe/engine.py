"""Main Counterfactual World Engine."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from uuid import UUID

import structlog

from cwe.models.incident import Incident, IncidentStatus
from cwe.models.artifact import Artifact, ArtifactType
from cwe.models.timeline import Timeline
from cwe.models.counterfactual import Counterfactual, Intervention
from cwe.reasoning.providers.base import VLMProvider, VLMConfig, VLMProviderType
from cwe.reasoning.providers.gemini import GeminiProvider
from cwe.reasoning.providers.anthropic import AnthropicProvider
from cwe.reasoning.providers.openai import OpenAIProvider
from cwe.reasoning.reasoner import TimelineReasoner
from cwe.agents.marathon import MarathonAgent, ExplorationConfig, ExplorationResult

logger = structlog.get_logger()


def get_provider(
    provider_type: str | VLMProviderType,
    api_key: str | None = None,
    model: str | None = None,
    **kwargs,
) -> VLMProvider:
    """
    Get a VLM provider instance.
    
    Args:
        provider_type: Provider type (gemini, anthropic, openai)
        api_key: API key (defaults to environment variable)
        model: Model to use (defaults to provider's default)
        **kwargs: Additional provider configuration
        
    Returns:
        Configured VLMProvider instance
    """
    if isinstance(provider_type, str):
        provider_type = VLMProviderType(provider_type.lower())
    
    # Get API key from environment if not provided
    env_keys = {
        VLMProviderType.GEMINI: "GEMINI_API_KEY",
        VLMProviderType.ANTHROPIC: "ANTHROPIC_API_KEY",
        VLMProviderType.OPENAI: "OPENAI_API_KEY",
        VLMProviderType.XAI: "XAI_API_KEY",
    }
    
    if not api_key:
        api_key = os.environ.get(env_keys.get(provider_type, ""))
    
    if not api_key:
        raise ValueError(f"API key required for {provider_type.value}")
    
    config = VLMConfig(
        provider=provider_type,
        api_key=api_key,
        model=model,
        **kwargs,
    )
    
    providers = {
        VLMProviderType.GEMINI: GeminiProvider,
        VLMProviderType.ANTHROPIC: AnthropicProvider,
        VLMProviderType.OPENAI: OpenAIProvider,
    }
    
    provider_class = providers.get(provider_type)
    if not provider_class:
        raise ValueError(f"Unsupported provider: {provider_type}")
    
    return provider_class(config)


class CounterfactualEngine:
    """
    Main interface for the Counterfactual World Engine.
    
    Provides high-level API for:
    - Creating and managing incidents
    - Ingesting multimodal artifacts
    - Building canonical timelines
    - Running counterfactual simulations
    - Autonomous exploration via Marathon agent
    
    Example:
        ```python
        engine = CounterfactualEngine(vlm_provider="gemini")
        
        # Create incident and ingest data
        incident = engine.create_incident("traffic_collision_001")
        incident.ingest_video("dashcam.mp4")
        incident.ingest_logs("vehicle_telemetry.jsonl")
        
        # Build timeline
        timeline = await incident.build_timeline()
        
        # Run counterfactual
        result = await incident.what_if("Driver braked 2 seconds earlier")
        
        # Autonomous exploration
        results = await incident.explore()
        ```
    """
    
    def __init__(
        self,
        vlm_provider: str | VLMProvider = "gemini",
        api_key: str | None = None,
        secondary_provider: str | VLMProvider | None = None,
        enable_ensemble: bool = False,
        **kwargs,
    ):
        """
        Initialize the Counterfactual World Engine.
        
        Args:
            vlm_provider: Primary VLM provider (name or instance)
            api_key: API key for the primary provider
            secondary_provider: Optional secondary provider for ensemble
            enable_ensemble: Enable ensemble voting across providers
            **kwargs: Additional configuration
        """
        # Set up primary provider
        if isinstance(vlm_provider, str):
            self.provider = get_provider(vlm_provider, api_key=api_key)
        else:
            self.provider = vlm_provider
        
        # Set up secondary provider if specified
        self.secondary_provider = None
        if secondary_provider:
            if isinstance(secondary_provider, str):
                self.secondary_provider = get_provider(secondary_provider)
            else:
                self.secondary_provider = secondary_provider
        
        self.enable_ensemble = enable_ensemble
        
        # Initialize components
        self.reasoner = TimelineReasoner(
            provider=self.provider,
            secondary_provider=self.secondary_provider,
            enable_ensemble=enable_ensemble,
        )
        
        # Storage (in-memory for now - would be replaced with DB in production)
        self._incidents: dict[UUID, Incident] = {}
        self._artifacts: dict[UUID, list[Artifact]] = {}
        self._timelines: dict[UUID, Timeline] = {}
        self._counterfactuals: dict[UUID, list[Counterfactual]] = {}
        
        logger.info(
            "CounterfactualEngine initialized",
            provider=self.provider.provider_type.value,
            model=self.provider.model,
        )
    
    def create_incident(
        self,
        name: str,
        description: str | None = None,
        **metadata,
    ) -> "IncidentHandle":
        """
        Create a new incident for analysis.
        
        Args:
            name: Name/identifier for the incident
            description: Optional description
            **metadata: Additional metadata (domain, severity, location, etc.)
            
        Returns:
            IncidentHandle for further operations
        """
        from cwe.models.incident import IncidentMetadata
        
        incident = Incident(
            name=name,
            description=description,
            metadata=IncidentMetadata(**metadata),
        )
        
        self._incidents[incident.id] = incident
        self._artifacts[incident.id] = []
        self._counterfactuals[incident.id] = []
        
        logger.info("Incident created", incident_id=str(incident.id), name=name)
        
        return IncidentHandle(self, incident)
    
    def get_incident(self, incident_id: UUID | str) -> "IncidentHandle | None":
        """Get an existing incident by ID."""
        if isinstance(incident_id, str):
            incident_id = UUID(incident_id)
        
        incident = self._incidents.get(incident_id)
        if incident:
            return IncidentHandle(self, incident)
        return None
    
    def list_incidents(self) -> list[Incident]:
        """List all incidents."""
        return list(self._incidents.values())


class IncidentHandle:
    """
    Handle for working with a specific incident.
    
    Provides fluent API for incident operations.
    """
    
    def __init__(self, engine: CounterfactualEngine, incident: Incident):
        self._engine = engine
        self._incident = incident
    
    @property
    def id(self) -> UUID:
        return self._incident.id
    
    @property
    def incident(self) -> Incident:
        return self._incident
    
    def ingest_video(
        self,
        path: str | Path,
        **metadata,
    ) -> "IncidentHandle":
        """
        Ingest a video file.
        
        Args:
            path: Path to the video file
            **metadata: Additional metadata
            
        Returns:
            Self for chaining
        """
        from cwe.models.artifact import VideoArtifact
        
        path = Path(path)
        
        artifact = VideoArtifact(
            incident_id=self._incident.id,
            filename=path.name,
            file_path=str(path),
            metadata=metadata,
        )
        
        # TODO: Process video (extract frames, detect keyframes, etc.)
        
        self._engine._artifacts[self._incident.id].append(artifact)
        self._incident.artifact_ids.append(artifact.id)
        
        logger.info("Video ingested", artifact_id=str(artifact.id), filename=path.name)
        
        return self
    
    def ingest_logs(
        self,
        path: str | Path,
        log_format: str | None = None,
        **metadata,
    ) -> "IncidentHandle":
        """
        Ingest a log file.
        
        Args:
            path: Path to the log file
            log_format: Log format (json, syslog, csv, etc.)
            **metadata: Additional metadata
            
        Returns:
            Self for chaining
        """
        from cwe.models.artifact import LogArtifact
        
        path = Path(path)
        
        artifact = LogArtifact(
            incident_id=self._incident.id,
            filename=path.name,
            file_path=str(path),
            log_format=log_format,
            metadata=metadata,
        )
        
        # TODO: Parse logs (detect format, extract entries, etc.)
        
        self._engine._artifacts[self._incident.id].append(artifact)
        self._incident.artifact_ids.append(artifact.id)
        
        logger.info("Logs ingested", artifact_id=str(artifact.id), filename=path.name)
        
        return self
    
    def ingest_report(
        self,
        path: str | Path,
        **metadata,
    ) -> "IncidentHandle":
        """
        Ingest a report/document.
        
        Args:
            path: Path to the document
            **metadata: Additional metadata
            
        Returns:
            Self for chaining
        """
        from cwe.models.artifact import ReportArtifact
        
        path = Path(path)
        
        artifact = ReportArtifact(
            incident_id=self._incident.id,
            filename=path.name,
            file_path=str(path),
            metadata=metadata,
        )
        
        # TODO: Extract document content
        
        self._engine._artifacts[self._incident.id].append(artifact)
        self._incident.artifact_ids.append(artifact.id)
        
        logger.info("Report ingested", artifact_id=str(artifact.id), filename=path.name)
        
        return self
    
    def ingest_sensor_data(
        self,
        path: str | Path,
        sensor_type: str | None = None,
        **metadata,
    ) -> "IncidentHandle":
        """
        Ingest sensor/telemetry data.
        
        Args:
            path: Path to the sensor data file
            sensor_type: Type of sensor (gps, accelerometer, etc.)
            **metadata: Additional metadata
            
        Returns:
            Self for chaining
        """
        from cwe.models.artifact import SensorArtifact
        
        path = Path(path)
        
        artifact = SensorArtifact(
            incident_id=self._incident.id,
            filename=path.name,
            file_path=str(path),
            sensor_type=sensor_type,
            metadata=metadata,
        )
        
        # TODO: Parse sensor data
        
        self._engine._artifacts[self._incident.id].append(artifact)
        self._incident.artifact_ids.append(artifact.id)
        
        logger.info("Sensor data ingested", artifact_id=str(artifact.id), filename=path.name)
        
        return self
    
    async def build_timeline(self) -> Timeline:
        """
        Build the canonical timeline from ingested artifacts.
        
        Uses VLM to analyze artifacts and construct a structured timeline
        with events, entities, and causal relationships.
        
        Returns:
            The canonical Timeline
        """
        self._incident.status = IncidentStatus.ANALYZING
        
        artifacts = self._engine._artifacts[self._incident.id]
        
        if not artifacts:
            raise ValueError("No artifacts ingested. Ingest data before building timeline.")
        
        timeline = await self._engine.reasoner.build_timeline(
            incident=self._incident,
            artifacts=artifacts,
        )
        
        self._engine._timelines[self._incident.id] = timeline
        self._incident.canonical_timeline_id = timeline.id
        self._incident.status = IncidentStatus.READY
        
        return timeline
    
    async def what_if(
        self,
        intervention: str | Intervention,
    ) -> Counterfactual:
        """
        Run a counterfactual simulation.
        
        Args:
            intervention: Natural language description or Intervention object
            
        Returns:
            Counterfactual result with alternate timeline and comparison
        """
        timeline = self._engine._timelines.get(self._incident.id)
        if not timeline:
            raise ValueError("Build timeline first before running counterfactuals.")
        
        # Convert string to Intervention
        if isinstance(intervention, str):
            from cwe.models.counterfactual import InterventionType
            intervention = Intervention(
                intervention_type=InterventionType.NATURAL_LANGUAGE,
                description=intervention,
            )
        
        # Run counterfactual simulation
        # TODO: Use dedicated CounterfactualSimulator
        from cwe.reasoning.providers.base import Message
        from cwe.reasoning.function_schema import get_counterfactual_functions
        
        prompt = f"""# Canonical Timeline

{self._format_timeline(timeline)}

# Intervention

{intervention.description}

# Task

Simulate what would have happened if this intervention had occurred.
Think step by step about which events would change, what new events might occur,
and how the final outcome differs from the canonical timeline.

Call `compare_outcomes` to summarize the comparison."""
        
        response = await self._engine.provider.generate(
            messages=[
                Message.system(
                    "You are simulating an alternate timeline. Given the canonical events "
                    "and an intervention, reason about what would have happened differently."
                ),
                Message.user(prompt),
            ],
            functions=[f.to_schema() for f in get_counterfactual_functions()],
            enable_thinking=True,
        )
        
        # Build counterfactual from response
        from cwe.models.counterfactual import OutcomeComparison, OutcomeSeverity
        
        counterfactual = Counterfactual(
            incident_id=self._incident.id,
            canonical_timeline_id=timeline.id,
            intervention=intervention,
            explanation=response.text,
            vlm_provider=self._engine.provider.provider_type.value,
            vlm_model=self._engine.provider.model,
        )
        
        # Parse function calls
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
                    improvement_magnitude=0.0,
                    key_differences=fc.arguments.get("key_differences", []),
                )
                counterfactual.confidence = fc.arguments.get("confidence", 0.5)
        
        self._engine._counterfactuals[self._incident.id].append(counterfactual)
        self._incident.counterfactual_ids.append(counterfactual.id)
        
        return counterfactual
    
    async def explore(
        self,
        strategy: str = "llm_guided",
        max_iterations: int = 10,
        seed_hypotheses: list[str] | None = None,
        **config,
    ) -> ExplorationResult:
        """
        Autonomously explore counterfactual scenarios.
        
        Uses the Marathon agent to generate and evaluate hypotheses.
        
        Args:
            strategy: Exploration strategy (llm_guided, exhaustive, greedy, mcts)
            max_iterations: Maximum number of hypotheses to explore
            seed_hypotheses: Optional user-provided hypotheses to prioritize
            **config: Additional exploration configuration
            
        Returns:
            ExplorationResult with ranked counterfactuals and recommendations
        """
        timeline = self._engine._timelines.get(self._incident.id)
        if not timeline:
            raise ValueError("Build timeline first before exploring.")
        
        self._incident.status = IncidentStatus.EXPLORING
        
        from cwe.agents.marathon import ExplorationStrategy
        
        exploration_config = ExplorationConfig(
            strategy=ExplorationStrategy(strategy),
            max_iterations=max_iterations,
            **config,
        )
        
        agent = MarathonAgent(
            provider=self._engine.provider,
            config=exploration_config,
        )
        
        results = await agent.explore(
            timeline=timeline,
            seed_hypotheses=seed_hypotheses,
        )
        
        # Store results
        for cf in results.counterfactuals:
            self._engine._counterfactuals[self._incident.id].append(cf)
            self._incident.counterfactual_ids.append(cf.id)
        
        self._incident.status = IncidentStatus.COMPLETED
        
        return results
    
    def get_timeline(self) -> Timeline | None:
        """Get the canonical timeline."""
        return self._engine._timelines.get(self._incident.id)
    
    def get_counterfactuals(self) -> list[Counterfactual]:
        """Get all counterfactuals for this incident."""
        return self._engine._counterfactuals.get(self._incident.id, [])
    
    def _format_timeline(self, timeline: Timeline) -> str:
        """Format timeline for prompts."""
        events_text = "\n".join(
            f"- [{e.timestamp.isoformat()}] {e.event_type.value}: {e.description}"
            for e in sorted(timeline.events, key=lambda e: e.timestamp)
        )
        
        causal_text = "\n".join(
            f"- {link.source_event_id} {link.relation.value} {link.target_event_id}: {link.mechanism}"
            for link in timeline.causal_links
        ) if timeline.causal_links else "No causal links established."
        
        return f"""## Events
{events_text}

## Causal Relationships
{causal_text}"""
