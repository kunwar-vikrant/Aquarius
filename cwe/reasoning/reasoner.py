"""Timeline reasoner - orchestrates VLM for timeline construction."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

import structlog

from cwe.models.timeline import (
    Timeline, Event, EventType, CausalLink, CausalRelation, Entity, EntityState
)
from cwe.models.incident import Incident
from cwe.models.artifact import Artifact
from cwe.reasoning.providers.base import VLMProvider, VLMResponse, Message
from cwe.reasoning.function_schema import get_timeline_functions, FunctionRegistry
from cwe.reasoning.context_manager import ContextManager, ContextWindow

logger = structlog.get_logger()


TIMELINE_SYSTEM_PROMPT = """You are an expert incident analyst tasked with constructing a detailed timeline from multimodal evidence.

Your job is to:
1. Analyze the provided artifacts (video frames, logs, reports, sensor data)
2. Identify all significant events and their timestamps
3. Establish causal relationships between events
4. Track entities (vehicles, people, systems) through the timeline
5. Flag any uncertainties or areas requiring human review

For each event you identify, call the `emit_event` function with:
- Precise timestamp (use ISO 8601 format)
- Event type classification
- Detailed description
- Entities involved
- Confidence score (0.0-1.0)
- Evidence references

For causal relationships, call `add_causal_link` with:
- Source and target event IDs
- Relationship type (causes, enables, prevents, delays)
- Mechanism explaining how/why
- Confidence score

Be thorough but precise. Only emit events you have evidence for.
Flag uncertainties with `flag_uncertainty` rather than guessing.

IMPORTANT: Think step by step. First scan all evidence to understand the incident, 
then systematically build the timeline from earliest to latest event."""


class TimelineReasoner:
    """
    Orchestrates VLM reasoning to construct timelines from incident artifacts.
    
    Handles:
    - Multi-pass analysis (coarse → detailed)
    - Function call execution
    - State management across reasoning steps
    - Ensemble voting (optional)
    """
    
    def __init__(
        self,
        provider: VLMProvider,
        secondary_provider: VLMProvider | None = None,
        enable_ensemble: bool = False,
    ):
        self.provider = provider
        self.secondary_provider = secondary_provider
        self.enable_ensemble = enable_ensemble
        
        self.context_manager = ContextManager(
            max_tokens=provider.max_context_tokens()
        )
        
        # Function registry with handlers
        self.function_registry = FunctionRegistry()
        for func in get_timeline_functions():
            self.function_registry.register(func)
        
        # Session state
        self._session_id: str | None = None
        self._incident: Incident | None = None
        self._timeline: Timeline | None = None
        self._entities: dict[str, Entity] = {}
        self._events: dict[str, Event] = {}
        self._uncertainties: list[dict[str, Any]] = []

    @property
    def timeline(self) -> Timeline | None:
        """Get the current timeline."""
        return self._timeline
        self._messages: list[Message] = []
    
    async def build_timeline(
        self,
        incident: Incident,
        artifacts: list[Artifact],
    ) -> Timeline:
        """
        Build a canonical timeline from incident artifacts.
        
        This is the main entry point for timeline construction.
        """
        logger.info("Starting timeline construction", incident_id=str(incident.id))
        
        self._session_id = self.provider.start_session()
        self._incident = incident
        self._timeline = Timeline(
            incident_id=incident.id,
            is_canonical=True,
            start_time=datetime.min,
            end_time=datetime.max,
            vlm_provider=self.provider.provider_type.value,
            vlm_model=self.provider.model,
            vlm_session_id=self._session_id,
        )
        
        # Add artifacts to context manager
        for artifact in artifacts:
            self.context_manager.add_artifact(artifact)
        
        # Build initial context
        context = self.context_manager.build_initial_context()
        
        # Run multi-pass analysis
        await self._run_coarse_pass(context)
        await self._run_detailed_pass(context)
        await self._run_causal_pass(context)
        
        # Finalize timeline
        self._timeline.events = list(self._events.values())
        self._timeline.entities = list(self._entities.values())
        
        # Update timeline bounds
        if self._events:
            timestamps = [e.timestamp for e in self._events.values()]
            self._timeline.start_time = min(timestamps)
            self._timeline.end_time = max(timestamps)
        
        # Calculate overall confidence
        if self._timeline.events:
            self._timeline.confidence = sum(
                e.confidence for e in self._timeline.events
            ) / len(self._timeline.events)
        
        logger.info(
            "Timeline construction complete",
            incident_id=str(incident.id),
            events=len(self._timeline.events),
            causal_links=len(self._timeline.causal_links),
            entities=len(self._timeline.entities),
            confidence=self._timeline.confidence,
        )
        
        return self._timeline
    
    async def _run_coarse_pass(self, context: ContextWindow) -> None:
        """First pass: Identify major events and entities."""
        logger.info("Running coarse pass")
        
        prompt = self._build_prompt(
            context,
            task=(
                "COARSE PASS: Scan all evidence and identify the major events in this incident. "
                "Focus on:\n"
                "1. Key entities (vehicles, people, systems involved)\n"
                "2. Major state changes and actions\n"
                "3. The overall sequence of events\n\n"
                "Call `register_entity` for each entity and `emit_event` for major events."
            ),
        )
        
        await self._reasoning_loop(prompt)
    
    async def _run_detailed_pass(self, context: ContextWindow) -> None:
        """Second pass: Fill in detailed events."""
        logger.info("Running detailed pass")
        
        # Request detail for areas with events
        for event in list(self._events.values()):
            if event.evidence:
                for evidence in event.evidence:
                    if evidence.artifact_id:
                        self.context_manager.request_detail(
                            artifact_id=evidence.artifact_id,
                            time_range=(event.timestamp, event.timestamp),
                        )
        
        prompt = self._build_prompt(
            context,
            task=(
                "DETAILED PASS: Now fill in the detailed events between the major events "
                "you identified. Look for:\n"
                "1. Intermediate state changes\n"
                "2. Subtle observations\n"
                "3. Entity position/velocity changes\n"
                "4. Any events you may have missed\n\n"
                "Call `emit_event` for each new event and `update_entity_state` for state changes."
            ),
            include_current_state=True,
        )
        
        await self._reasoning_loop(prompt)
    
    async def _run_causal_pass(self, context: ContextWindow) -> None:
        """Third pass: Establish causal relationships."""
        logger.info("Running causal pass")
        
        prompt = self._build_prompt(
            context,
            task=(
                "CAUSAL PASS: Now analyze the causal relationships between events. "
                "For each pair of events that have a causal relationship, call `add_causal_link`.\n\n"
                "Consider:\n"
                "1. Direct causation (A directly causes B)\n"
                "2. Enabling relationships (A makes B possible)\n"
                "3. Prevention (A stops B from happening)\n"
                "4. Temporal relationships (A delays/accelerates B)\n\n"
                "Be rigorous - only assert causal links you have evidence for."
            ),
            include_current_state=True,
        )
        
        await self._reasoning_loop(prompt)
    
    def _build_prompt(
        self,
        context: ContextWindow,
        task: str,
        include_current_state: bool = False,
    ) -> str:
        """Build the prompt for a reasoning pass."""
        sections = context.to_prompt_sections()
        
        prompt_parts = [
            "# Incident Analysis Task\n",
            task,
            "\n\n# Evidence\n",
        ]
        
        for section in sections:
            prompt_parts.append(f"\n## {section['title']}\n{section['content']}\n")
        
        if include_current_state and self._events:
            prompt_parts.append("\n# Current Timeline State\n")
            prompt_parts.append(f"Entities registered: {len(self._entities)}\n")
            prompt_parts.append(f"Events identified: {len(self._events)}\n")
            prompt_parts.append(f"Causal links: {len(self._timeline.causal_links)}\n")
            
            prompt_parts.append("\nEvents so far:\n")
            for event in sorted(self._events.values(), key=lambda e: e.timestamp):
                prompt_parts.append(
                    f"- [{event.id}] {event.timestamp.isoformat()}: {event.description[:100]}\n"
                )
        
        return "".join(prompt_parts)
    
    async def _reasoning_loop(self, initial_prompt: str) -> None:
        """Run the reasoning loop with function calling."""
        # Initialize messages
        self._messages = [
            Message.system(TIMELINE_SYSTEM_PROMPT),
            Message.user(initial_prompt),
        ]
        
        max_iterations = 20
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            
            # Generate response
            response = await self.provider.generate(
                messages=self._messages,
                functions=self.function_registry.get_schemas(),
                enable_thinking=True,
            )
            
            logger.debug(
                "VLM response",
                iteration=iteration,
                has_text=bool(response.text),
                function_calls=len(response.function_calls),
                tokens=response.total_tokens,
            )
            
            # Handle function calls
            if response.function_calls:
                function_results = []
                for fc in response.function_calls:
                    result = await self._handle_function_call(fc.name, fc.arguments)
                    function_results.append({
                        "call_id": fc.id,
                        "name": fc.name,
                        "result": result,
                    })
                
                # Add assistant message with function calls
                self._messages.append(Message(
                    role="assistant",
                    content=[],
                    function_calls=[{
                        "id": fc.id,
                        "name": fc.name,
                        "arguments": fc.arguments,
                    } for fc in response.function_calls],
                ))
                
                # Add function results
                self._messages.append(Message(
                    role="user",
                    content=[],
                    function_results=function_results,
                ))
            else:
                # No function calls - model is done
                if response.text:
                    self._messages.append(Message.assistant(response.text))
                break
            
            if not response.requires_continuation:
                break
        
        if iteration >= max_iterations:
            logger.warning("Reached max reasoning iterations")
    
    async def _handle_function_call(
        self, 
        name: str, 
        arguments: dict[str, Any]
    ) -> dict[str, Any]:
        """Handle a function call from the VLM."""
        logger.debug("Handling function call", name=name, arguments=arguments)
        
        try:
            if name == "emit_event":
                return await self._handle_emit_event(arguments)
            elif name == "add_causal_link":
                return await self._handle_add_causal_link(arguments)
            elif name == "register_entity":
                return await self._handle_register_entity(arguments)
            elif name == "update_entity_state":
                return await self._handle_update_entity_state(arguments)
            elif name == "flag_uncertainty":
                return await self._handle_flag_uncertainty(arguments)
            elif name == "request_frame_analysis":
                return await self._handle_request_frame_analysis(arguments)
            elif name == "set_timeline_bounds":
                return await self._handle_set_timeline_bounds(arguments)
            else:
                return {"error": f"Unknown function: {name}"}
        except Exception as e:
            logger.error("Function call error", name=name, error=str(e))
            return {"error": str(e)}
    
    async def _handle_emit_event(self, args: dict[str, Any]) -> dict[str, Any]:
        """Handle emit_event function call."""
        event_id = str(uuid4())
        
        # Parse timestamp
        timestamp = datetime.fromisoformat(args["timestamp"].replace("Z", "+00:00"))
        
        # Parse event type
        event_type = EventType(args["event_type"])
        
        # Parse entities
        entity_refs = []
        for entity_data in args.get("entities", []):
            entity_id = entity_data.get("id")
            if entity_id and entity_id in self._entities:
                entity_refs.append(UUID(entity_id))
        
        # Create event
        event = Event(
            id=UUID(event_id),
            timestamp=timestamp,
            event_type=event_type,
            description=args["description"],
            entities=entity_refs,
            confidence=args.get("confidence", 0.8),
            reasoning=args.get("reasoning"),
        )
        
        self._events[event_id] = event
        
        return {
            "success": True,
            "event_id": event_id,
            "message": f"Event registered: {event.description[:50]}...",
        }
    
    async def _handle_add_causal_link(self, args: dict[str, Any]) -> dict[str, Any]:
        """Handle add_causal_link function call."""
        source_id = args["source_event_id"]
        target_id = args["target_event_id"]
        
        # Validate events exist
        if source_id not in self._events:
            return {"error": f"Source event not found: {source_id}"}
        if target_id not in self._events:
            return {"error": f"Target event not found: {target_id}"}
        
        # Create causal link
        link = CausalLink(
            source_event_id=UUID(source_id),
            target_event_id=UUID(target_id),
            relation=CausalRelation(args["relation"]),
            mechanism=args["mechanism"],
            confidence=args.get("confidence", 0.8),
            delay_seconds=args.get("delay_seconds"),
            reasoning=args.get("reasoning"),
        )
        
        self._timeline.causal_links.append(link)
        
        return {
            "success": True,
            "link_id": str(link.id),
            "message": f"Causal link added: {source_id} {args['relation']} {target_id}",
        }
    
    async def _handle_register_entity(self, args: dict[str, Any]) -> dict[str, Any]:
        """Handle register_entity function call."""
        entity_id = args["entity_id"]
        
        entity = Entity(
            id=UUID(entity_id) if len(entity_id) == 36 else uuid4(),
            name=args["name"],
            entity_type=args["entity_type"],
            properties=args.get("properties", {}),
        )
        
        self._entities[str(entity.id)] = entity
        
        return {
            "success": True,
            "entity_id": str(entity.id),
            "message": f"Entity registered: {entity.name} ({entity.entity_type})",
        }
    
    async def _handle_update_entity_state(self, args: dict[str, Any]) -> dict[str, Any]:
        """Handle update_entity_state function call."""
        entity_id = args["entity_id"]
        
        if entity_id not in self._entities:
            return {"error": f"Entity not found: {entity_id}"}
        
        entity = self._entities[entity_id]
        timestamp = datetime.fromisoformat(args["timestamp"].replace("Z", "+00:00"))
        
        # Parse position/velocity if provided
        position = None
        if "position" in args:
            p = args["position"]
            position = (p.get("x", 0), p.get("y", 0), p.get("z", 0))
        
        velocity = None
        if "velocity" in args:
            v = args["velocity"]
            velocity = (v.get("vx", 0), v.get("vy", 0), v.get("vz", 0))
        
        # Create state
        state = EntityState(
            timestamp=timestamp,
            position=position,
            velocity=velocity,
            properties={
                args.get("property_name", "unknown"): args.get("new_value")
            } if "property_name" in args else {},
        )
        
        entity.states.append(state)
        
        return {
            "success": True,
            "message": f"Entity state updated: {entity.name} at {timestamp.isoformat()}",
        }
    
    async def _handle_flag_uncertainty(self, args: dict[str, Any]) -> dict[str, Any]:
        """Handle flag_uncertainty function call."""
        uncertainty = {
            "context": args["context"],
            "type": args["uncertainty_type"],
            "description": args["description"],
            "possible_interpretations": args.get("possible_interpretations", []),
            "data_needed": args.get("data_needed"),
            "impact": args.get("impact", "medium"),
        }
        
        self._uncertainties.append(uncertainty)
        
        return {
            "success": True,
            "message": f"Uncertainty flagged: {args['uncertainty_type']} - {args['description'][:50]}...",
        }
    
    async def _handle_request_frame_analysis(self, args: dict[str, Any]) -> dict[str, Any]:
        """Handle request_frame_analysis function call."""
        video_id = UUID(args["video_id"])
        start = datetime.fromisoformat(args["start_timestamp"].replace("Z", "+00:00"))
        end = datetime.fromisoformat(args["end_timestamp"].replace("Z", "+00:00"))
        
        # Request detailed context
        items = self.context_manager.request_detail(
            artifact_id=video_id,
            time_range=(start, end),
            query=args.get("query"),
        )
        
        return {
            "success": True,
            "frames_added": len(items),
            "message": f"Added {len(items)} frames to context for analysis",
        }
    
    async def _handle_set_timeline_bounds(self, args: dict[str, Any]) -> dict[str, Any]:
        """Handle set_timeline_bounds function call."""
        start = datetime.fromisoformat(args["start_time"].replace("Z", "+00:00"))
        end = datetime.fromisoformat(args["end_time"].replace("Z", "+00:00"))
        
        self._timeline.start_time = start
        self._timeline.end_time = end
        
        return {
            "success": True,
            "message": f"Timeline bounds set: {start.isoformat()} to {end.isoformat()}",
        }
