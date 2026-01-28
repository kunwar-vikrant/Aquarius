"""Context management for long-form VLM reasoning."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from cwe.models.artifact import Artifact, VideoArtifact, LogArtifact
from cwe.models.timeline import Event


class ContextPriority(str, Enum):
    """Priority levels for context items."""
    
    CRITICAL = "critical"  # Must be included
    HIGH = "high"          # Include if space allows
    MEDIUM = "medium"      # Include if needed
    LOW = "low"            # Summarize or omit


@dataclass
class ContextItem:
    """An item in the context window."""
    
    id: str = field(default_factory=lambda: str(uuid4()))
    content_type: str = ""  # "video_frame", "log_entry", "event", "summary", etc.
    content: str = ""  # Serialized content
    
    # Temporal info
    timestamp: datetime | None = None
    time_range: tuple[datetime, datetime] | None = None
    
    # Priority and size
    priority: ContextPriority = ContextPriority.MEDIUM
    estimated_tokens: int = 0
    
    # Source reference
    artifact_id: UUID | None = None
    artifact_type: str | None = None
    
    # Whether this is a summary of other items
    is_summary: bool = False
    summarizes: list[str] = field(default_factory=list)  # IDs of summarized items


@dataclass
class ContextWindow:
    """
    A window of context to send to the VLM.
    
    Manages prioritization, token budgets, and hierarchical summarization
    for incidents that exceed the context window.
    """
    
    max_tokens: int = 1_000_000
    items: list[ContextItem] = field(default_factory=list)
    
    # Running token count
    _current_tokens: int = 0
    
    # Summarization state
    summaries: dict[str, ContextItem] = field(default_factory=dict)
    
    def add_item(self, item: ContextItem) -> bool:
        """
        Add an item to the context window.
        
        Returns True if the item was added, False if it was summarized or dropped.
        """
        if self._current_tokens + item.estimated_tokens <= self.max_tokens:
            self.items.append(item)
            self._current_tokens += item.estimated_tokens
            return True
        
        # Try to make room by summarizing lower-priority items
        if item.priority in (ContextPriority.CRITICAL, ContextPriority.HIGH):
            freed = self._summarize_low_priority()
            if self._current_tokens + item.estimated_tokens <= self.max_tokens:
                self.items.append(item)
                self._current_tokens += item.estimated_tokens
                return True
        
        return False
    
    def _summarize_low_priority(self) -> int:
        """Summarize low-priority items to free up tokens."""
        # Group low-priority items by type and time range
        low_priority = [
            item for item in self.items
            if item.priority in (ContextPriority.LOW, ContextPriority.MEDIUM)
            and not item.is_summary
        ]
        
        if not low_priority:
            return 0
        
        # For now, just remove the lowest priority items
        # TODO: Implement actual summarization with VLM
        freed = 0
        for item in sorted(low_priority, key=lambda x: x.priority.value):
            self.items.remove(item)
            freed += item.estimated_tokens
            self._current_tokens -= item.estimated_tokens
            
            if freed > self.max_tokens * 0.2:  # Free up at least 20%
                break
        
        return freed
    
    def get_items_in_range(
        self, 
        start: datetime, 
        end: datetime
    ) -> list[ContextItem]:
        """Get items within a time range."""
        result = []
        for item in self.items:
            if item.timestamp and start <= item.timestamp <= end:
                result.append(item)
            elif item.time_range:
                item_start, item_end = item.time_range
                if not (item_end < start or item_start > end):
                    result.append(item)
        return result
    
    def to_prompt_sections(self) -> list[dict[str, Any]]:
        """Convert context to prompt sections for the VLM."""
        # Group items by type
        by_type: dict[str, list[ContextItem]] = {}
        for item in self.items:
            by_type.setdefault(item.content_type, []).append(item)
        
        sections = []
        
        # Order: summaries first, then by type
        if "summary" in by_type:
            sections.append({
                "title": "Overview",
                "content": "\n\n".join(item.content for item in by_type["summary"]),
            })
        
        for content_type in ["video_frame", "log_entry", "event", "report_excerpt"]:
            if content_type in by_type:
                items = sorted(by_type[content_type], key=lambda x: x.timestamp or datetime.min)
                sections.append({
                    "title": f"{content_type.replace('_', ' ').title()}s",
                    "content": "\n\n".join(item.content for item in items),
                })
        
        return sections
    
    @property
    def current_tokens(self) -> int:
        return self._current_tokens
    
    @property
    def remaining_tokens(self) -> int:
        return self.max_tokens - self._current_tokens


class ContextManager:
    """
    Manages context for VLM reasoning across an incident analysis session.
    
    Implements:
    - Hierarchical summarization for long incidents
    - Priority-based context allocation
    - Rolling windows for extended timelines
    - VLM-guided context navigation
    """
    
    def __init__(self, max_tokens: int = 1_000_000):
        self.max_tokens = max_tokens
        self.window = ContextWindow(max_tokens=max_tokens)
        
        # Full artifact store (not in context, but available for retrieval)
        self._artifacts: dict[UUID, Artifact] = {}
        self._events: dict[UUID, Event] = {}
        
        # Summarization cache
        self._summaries: dict[str, str] = {}
    
    def add_artifact(self, artifact: Artifact) -> None:
        """Add an artifact to the context manager."""
        self._artifacts[artifact.id] = artifact
    
    def add_event(self, event: Event) -> None:
        """Add an event to the context manager."""
        self._events[event.id] = event
    
    def build_initial_context(
        self,
        focus_time: datetime | None = None,
        focus_radius_seconds: float = 300,
    ) -> ContextWindow:
        """
        Build the initial context window for analysis.
        
        If focus_time is provided, prioritize context around that time.
        """
        self.window = ContextWindow(max_tokens=self.max_tokens)
        
        # Add artifact summaries first
        for artifact in self._artifacts.values():
            summary = self._get_artifact_summary(artifact)
            self.window.add_item(ContextItem(
                content_type="summary",
                content=summary,
                priority=ContextPriority.HIGH,
                estimated_tokens=len(summary) // 4,
                artifact_id=artifact.id,
                artifact_type=artifact.artifact_type.value,
                is_summary=True,
            ))
        
        # Add events
        for event in sorted(self._events.values(), key=lambda e: e.timestamp):
            priority = ContextPriority.MEDIUM
            
            # Higher priority for events near focus time
            if focus_time and event.timestamp:
                delta = abs((event.timestamp - focus_time).total_seconds())
                if delta <= focus_radius_seconds:
                    priority = ContextPriority.HIGH
                elif delta > focus_radius_seconds * 3:
                    priority = ContextPriority.LOW
            
            event_content = self._format_event(event)
            self.window.add_item(ContextItem(
                content_type="event",
                content=event_content,
                timestamp=event.timestamp,
                priority=priority,
                estimated_tokens=len(event_content) // 4,
            ))
        
        return self.window
    
    def request_detail(
        self,
        artifact_id: UUID,
        time_range: tuple[datetime, datetime] | None = None,
        query: str | None = None,
    ) -> list[ContextItem]:
        """
        Request detailed context from a specific artifact.
        
        Called when VLM uses the request_frame_analysis function.
        """
        artifact = self._artifacts.get(artifact_id)
        if not artifact:
            return []
        
        items = []
        
        if isinstance(artifact, VideoArtifact):
            items.extend(self._get_video_frames(artifact, time_range))
        elif isinstance(artifact, LogArtifact):
            items.extend(self._get_log_entries(artifact, time_range))
        
        # Add new items to context
        for item in items:
            self.window.add_item(item)
        
        return items
    
    def _get_artifact_summary(self, artifact: Artifact) -> str:
        """Get or generate a summary for an artifact."""
        cache_key = f"summary_{artifact.id}"
        if cache_key in self._summaries:
            return self._summaries[cache_key]
        
        # Generate summary based on artifact type
        summary_parts = [
            f"## {artifact.artifact_type.value.title()}: {artifact.filename}",
        ]
        
        if artifact.start_time and artifact.end_time:
            summary_parts.append(
                f"Time range: {artifact.start_time.isoformat()} to {artifact.end_time.isoformat()}"
            )
        
        if isinstance(artifact, VideoArtifact):
            summary_parts.extend([
                f"Resolution: {artifact.width}x{artifact.height}",
                f"Duration: {artifact.duration} seconds" if artifact.duration else "",
                f"FPS: {artifact.fps}",
                f"Keyframes: {len(artifact.keyframes)}",
            ])
        elif isinstance(artifact, LogArtifact):
            summary_parts.extend([
                f"Format: {artifact.log_format}",
                f"Entries: {artifact.entry_count}",
            ])
        
        summary = "\n".join(p for p in summary_parts if p)
        self._summaries[cache_key] = summary
        return summary
    
    def _format_event(self, event: Event) -> str:
        """Format an event for inclusion in context."""
        return (
            f"[{event.timestamp.isoformat()}] {event.event_type.value}: {event.description}\n"
            f"Confidence: {event.confidence:.2f}"
        )
    
    def _get_video_frames(
        self,
        artifact: VideoArtifact,
        time_range: tuple[datetime, datetime] | None = None,
    ) -> list[ContextItem]:
        """Extract video frames for context."""
        items = []
        
        for frame in artifact.keyframes:
            if time_range:
                start, end = time_range
                if not (start <= frame.timestamp <= end):
                    continue
            
            # Create a text description of the frame
            # In practice, this would include the actual image
            content = (
                f"Video frame {frame.frame_number} at {frame.timestamp.isoformat()}\n"
                f"Scene: {frame.scene_id or 'unknown'}\n"
                f"[Image: {frame.file_path}]"
            )
            
            items.append(ContextItem(
                content_type="video_frame",
                content=content,
                timestamp=frame.timestamp,
                priority=ContextPriority.HIGH,
                estimated_tokens=500,  # Approximate for image
                artifact_id=artifact.id,
                artifact_type="video",
            ))
        
        return items
    
    def _get_log_entries(
        self,
        artifact: LogArtifact,
        time_range: tuple[datetime, datetime] | None = None,
    ) -> list[ContextItem]:
        """Extract log entries for context."""
        items = []
        
        for entry in artifact.sample_entries:
            if time_range and entry.timestamp:
                start, end = time_range
                if not (start <= entry.timestamp <= end):
                    continue
            
            content = (
                f"[{entry.timestamp.isoformat() if entry.timestamp else 'unknown'}] "
                f"{entry.level or 'INFO'}: {entry.message}"
            )
            
            items.append(ContextItem(
                content_type="log_entry",
                content=content,
                timestamp=entry.timestamp,
                priority=ContextPriority.MEDIUM,
                estimated_tokens=len(content) // 4,
                artifact_id=artifact.id,
                artifact_type="log",
            ))
        
        return items
