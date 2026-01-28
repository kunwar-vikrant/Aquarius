"""Base VLM provider interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, AsyncIterator
from uuid import UUID, uuid4

from pydantic import BaseModel


class VLMProviderType(str, Enum):
    """Supported VLM providers."""
    
    GEMINI = "gemini"
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    XAI = "xai"


@dataclass
class VLMConfig:
    """Configuration for a VLM provider."""
    
    provider: VLMProviderType
    api_key: str
    model: str | None = None  # Use provider default if None
    
    # Generation parameters
    temperature: float = 0.2
    max_tokens: int = 8192
    top_p: float = 0.95
    
    # Timeout and retry
    timeout_seconds: int = 300
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    
    # Provider-specific options
    options: dict[str, Any] = field(default_factory=dict)


class ContentPart(BaseModel):
    """A part of a message (text, image, video frame, etc.)."""
    
    type: str  # "text", "image", "video_frame"
    
    # For text
    text: str | None = None
    
    # For images/video frames
    image_path: str | None = None
    image_base64: str | None = None
    image_url: str | None = None
    mime_type: str | None = None
    
    # Metadata
    metadata: dict[str, Any] = {}


class Message(BaseModel):
    """A message in a conversation."""
    
    role: str  # "user", "assistant", "system"
    content: list[ContentPart]
    
    # Function call info
    function_calls: list[dict[str, Any]] | None = None
    function_results: list[dict[str, Any]] | None = None
    
    @classmethod
    def user(cls, text: str, images: list[str] | None = None) -> Message:
        """Create a user message."""
        parts = [ContentPart(type="text", text=text)]
        if images:
            for img_path in images:
                parts.append(ContentPart(
                    type="image",
                    image_path=img_path,
                ))
        return cls(role="user", content=parts)
    
    @classmethod
    def assistant(cls, text: str) -> Message:
        """Create an assistant message."""
        return cls(role="assistant", content=[ContentPart(type="text", text=text)])
    
    @classmethod
    def system(cls, text: str) -> Message:
        """Create a system message."""
        return cls(role="system", content=[ContentPart(type="text", text=text)])


@dataclass
class FunctionCall:
    """A function call made by the VLM."""
    
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class VLMResponse:
    """Response from a VLM provider."""
    
    id: str = field(default_factory=lambda: str(uuid4()))
    
    # Text response
    text: str | None = None
    
    # Function calls (if any)
    function_calls: list[FunctionCall] = field(default_factory=list)
    
    # Whether the model wants to continue (after function results)
    requires_continuation: bool = False
    
    # Finish reason
    finish_reason: str | None = None  # "stop", "function_call", "max_tokens", etc.
    
    # Usage stats
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    
    # Thinking/reasoning tokens (if supported)
    thinking_tokens: int = 0
    
    # Raw response for debugging
    raw_response: Any = None
    
    # Timing
    latency_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # Provider info
    provider: str | None = None
    model: str | None = None


class VLMProvider(ABC):
    """Abstract base class for VLM providers."""
    
    def __init__(self, config: VLMConfig):
        self.config = config
        self._session_id: str | None = None
    
    @property
    @abstractmethod
    def provider_type(self) -> VLMProviderType:
        """Get the provider type."""
        pass
    
    @property
    @abstractmethod
    def default_model(self) -> str:
        """Get the default model for this provider."""
        pass
    
    @property
    def model(self) -> str:
        """Get the model to use."""
        return self.config.model or self.default_model
    
    @abstractmethod
    async def generate(
        self,
        messages: list[Message],
        functions: list[dict[str, Any]] | None = None,
        **kwargs,
    ) -> VLMResponse:
        """
        Generate a response from the VLM.
        
        Args:
            messages: Conversation history
            functions: Function schemas for function calling
            **kwargs: Additional provider-specific options
            
        Returns:
            VLMResponse with text and/or function calls
        """
        pass
    
    @abstractmethod
    async def generate_stream(
        self,
        messages: list[Message],
        functions: list[dict[str, Any]] | None = None,
        **kwargs,
    ) -> AsyncIterator[VLMResponse]:
        """
        Generate a streaming response from the VLM.
        
        Args:
            messages: Conversation history
            functions: Function schemas for function calling
            **kwargs: Additional provider-specific options
            
        Yields:
            Partial VLMResponse chunks
        """
        pass
    
    @abstractmethod
    async def count_tokens(self, messages: list[Message]) -> int:
        """Count tokens in the given messages."""
        pass
    
    @abstractmethod
    def supports_vision(self) -> bool:
        """Whether this provider supports image/video input."""
        pass
    
    @abstractmethod
    def supports_function_calling(self) -> bool:
        """Whether this provider supports function calling."""
        pass
    
    @abstractmethod
    def max_context_tokens(self) -> int:
        """Maximum context window size in tokens."""
        pass
    
    def start_session(self) -> str:
        """Start a new reasoning session."""
        self._session_id = str(uuid4())
        return self._session_id
    
    def end_session(self) -> None:
        """End the current session."""
        self._session_id = None
