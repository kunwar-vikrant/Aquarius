"""VLM Reasoning Core - Provider abstraction and function calling."""

from cwe.reasoning.providers.base import VLMProvider, VLMResponse, VLMConfig
from cwe.reasoning.providers.gemini import GeminiProvider
from cwe.reasoning.providers.anthropic import AnthropicProvider
from cwe.reasoning.providers.openai import OpenAIProvider
from cwe.reasoning.function_schema import (
    VLMFunction,
    FunctionRegistry,
    get_timeline_functions,
)
from cwe.reasoning.context_manager import ContextManager, ContextWindow
from cwe.reasoning.reasoner import TimelineReasoner

__all__ = [
    # Providers
    "VLMProvider",
    "VLMResponse",
    "VLMConfig",
    "GeminiProvider",
    "AnthropicProvider",
    "OpenAIProvider",
    # Functions
    "VLMFunction",
    "FunctionRegistry",
    "get_timeline_functions",
    # Context
    "ContextManager",
    "ContextWindow",
    # Reasoner
    "TimelineReasoner",
]
