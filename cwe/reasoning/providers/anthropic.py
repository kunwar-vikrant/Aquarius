"""Anthropic Claude VLM provider."""

from __future__ import annotations

import base64
import time
from pathlib import Path
from typing import Any, AsyncIterator

from cwe.reasoning.providers.base import (
    VLMProvider,
    VLMProviderType,
    VLMConfig,
    VLMResponse,
    Message,
    FunctionCall,
)


class AnthropicProvider(VLMProvider):
    """
    Anthropic Claude VLM provider.
    
    Supports Claude 3.5/4 Opus, Sonnet, and Haiku models.
    Best-in-class for complex reasoning and structured output.
    """
    
    def __init__(self, config: VLMConfig):
        super().__init__(config)
        self._client = None
    
    @property
    def provider_type(self) -> VLMProviderType:
        return VLMProviderType.ANTHROPIC
    
    @property
    def default_model(self) -> str:
        return "claude-sonnet-4-20250514"
    
    def _get_client(self):
        """Get or create the Anthropic client."""
        if self._client is None:
            import anthropic
            self._client = anthropic.AsyncAnthropic(api_key=self.config.api_key)
        return self._client
    
    def _convert_messages(self, messages: list[Message]) -> tuple[str | None, list[dict]]:
        """Convert our Message format to Anthropic format."""
        system = None
        anthropic_messages = []
        
        for msg in messages:
            if msg.role == "system":
                system = msg.content[0].text
                continue
            
            content = []
            for part in msg.content:
                if part.type == "text":
                    content.append({"type": "text", "text": part.text})
                elif part.type == "image":
                    if part.image_path:
                        with open(part.image_path, "rb") as f:
                            image_data = base64.standard_b64encode(f.read()).decode()
                        mime_type = part.mime_type or self._guess_mime_type(part.image_path)
                        content.append({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": mime_type,
                                "data": image_data,
                            }
                        })
                    elif part.image_base64:
                        content.append({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": part.mime_type or "image/jpeg",
                                "data": part.image_base64,
                            }
                        })
                    elif part.image_url:
                        content.append({
                            "type": "image",
                            "source": {
                                "type": "url",
                                "url": part.image_url,
                            }
                        })
            
            anthropic_messages.append({
                "role": msg.role,
                "content": content,
            })
        
        return system, anthropic_messages
    
    def _guess_mime_type(self, path: str) -> str:
        """Guess MIME type from file extension."""
        ext = Path(path).suffix.lower()
        return {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }.get(ext, "image/jpeg")
    
    def _convert_functions(self, functions: list[dict[str, Any]]) -> list[dict]:
        """Convert function schemas to Anthropic tool format."""
        if not functions:
            return []
        
        tools = []
        for func in functions:
            tools.append({
                "name": func["name"],
                "description": func.get("description", ""),
                "input_schema": func.get("parameters", {}),
            })
        
        return tools
    
    async def generate(
        self,
        messages: list[Message],
        functions: list[dict[str, Any]] | None = None,
        **kwargs,
    ) -> VLMResponse:
        """Generate a response from Claude."""
        client = self._get_client()
        system, anthropic_messages = self._convert_messages(messages)
        
        # Build request params
        params = {
            "model": self.model,
            "messages": anthropic_messages,
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
        }
        
        if system:
            params["system"] = system
        
        if functions:
            params["tools"] = self._convert_functions(functions)
        
        # Enable extended thinking if requested
        if kwargs.get("enable_thinking", False):
            params["thinking"] = {
                "type": "enabled",
                "budget_tokens": kwargs.get("thinking_budget", 10000),
            }
        
        start_time = time.time()
        
        response = await client.messages.create(**params)
        
        latency_ms = (time.time() - start_time) * 1000
        
        # Parse response
        text = None
        function_calls = []
        thinking_tokens = 0
        
        for block in response.content:
            if block.type == "text":
                text = block.text
            elif block.type == "tool_use":
                function_calls.append(FunctionCall(
                    id=block.id,
                    name=block.name,
                    arguments=block.input,
                ))
            elif block.type == "thinking":
                thinking_tokens = getattr(block, "token_count", 0)
        
        return VLMResponse(
            text=text,
            function_calls=function_calls,
            requires_continuation=len(function_calls) > 0,
            finish_reason=response.stop_reason,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            total_tokens=response.usage.input_tokens + response.usage.output_tokens,
            thinking_tokens=thinking_tokens,
            raw_response=response,
            latency_ms=latency_ms,
            provider="anthropic",
            model=self.model,
        )
    
    async def generate_stream(
        self,
        messages: list[Message],
        functions: list[dict[str, Any]] | None = None,
        **kwargs,
    ) -> AsyncIterator[VLMResponse]:
        """Generate a streaming response from Claude."""
        client = self._get_client()
        system, anthropic_messages = self._convert_messages(messages)
        
        params = {
            "model": self.model,
            "messages": anthropic_messages,
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
        }
        
        if system:
            params["system"] = system
        
        if functions:
            params["tools"] = self._convert_functions(functions)
        
        async with client.messages.stream(**params) as stream:
            async for text in stream.text_stream:
                yield VLMResponse(
                    text=text,
                    provider="anthropic",
                    model=self.model,
                )
    
    async def count_tokens(self, messages: list[Message]) -> int:
        """Count tokens in the given messages."""
        client = self._get_client()
        system, anthropic_messages = self._convert_messages(messages)
        
        # Use the count_tokens endpoint
        result = await client.messages.count_tokens(
            model=self.model,
            messages=anthropic_messages,
            system=system,
        )
        
        return result.input_tokens
    
    def supports_vision(self) -> bool:
        return True
    
    def supports_function_calling(self) -> bool:
        return True
    
    def max_context_tokens(self) -> int:
        return 200_000  # Claude 3.5/4 supports 200K tokens
