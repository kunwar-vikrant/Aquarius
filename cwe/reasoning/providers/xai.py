"""xAI Grok VLM provider."""

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


class XAIProvider(VLMProvider):
    """
    xAI Grok VLM provider.
    
    Supports Grok models via xAI's OpenAI-compatible API.
    Best-in-class for reasoning and real-time knowledge.
    """
    
    # xAI API base URL
    BASE_URL = "https://api.x.ai/v1"
    
    def __init__(self, config: VLMConfig):
        super().__init__(config)
        self._client = None
    
    @property
    def provider_type(self) -> VLMProviderType:
        return VLMProviderType.XAI
    
    @property
    def default_model(self) -> str:
        return "grok-4-1-fast-reasoning"
    
    def _get_client(self):
        """Get or create the xAI client (OpenAI-compatible)."""
        if self._client is None:
            import openai
            self._client = openai.AsyncOpenAI(
                api_key=self.config.api_key,
                base_url=self.BASE_URL,
            )
        return self._client
    
    def _convert_messages(self, messages: list[Message]) -> list[dict]:
        """Convert our Message format to OpenAI-compatible format."""
        xai_messages = []
        
        for msg in messages:
            # Handle assistant messages with function/tool calls
            if msg.role == "assistant" and msg.function_calls:
                tool_calls = []
                for fc in msg.function_calls:
                    import json
                    tool_calls.append({
                        "id": fc.get("id", f"call_{len(tool_calls)}"),
                        "type": "function",
                        "function": {
                            "name": fc["name"],
                            "arguments": json.dumps(fc["arguments"]) if isinstance(fc["arguments"], dict) else fc["arguments"],
                        }
                    })
                
                # xAI requires content to be null (not empty) for tool call messages
                xai_messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": tool_calls,
                })
                continue
            
            # Handle user messages with function/tool results
            if msg.role == "user" and msg.function_results:
                for result in msg.function_results:
                    import json
                    xai_messages.append({
                        "role": "tool",
                        "tool_call_id": result.get("call_id", ""),
                        "content": json.dumps(result.get("result", {})) if isinstance(result.get("result"), dict) else str(result.get("result", "")),
                    })
                continue
            
            # Handle regular content messages
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
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{image_data}",
                            }
                        })
                    elif part.image_base64:
                        mime_type = part.mime_type or "image/jpeg"
                        content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{part.image_base64}",
                            }
                        })
                    elif part.image_url:
                        content.append({
                            "type": "image_url",
                            "image_url": {"url": part.image_url},
                        })
            
            # Skip empty content messages (shouldn't happen for regular messages)
            if not content:
                continue
                
            # Simplify if only text
            if len(content) == 1 and content[0]["type"] == "text":
                content = content[0]["text"]
            
            xai_messages.append({
                "role": msg.role,
                "content": content,
            })
        
        return xai_messages
    
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
        """Convert function schemas to OpenAI-compatible tool format."""
        if not functions:
            return []
        
        tools = []
        for func in functions:
            tools.append({
                "type": "function",
                "function": {
                    "name": func["name"],
                    "description": func.get("description", ""),
                    "parameters": func.get("parameters", {}),
                }
            })
        
        return tools
    
    async def generate(
        self,
        messages: list[Message],
        functions: list[dict[str, Any]] | None = None,
        **kwargs,
    ) -> VLMResponse:
        """Generate a response from Grok."""
        import json
        
        client = self._get_client()
        xai_messages = self._convert_messages(messages)
        
        params = {
            "model": self.model,
            "messages": xai_messages,
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
        }
        
        if functions:
            params["tools"] = self._convert_functions(functions)
        
        start_time = time.time()
        
        response = await client.chat.completions.create(**params)
        
        latency_ms = (time.time() - start_time) * 1000
        
        # Parse response
        message = response.choices[0].message
        text = message.content
        function_calls = []
        
        if message.tool_calls:
            for tc in message.tool_calls:
                function_calls.append(FunctionCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=json.loads(tc.function.arguments),
                ))
        
        # Extract reasoning tokens if available (Grok reasoning models)
        thinking_tokens = 0
        if hasattr(response, 'usage') and hasattr(response.usage, 'reasoning_tokens'):
            thinking_tokens = response.usage.reasoning_tokens or 0
        
        return VLMResponse(
            text=text,
            function_calls=function_calls,
            requires_continuation=len(function_calls) > 0,
            finish_reason=response.choices[0].finish_reason,
            input_tokens=response.usage.prompt_tokens if response.usage else 0,
            output_tokens=response.usage.completion_tokens if response.usage else 0,
            total_tokens=response.usage.total_tokens if response.usage else 0,
            thinking_tokens=thinking_tokens,
            raw_response=response,
            latency_ms=latency_ms,
            provider="xai",
            model=self.model,
        )
    
    async def generate_stream(
        self,
        messages: list[Message],
        functions: list[dict[str, Any]] | None = None,
        **kwargs,
    ) -> AsyncIterator[VLMResponse]:
        """Generate a streaming response from Grok."""
        client = self._get_client()
        xai_messages = self._convert_messages(messages)
        
        params = {
            "model": self.model,
            "messages": xai_messages,
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
            "stream": True,
        }
        
        if functions:
            params["tools"] = self._convert_functions(functions)
        
        stream = await client.chat.completions.create(**params)
        
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield VLMResponse(
                    text=chunk.choices[0].delta.content,
                    provider="xai",
                    model=self.model,
                )
    
    async def count_tokens(self, messages: list[Message]) -> int:
        """Count tokens in the given messages."""
        # Use tiktoken with gpt-4 encoding as approximation
        # xAI likely uses similar tokenization
        import tiktoken
        
        try:
            encoding = tiktoken.encoding_for_model("gpt-4")
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
        
        xai_messages = self._convert_messages(messages)
        
        total = 0
        for msg in xai_messages:
            content = msg["content"]
            if isinstance(content, str):
                total += len(encoding.encode(content))
            elif isinstance(content, list):
                for part in content:
                    if part["type"] == "text":
                        total += len(encoding.encode(part["text"]))
                    elif part["type"] == "image_url":
                        # Rough estimate for images
                        total += 1000
        
        return total
    
    def supports_vision(self) -> bool:
        """Grok models support vision."""
        return True
    
    def supports_function_calling(self) -> bool:
        """Grok models support function calling."""
        return True
    
    def max_context_tokens(self) -> int:
        """Grok models have large context windows."""
        # grok-4-1-fast-reasoning has 131072 token context
        return 131_072
