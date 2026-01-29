"""OpenAI GPT-4 VLM provider."""

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


class OpenAIProvider(VLMProvider):
    """
    OpenAI GPT-4 VLM provider.
    
    Supports GPT-4o, GPT-4 Turbo, and other OpenAI models.
    """
    
    def __init__(self, config: VLMConfig):
        super().__init__(config)
        self._client = None
    
    @property
    def provider_type(self) -> VLMProviderType:
        return VLMProviderType.OPENAI
    
    @property
    def default_model(self) -> str:
        return "gpt-4o"
    
    def _get_client(self):
        """Get or create the OpenAI client."""
        if self._client is None:
            import openai
            self._client = openai.AsyncOpenAI(api_key=self.config.api_key)
        return self._client
    
    def _convert_messages(self, messages: list[Message]) -> list[dict]:
        """Convert our Message format to OpenAI format."""
        openai_messages = []
        
        for msg in messages:
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
            
            # Simplify if only text
            if len(content) == 1 and content[0]["type"] == "text":
                content = content[0]["text"]
            
            openai_messages.append({
                "role": msg.role,
                "content": content,
            })
        
        return openai_messages
    
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
    
    def _convert_functions(self, functions: list) -> list[dict]:
        """Convert function schemas to OpenAI tool format.
        
        Handles both VLMFunction objects and dict schemas.
        """
        from cwe.reasoning.function_schema import VLMFunction
        
        if not functions:
            return []
        
        tools = []
        for func in functions:
            # Handle VLMFunction objects
            if isinstance(func, VLMFunction):
                name = func.name
                description = func.description
                parameters = func.parameters
            else:
                # Handle dict schemas
                name = func["name"]
                description = func.get("description", "")
                parameters = func.get("parameters", {})
            
            tools.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": parameters,
                }
            })
        
        return tools
    
    async def generate(
        self,
        messages: list[Message],
        functions: list[dict[str, Any]] | None = None,
        **kwargs,
    ) -> VLMResponse:
        """Generate a response from GPT-4."""
        import json
        
        client = self._get_client()
        openai_messages = self._convert_messages(messages)
        
        params = {
            "model": self.model,
            "messages": openai_messages,
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
        
        return VLMResponse(
            text=text,
            function_calls=function_calls,
            requires_continuation=len(function_calls) > 0,
            finish_reason=response.choices[0].finish_reason,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens,
            raw_response=response,
            latency_ms=latency_ms,
            provider="openai",
            model=self.model,
        )
    
    async def generate_stream(
        self,
        messages: list[Message],
        functions: list[dict[str, Any]] | None = None,
        **kwargs,
    ) -> AsyncIterator[VLMResponse]:
        """Generate a streaming response from GPT-4."""
        client = self._get_client()
        openai_messages = self._convert_messages(messages)
        
        params = {
            "model": self.model,
            "messages": openai_messages,
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
                    provider="openai",
                    model=self.model,
                )
    
    async def count_tokens(self, messages: list[Message]) -> int:
        """Count tokens in the given messages."""
        # OpenAI doesn't have a direct token count API for messages
        # Use tiktoken for estimation
        import tiktoken
        
        encoding = tiktoken.encoding_for_model(self.model)
        openai_messages = self._convert_messages(messages)
        
        total = 0
        for msg in openai_messages:
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
        return "gpt-4" in self.model.lower()
    
    def supports_function_calling(self) -> bool:
        return True
    
    def max_context_tokens(self) -> int:
        if "gpt-4o" in self.model:
            return 128_000
        elif "gpt-4-turbo" in self.model:
            return 128_000
        else:
            return 8_192
