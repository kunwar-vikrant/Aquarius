"""Google Gemini VLM provider."""

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


class GeminiProvider(VLMProvider):
    """
    Google Gemini VLM provider.
    
    Supports Gemini 2.5 Pro, 2.5 Flash, and other Gemini models.
    Best-in-class for multimodal understanding, especially video.
    """
    
    def __init__(self, config: VLMConfig):
        super().__init__(config)
        self._client = None
    
    @property
    def provider_type(self) -> VLMProviderType:
        return VLMProviderType.GEMINI
    
    @property
    def default_model(self) -> str:
        return "gemini-3-flash-preview"
    
    def _get_client(self):
        """Get or create the Gemini client."""
        if self._client is None:
            from google import genai
            self._client = genai.Client(api_key=self.config.api_key)
        return self._client
    
    def _convert_messages(self, messages: list[Message]) -> tuple[str | None, list[dict]]:
        """Convert our Message format to Gemini format."""
        system_instruction = None
        contents = []
        
        for msg in messages:
            if msg.role == "system":
                # Gemini uses system_instruction separately
                system_instruction = msg.content[0].text
                continue
            
            parts = []
            for part in msg.content:
                if part.type == "text":
                    parts.append({"text": part.text})
                elif part.type == "image":
                    # Load and encode image
                    if part.image_path:
                        with open(part.image_path, "rb") as f:
                            image_data = base64.standard_b64encode(f.read()).decode()
                        mime_type = part.mime_type or self._guess_mime_type(part.image_path)
                        parts.append({
                            "inline_data": {
                                "mime_type": mime_type,
                                "data": image_data,
                            }
                        })
                    elif part.image_base64:
                        parts.append({
                            "inline_data": {
                                "mime_type": part.mime_type or "image/jpeg",
                                "data": part.image_base64,
                            }
                        })
                    elif part.image_url:
                        # Gemini supports URLs directly in some cases
                        parts.append({"file_uri": part.image_url})
            
            role = "user" if msg.role == "user" else "model"
            contents.append({"role": role, "parts": parts})
        
        return system_instruction, contents
    
    def _guess_mime_type(self, path: str) -> str:
        """Guess MIME type from file extension."""
        ext = Path(path).suffix.lower()
        return {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
            ".mp4": "video/mp4",
            ".avi": "video/avi",
            ".mov": "video/quicktime",
        }.get(ext, "application/octet-stream")
    
    def _convert_functions(self, functions: list[dict[str, Any]]) -> list[dict]:
        """Convert function schemas to Gemini tool format."""
        if not functions:
            return []
        
        tools = []
        for func in functions:
            tool = {
                "function_declarations": [{
                    "name": func["name"],
                    "description": func.get("description", ""),
                    "parameters": func.get("parameters", {}),
                }]
            }
            tools.append(tool)
        
        return tools
    
    async def generate(
        self,
        messages: list[Message],
        functions: list[dict[str, Any]] | None = None,
        **kwargs,
    ) -> VLMResponse:
        """Generate a response from Gemini."""
        from google.genai import types
        
        client = self._get_client()
        system_instruction, contents = self._convert_messages(messages)
        
        # Build generation config
        generation_config = types.GenerateContentConfig(
            temperature=kwargs.get("temperature", self.config.temperature),
            max_output_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            top_p=kwargs.get("top_p", self.config.top_p),
        )
        
        # Add system instruction if present
        if system_instruction:
            generation_config.system_instruction = system_instruction
        
        # Add tools if functions provided
        if functions:
            tools = self._convert_functions(functions)
            generation_config.tools = tools
        
        # Enable thinking if requested
        if kwargs.get("enable_thinking", True):
            generation_config.thinking_config = types.ThinkingConfig(
                thinking_budget=kwargs.get("thinking_budget", 10000)
            )
        
        start_time = time.time()
        
        # Make the API call
        response = await client.aio.models.generate_content(
            model=self.model,
            contents=contents,
            config=generation_config,
        )
        
        latency_ms = (time.time() - start_time) * 1000
        
        # Parse response
        text = None
        function_calls = []
        
        for candidate in response.candidates:
            for part in candidate.content.parts:
                if hasattr(part, "text") and part.text:
                    text = part.text
                elif hasattr(part, "function_call") and part.function_call:
                    fc = part.function_call
                    function_calls.append(FunctionCall(
                        id=f"call_{len(function_calls)}",
                        name=fc.name,
                        arguments=dict(fc.args) if fc.args else {},
                    ))
        
        # Get usage stats
        usage = response.usage_metadata if hasattr(response, "usage_metadata") else None
        
        return VLMResponse(
            text=text,
            function_calls=function_calls,
            requires_continuation=len(function_calls) > 0,
            finish_reason=str(response.candidates[0].finish_reason) if response.candidates else None,
            input_tokens=usage.prompt_token_count if usage else 0,
            output_tokens=usage.candidates_token_count if usage else 0,
            total_tokens=usage.total_token_count if usage else 0,
            thinking_tokens=getattr(usage, "thinking_token_count", 0) if usage else 0,
            raw_response=response,
            latency_ms=latency_ms,
            provider="gemini",
            model=self.model,
        )
    
    async def generate_stream(
        self,
        messages: list[Message],
        functions: list[dict[str, Any]] | None = None,
        **kwargs,
    ) -> AsyncIterator[VLMResponse]:
        """Generate a streaming response from Gemini."""
        from google.genai import types
        
        client = self._get_client()
        system_instruction, contents = self._convert_messages(messages)
        
        generation_config = types.GenerateContentConfig(
            temperature=kwargs.get("temperature", self.config.temperature),
            max_output_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            top_p=kwargs.get("top_p", self.config.top_p),
        )
        
        if system_instruction:
            generation_config.system_instruction = system_instruction
        
        if functions:
            tools = self._convert_functions(functions)
            generation_config.tools = tools
        
        async for chunk in client.aio.models.generate_content_stream(
            model=self.model,
            contents=contents,
            config=generation_config,
        ):
            text = None
            if chunk.candidates:
                for part in chunk.candidates[0].content.parts:
                    if hasattr(part, "text"):
                        text = part.text
            
            yield VLMResponse(
                text=text,
                provider="gemini",
                model=self.model,
            )
    
    async def count_tokens(self, messages: list[Message]) -> int:
        """Count tokens in the given messages."""
        client = self._get_client()
        _, contents = self._convert_messages(messages)
        
        response = await client.aio.models.count_tokens(
            model=self.model,
            contents=contents,
        )
        
        return response.total_tokens
    
    def supports_vision(self) -> bool:
        return True
    
    def supports_function_calling(self) -> bool:
        return True
    
    def max_context_tokens(self) -> int:
        # Gemini 2.5 Pro supports 2M tokens
        if "2.5-pro" in self.model or "2.0-pro" in self.model:
            return 2_000_000
        elif "2.5-flash" in self.model or "2.0-flash" in self.model:
            return 1_000_000
        else:
            return 128_000  # Conservative default
