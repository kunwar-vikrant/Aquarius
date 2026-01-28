# VLM Provider Integration Guide

Guide to configuring and using Vision Language Model providers with the Counterfactual World Engine.

## Supported Providers

| Provider | Model | Best For | Context Window |
|----------|-------|----------|----------------|
| **Gemini** | gemini-2.5-pro | Video understanding | 2M tokens |
| **xAI (Grok)** | grok-4-1-fast-reasoning | Fast reasoning, real-time | 128K tokens |
| **Claude** | claude-3.5-sonnet | Complex analysis | 200K tokens |
| **OpenAI** | gpt-4o | Balanced performance | 128K tokens |

---

## Configuration

### Environment Variables

```bash
# .env file
# Gemini (Google)
GOOGLE_API_KEY=your-gemini-api-key

# xAI (Grok)
XAI_API_KEY=your-xai-api-key

# Anthropic (Claude)
ANTHROPIC_API_KEY=your-anthropic-api-key

# OpenAI (GPT-4)
OPENAI_API_KEY=your-openai-api-key
```

### Provider Selection

```python
from cwe.reasoning.providers import get_provider, VLMProviderType

# By type
provider = get_provider(VLMProviderType.GEMINI)
provider = get_provider(VLMProviderType.XAI)
provider = get_provider(VLMProviderType.CLAUDE)
provider = get_provider(VLMProviderType.OPENAI)

# With custom config
from cwe.reasoning.providers.base import VLMConfig

config = VLMConfig(
    model="grok-4-1-fast-reasoning",
    temperature=0.3,
    max_tokens=4096
)
provider = get_provider(VLMProviderType.XAI, config)
```

---

## Provider Details

### xAI (Grok)

**Implementation:** `cwe/reasoning/providers/xai.py`

**API Endpoint:** `https://api.x.ai/v1`

**Models:**
- `grok-4-1-fast-reasoning` (default) - Optimized for speed
- `grok-3` - Previous generation

**Key Features:**
- OpenAI-compatible API
- Strong reasoning capabilities
- Real-time data access (when enabled)

**Configuration:**

```python
from cwe.reasoning.providers.xai import XAIProvider
from cwe.reasoning.providers.base import VLMConfig

config = VLMConfig(
    model="grok-4-1-fast-reasoning",
    temperature=0.3,
    max_tokens=4096,
    api_key="your-xai-api-key"  # Or use XAI_API_KEY env var
)

provider = XAIProvider(config)
```

**Implementation Notes:**

```python
# xAI uses OpenAI-compatible client
self.client = openai.AsyncOpenAI(
    api_key=api_key,
    base_url="https://api.x.ai/v1"
)
```

**Message Conversion Quirks:**
- Assistant messages with tool calls must have `content=None` handled properly
- Tool results require specific formatting

```python
# In xai.py - handling tool call messages
def _convert_messages(self, messages: list[Message]) -> list[dict]:
    converted = []
    for msg in messages:
        if msg.role == "assistant" and msg.tool_calls:
            # xAI requires explicit handling of null content
            converted.append({
                "role": "assistant",
                "content": msg.content or "",  # Convert None to empty string
                "tool_calls": msg.tool_calls
            })
        # ...
```

**Function Calling:**

```python
# xAI tool schema format (OpenAI-compatible)
tools = [
    {
        "type": "function",
        "function": {
            "name": "emit_event",
            "description": "Register an event in the timeline",
            "parameters": {
                "type": "object",
                "properties": {...},
                "required": [...]
            }
        }
    }
]
```

---

### Gemini (Google)

**Implementation:** `cwe/reasoning/providers/gemini.py`

**Models:**
- `gemini-2.5-pro` (default) - Best multimodal
- `gemini-2.5-flash` - Faster, cheaper

**Key Features:**
- 2M token context window
- Superior video understanding
- Native function calling

**Configuration:**

```python
from cwe.reasoning.providers.gemini import GeminiProvider
from cwe.reasoning.providers.base import VLMConfig

config = VLMConfig(
    model="gemini-2.5-pro",
    temperature=0.3,
    max_tokens=8192
)

provider = GeminiProvider(config)
```

**Video Frame Analysis:**

```python
# Gemini excels at video frame analysis
from cwe.reasoning.providers.base import Message, ContentPart

# Create message with image
message = Message.user(
    text="Analyze this dashcam frame for collision indicators",
    images=["base64_encoded_frame_data"]
)

response = await provider.generate(
    messages=[message],
    functions=[]
)
```

---

### Claude (Anthropic)

**Implementation:** `cwe/reasoning/providers/claude.py`

**Models:**
- `claude-3.5-sonnet` (default)
- `claude-3-opus` - Most capable

**Key Features:**
- Best reasoning and analysis
- Excellent structured output
- Strong at causal reasoning

**Configuration:**

```python
from cwe.reasoning.providers.claude import ClaudeProvider

provider = ClaudeProvider(VLMConfig(
    model="claude-3.5-sonnet",
    temperature=0.3
))
```

**Tool Use:**

```python
# Claude returns tool use in content array
# Response format:
{
    "content": [
        {"type": "text", "text": "I'll analyze..."},
        {
            "type": "tool_use",
            "id": "toolu_xxx",
            "name": "emit_event",
            "input": {...}
        }
    ]
}
```

---

### OpenAI (GPT-4)

**Implementation:** `cwe/reasoning/providers/openai.py`

**Models:**
- `gpt-4o` (default)
- `gpt-4-turbo`

**Key Features:**
- Reliable function calling
- Good balance of speed/quality
- Wide tooling support

**Configuration:**

```python
from cwe.reasoning.providers.openai import OpenAIProvider

provider = OpenAIProvider(VLMConfig(
    model="gpt-4o",
    temperature=0.3
))
```

---

## Provider Abstraction

### Base Interface

All providers implement `VLMProvider`:

```python
class VLMProvider(ABC):
    """Abstract base class for VLM providers."""
    
    def __init__(self, config: VLMConfig):
        self.config = config
    
    @property
    @abstractmethod
    def provider_type(self) -> VLMProviderType:
        """Return the provider type."""
        pass
    
    @abstractmethod
    async def generate(
        self,
        messages: list[Message],
        functions: list[dict] | None = None,
        function_call: str | dict | None = None
    ) -> VLMResponse:
        """Generate a response from the VLM."""
        pass
    
    async def generate_stream(
        self,
        messages: list[Message],
        functions: list[dict] | None = None
    ) -> AsyncIterator[str]:
        """Stream a response (optional implementation)."""
        raise NotImplementedError("Streaming not supported")
```

### Response Format

```python
@dataclass
class VLMResponse:
    content: str | None           # Text response
    function_calls: list[FunctionCall]  # Tool calls
    usage: dict                   # Token usage
    raw_response: Any             # Provider-specific response
    
    @property
    def has_function_calls(self) -> bool:
        return len(self.function_calls) > 0
```

### Function Call Format

```python
@dataclass
class FunctionCall:
    id: str           # Unique ID for tool call
    name: str         # Function name
    arguments: dict   # Parsed arguments
```

---

## Multi-Provider Strategy

### Ensemble Voting (Planned)

For high-stakes analysis, run through multiple providers:

```python
# Conceptual - not yet implemented
async def ensemble_analysis(timeline, providers):
    results = await asyncio.gather(*[
        provider.generate(messages, functions)
        for provider in providers
    ])
    
    # Consensus scoring
    consensus = find_consensus(results)
    disagreements = find_disagreements(results)
    
    return EnsebleResult(
        consensus=consensus,
        disagreements=disagreements,  # Flag for human review
        confidence=calculate_confidence(results)
    )
```

### Provider Fallback

```python
# In cwe/reasoning/reasoner.py
async def generate_with_fallback(
    self,
    messages: list[Message],
    functions: list[dict]
) -> VLMResponse:
    providers = [self.primary_provider, self.fallback_provider]
    
    for provider in providers:
        try:
            return await provider.generate(messages, functions)
        except Exception as e:
            logger.warning(f"Provider {provider.provider_type} failed: {e}")
            continue
    
    raise AllProvidersFailedError()
```

---

## Best Practices

### Provider Selection

| Use Case | Recommended Provider |
|----------|---------------------|
| Video frame analysis | Gemini |
| Fast reasoning | xAI Grok |
| Complex causal analysis | Claude |
| General purpose | OpenAI GPT-4o |
| Long incidents (>100K tokens) | Gemini |

### Cost Optimization

1. **Use flash models for initial passes** - Coarse pass can use cheaper models
2. **Cache VLM responses** - Avoid re-analyzing same content
3. **Batch similar requests** - Group related queries
4. **Use context prioritization** - Send only relevant context

### Reliability

1. **Implement retry logic** - Transient failures are common
2. **Set reasonable timeouts** - Long analyses may timeout
3. **Monitor token usage** - Avoid context overflow
4. **Validate function calls** - VLMs can generate invalid schemas

---

## Adding New Providers

### 1. Create Provider Class

```python
# cwe/reasoning/providers/new_provider.py
from .base import VLMProvider, VLMConfig, VLMResponse, Message

class NewProvider(VLMProvider):
    def __init__(self, config: VLMConfig):
        super().__init__(config)
        # Initialize client
    
    @property
    def provider_type(self) -> VLMProviderType:
        return VLMProviderType.NEW_PROVIDER
    
    async def generate(
        self,
        messages: list[Message],
        functions: list[dict] | None = None,
        function_call: str | dict | None = None
    ) -> VLMResponse:
        # Implementation
        pass
```

### 2. Register Provider

```python
# cwe/reasoning/providers/__init__.py
from .new_provider import NewProvider

PROVIDERS = {
    VLMProviderType.GEMINI: GeminiProvider,
    VLMProviderType.XAI: XAIProvider,
    VLMProviderType.CLAUDE: ClaudeProvider,
    VLMProviderType.OPENAI: OpenAIProvider,
    VLMProviderType.NEW_PROVIDER: NewProvider,  # Add here
}
```

### 3. Add Provider Type

```python
# cwe/reasoning/providers/base.py
class VLMProviderType(str, Enum):
    GEMINI = "gemini"
    XAI = "xai"
    CLAUDE = "claude"
    OPENAI = "openai"
    NEW_PROVIDER = "new_provider"  # Add here
```

---

## Troubleshooting

### Common Issues

**xAI: "content cannot be null"**
- Ensure assistant messages with tool calls have content set to empty string, not None

**Gemini: Context too long**
- Use hierarchical summarization
- Prioritize recent/critical context

**Claude: Tool use not returned**
- Ensure `tool_choice: "auto"` is set
- Check function schema validity

**OpenAI: Rate limiting**
- Implement exponential backoff
- Use organization tier with higher limits

### Debug Mode

```python
# Enable verbose logging
import logging
logging.getLogger("cwe.reasoning.providers").setLevel(logging.DEBUG)
```
