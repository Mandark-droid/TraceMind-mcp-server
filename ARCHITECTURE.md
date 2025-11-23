# TraceMind MCP Server - Technical Architecture

This document provides a deep technical dive into the TraceMind MCP Server architecture, implementation details, and deployment configuration.

## Table of Contents

- [System Overview](#system-overview)
- [Project Structure](#project-structure)
- [Core Components](#core-components)
- [MCP Protocol Implementation](#mcp-protocol-implementation)
- [Gemini Integration](#gemini-integration)
- [Data Flow](#data-flow)
- [Deployment Architecture](#deployment-architecture)
- [Development Workflow](#development-workflow)
- [Performance Considerations](#performance-considerations)
- [Security](#security)

---

## System Overview

TraceMind MCP Server is a Gradio-based MCP (Model Context Protocol) server that provides AI-powered analysis tools for agent evaluation data. It serves as the backend intelligence layer for the TraceMind ecosystem.

### Technology Stack

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| **Framework** | Gradio | 6.x | Native MCP support with `@gr.mcp.*` decorators |
| **AI Model** | Google Gemini | 2.5 Flash Lite | AI-powered analysis and insights |
| **Data Source** | HuggingFace Datasets | Latest | Load evaluation datasets |
| **Protocol** | MCP | 1.0 | Model Context Protocol for tool exposure |
| **Transport** | SSE | - | Server-Sent Events for real-time communication |
| **Deployment** | Docker | - | HuggingFace Spaces containerized deployment |
| **Language** | Python | 3.10+ | Core implementation |

### Architecture Diagram

```
┌──────────────────────────────────────────────────────────────┐
│ MCP Clients (External)                                        │
│  - Claude Desktop                                             │
│  - VS Code (Continue, Cursor, Cline)                         │
│  - TraceMind-AI (Track 2)                                    │
└────────────────┬─────────────────────────────────────────────┘
                 │
                 │ MCP Protocol
                 │ (SSE Transport)
                 ↓
┌──────────────────────────────────────────────────────────────┐
│ TraceMind MCP Server (HuggingFace Spaces)                    │
│                                                               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Gradio App (app.py)                                   │   │
│  │  - MCP Server Endpoint (mcp_server=True)             │   │
│  │  - Testing UI (Gradio Blocks)                        │   │
│  │  - Configuration Management                           │   │
│  └─────────────┬────────────────────────────────────────┘   │
│                │                                              │
│                ↓                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ MCP Tools (mcp_tools.py)                             │   │
│  │  - 11 Tools (@gr.mcp.tool())                         │   │
│  │  - 3 Resources (@gr.mcp.resource())                  │   │
│  │  - 3 Prompts (@gr.mcp.prompt())                      │   │
│  └─────────────┬────────────────────────────────────────┘   │
│                │                                              │
│                ↓                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Gemini Client (gemini_client.py)                     │   │
│  │  - API Authentication                                 │   │
│  │  - Prompt Engineering                                 │   │
│  │  - Response Parsing                                   │   │
│  └─────────────┬────────────────────────────────────────┘   │
│                │                                              │
└────────────────┼──────────────────────────────────────────────┘
                 │
                 ↓
        ┌────────────────┐
        │ External APIs  │
        │  - Gemini API  │
        │  - HF Datasets │
        └────────────────┘
```

---

## Project Structure

```
TraceMind-mcp-server/
├── app.py                      # Main entry point, Gradio UI
├── mcp_tools.py                # MCP tool implementations (11 tools + 3 resources + 3 prompts)
├── gemini_client.py            # Google Gemini API client
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Container configuration
├── .env.example                # Environment variable template
├── .gitignore                  # Git ignore rules
├── README.md                   # Project documentation
└── DOCUMENTATION.md            # Complete API reference

Total: 8 files (excluding docs)
Lines of Code: ~3,500 lines (breakdown below)
```

### File Sizes

| File | Lines | Purpose |
|------|-------|---------|
| `app.py` | ~1,200 | Gradio UI + MCP server setup + testing interface |
| `mcp_tools.py` | ~2,100 | All 17 MCP components (tools, resources, prompts) |
| `gemini_client.py` | ~200 | Gemini API integration |
| `requirements.txt` | ~20 | Dependencies |
| `Dockerfile` | ~30 | Deployment configuration |

---

## Core Components

### 1. app.py - Main Application

**Purpose**: Entry point for HuggingFace Spaces deployment, provides both MCP server and testing UI.

**Key Responsibilities**:
- Initialize Gradio app with `mcp_server=True`
- Create testing interface for all MCP tools
- Handle configuration (API keys, settings)
- Manage client connections

**Architecture**:

```python
# app.py structure
import gradio as gr
from gemini_client import GeminiClient
from mcp_tools import *  # All tool implementations

# 1. Initialize Gemini client (with fallback)
default_gemini_client = GeminiClient()

# 2. Create Gradio UI for testing
def create_gradio_ui():
    with gr.Blocks() as demo:
        # Settings tab for API key configuration
        # Tab for each MCP tool (11 tabs)
        # Tab for testing resources
        # Tab for testing prompts
        # API documentation tab
    return demo

# 3. Launch with MCP server enabled
if __name__ == "__main__":
    demo = create_gradio_ui()
    demo.launch(
        mcp_server=True,  # ← Enables MCP endpoint
        share=False,
        server_name="0.0.0.0",
        server_port=7860
    )
```

**MCP Enablement**:
- `mcp_server=True` in `demo.launch()` automatically:
  - Exposes `/gradio_api/mcp/sse` endpoint
  - Discovers all `@gr.mcp.tool()`, `@gr.mcp.resource()`, `@gr.mcp.prompt()` decorated functions
  - Generates MCP tool schemas from function signatures and docstrings
  - Handles MCP protocol communication (SSE transport)

**Testing Interface**:
- **Settings Tab**: Configure Gemini API key and HF token
- **Tool Tabs** (11): One tab per tool for manual testing
  - Input fields for all parameters
  - Submit button
  - Output display (Markdown or JSON)
- **Resources Tab**: Test resource URIs
- **Prompts Tab**: Test prompt templates
- **API Documentation Tab**: Generated from tool docstrings

---

### 2. mcp_tools.py - MCP Components

**Purpose**: Implements all 17 MCP components (11 tools + 3 resources + 3 prompts).

**Structure**:

```python
# mcp_tools.py structure
import gradio as gr
from gemini_client import GeminiClient
from datasets import load_dataset

# ============ TOOLS (11) ============

@gr.mcp.tool()
async def analyze_leaderboard(...) -> str:
    """Tool docstring (becomes MCP description)"""
    # 1. Load data from HuggingFace
    # 2. Process/filter data
    # 3. Call Gemini for AI analysis
    # 4. Return formatted response
    pass

@gr.mcp.tool()
async def debug_trace(...) -> str:
    """Debug traces with AI assistance"""
    pass

# ... (9 more tools)

# ============ RESOURCES (3) ============

@gr.mcp.resource()
def get_leaderboard_data(uri: str) -> str:
    """URI: leaderboard://{repo}"""
    # Parse URI
    # Load dataset
    # Return raw JSON
    pass

@gr.mcp.resource()
def get_trace_data(uri: str) -> str:
    """URI: trace://{trace_id}/{repo}"""
    pass

@gr.mcp.resource()
def get_cost_data(uri: str) -> str:
    """URI: cost://model/{model_name}"""
    pass

# ============ PROMPTS (3) ============

@gr.mcp.prompt()
def analysis_prompt(analysis_type: str, ...) -> str:
    """Generate analysis prompt templates"""
    pass

@gr.mcp.prompt()
def debug_prompt(debug_type: str, ...) -> str:
    """Generate debug prompt templates"""
    pass

@gr.mcp.prompt()
def optimization_prompt(optimization_goal: str, ...) -> str:
    """Generate optimization prompt templates"""
    pass
```

**Design Patterns**:

1. **Decorator-Based Registration**:
   ```python
   @gr.mcp.tool()  # Gradio automatically registers as MCP tool
   async def tool_name(...) -> str:
       """Docstring becomes tool description in MCP schema"""
       pass
   ```

2. **Structured Docstrings**:
   ```python
   """
   Brief one-line description.

   Longer detailed description explaining purpose and behavior.

   Args:
       param1 (type): Description of param1
       param2 (type): Description of param2. Default: value

   Returns:
       type: Description of return value
   """
   ```
   Gradio parses this to generate MCP tool schema automatically.

3. **Error Handling**:
   ```python
   try:
       # Tool implementation
       return result
   except Exception as e:
       return f"❌ **Error**: {str(e)}"
   ```
   All errors returned as user-friendly strings.

4. **Async/Await**:
   All tools are `async` for efficient I/O operations (API calls, dataset loading).

---

### 3. gemini_client.py - AI Integration

**Purpose**: Handles all interactions with Google Gemini 2.5 Flash Lite API.

**Key Features**:
- API authentication
- Prompt engineering for different analysis types
- Response parsing and formatting
- Error handling and retries
- Token optimization

**Class Structure**:

```python
class GeminiClient:
    def __init__(self, api_key: str, model_name: str):
        """Initialize with API key and model"""
        self.api_key = api_key
        self.model = genai.GenerativeModel(model_name)
        self.generation_config = {
            "temperature": 0.7,
            "top_p": 0.95,
            "max_output_tokens": 4096,  # Optimized for HF Spaces
        }
        self.request_timeout = 30  # 30s timeout

    async def analyze_with_context(
        self,
        data: Dict,
        analysis_type: str,
        specific_question: Optional[str] = None
    ) -> str:
        """
        Core analysis method used by all AI-powered tools

        Args:
            data: Data to analyze (dict or JSON)
            analysis_type: "leaderboard", "trace", "cost_estimate", "comparison", "results"
            specific_question: Optional specific question

        Returns:
            Markdown-formatted analysis
        """
        # 1. Build system prompt based on analysis_type
        system_prompt = self._get_system_prompt(analysis_type)

        # 2. Format data for context
        data_str = json.dumps(data, indent=2)

        # 3. Build user prompt
        user_prompt = f"{system_prompt}\n\nData:\n{data_str}"
        if specific_question:
            user_prompt += f"\n\nSpecific Question: {specific_question}"

        # 4. Call Gemini API
        response = await self.model.generate_content_async(
            user_prompt,
            generation_config=self.generation_config,
            request_options={"timeout": self.request_timeout}
        )

        # 5. Extract and return text
        return response.text

    def _get_system_prompt(self, analysis_type: str) -> str:
        """Get specialized system prompt for each analysis type"""
        prompts = {
            "leaderboard": """You are an expert AI agent performance analyst.
                Analyze evaluation leaderboard data and provide:
                - Top performers by key metrics
                - Trade-off analysis (cost vs accuracy)
                - Trend identification
                - Actionable recommendations
                Format: Markdown with clear sections.""",

            "trace": """You are an expert at debugging AI agent executions.
                Analyze OpenTelemetry trace data and:
                - Answer specific questions about execution
                - Identify performance bottlenecks
                - Explain reasoning chain
                - Provide optimization suggestions
                Format: Clear, concise explanation.""",

            "cost_estimate": """You are a cost optimization expert.
                Analyze cost estimation data and provide:
                - Detailed cost breakdown
                - Hardware recommendations
                - Cost optimization opportunities
                - ROI analysis
                Format: Structured breakdown with recommendations.""",

            # ... more prompts for other analysis types
        }
        return prompts.get(analysis_type, prompts["leaderboard"])
```

**Optimization Strategies**:
- **Token Reduction**: `max_output_tokens: 4096` (reduced from 8192) for faster responses
- **Request Timeout**: 30s timeout for HF Spaces compatibility
- **Temperature**: 0.7 for balanced creativity and consistency
- **Model Selection**: `gemini-2.5-flash-lite` for speed (can switch to `gemini-2.5-flash` for quality)

---

## MCP Protocol Implementation

### How Gradio's Native MCP Support Works

Gradio 6+ provides native MCP server capabilities through decorators and automatic schema generation.

**1. Tool Registration**:
```python
@gr.mcp.tool()  # ← This decorator tells Gradio to expose this as an MCP tool
async def my_tool(param1: str, param2: int = 10) -> str:
    """
    Brief description (used in MCP tool schema).

    Args:
        param1 (str): Description of param1
        param2 (int): Description of param2. Default: 10

    Returns:
        str: Description of return value
    """
    return f"Result: {param1}, {param2}"
```

**What Gradio does automatically**:
- Parses function signature to extract parameter names and types
- Parses docstring to extract descriptions
- Generates MCP tool schema:
  ```json
  {
    "name": "my_tool",
    "description": "Brief description (used in MCP tool schema).",
    "inputSchema": {
      "type": "object",
      "properties": {
        "param1": {
          "type": "string",
          "description": "Description of param1"
        },
        "param2": {
          "type": "integer",
          "default": 10,
          "description": "Description of param2. Default: 10"
        }
      },
      "required": ["param1"]
    }
  }
  ```

**2. Resource Registration**:
```python
@gr.mcp.resource()
def get_resource(uri: str) -> str:
    """
    Resource description.

    Args:
        uri (str): Resource URI (e.g., "leaderboard://repo/name")

    Returns:
        str: JSON data
    """
    # Parse URI
    # Load data
    # Return JSON string
    pass
```

**3. Prompt Registration**:
```python
@gr.mcp.prompt()
def generate_prompt(prompt_type: str, context: str) -> str:
    """
    Generate reusable prompt templates.

    Args:
        prompt_type (str): Type of prompt
        context (str): Context for prompt generation

    Returns:
        str: Generated prompt text
    """
    return f"Prompt template for {prompt_type} with {context}"
```

### MCP Endpoint URLs

When `demo.launch(mcp_server=True)` is called:

**SSE Endpoint** (Primary):
```
https://mcp-1st-birthday-tracemind-mcp-server.hf.space/gradio_api/mcp/sse
```

**Streamable HTTP Endpoint** (Alternative):
```
https://mcp-1st-birthday-tracemind-mcp-server.hf.space/gradio_api/mcp/
```

### Client Configuration

**Claude Desktop** (`claude_desktop_config.json`):
```json
{
  "mcpServers": {
    "tracemind": {
      "url": "https://mcp-1st-birthday-tracemind-mcp-server.hf.space/gradio_api/mcp/sse",
      "transport": "sse"
    }
  }
}
```

**Python MCP Client**:
```python
from mcp import ClientSession, ServerParameters

session = ClientSession(
    ServerParameters(
        url="https://mcp-1st-birthday-tracemind-mcp-server.hf.space/gradio_api/mcp/sse",
        transport="sse"
    )
)
await session.__aenter__()

# List tools
tools = await session.list_tools()

# Call tool
result = await session.call_tool("analyze_leaderboard", arguments={
    "metric_focus": "cost",
    "top_n": 5
})
```

---

## Gemini Integration

### API Configuration

**Environment Variable**:
```bash
GEMINI_API_KEY=your_api_key_here
```

**Initialization**:
```python
import google.generativeai as genai

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.5-flash-lite")
```

### Prompt Engineering Strategy

**1. System Prompts by Analysis Type**:
Each analysis type (leaderboard, trace, cost, comparison, results) has a specialized system prompt that:
- Defines the AI's role and expertise
- Specifies output format (markdown, structured sections)
- Lists key insights to include
- Sets tone (professional, concise, actionable)

**2. Context Injection**:
```python
user_prompt = f"""
{system_prompt}

Data to Analyze:
{json.dumps(data, indent=2)}

Specific Question: {question}
"""
```

**3. Output Formatting**:
- All responses in Markdown
- Clear sections: Top Performers, Key Insights, Trade-offs, Recommendations
- Bullet points for readability
- Code blocks for technical details

### Rate Limiting & Error Handling

**Rate Limits** (Gemini 2.5 Flash Lite free tier):
- 1,500 requests per day
- 1 request per second

**Error Handling Strategy**:
```python
try:
    response = await model.generate_content_async(...)
    return response.text
except google.api_core.exceptions.ResourceExhausted:
    return "❌ **Rate limit exceeded**. Please try again in a few seconds."
except google.api_core.exceptions.DeadlineExceeded:
    return "❌ **Request timeout**. The analysis is taking too long. Try with less data."
except Exception as e:
    return f"❌ **Error**: {str(e)}"
```

---

## Data Flow

### Tool Execution Flow

```
1. MCP Client                    (e.g., Claude Desktop, TraceMind-AI)
   └─→ Calls: analyze_leaderboard(metric_focus="cost", top_n=5)

2. Gradio MCP Server             (app.py)
   └─→ Routes to: analyze_leaderboard() in mcp_tools.py

3. MCP Tool Function             (mcp_tools.py)
   ├─→ Load data from HuggingFace Datasets
   │   └─→ ds = load_dataset("kshitijthakkar/smoltrace-leaderboard")
   │
   ├─→ Process/filter data
   │   └─→ Filter by time range, sort by metric
   │
   ├─→ Call Gemini Client
   │   └─→ gemini_client.analyze_with_context(data, "leaderboard")
   │
   └─→ Return formatted response

4. Gemini Client                 (gemini_client.py)
   ├─→ Build system prompt
   ├─→ Format data as JSON
   ├─→ Call Gemini API
   │   └─→ model.generate_content_async(prompt)
   └─→ Return AI-generated analysis

5. Response Path                 (back through stack)
   └─→ Gemini → gemini_client → mcp_tool → Gradio → MCP Client

6. MCP Client                    (displays result to user)
   └─→ Shows markdown-formatted analysis
```

### Resource Access Flow

```
1. MCP Client
   └─→ Accesses: leaderboard://kshitijthakkar/smoltrace-leaderboard

2. Gradio MCP Server
   └─→ Routes to: get_leaderboard_data(uri)

3. Resource Function
   ├─→ Parse URI to extract repo name
   ├─→ Load dataset from HuggingFace
   ├─→ Convert to JSON
   └─→ Return raw JSON string

4. MCP Client
   └─→ Receives raw JSON data (no AI processing)
```

---

## Deployment Architecture

### HuggingFace Spaces Deployment

**Platform**: HuggingFace Spaces
**SDK**: Docker (for custom dependencies)
**Hardware**: CPU Basic (free tier) - sufficient for API calls and dataset loading
**URL**: https://huggingface.co/spaces/MCP-1st-Birthday/TraceMind-mcp-server

### Dockerfile

```dockerfile
# Base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY app.py .
COPY mcp_tools.py .
COPY gemini_client.py .

# Expose port
EXPOSE 7860

# Set environment variables
ENV GRADIO_SERVER_NAME="0.0.0.0"
ENV GRADIO_SERVER_PORT="7860"

# Run application
CMD ["python", "app.py"]
```

### Environment Variables (HF Spaces Secrets)

```bash
# Required
GEMINI_API_KEY=your_gemini_api_key_here

# Optional (for testing)
HF_TOKEN=your_huggingface_token_here
```

### Scaling Considerations

**Current Setup** (Free Tier):
- Hardware: CPU Basic
- Concurrent Users: ~10-20
- Request Latency: 2-5 seconds (AI analysis)
- Rate Limit: Gemini API (1,500 req/day)

**If Scaling Needed**:
1. **Upgrade Hardware**: CPU Basic → CPU Upgrade (2x performance)
2. **Caching**: Add Redis for caching frequent queries
3. **API Key Pool**: Rotate multiple Gemini API keys to bypass rate limits
4. **Load Balancing**: Deploy multiple Spaces instances with load balancer

---

## Development Workflow

### Local Development Setup

```bash
# 1. Clone repository
git clone https://github.com/Mandark-droid/TraceMind-mcp-server.git
cd TraceMind-mcp-server

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env with your API keys

# 5. Run locally
python app.py

# 6. Access
# - Gradio UI: http://localhost:7860
# - MCP Endpoint: http://localhost:7860/gradio_api/mcp/sse
```

### Testing MCP Tools

**Option 1: Gradio UI** (Easiest):
```
1. Run app.py
2. Open http://localhost:7860
3. Navigate to tool tab (e.g., "📊 Analyze Leaderboard")
4. Fill in parameters
5. Click submit button
6. View results
```

**Option 2: Python MCP Client**:
```python
from mcp import ClientSession, ServerParameters

async def test_tool():
    session = ClientSession(
        ServerParameters(
            url="http://localhost:7860/gradio_api/mcp/sse",
            transport="sse"
        )
    )
    await session.__aenter__()

    result = await session.call_tool("analyze_leaderboard", {
        "metric_focus": "cost",
        "top_n": 3
    })

    print(result.content[0].text)

import asyncio
asyncio.run(test_tool())
```

### Adding New MCP Tools

**Step 1: Add function to mcp_tools.py**:
```python
@gr.mcp.tool()
async def new_tool_name(
    param1: str,
    param2: int = 10
) -> str:
    """
    Brief description of what this tool does.

    Detailed explanation of the tool's purpose and behavior.

    Args:
        param1 (str): Description of param1 with examples
        param2 (int): Description of param2. Default: 10

    Returns:
        str: Description of what the function returns
    """
    try:
        # Implementation
        result = f"Processed: {param1} with {param2}"
        return result
    except Exception as e:
        return f"❌ **Error**: {str(e)}"
```

**Step 2: Add testing UI to app.py** (optional):
```python
with gr.Tab("🆕 New Tool"):
    gr.Markdown("## New Tool Name")
    param1_input = gr.Textbox(label="Param 1")
    param2_input = gr.Number(label="Param 2", value=10)
    submit_btn = gr.Button("Execute")
    output = gr.Markdown()

    submit_btn.click(
        fn=new_tool_name,
        inputs=[param1_input, param2_input],
        outputs=output
    )
```

**Step 3: Test**:
```bash
python app.py
# Visit http://localhost:7860
# Test in new tab
```

**Step 4: Deploy**:
```bash
git add mcp_tools.py app.py
git commit -m "feat: Add new_tool_name MCP tool"
git push origin main
# HF Spaces auto-deploys
```

---

## Performance Considerations

### 1. Token Optimization

**Problem**: Loading full datasets consumes excessive tokens in AI analysis.

**Solutions**:
- **get_top_performers**: Returns only top N models (90% token reduction)
- **get_leaderboard_summary**: Returns aggregated stats (99% token reduction)
- **Data sampling**: Limit rows when loading datasets (max_rows parameter)

**Example**:
```python
# ❌ BAD: Loads 51 rows, ~50K tokens
full_data = load_dataset("kshitijthakkar/smoltrace-leaderboard")

# ✅ GOOD: Returns top 5, ~5K tokens (90% reduction)
top_5 = await get_top_performers(top_n=5)

# ✅ BETTER: Returns summary, ~500 tokens (99% reduction)
summary = await get_leaderboard_summary()
```

### 2. Async Operations

All tools are `async` for efficient I/O:
```python
@gr.mcp.tool()
async def tool_name(...):  # ← async
    ds = load_dataset(...)  # ← Blocks on I/O
    result = await gemini_client.analyze(...)  # ← async API call
    return result
```

Benefits:
- Non-blocking API calls
- Multiple concurrent requests
- Better resource utilization

### 3. Caching (Future Enhancement)

**Current**: No caching (stateless)
**Future**: Add Redis for caching frequent queries

```python
import redis
from functools import wraps

redis_client = redis.Redis(...)

def cache_result(ttl=300):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = f"{func.__name__}:{hash((args, tuple(kwargs.items())))}"

            # Check cache
            cached = redis_client.get(cache_key)
            if cached:
                return cached.decode()

            # Execute function
            result = await func(*args, **kwargs)

            # Store in cache
            redis_client.setex(cache_key, ttl, result)

            return result
        return wrapper
    return decorator

@gr.mcp.tool()
@cache_result(ttl=300)  # 5-minute cache
async def analyze_leaderboard(...):
    pass
```

---

## Security

### API Key Management

**Storage**:
- Development: `.env` file (gitignored)
- Production: HuggingFace Spaces Secrets (encrypted)

**Access**:
```python
# gemini_client.py
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not set")
```

**Never**:
- ❌ Hardcode API keys in source code
- ❌ Commit `.env` to git
- ❌ Expose keys in client-side JavaScript
- ❌ Log API keys in console/files

### Input Validation

**Dataset Repository Validation**:
```python
# Only allow "smoltrace-" prefix datasets
if "smoltrace-" not in dataset_repo:
    return "❌ Error: Dataset must contain 'smoltrace-' prefix for security"
```

**Parameter Validation**:
```python
# Constrain ranges
top_n = max(1, min(20, top_n))  # Clamp between 1-20
max_rows = max(10, min(500, max_rows))  # Clamp between 10-500
```

### Rate Limiting

**Gemini API**:
- Free tier: 1,500 requests/day
- Handled by Google (automatic)
- Errors returned as user-friendly messages

**HuggingFace Datasets**:
- No rate limits for public datasets
- Private datasets require HF token

---

## Related Documentation

- [README.md](PROPOSED_README_MCP_SERVER.md) - Overview and quick start
- [DOCUMENTATION.md](DOCUMENTATION_MCP_SERVER.md) - Complete API reference
- [TraceMind-AI Architecture](ARCHITECTURE_TRACEMIND_AI.md) - Client-side architecture

---

**Last Updated**: November 21, 2025
**Version**: 1.0.0
**Track**: Building MCP (Enterprise)
