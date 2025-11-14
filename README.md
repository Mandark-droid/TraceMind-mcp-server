---
title: TraceMind MCP Server
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 5.49.1
app_port: 7860
pinned: false
license: agpl-3.0
short_description: MCP server for agent evaluation with Gemini 2.5 Pro
tags:
  - building-mcp-track-enterprise
  - mcp
  - gradio
  - gemini
  - agent-evaluation
  - leaderboard
---

# TraceMind MCP Server

**AI-Powered Analysis Tools for Agent Evaluation Data**

[![MCP's 1st Birthday Hackathon](https://img.shields.io/badge/MCP%27s%201st%20Birthday-Hackathon-blue)](https://github.com/modelcontextprotocol)
[![Track](https://img.shields.io/badge/Track-Building%20MCP%20(Enterprise)-green)](https://github.com/modelcontextprotocol/hackathon)
[![Google Gemini](https://img.shields.io/badge/Powered%20by-Google%20Gemini%202.5%20Pro-orange)](https://ai.google.dev/)

> **🎯 Track 1 Submission**: Building MCP (Enterprise)
> **📅 MCP's 1st Birthday Hackathon**: November 14-30, 2025

## Overview

TraceMind MCP Server is a Gradio-based MCP (Model Context Protocol) server that provides a complete MCP implementation with:

### 🛠️ **5 AI-Powered Tools**
1. **📊 analyze_leaderboard**: Generate insights from evaluation leaderboard data
2. **🐛 debug_trace**: Debug specific agent execution traces using OpenTelemetry data
3. **💰 estimate_cost**: Predict evaluation costs before running
4. **⚖️ compare_runs**: Compare two evaluation runs with AI-powered analysis
5. **📦 get_dataset**: Load SMOLTRACE datasets (smoltrace-* prefix only) as JSON for flexible analysis

### 📦 **3 Data Resources**
1. **leaderboard data**: Direct JSON access to evaluation results
2. **trace data**: Raw OpenTelemetry trace data with spans
3. **cost data**: Model pricing and hardware cost information

### 📝 **3 Prompt Templates**
1. **analysis prompts**: Standardized templates for different analysis types
2. **debug prompts**: Templates for debugging scenarios
3. **optimization prompts**: Templates for optimization goals

All analysis is powered by **Google Gemini 2.5 Pro** for intelligent, context-aware insights.

## 🔗 Quick Links

### Current Deployment (Development)
- **Gradio UI**: https://huggingface.co/spaces/kshitijthakkar/TraceMind-mcp-server
- **MCP Endpoint**: `https://kshitijthakkar-tracemind-mcp-server.hf.space/gradio_api/mcp/`
- **Auto-Config**: Add `kshitijthakkar-tracemind-mcp-server` at https://huggingface.co/settings/mcp

### After Hackathon Submission (Production)
- **Gradio UI**: https://huggingface.co/spaces/MCP-1st-Birthday/TraceMind-mcp-server
- **MCP Endpoint**: `https://mcp-1st-birthday-tracemind-mcp-server.hf.space/gradio_api/mcp/`
- **Auto-Config**: Add `mcp-1st-birthday-tracemind-mcp-server` at https://huggingface.co/settings/mcp

> 💡 **Tip**: Use the Auto-Config link above for the easiest setup! It generates the correct config for your MCP client automatically.

## 📱 Social Media & Demo

**📢 Announcement Post**: [Coming Soon - X/LinkedIn post]

**🎥 Demo Video**: [Coming Soon - YouTube/Loom link showing MCP server integration with Claude Desktop]

---

## Why This MCP Server?

**Problem**: Agent evaluation generates massive amounts of data (leaderboards, traces, metrics), but developers struggle to:
- Understand which models perform best for their use case
- Debug why specific agent executions failed
- Estimate costs before running expensive evaluations

**Solution**: This MCP server provides AI-powered analysis tools that connect to HuggingFace datasets and deliver actionable insights in natural language.

**Impact**: Developers can make informed decisions about agent configurations, debug issues faster, and optimize costs—all through a simple MCP interface.

## Features

### 🎯 Track 1 Compliance: Building MCP (Enterprise)

- ✅ **Complete MCP Implementation**: Tools, Resources, AND Prompts
- ✅ **MCP Standard Compliant**: Built with Gradio's native MCP support (`@gr.mcp.*` decorators)
- ✅ **Production-Ready**: Deployable to HuggingFace Spaces with SSE transport
- ✅ **Testing Interface**: Beautiful Gradio UI for testing all components
- ✅ **Enterprise Focus**: Cost optimization, debugging, and decision support
- ✅ **Google Gemini Powered**: Leverages Gemini 2.5 Pro for intelligent analysis
- ✅ **11 Total Components**: 5 Tools + 3 Resources + 3 Prompts

### 🛠️ Five Production-Ready Tools

#### 1. analyze_leaderboard

Analyzes evaluation leaderboard data from HuggingFace datasets and provides:
- Top performers by selected metric (accuracy, cost, latency, CO2)
- Trade-off analysis (e.g., "GPT-4 is most accurate but Llama-3.1 is 25x cheaper")
- Trend identification
- Actionable recommendations

**Example Use Case**: Before choosing a model for production, get AI-powered insights on which configuration offers the best cost/performance for your requirements.

#### 2. debug_trace

Analyzes OpenTelemetry trace data and answers specific questions like:
- "Why was tool X called twice?"
- "Which step took the most time?"
- "Why did this test fail?"

**Example Use Case**: When an agent test fails, understand exactly what happened without manually parsing trace spans.

#### 3. estimate_cost

Predicts costs before running evaluations:
- LLM API costs (token-based)
- HuggingFace Jobs compute costs
- CO2 emissions estimate
- Hardware recommendations

**Example Use Case**: Compare the cost of evaluating GPT-4 vs Llama-3.1 across 1000 tests before committing resources.

#### 4. compare_runs

Compares two evaluation runs with AI-powered analysis across multiple dimensions:
- Success rate comparison with statistical significance
- Cost efficiency analysis (total cost, cost per test, cost per successful test)
- Speed comparison (average duration, throughput)
- Environmental impact (CO2 emissions per test)
- GPU efficiency (for GPU jobs)

**Focus Options**:
- `comprehensive`: Complete comparison across all dimensions
- `cost`: Detailed cost efficiency and ROI analysis
- `performance`: Speed and accuracy trade-off analysis
- `eco_friendly`: Environmental impact and carbon footprint comparison

**Example Use Case**: After running evaluations with two different models, compare them head-to-head to determine which is better for production deployment based on your priorities (accuracy, cost, speed, or environmental impact).

#### 5. get_dataset

Loads SMOLTRACE datasets from HuggingFace and returns raw data as JSON:
- Simple, flexible tool that returns complete dataset with metadata
- Works with any dataset containing "smoltrace-" prefix
- Returns total rows, columns list, and data array
- Automatically sorts by timestamp if available
- Configurable row limit (1-200) to manage token usage

**Security Restriction**: Only datasets with "smoltrace-" in the repository name are allowed.

**Primary Use Cases**:
- Load `smoltrace-leaderboard` to find run IDs and model names
- Discover supporting datasets via `results_dataset`, `traces_dataset`, `metrics_dataset` fields
- Load `smoltrace-results-*` datasets to see individual test case details
- Load `smoltrace-traces-*` datasets to access OpenTelemetry trace data
- Load `smoltrace-metrics-*` datasets to get GPU performance data
- Answer specific questions requiring raw data access

**Example Workflow**:
1. LLM calls `get_dataset("kshitijthakkar/smoltrace-leaderboard")` to see all runs
2. Examines the JSON response to find run IDs, models, and supporting dataset names
3. Calls `get_dataset("username/smoltrace-results-gpt4")` to load detailed results
4. Can now answer questions like "What are the last 10 run IDs?" or "Which models were tested?"

**Example Use Case**: When the user asks "Can you provide me with the list of last 10 runIds and model names?", the LLM loads the leaderboard dataset and extracts the requested information from the JSON response.

## MCP Resources Usage

Resources provide direct data access without AI analysis:

```python
# Access leaderboard data
GET leaderboard://kshitijthakkar/smoltrace-leaderboard
# Returns: JSON with all evaluation runs

# Access specific trace
GET trace://trace_abc123/username/agent-traces-gpt4
# Returns: JSON with trace spans and attributes

# Get model cost information
GET cost://model/openai/gpt-4
# Returns: JSON with pricing and hardware costs
```

## MCP Prompts Usage

Prompts provide reusable templates for standardized interactions:

```python
# Get analysis prompt template
analysis_prompt(analysis_type="leaderboard", focus_area="cost", detail_level="detailed")
# Returns: "Provide a detailed analysis. Analyze cost efficiency in the leaderboard..."

# Get debug prompt template
debug_prompt(debug_type="performance", context="tool_calling")
# Returns: "Analyze tool calling performance. Identify which tools are slow..."

# Get optimization prompt template
optimization_prompt(optimization_goal="cost", constraints="maintain_quality")
# Returns: "Analyze this evaluation setup and recommend cost optimizations..."
```

Use these prompts when interacting with the tools to get consistent, high-quality analysis.

## Quick Start

### 1. Installation

```bash
git clone https://github.com/Mandark-droid/TraceMind-mcp-server.git
cd TraceMind-mcp-server

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies (note: gradio[mcp] includes MCP support)
pip install -r requirements.txt
```

### 2. Environment Setup

Create `.env` file:

```bash
cp .env.example .env
# Edit .env and add your API keys
```

Get your keys:
- **Gemini API Key**: https://ai.google.dev/
- **HuggingFace Token**: https://huggingface.co/settings/tokens

### 3. Run Locally

```bash
python app.py
```

Open http://localhost:7860 to test the tools via Gradio interface.

### 4. Test with Live Data

Try the live example with real HuggingFace dataset:

**In the Gradio UI, Tab "📊 Analyze Leaderboard":**

```
Leaderboard Repository: kshitijthakkar/smoltrace-leaderboard
Metric Focus: overall
Time Range: last_week
Top N Models: 5
```

Click "🔍 Analyze" and get AI-powered insights from live data!

## MCP Integration

### How It Works

This Gradio app uses `mcp_server=True` in the launch configuration, which automatically:
- Exposes all async functions with proper docstrings as MCP tools
- Handles MCP protocol communication
- Provides MCP interfaces via:
  - **Streamable HTTP** (recommended) - Modern streaming protocol
  - **SSE** (deprecated) - Server-Sent Events for legacy compatibility

### Connecting from MCP Clients

Once deployed to HuggingFace Spaces, your MCP server will be available at:

#### Current Space (Development)
**🎯 MCP Endpoint (Streamable HTTP - Recommended)**:
```
https://kshitijthakkar-tracemind-mcp-server.hf.space/gradio_api/mcp/
```

**⚠️ MCP Endpoint (SSE - Deprecated)**:
```
https://kshitijthakkar-tracemind-mcp-server.hf.space/gradio_api/mcp/sse
```

#### After Hackathon Submission
**🎯 MCP Endpoint (Streamable HTTP - Recommended)**:
```
https://mcp-1st-birthday-tracemind-mcp-server.hf.space/gradio_api/mcp/
```

**⚠️ MCP Endpoint (SSE - Deprecated)**:
```
https://mcp-1st-birthday-tracemind-mcp-server.hf.space/gradio_api/mcp/sse
```

**Note**: Use the **Streamable HTTP endpoint** (recommended) for all new integrations. The SSE endpoint is deprecated and maintained for backward compatibility only. After submission to the hackathon org, use the `mcp-1st-birthday-tracemind-mcp-server` URLs.

Configure your MCP client (Claude Desktop, Cursor, Cline, etc.) with the streamable HTTP endpoint.

### ✨ Easiest Way to Connect

**Recommended for all users** - HuggingFace provides an automatic configuration generator:

1. **Visit**: https://huggingface.co/settings/mcp (while logged in)
2. **Add Space**: Enter one of the following:
   - **Development**: `kshitijthakkar-tracemind-mcp-server`
   - **After Hackathon Submission**: `mcp-1st-birthday-tracemind-mcp-server`
3. **Select Client**: Choose Claude Desktop, VSCode, Cursor, etc.
4. **Copy Config**: Get the auto-generated configuration snippet
5. **Paste & Restart**: Add to your client's config file and restart

This automatically configures the correct endpoint URL and transport method for your chosen client!

### 🔧 Manual Configuration (Advanced)

If you prefer to manually configure your MCP client:

**Claude Desktop (`claude_desktop_config.json`)**:
```json
{
  "mcpServers": {
    "tracemind": {
      "url": "https://kshitijthakkar-tracemind-mcp-server.hf.space/gradio_api/mcp/",
      "transport": "streamable-http"
    }
  }
}
```

**After Hackathon Submission - Use this URL instead**:
```json
{
  "mcpServers": {
    "tracemind": {
      "url": "https://mcp-1st-birthday-tracemind-mcp-server.hf.space/gradio_api/mcp/",
      "transport": "streamable-http"
    }
  }
}
```

**VSCode / Cursor (`settings.json` or `.cursor/mcp.json`)**:
```json
{
  "mcp.servers": {
    "tracemind": {
      "url": "https://kshitijthakkar-tracemind-mcp-server.hf.space/gradio_api/mcp/",
      "transport": "streamable-http"
    }
  }
}
```

**After Hackathon Submission - Use this URL instead**:
```json
{
  "mcp.servers": {
    "tracemind": {
      "url": "https://mcp-1st-birthday-tracemind-mcp-server.hf.space/gradio_api/mcp/",
      "transport": "streamable-http"
    }
  }
}
```

**Cline / Other MCP Clients**:
- **Current URL**: `https://kshitijthakkar-tracemind-mcp-server.hf.space/gradio_api/mcp/`
- **After Hackathon Submission**: `https://mcp-1st-birthday-tracemind-mcp-server.hf.space/gradio_api/mcp/`
- **Transport**: `streamable-http` (or `http` depending on client)

### ❓ Connection FAQ

**Q: Which endpoint should I use?**
A: Use the **Streamable HTTP endpoint** (`/gradio_api/mcp/`) for all new connections. It's the modern, recommended protocol.

**Q: My client only supports SSE. What should I do?**
A: Use the SSE endpoint (`/gradio_api/mcp/sse`) for now, but note that it's deprecated. Consider upgrading your client if possible.

**Q: What's the difference between the two transports?**
A: Streamable HTTP is the newer, more efficient protocol with better error handling and performance. SSE is the legacy protocol being phased out.

**Q: How do I test if my connection works?**
A: After configuring your client, restart it and look for "tracemind" in your available MCP tools/servers. You should see 5 tools, 3 resources, and 3 prompts.

**Q: Can I use this MCP server without authentication?**
A: The MCP endpoint is publicly accessible. However, the tools may require HuggingFace datasets to be public or accessible with your HF token (configured server-side).

### Available MCP Components

**Tools** (5):
1. **analyze_leaderboard**: AI-powered leaderboard analysis with Gemini 2.5 Pro
2. **debug_trace**: Trace debugging with AI insights
3. **estimate_cost**: Cost estimation with optimization recommendations
4. **compare_runs**: Compare two evaluation runs with AI-powered analysis
5. **get_dataset**: Load SMOLTRACE datasets (smoltrace-* only) as JSON

**Resources** (3):
1. **leaderboard://{repo}**: Direct access to raw leaderboard data in JSON
2. **trace://{trace_id}/{repo}**: Direct access to trace data with spans
3. **cost://model/{model_name}**: Model pricing and hardware cost information

**Prompts** (3):
1. **analysis_prompt**: Reusable templates for different analysis types
2. **debug_prompt**: Reusable templates for debugging scenarios
3. **optimization_prompt**: Reusable templates for optimization goals

See full API documentation in the Gradio interface under "📖 API Documentation" tab.

## Architecture

```
TraceMind-mcp-server/
├── app.py                      # Gradio UI + MCP server (mcp_server=True)
├── gemini_client.py            # Google Gemini 2.5 Pro integration
├── mcp_tools.py                # 3 tool implementations
├── requirements.txt            # Python dependencies
├── .env.example                # Environment variable template
├── .gitignore
└── README.md
```

**Key Technologies**:
- **Gradio 6 with MCP support**: `gradio[mcp]` provides native MCP server capabilities
- **Google Gemini 2.5 Pro**: Latest AI model for intelligent analysis
- **HuggingFace Datasets**: Data source for evaluations
- **Streamable HTTP Transport**: Modern streaming protocol for MCP communication (recommended)
- **SSE Transport**: Server-Sent Events for legacy MCP compatibility (deprecated)

## Deploy to HuggingFace Spaces

### 1. Create Space

Go to https://huggingface.co/new-space

- **Space name**: `TraceMind-mcp-server`
- **License**: MIT
- **SDK**: Gradio
- **Hardware**: CPU Basic (free tier works fine)

### 2. Add Files

Upload all files from this repository to your Space:
- `app.py`
- `gemini_client.py`
- `mcp_tools.py`
- `requirements.txt`
- `README.md`

### 3. Add Secrets

In Space settings → Variables and secrets, add:
- `GEMINI_API_KEY`: Your Gemini API key
- `HF_TOKEN`: Your HuggingFace token

### 4. Add Hackathon Tag

In Space settings → Tags, add:
- `building-mcp-track-enterprise`

### 5. Access Your MCP Server

Your MCP server will be publicly available at:

#### Current Space (Development)

**Gradio UI**:
```
https://huggingface.co/spaces/kshitijthakkar/TraceMind-mcp-server
```

**MCP Endpoint (Streamable HTTP)**:
```
https://kshitijthakkar-tracemind-mcp-server.hf.space/gradio_api/mcp/
```

#### After Hackathon Submission

**Gradio UI**:
```
https://huggingface.co/spaces/MCP-1st-Birthday/TraceMind-mcp-server
```

**MCP Endpoint (Streamable HTTP)**:
```
https://mcp-1st-birthday-tracemind-mcp-server.hf.space/gradio_api/mcp/
```

Use the **Easiest Way to Connect** section above to configure your MCP client automatically!

## Testing

### Test 1: Analyze Leaderboard (Live Data)

```bash
# In Gradio UI - Tab "📊 Analyze Leaderboard":
Repository: kshitijthakkar/smoltrace-leaderboard
Metric: overall
Time Range: last_week
Top N: 5
Click "🔍 Analyze"
```

**Expected**: AI-generated analysis of top performing models from live HuggingFace dataset

### Test 2: Estimate Cost

```bash
# In Gradio UI - Tab "💰 Estimate Cost":
Model: openai/gpt-4
Agent Type: both
Number of Tests: 100
Hardware: auto
Click "💰 Estimate"
```

**Expected**: Cost breakdown with LLM costs, HF Jobs costs, duration, and CO2 estimate

### Test 3: Debug Trace

Note: This requires actual trace data from an evaluation run. For testing purposes, this will show an error about missing data, which is expected behavior.

## Hackathon Submission

### Track 1: Building MCP (Enterprise)

**Tag**: `building-mcp-track-enterprise`

**Why Enterprise Track?**
- Solves real business problems (cost optimization, debugging, decision support)
- Production-ready tools with clear ROI
- Integrates with enterprise data infrastructure (HuggingFace datasets)

**Technology Stack**
- **AI Analysis**: Google Gemini 2.5 Pro for all intelligent insights
- **MCP Framework**: Gradio 6 with native MCP support
- **Data Source**: HuggingFace Datasets
- **Transport**: Streamable HTTP (recommended) and SSE (deprecated)

## Related Project: TraceMind UI (Track 2)

This MCP server is designed to be consumed by **TraceMind UI** (separate submission for Track 2: MCP in Action).

TraceMind UI is a Gradio-based agent evaluation platform that uses these MCP tools to provide:
- AI-powered leaderboard insights
- Interactive trace debugging
- Pre-evaluation cost estimates

## File Descriptions

### app.py
Main Gradio application with:
- Testing UI for all 5 tools
- MCP server enabled via `mcp_server=True`
- API documentation

### gemini_client.py
Google Gemini 2.5 Pro client that:
- Handles API authentication
- Provides specialized analysis methods for different data types
- Formats prompts for optimal results
- Uses `gemini-2.5-pro-latest` model (can switch to `gemini-2.5-flash-latest`)

### mcp_tools.py
Complete MCP implementation with 11 components:

**Tools** (5 async functions):
- `analyze_leaderboard()`: AI-powered leaderboard analysis
- `debug_trace()`: AI-powered trace debugging
- `estimate_cost()`: AI-powered cost estimation
- `compare_runs()`: AI-powered run comparison
- `get_dataset()`: Load SMOLTRACE datasets as JSON

**Resources** (3 decorated functions with `@gr.mcp.resource()`):
- `get_leaderboard_data()`: Raw leaderboard JSON data
- `get_trace_data()`: Raw trace JSON data with spans
- `get_cost_data()`: Model pricing and hardware cost JSON

**Prompts** (3 decorated functions with `@gr.mcp.prompt()`):
- `analysis_prompt()`: Templates for different analysis types
- `debug_prompt()`: Templates for debugging scenarios
- `optimization_prompt()`: Templates for optimization goals

Each function includes:
- Appropriate decorator (`@gr.mcp.tool()`, `@gr.mcp.resource()`, or `@gr.mcp.prompt()`)
- Detailed docstring with "Args:" section
- Type hints for all parameters and return values
- Descriptive function name (becomes the MCP component name)

## Environment Variables

Required environment variables:

```bash
GEMINI_API_KEY=your_gemini_api_key_here
HF_TOKEN=your_huggingface_token_here
```

## Development

### Running Tests

```bash
# Test Gemini client
python -c "from gemini_client import GeminiClient; client = GeminiClient(); print('✅ Gemini client initialized')"

# Test with live leaderboard data
python app.py
# Open browser, test "Analyze Leaderboard" tab
```

### Adding New Tools

To add a new MCP tool (with Gradio's native MCP support):

1. **Add function to `mcp_tools.py`** with proper docstring:
```python
async def your_new_tool(
    gemini_client: GeminiClient,
    param1: str,
    param2: int = 10
) -> str:
    """
    Brief description of what the tool does.

    Longer description explaining the tool's purpose and behavior.

    Args:
        gemini_client (GeminiClient): Initialized Gemini client for AI analysis
        param1 (str): Description of param1 with examples if helpful
        param2 (int): Description of param2. Default: 10

    Returns:
        str: Description of what the function returns
    """
    # Your implementation
    return result
```

2. **Add UI tab in `app.py`** (optional, for testing):
```python
with gr.Tab("Your Tool"):
    # Add UI components
    # Wire up to your_new_tool()
```

3. That's it! Gradio automatically exposes it as an MCP tool based on:
   - Function name (becomes tool name)
   - Docstring (becomes tool description)
   - Args section (becomes parameter descriptions)
   - Type hints (become parameter types)

### Switching to Gemini 2.5 Flash

For faster (but slightly less capable) responses, switch to Gemini 2.5 Flash:

```python
# In app.py, change:
gemini_client = GeminiClient(model_name="gemini-2.5-flash-latest")
```

## 🙏 Credits & Acknowledgments

### Hackathon Sponsors

Special thanks to the sponsors of **MCP's 1st Birthday Hackathon** (November 14-30, 2025):

- **🤗 HuggingFace** - Hosting platform and dataset infrastructure
- **🧠 Google Gemini** - AI analysis powered by Gemini 2.5 Pro API
- **⚡ Modal** - Serverless infrastructure partner
- **🏢 Anthropic** - MCP protocol creators
- **🎨 Gradio** - Native MCP framework support
- **🎙️ ElevenLabs** - Audio AI capabilities
- **🦙 SambaNova** - High-performance AI infrastructure
- **🎯 Blaxel** - Additional compute credits

### Related Open Source Projects

This MCP server builds upon our open source agent evaluation ecosystem:

#### 📊 SMOLTRACE - Agent Evaluation Engine
- **Description**: Lightweight, production-ready evaluation framework for AI agents with OpenTelemetry instrumentation
- **GitHub**: [https://github.com/Mandark-droid/SMOLTRACE](https://github.com/Mandark-droid/SMOLTRACE)
- **PyPI**: [https://pypi.org/project/smoltrace/](https://pypi.org/project/smoltrace/)
- **Social**: [@smoltrace on X](https://twitter.com/smoltrace)

#### 🔭 TraceVerde - GenAI OpenTelemetry Instrumentation
- **Description**: Automatic OpenTelemetry instrumentation for LLM frameworks (LiteLLM, Transformers, LangChain, etc.)
- **GitHub**: [https://github.com/Mandark-droid/genai_otel_instrument](https://github.com/Mandark-droid/genai_otel_instrument)
- **PyPI**: [https://pypi.org/project/genai-otel-instrument](https://pypi.org/project/genai-otel-instrument)
- **Social**: [@genai_otel on X](https://twitter.com/genai_otel)

### Built By

**Track**: Building MCP (Enterprise)
**Author**: Kshitij Thakkar
**Powered by**: Google Gemini 2.5 Pro
**Built with**: Gradio 6 (native MCP support)

---

## 📄 License

AGPL-3.0 License

This project is licensed under the GNU Affero General Public License v3.0. See the LICENSE file for details.

---

## 💬 Support

For issues or questions:
- 📧 Open an issue on GitHub
- 💬 Join the [HuggingFace Discord](https://discord.gg/huggingface) - Channel: `#agents-mcp-hackathon-winter25`
- 🏷️ Tag `building-mcp-track-enterprise` for hackathon-related questions
- 🐦 Follow us on X: [@TraceMindAI](https://twitter.com/TraceMindAI) (placeholder)

## Changelog

### v1.0.0 (2025-11-14)
- Initial release for MCP Hackathon
- **Complete MCP Implementation**: 11 components total
  - 5 AI-powered tools (analyze_leaderboard, debug_trace, estimate_cost, compare_runs, get_dataset)
  - 3 data resources (leaderboard, trace, cost data)
  - 3 prompt templates (analysis, debug, optimization)
- Gradio native MCP support with decorators (`@gr.mcp.*`)
- Google Gemini 2.5 Pro integration for all AI analysis
- Live HuggingFace dataset integration
- SSE transport for MCP communication
- Production-ready for HuggingFace Spaces deployment
