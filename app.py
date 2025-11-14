"""
TraceMind MCP Server - Gradio Interface with MCP Support

This server provides AI-powered analysis tools for agent evaluation data:
1. analyze_leaderboard: Summarize trends and insights from leaderboard
2. debug_trace: Debug specific agent execution traces
3. estimate_cost: Predict evaluation costs before running
4. compare_runs: Compare two evaluation runs with AI-powered analysis
5. get_dataset: Load any HuggingFace dataset as JSON for flexible analysis
"""

import os
import gradio as gr
from typing import Optional, Dict, Any
from datetime import datetime

# Local imports
from gemini_client import GeminiClient
from mcp_tools import (
    analyze_leaderboard,
    debug_trace,
    estimate_cost,
    compare_runs,
    analyze_results,
    get_dataset
)

# Initialize default Gemini client (fallback if user doesn't provide key)
try:
    default_gemini_client = GeminiClient()
except ValueError:
    default_gemini_client = None  # Will prompt user to enter API key

# Gradio Interface for Testing
def create_gradio_ui():
    """Create Gradio UI for testing MCP tools"""

    with gr.Blocks(title="TraceMind MCP Server", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🤖 TraceMind MCP Server

        **AI-Powered Analysis for Agent Evaluation Data**

        This server provides **6 MCP Tools + 3 MCP Resources + 3 MCP Prompts**:

        ### MCP Tools (AI-Powered)
        - 📊 **Analyze Leaderboard**: Get insights from evaluation results
        - 🐛 **Debug Trace**: Understand what happened in a specific test
        - 💰 **Estimate Cost**: Predict evaluation costs before running
        - ⚖️ **Compare Runs**: Compare two evaluation runs with AI-powered analysis
        - 🔍 **Analyze Results**: Deep dive into test results with optimization recommendations
        - 📦 **Get Dataset**: Load any HuggingFace dataset as JSON for flexible analysis

        ### MCP Resources (Data Access)
        - 📊 **leaderboard://{repo}**: Raw leaderboard data
        - 🔍 **trace://{trace_id}/{repo}**: Raw trace data
        - 💰 **cost://model/{model_name}**: Model pricing data

        ### MCP Prompts (Templates)
        - 📝 **analysis_prompt**: Templates for analysis requests
        - 🐛 **debug_prompt**: Templates for debugging traces
        - ⚡ **optimization_prompt**: Templates for optimization recommendations

        All powered by **Google Gemini 2.5 Pro**.

        ## MCP Connection

        ### Current Space (Development)
        **HuggingFace Space**: `https://huggingface.co/spaces/kshitijthakkar/TraceMind-mcp-server`

        **MCP Endpoint (Streamable HTTP - Recommended)**: `https://kshitijthakkar-tracemind-mcp-server.hf.space/gradio_api/mcp/`

        **MCP Endpoint (SSE - Deprecated)**: `https://kshitijthakkar-tracemind-mcp-server.hf.space/gradio_api/mcp/sse`

        ### After Hackathon Submission
        **HuggingFace Space**: `https://huggingface.co/spaces/MCP-1st-Birthday/TraceMind-mcp-server`

        **MCP Endpoint (Streamable HTTP - Recommended)**: `https://mcp-1st-birthday-tracemind-mcp-server.hf.space/gradio_api/mcp/`

        **MCP Endpoint (SSE - Deprecated)**: `https://mcp-1st-birthday-tracemind-mcp-server.hf.space/gradio_api/mcp/sse`
        """)

        # Session state for API keys
        gemini_key_state = gr.State(value=os.getenv("GEMINI_API_KEY", ""))
        hf_token_state = gr.State(value=os.getenv("HF_TOKEN", ""))

        with gr.Tabs():
            # Tab 0: Settings (API Keys)
            with gr.Tab("⚙️ Settings"):
                gr.Markdown("""
                ## 🔑 API Key Configuration

                Configure your API keys here. These will override environment variables for this session only.

                **Why configure here?**
                - No need to set environment variables
                - Test with different API keys easily
                - Secure session-only storage (not persisted)

                **Security Note**: API keys are stored in session state only and are not saved permanently.
                """)

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Google Gemini API Key")
                        gemini_key_input = gr.Textbox(
                            label="Gemini API Key",
                            placeholder="Enter your Google Gemini API key",
                            type="password",
                            value=os.getenv("GEMINI_API_KEY", ""),
                            info="Get your key from: https://aistudio.google.com/app/apikey"
                        )
                        gemini_status = gr.Markdown("Status: Using environment variable" if os.getenv("GEMINI_API_KEY") else "⚠️ Status: No API key configured")

                    with gr.Column():
                        gr.Markdown("### HuggingFace Token")
                        hf_token_input = gr.Textbox(
                            label="HuggingFace Token",
                            placeholder="Enter your HuggingFace token",
                            type="password",
                            value=os.getenv("HF_TOKEN", ""),
                            info="Get your token from: https://huggingface.co/settings/tokens"
                        )
                        hf_status = gr.Markdown("Status: Using environment variable" if os.getenv("HF_TOKEN") else "⚠️ Status: No token configured")

                with gr.Row():
                    save_keys_button = gr.Button("💾 Save API Keys for This Session", variant="primary", size="lg")
                    clear_keys_button = gr.Button("🗑️ Clear Session Keys", variant="secondary")

                keys_save_status = gr.Markdown("")

                def save_api_keys(gemini_key, hf_token):
                    """
                    Save API keys to session state.

                    Args:
                        gemini_key (str): Google Gemini API key
                        hf_token (str): HuggingFace token

                    Returns:
                        tuple: Updated state values and status message
                    """
                    status_messages = []

                    # Validate and save Gemini key
                    if gemini_key and gemini_key.strip():
                        try:
                            # Test the key by creating a client
                            test_client = GeminiClient(api_key=gemini_key.strip())
                            gemini_saved = gemini_key.strip()
                            status_messages.append("✅ Gemini API key validated and saved")
                        except Exception as e:
                            gemini_saved = os.getenv("GEMINI_API_KEY", "")
                            status_messages.append(f"❌ Gemini API key invalid: {str(e)}")
                    else:
                        gemini_saved = os.getenv("GEMINI_API_KEY", "")
                        status_messages.append("ℹ️ Gemini API key cleared (using environment variable if set)")

                    # Validate and save HF token
                    if hf_token and hf_token.strip():
                        hf_saved = hf_token.strip()
                        status_messages.append("✅ HuggingFace token saved")
                    else:
                        hf_saved = os.getenv("HF_TOKEN", "")
                        status_messages.append("ℹ️ HuggingFace token cleared (using environment variable if set)")

                    status_markdown = "\n\n".join(status_messages)

                    return gemini_saved, hf_saved, f"### Save Status\n\n{status_markdown}"

                def clear_api_keys():
                    """
                    Clear session API keys and revert to environment variables.

                    Returns:
                        tuple: Cleared state values and status message
                    """
                    env_gemini = os.getenv("GEMINI_API_KEY", "")
                    env_hf = os.getenv("HF_TOKEN", "")

                    status = "### Keys Cleared\n\nReverted to environment variables.\n\n"
                    if env_gemini:
                        status += "✅ Using GEMINI_API_KEY from environment\n\n"
                    else:
                        status += "⚠️ No GEMINI_API_KEY in environment\n\n"

                    if env_hf:
                        status += "✅ Using HF_TOKEN from environment"
                    else:
                        status += "⚠️ No HF_TOKEN in environment"

                    return env_gemini, env_hf, status

                save_keys_button.click(
                    fn=save_api_keys,
                    inputs=[gemini_key_input, hf_token_input],
                    outputs=[gemini_key_state, hf_token_state, keys_save_status]
                )

                clear_keys_button.click(
                    fn=clear_api_keys,
                    inputs=[],
                    outputs=[gemini_key_state, hf_token_state, keys_save_status]
                )

                gr.Markdown("""
                ---

                ### How It Works

                1. **Enter your API keys** in the fields above
                2. **Click "Save API Keys"** to validate and store them for this session
                3. **Use any tool** - they will automatically use your configured keys
                4. **Keys are session-only** - they won't be saved when you close the browser

                ### Environment Variables (Alternative)

                You can also set these as environment variables:
                ```bash
                export GEMINI_API_KEY="your-key-here"
                export HF_TOKEN="your-token-here"
                ```

                UI-configured keys will always override environment variables.
                """)

            # Tab 1: Analyze Leaderboard
            with gr.Tab("📊 Analyze Leaderboard"):
                gr.Markdown("### Get AI-powered insights from evaluation leaderboard")

                with gr.Row():
                    with gr.Column():
                        lb_repo = gr.Textbox(
                            label="Leaderboard Repository",
                            value="kshitijthakkar/smoltrace-leaderboard",
                            placeholder="username/dataset-name"
                        )
                        lb_metric = gr.Dropdown(
                            label="Metric Focus",
                            choices=["overall", "accuracy", "cost", "latency", "co2"],
                            value="overall"
                        )
                        lb_time = gr.Dropdown(
                            label="Time Range",
                            choices=["last_week", "last_month", "all_time"],
                            value="last_week"
                        )
                        lb_top_n = gr.Slider(
                            label="Top N Models",
                            minimum=3,
                            maximum=10,
                            value=5,
                            step=1
                        )
                        lb_button = gr.Button("🔍 Analyze", variant="primary")

                    with gr.Column():
                        lb_output = gr.Markdown(label="Analysis Results")

                async def run_analyze_leaderboard(repo, metric, time_range, top_n, gemini_key, hf_token):
                    """
                    Analyze agent evaluation leaderboard and generate AI-powered insights.

                    This tool loads agent evaluation data from HuggingFace datasets and uses
                    Google Gemini 2.5 Pro to provide intelligent analysis of top performers,
                    trends, cost/performance trade-offs, and actionable recommendations.

                    Args:
                        repo (str): HuggingFace dataset repository containing leaderboard data
                        metric (str): Primary metric to focus analysis on - "overall", "accuracy", "cost", "latency", or "co2"
                        time_range (str): Time range for analysis - "last_week", "last_month", or "all_time"
                        top_n (int): Number of top models to highlight in analysis (3-10)
                        gemini_key (str): Gemini API key from session state
                        hf_token (str): HuggingFace token from session state

                    Returns:
                        str: Markdown-formatted analysis with top performers, trends, and recommendations
                    """
                    try:
                        # Use user-provided key or fall back to environment variable
                        api_key = gemini_key if gemini_key and gemini_key.strip() else None

                        result = await analyze_leaderboard(
                            leaderboard_repo=repo,
                            metric_focus=metric,
                            time_range=time_range,
                            top_n=int(top_n),
                            hf_token=hf_token if hf_token and hf_token.strip() else None,
                            gemini_api_key=api_key
                        )
                        return result
                    except Exception as e:
                        return f"❌ **Error**: {str(e)}"

                lb_button.click(
                    fn=run_analyze_leaderboard,
                    inputs=[lb_repo, lb_metric, lb_time, lb_top_n, gemini_key_state, hf_token_state],
                    outputs=[lb_output]
                )

            # Tab 2: Debug Trace
            with gr.Tab("🐛 Debug Trace"):
                gr.Markdown("### Ask questions about specific agent execution traces")

                with gr.Row():
                    with gr.Column():
                        trace_id = gr.Textbox(
                            label="Trace ID",
                            placeholder="trace_abc123",
                            info="Get this from the Run Detail screen"
                        )
                        traces_repo = gr.Textbox(
                            label="Traces Repository",
                            placeholder="username/agent-traces-model-timestamp",
                            info="Dataset containing trace data"
                        )
                        question = gr.Textbox(
                            label="Your Question",
                            placeholder="Why was tool X called twice?",
                            lines=3
                        )
                        trace_button = gr.Button("🔍 Analyze", variant="primary")

                    with gr.Column():
                        trace_output = gr.Markdown(label="Debug Analysis")

                async def run_debug_trace(trace_id_val, traces_repo_val, question_val, gemini_key, hf_token):
                    """
                    Debug a specific agent execution trace using OpenTelemetry data.

                    This tool analyzes OpenTelemetry trace data from agent executions and uses
                    Google Gemini 2.5 Pro to answer specific questions about the execution flow,
                    identify bottlenecks, explain agent behavior, and provide debugging insights.

                    Args:
                        trace_id_val (str): Unique identifier for the trace to analyze (e.g., "trace_abc123")
                        traces_repo_val (str): HuggingFace dataset repository containing trace data
                        question_val (str): Specific question about the trace (optional, defaults to general analysis)
                        gemini_key (str): Gemini API key from session state
                        hf_token (str): HuggingFace token from session state

                    Returns:
                        str: Markdown-formatted debug analysis with step-by-step breakdown and answers
                    """
                    try:
                        if not trace_id_val or not traces_repo_val:
                            return "❌ **Error**: Please provide both Trace ID and Traces Repository"

                        # Use user-provided key or fall back to environment variable
                        api_key = gemini_key if gemini_key and gemini_key.strip() else None

                        result = await debug_trace(
                            trace_id=trace_id_val,
                            traces_repo=traces_repo_val,
                            question=question_val or "Analyze this trace",
                            hf_token=hf_token if hf_token and hf_token.strip() else None,
                            gemini_api_key=api_key
                        )
                        return result
                    except Exception as e:
                        return f"❌ **Error**: {str(e)}"

                trace_button.click(
                    fn=run_debug_trace,
                    inputs=[trace_id, traces_repo, question, gemini_key_state, hf_token_state],
                    outputs=[trace_output]
                )

            # Tab 3: Estimate Cost
            with gr.Tab("💰 Estimate Cost"):
                gr.Markdown("### Predict evaluation costs before running")

                with gr.Row():
                    with gr.Column():
                        cost_model = gr.Textbox(
                            label="Model",
                            placeholder="openai/gpt-4 or meta-llama/Llama-3.1-8B",
                            info="Use litellm format (provider/model)"
                        )
                        cost_agent_type = gr.Dropdown(
                            label="Agent Type",
                            choices=["tool", "code", "both"],
                            value="both"
                        )
                        cost_num_tests = gr.Slider(
                            label="Number of Tests",
                            minimum=10,
                            maximum=1000,
                            value=100,
                            step=10
                        )
                        cost_hardware = gr.Dropdown(
                            label="Hardware Type",
                            choices=["auto", "cpu", "gpu_a10", "gpu_h200"],
                            value="auto",
                            info="'auto' will choose based on model type"
                        )
                        cost_button = gr.Button("💰 Estimate", variant="primary")

                    with gr.Column():
                        cost_output = gr.Markdown(label="Cost Estimate")

                async def run_estimate_cost(model, agent_type, num_tests, hardware, gemini_key):
                    """
                    Estimate the cost, duration, and CO2 emissions of running agent evaluations.

                    This tool predicts costs before running evaluations by calculating LLM API costs,
                    HuggingFace Jobs compute costs, and CO2 emissions. Uses Google Gemini 2.5 Pro
                    to provide detailed cost breakdown and optimization recommendations.

                    Args:
                        model (str): Model identifier in litellm format (e.g., "openai/gpt-4", "meta-llama/Llama-3.1-8B")
                        agent_type (str): Type of agent capabilities to test - "tool", "code", or "both"
                        num_tests (int): Number of test cases to run (10-1000)
                        hardware (str): Hardware type for HF Jobs - "auto", "cpu", "gpu_a10", or "gpu_h200"
                        gemini_key (str): Gemini API key from session state

                    Returns:
                        str: Markdown-formatted cost estimate with LLM costs, HF Jobs costs, duration, CO2, and tips
                    """
                    try:
                        if not model:
                            return "❌ **Error**: Please provide a model name"

                        # Use user-provided key or fall back to environment variable
                        api_key = gemini_key if gemini_key and gemini_key.strip() else None

                        result = await estimate_cost(
                            model=model,
                            agent_type=agent_type,
                            num_tests=int(num_tests),
                            hardware=hardware,
                            gemini_api_key=api_key
                        )
                        return result
                    except Exception as e:
                        return f"❌ **Error**: {str(e)}"

                cost_button.click(
                    fn=run_estimate_cost,
                    inputs=[cost_model, cost_agent_type, cost_num_tests, cost_hardware, gemini_key_state],
                    outputs=[cost_output]
                )

            # Tab 4: Compare Runs
            with gr.Tab("⚖️ Compare Runs"):
                gr.Markdown("""
                ## Compare Two Evaluation Runs

                Compare two evaluation runs with AI-powered analysis across multiple dimensions:
                success rate, cost efficiency, speed, environmental impact, and more.
                """)

                with gr.Row():
                    with gr.Column():
                        compare_run_id_1 = gr.Textbox(
                            label="First Run ID",
                            placeholder="e.g., run_abc123",
                            info="Enter the run_id from the leaderboard"
                        )
                    with gr.Column():
                        compare_run_id_2 = gr.Textbox(
                            label="Second Run ID",
                            placeholder="e.g., run_xyz789",
                            info="Enter the run_id to compare against"
                        )

                with gr.Row():
                    compare_focus = gr.Dropdown(
                        choices=["comprehensive", "cost", "performance", "eco_friendly"],
                        value="comprehensive",
                        label="Comparison Focus",
                        info="Choose what aspect to focus the comparison on"
                    )
                    compare_repo = gr.Textbox(
                        label="Leaderboard Repository",
                        value="kshitijthakkar/smoltrace-leaderboard",
                        info="HuggingFace dataset containing leaderboard data"
                    )

                compare_button = gr.Button("🔍 Compare Runs", variant="primary")
                compare_output = gr.Markdown()

                async def run_compare_runs(run_id_1, run_id_2, focus, repo, gemini_key, hf_token):
                    """
                    Compare two evaluation runs and generate AI-powered comparative analysis.

                    This tool fetches data for two evaluation runs from the leaderboard and uses
                    Google Gemini 2.5 Pro to provide intelligent comparison across multiple dimensions:
                    success rate, cost efficiency, speed, environmental impact, and use case recommendations.

                    Args:
                        run_id_1 (str): First run ID from the leaderboard to compare
                        run_id_2 (str): Second run ID from the leaderboard to compare against
                        focus (str): Focus area - "comprehensive", "cost", "performance", or "eco_friendly"
                        repo (str): HuggingFace dataset repository containing leaderboard data
                        gemini_key (str): Gemini API key from session state
                        hf_token (str): HuggingFace token from session state

                    Returns:
                        str: Markdown-formatted comparative analysis with winners, trade-offs, and recommendations
                    """
                    try:
                        # Use user-provided key or fall back to environment variable
                        api_key = gemini_key if gemini_key and gemini_key.strip() else None

                        result = await compare_runs(
                            run_id_1=run_id_1,
                            run_id_2=run_id_2,
                            leaderboard_repo=repo,
                            comparison_focus=focus,
                            hf_token=hf_token if hf_token and hf_token.strip() else None,
                            gemini_api_key=api_key
                        )
                        return result
                    except Exception as e:
                        return f"❌ **Error**: {str(e)}"

                compare_button.click(
                    fn=run_compare_runs,
                    inputs=[compare_run_id_1, compare_run_id_2, compare_focus, compare_repo, gemini_key_state, hf_token_state],
                    outputs=[compare_output]
                )

            # Tab 5: Analyze Results
            with gr.Tab("🔍 Analyze Results"):
                gr.Markdown("""
                ## Analyze Test Results & Get Optimization Recommendations

                Deep dive into individual test case results to identify failure patterns,
                performance bottlenecks, and cost optimization opportunities.
                """)

                with gr.Row():
                    results_repo_input = gr.Textbox(
                        label="Results Repository",
                        placeholder="e.g., username/smoltrace-results-gpt4-20251114",
                        info="HuggingFace dataset containing results data"
                    )
                    results_focus = gr.Dropdown(
                        choices=["comprehensive", "failures", "performance", "cost"],
                        value="comprehensive",
                        label="Analysis Focus",
                        info="What aspect to focus the analysis on"
                    )

                with gr.Row():
                    results_max_rows = gr.Slider(
                        minimum=10,
                        maximum=500,
                        value=100,
                        step=10,
                        label="Max Test Cases to Analyze",
                        info="Limit number of test cases for analysis"
                    )

                results_button = gr.Button("🔍 Analyze Results", variant="primary")
                results_output = gr.Markdown()

                async def run_analyze_results(repo, focus, max_rows, gemini_key, hf_token):
                    """
                    Analyze detailed test results and provide optimization recommendations.

                    Args:
                        repo (str): HuggingFace dataset repository containing results
                        focus (str): Analysis focus area
                        max_rows (int): Maximum test cases to analyze
                        gemini_key (str): Gemini API key from session state
                        hf_token (str): HuggingFace token from session state

                    Returns:
                        str: Markdown-formatted analysis with recommendations
                    """
                    try:
                        if not repo:
                            return "❌ **Error**: Please provide a results repository"

                        # Use user-provided key or fall back to environment variable
                        api_key = gemini_key if gemini_key and gemini_key.strip() else None

                        result = await analyze_results(
                            results_repo=repo,
                            analysis_focus=focus,
                            max_rows=int(max_rows),
                            hf_token=hf_token if hf_token and hf_token.strip() else None,
                            gemini_api_key=api_key
                        )
                        return result
                    except Exception as e:
                        return f"❌ **Error**: {str(e)}"

                results_button.click(
                    fn=run_analyze_results,
                    inputs=[results_repo_input, results_focus, results_max_rows, gemini_key_state, hf_token_state],
                    outputs=[results_output]
                )

            # Tab 6: Get Dataset
            with gr.Tab("📦 Get Dataset"):
                gr.Markdown("""
                ## Load SMOLTRACE Datasets as JSON

                This tool loads datasets with the **smoltrace-** prefix and returns the raw data as JSON.
                Use this to access leaderboard data, results datasets, traces datasets, or metrics datasets.

                **Restriction**: Only datasets with "smoltrace-" in the name are allowed for security.

                **Tip**: If you don't know which dataset to load, first load the leaderboard to see
                dataset references in the `results_dataset`, `traces_dataset`, `metrics_dataset` fields.
                """)

                with gr.Row():
                    dataset_repo_input = gr.Textbox(
                        label="Dataset Repository (must contain 'smoltrace-')",
                        placeholder="e.g., kshitijthakkar/smoltrace-leaderboard",
                        value="kshitijthakkar/smoltrace-leaderboard",
                        info="HuggingFace dataset repository path with smoltrace- prefix"
                    )
                    dataset_max_rows = gr.Slider(
                        minimum=1,
                        maximum=200,
                        value=50,
                        step=1,
                        label="Max Rows",
                        info="Limit rows to avoid token limits"
                    )

                dataset_button = gr.Button("📥 Load Dataset", variant="primary")
                dataset_output = gr.JSON(label="Dataset JSON Output")

                async def run_get_dataset(repo, max_rows, hf_token):
                    """
                    Load SMOLTRACE datasets from HuggingFace and return as JSON.

                    This tool loads datasets with the "smoltrace-" prefix and returns the raw data
                    as JSON. Use this to access leaderboard data, results datasets, traces datasets,
                    or metrics datasets. Only datasets with "smoltrace-" in the name are allowed.

                    Args:
                        repo (str): HuggingFace dataset repository path with "smoltrace-" prefix (e.g., "kshitijthakkar/smoltrace-leaderboard")
                        max_rows (int): Maximum number of rows to return (1-200, default 50)
                        hf_token (str): HuggingFace token from session state

                    Returns:
                        dict: JSON object with dataset data, metadata, total rows, and column names
                    """
                    try:
                        import json
                        result = await get_dataset(
                            dataset_repo=repo,
                            max_rows=int(max_rows),
                            hf_token=hf_token if hf_token and hf_token.strip() else None
                        )
                        # Parse JSON string back to dict for JSON component
                        return json.loads(result)
                    except Exception as e:
                        return {"error": str(e)}

                dataset_button.click(
                    fn=run_get_dataset,
                    inputs=[dataset_repo_input, dataset_max_rows, hf_token_state],
                    outputs=[dataset_output]
                )

            # Tab 6: MCP Resources & Prompts
            with gr.Tab("🔌 MCP Resources & Prompts"):
                gr.Markdown("""
                ## MCP Resources & Prompts

                Beyond the 5 MCP Tools, this server also exposes **MCP Resources** and **MCP Prompts**
                that MCP clients can use directly.

                ### MCP Resources (Read-Only Data Access)

                Resources provide direct access to data without AI processing:

                #### 1. `leaderboard://{repo}`
                Get raw leaderboard data in JSON format.

                **Example**: `leaderboard://kshitijthakkar/smoltrace-leaderboard`

                **Returns**: JSON with all evaluation runs

                #### 2. `trace://{trace_id}/{repo}`
                Get raw trace data for a specific trace.

                **Example**: `trace://trace_abc123/kshitijthakkar/smoltrace-traces-gpt4`

                **Returns**: JSON with OpenTelemetry spans

                #### 3. `cost://model/{model_name}`
                Get cost information for a specific model.

                **Example**: `cost://model/openai/gpt-4`

                **Returns**: JSON with pricing data

                ---

                ### MCP Prompts (Reusable Templates)

                Prompts provide standardized templates for common workflows:

                #### 1. `analysis_prompt(analysis_type, focus_area, detail_level)`
                Generate analysis prompt templates.

                **Parameters**:
                - `analysis_type`: "leaderboard", "trace", "cost"
                - `focus_area`: "overall", "performance", "cost", "efficiency"
                - `detail_level`: "summary", "detailed", "comprehensive"

                #### 2. `debug_prompt(debug_type, context)`
                Generate debugging prompt templates.

                **Parameters**:
                - `debug_type`: "error", "performance", "behavior", "optimization"
                - `context`: "agent_execution", "tool_calling", "llm_reasoning"

                #### 3. `optimization_prompt(optimization_goal, constraints)`
                Generate optimization prompt templates.

                **Parameters**:
                - `optimization_goal`: "cost", "speed", "quality", "efficiency"
                - `constraints`: "maintain_quality", "maintain_speed", "no_constraints"

                ---

                ### Testing MCP Resources

                Test resources directly from this UI:
                """)

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### Test Leaderboard Resource")
                        resource_lb_repo = gr.Textbox(
                            label="Repository",
                            value="kshitijthakkar/smoltrace-leaderboard"
                        )
                        resource_lb_button = gr.Button("Fetch Leaderboard Data", variant="primary")
                        resource_lb_output = gr.JSON(label="Resource Output")

                        def test_leaderboard_resource(repo):
                            """
                            Test the leaderboard MCP resource by fetching raw leaderboard data.

                            Args:
                                repo (str): HuggingFace dataset repository name

                            Returns:
                                dict: JSON object with leaderboard data
                            """
                            from mcp_tools import get_leaderboard_data
                            import json
                            result = get_leaderboard_data(repo)
                            return json.loads(result)

                        resource_lb_button.click(
                            fn=test_leaderboard_resource,
                            inputs=[resource_lb_repo],
                            outputs=[resource_lb_output]
                        )

                    with gr.Column():
                        gr.Markdown("#### Test Cost Resource")
                        resource_cost_model = gr.Textbox(
                            label="Model Name",
                            value="openai/gpt-4"
                        )
                        resource_cost_button = gr.Button("Fetch Cost Data", variant="primary")
                        resource_cost_output = gr.JSON(label="Resource Output")

                        def test_cost_resource(model):
                            """
                            Test the cost MCP resource by fetching model pricing data.

                            Args:
                                model (str): Model identifier (e.g., "openai/gpt-4")

                            Returns:
                                dict: JSON object with cost and pricing information
                            """
                            from mcp_tools import get_cost_data
                            import json
                            result = get_cost_data(model)
                            return json.loads(result)

                        resource_cost_button.click(
                            fn=test_cost_resource,
                            inputs=[resource_cost_model],
                            outputs=[resource_cost_output]
                        )

                gr.Markdown("---")
                gr.Markdown("### Testing MCP Prompts")
                gr.Markdown("Generate prompt templates for different scenarios:")

                with gr.Row():
                    with gr.Column():
                        prompt_type = gr.Radio(
                            label="Prompt Type",
                            choices=["analysis_prompt", "debug_prompt", "optimization_prompt"],
                            value="analysis_prompt"
                        )

                        # Analysis prompt params
                        with gr.Group(visible=True) as analysis_group:
                            analysis_type = gr.Dropdown(
                                label="Analysis Type",
                                choices=["leaderboard", "trace", "cost"],
                                value="leaderboard"
                            )
                            focus_area = gr.Dropdown(
                                label="Focus Area",
                                choices=["overall", "performance", "cost", "efficiency"],
                                value="overall"
                            )
                            detail_level = gr.Dropdown(
                                label="Detail Level",
                                choices=["summary", "detailed", "comprehensive"],
                                value="detailed"
                            )

                        # Debug prompt params
                        with gr.Group(visible=False) as debug_group:
                            debug_type = gr.Dropdown(
                                label="Debug Type",
                                choices=["error", "performance", "behavior", "optimization"],
                                value="error"
                            )
                            debug_context = gr.Dropdown(
                                label="Context",
                                choices=["agent_execution", "tool_calling", "llm_reasoning"],
                                value="agent_execution"
                            )

                        # Optimization prompt params
                        with gr.Group(visible=False) as optimization_group:
                            optimization_goal = gr.Dropdown(
                                label="Optimization Goal",
                                choices=["cost", "speed", "quality", "efficiency"],
                                value="cost"
                            )
                            constraints = gr.Dropdown(
                                label="Constraints",
                                choices=["maintain_quality", "maintain_speed", "no_constraints"],
                                value="maintain_quality"
                            )

                        prompt_button = gr.Button("Generate Prompt", variant="primary")

                    with gr.Column():
                        prompt_output = gr.Textbox(
                            label="Generated Prompt Template",
                            lines=10,
                            max_lines=20
                        )

                def toggle_prompt_groups(prompt_type):
                    """
                    Toggle visibility of prompt parameter groups based on selected prompt type.

                    Args:
                        prompt_type (str): The type of prompt selected

                    Returns:
                        dict: Gradio update objects for group visibility
                    """
                    return {
                        analysis_group: gr.update(visible=(prompt_type == "analysis_prompt")),
                        debug_group: gr.update(visible=(prompt_type == "debug_prompt")),
                        optimization_group: gr.update(visible=(prompt_type == "optimization_prompt"))
                    }

                prompt_type.change(
                    fn=toggle_prompt_groups,
                    inputs=[prompt_type],
                    outputs=[analysis_group, debug_group, optimization_group]
                )

                def generate_prompt(
                    prompt_type,
                    analysis_type_val, focus_area_val, detail_level_val,
                    debug_type_val, debug_context_val,
                    optimization_goal_val, constraints_val
                ):
                    """
                    Generate a prompt template based on the selected type and parameters.

                    Args:
                        prompt_type (str): Type of prompt to generate
                        analysis_type_val (str): Analysis type parameter
                        focus_area_val (str): Focus area parameter
                        detail_level_val (str): Detail level parameter
                        debug_type_val (str): Debug type parameter
                        debug_context_val (str): Debug context parameter
                        optimization_goal_val (str): Optimization goal parameter
                        constraints_val (str): Constraints parameter

                    Returns:
                        str: Generated prompt template text
                    """
                    from mcp_tools import analysis_prompt, debug_prompt, optimization_prompt

                    if prompt_type == "analysis_prompt":
                        return analysis_prompt(analysis_type_val, focus_area_val, detail_level_val)
                    elif prompt_type == "debug_prompt":
                        return debug_prompt(debug_type_val, debug_context_val)
                    elif prompt_type == "optimization_prompt":
                        return optimization_prompt(optimization_goal_val, constraints_val)

                prompt_button.click(
                    fn=generate_prompt,
                    inputs=[
                        prompt_type,
                        analysis_type, focus_area, detail_level,
                        debug_type, debug_context,
                        optimization_goal, constraints
                    ],
                    outputs=[prompt_output]
                )

            # Tab 7: API Documentation
            with gr.Tab("📖 API Documentation"):
                gr.Markdown("""
                ## MCP Tool Specifications

                ### 1. analyze_leaderboard

                **Description**: Generate AI-powered insights from evaluation leaderboard data

                **Parameters**:
                - `leaderboard_repo` (str): HuggingFace dataset repository (default: "kshitijthakkar/smoltrace-leaderboard")
                - `metric_focus` (str): "overall", "accuracy", "cost", "latency", or "co2" (default: "overall")
                - `time_range` (str): "last_week", "last_month", or "all_time" (default: "last_week")
                - `top_n` (int): Number of top models to highlight (default: 5, min: 3, max: 10)

                **Returns**: Markdown-formatted analysis with top performers, trends, and recommendations

                ---

                ### 2. debug_trace

                **Description**: Answer questions about specific agent execution traces

                **Parameters**:
                - `trace_id` (str, required): Unique identifier for the trace
                - `traces_repo` (str, required): HuggingFace dataset repository with trace data
                - `question` (str): Specific question about the trace (default: "Analyze this trace and explain what happened")

                **Returns**: Markdown-formatted debug analysis with step-by-step breakdown

                ---

                ### 3. estimate_cost

                **Description**: Predict evaluation costs before running

                **Parameters**:
                - `model` (str, required): Model identifier in litellm format (e.g., "openai/gpt-4")
                - `agent_type` (str, required): "tool", "code", or "both"
                - `num_tests` (int): Number of test cases (default: 100, min: 10, max: 1000)
                - `hardware` (str): "auto", "cpu", "gpu_a10", or "gpu_h200" (default: "auto")

                **Returns**: Markdown-formatted cost estimate with breakdown and optimization tips

                ---

                ### 4. compare_runs

                **Description**: Compare two evaluation runs with AI-powered analysis

                **Parameters**:
                - `run_id_1` (str, required): First run ID from the leaderboard
                - `run_id_2` (str, required): Second run ID to compare against
                - `leaderboard_repo` (str): HuggingFace dataset repository (default: "kshitijthakkar/smoltrace-leaderboard")
                - `comparison_focus` (str): "comprehensive", "cost", "performance", or "eco_friendly" (default: "comprehensive")

                **Returns**: Markdown-formatted comparative analysis with winner for each category, trade-offs, and recommendations

                **Focus Options**:
                - `comprehensive`: Complete comparison across all dimensions (success rate, cost, speed, CO2, GPU)
                - `cost`: Detailed cost efficiency analysis and ROI
                - `performance`: Speed and accuracy trade-off analysis
                - `eco_friendly`: Environmental impact and carbon footprint comparison

                ---

                ### 5. get_dataset

                **Description**: Load SMOLTRACE datasets from HuggingFace and return as JSON

                **Parameters**:
                - `dataset_repo` (str, required): HuggingFace dataset repository path with "smoltrace-" prefix (e.g., "kshitijthakkar/smoltrace-leaderboard")
                - `max_rows` (int): Maximum number of rows to return (default: 50, range: 1-200)

                **Returns**: JSON object with dataset data and metadata

                **Restriction**: Only datasets with "smoltrace-" in the repository name are allowed for security.

                **Use Cases**:
                - Load smoltrace-leaderboard to find run IDs, model names, and supporting dataset references
                - Load smoltrace-results-* datasets to see individual test case details
                - Load smoltrace-traces-* datasets to access OpenTelemetry trace data
                - Load smoltrace-metrics-* datasets to get GPU metrics and performance data

                **Workflow**:
                1. Call `get_dataset("kshitijthakkar/smoltrace-leaderboard")` to see all runs
                2. Find the `results_dataset`, `traces_dataset`, or `metrics_dataset` field for a specific run
                3. Call `get_dataset(dataset_repo)` with that smoltrace-* dataset name to get detailed data

                ---

                ## MCP Integration

                This Gradio app is MCP-enabled. When deployed to HuggingFace Spaces, it can be accessed via MCP clients.

                **Current Space**: `https://huggingface.co/spaces/kshitijthakkar/TraceMind-mcp-server`

                **After Hackathon Submission**: `https://huggingface.co/spaces/MCP-1st-Birthday/TraceMind-mcp-server`

                **MCP Endpoint (Streamable HTTP)**: Use `{space-name}.hf.space/gradio_api/mcp/` (see easiest connection method at https://huggingface.co/settings/mcp)

                ### What's Exposed via MCP:

                #### 5 MCP Tools (AI-Powered)
                The five tools above (`analyze_leaderboard`, `debug_trace`, `estimate_cost`, `compare_runs`, `get_dataset`)
                are automatically exposed as MCP tools and can be called from any MCP client.

                #### 3 MCP Resources (Data Access)
                - `leaderboard://{repo}` - Raw leaderboard data
                - `trace://{trace_id}/{repo}` - Raw trace data
                - `cost://model/{model_name}` - Model pricing data

                #### 3 MCP Prompts (Templates)
                - `analysis_prompt(analysis_type, focus_area, detail_level)` - Analysis templates
                - `debug_prompt(debug_type, context)` - Debug templates
                - `optimization_prompt(optimization_goal, constraints)` - Optimization templates

                **See the "🔌 MCP Resources & Prompts" tab to test these features.**
                """)

        gr.Markdown("""
        ---

        ## Environment Variables

        Required:
        - `GEMINI_API_KEY`: Your Google Gemini API key
        - `HF_TOKEN`: Your HuggingFace token (for dataset access)

        ## Source Code

        This server is part of the TraceMind project submission for MCP's 1st Birthday Hackathon.

        **Track 1**: Building MCP (Enterprise)
        **Tag**: `building-mcp-track-enterprise`
        """)

    return demo

if __name__ == "__main__":
    # Create Gradio interface
    demo = create_gradio_ui()

    # Launch with MCP server enabled
    # share=True creates a temporary public HTTPS URL for testing with Claude Code
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        #share=True,  # Creates temporary HTTPS URL (e.g., https://abc123.gradio.live)
        mcp_server=True  # Enable MCP server functionality
    )
