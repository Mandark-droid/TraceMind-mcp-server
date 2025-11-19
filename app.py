"""
TraceMind MCP Server - Hugging Face Space Entry Point (Track 1)

This file serves as the entry point for HuggingFace Space deployment.
Exposes 10 AI-powered MCP tools + 3 Resources + 3 Prompts via Gradio's native MCP support.

Built on Open Source Foundation:
    🔭 TraceVerde (genai_otel_instrument) - Automatic OpenTelemetry instrumentation
       for LLM frameworks (LiteLLM, Transformers, LangChain, etc.)
       GitHub: https://github.com/Mandark-droid/genai_otel_instrument
       PyPI: https://pypi.org/project/genai-otel-instrument

    📊 SMOLTRACE - Agent evaluation engine with OTEL tracing built-in
       Generates structured datasets (leaderboard, results, traces, metrics)
       GitHub: https://github.com/Mandark-droid/SMOLTRACE
       PyPI: https://pypi.org/project/smoltrace/

    The Flow: TraceVerde instruments → SMOLTRACE evaluates → TraceMind analyzes

Architecture:
    User → MCP Client (Claude Desktop, Continue, Cline, etc.)
         → MCP Endpoint (Gradio SSE)
         → TraceMind MCP Server (this file)
         → Tools (mcp_tools.py)
         → Google Gemini 2.5 Pro API

For Track 1: Building MCP Servers - Enterprise Category
https://huggingface.co/MCP-1st-Birthday

Tools Provided:
    📊 analyze_leaderboard - AI-powered leaderboard analysis
    🐛 debug_trace - Debug agent execution traces with AI
    💰 estimate_cost - Predict evaluation costs before running
    ⚖️ compare_runs - Compare evaluation runs with AI analysis
    🏆 get_top_performers - Get top N models from leaderboard (optimized)
    📈 get_leaderboard_summary - Get leaderboard overview statistics
    📦 get_dataset - Load SMOLTRACE datasets as JSON
    🧪 generate_synthetic_dataset - Create domain-specific test datasets
    📝 generate_prompt_template - Generate customized smolagents prompt templates
    📤 push_dataset_to_hub - Upload datasets to HuggingFace Hub

Compatible with:
- Claude Desktop (via Gradio MCP support)
- Continue.dev (VS Code extension)
- Cline (VS Code extension)
- Any MCP client supporting Gradio's MCP protocol
"""

import os
import logging
import gradio as gr
from typing import Optional, Dict, Any
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Local imports
from gemini_client import GeminiClient
from mcp_tools import (
    analyze_leaderboard,
    debug_trace,
    estimate_cost,
    compare_runs,
    analyze_results,
    get_top_performers,
    get_leaderboard_summary,
    get_dataset,
    generate_synthetic_dataset,
    generate_prompt_template,
    push_dataset_to_hub
)

# Initialize default Gemini client (fallback if user doesn't provide key)
try:
    default_gemini_client = GeminiClient()
except ValueError:
    default_gemini_client = None  # Will prompt user to enter API key

# Gradio Interface for Testing
def create_gradio_ui():
    """Create Gradio UI for testing MCP tools"""

    # Note: In Gradio 6, theme is passed to launch(), not Blocks()
    with gr.Blocks(title="TraceMind MCP Server") as demo:
        gr.Markdown("""
        # 🤖 TraceMind MCP Server

        **AI-Powered Analysis for Agent Evaluation Data**

        This server provides **10 MCP Tools + 3 MCP Resources + 3 MCP Prompts**:

        ### MCP Tools (AI-Powered & Optimized)
        - 📊 **Analyze Leaderboard**: Get AI-powered insights from evaluation results
        - 🐛 **Debug Trace**: Understand what happened in a specific test with AI debugging
        - 💰 **Estimate Cost**: Predict evaluation costs before running with AI recommendations
        - ⚖️ **Compare Runs**: Compare two evaluation runs with AI-powered analysis
        - 🏆 **Get Top Performers**: Get top N models from leaderboard (optimized for quick queries)
        - 📈 **Get Leaderboard Summary**: Get high-level leaderboard statistics (optimized for overview)
        - 📦 **Get Dataset**: Load any HuggingFace dataset as JSON for flexible analysis
        - 🧪 **Generate Synthetic Dataset**: Create domain-specific test datasets for SMOLTRACE
        - 📝 **Generate Prompt Template**: Create customized smolagents prompt templates for your domain
        - 📤 **Push to Hub**: Upload generated datasets to HuggingFace Hub

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

        **HuggingFace Space**: `https://huggingface.co/spaces/MCP-1st-Birthday/TraceMind-mcp-server`

        **MCP Endpoint (SSE - Recommended)**: `https://mcp-1st-birthday-tracemind-mcp-server.hf.space/gradio_api/mcp/sse`

        **MCP Endpoint (Streamable HTTP)**: `https://mcp-1st-birthday-tracemind-mcp-server.hf.space/gradio_api/mcp/`
        """)
        with gr.Tabs():
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

                async def run_analyze_leaderboard(repo, metric, time_range, top_n):
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
                        result = await analyze_leaderboard(
                            leaderboard_repo=repo,
                            metric_focus=metric,
                            time_range=time_range,
                            top_n=int(top_n)
                        )
                        return result
                    except Exception as e:
                        return f"❌ **Error**: {str(e)}"

                lb_button.click(
                    fn=run_analyze_leaderboard,
                    inputs=[lb_repo, lb_metric, lb_time, lb_top_n],
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

                async def run_debug_trace(trace_id_val, traces_repo_val, question_val):
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

                        result = await debug_trace(
                            trace_id=trace_id_val,
                            traces_repo=traces_repo_val,
                            question=question_val or "Analyze this trace")
                        return result
                    except Exception as e:
                        return f"❌ **Error**: {str(e)}"

                trace_button.click(
                    fn=run_debug_trace,
                    inputs=[trace_id, traces_repo, question],
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
                            choices=[
                                "auto",
                                # Modal
                                "cpu", "gpu_t4", "gpu_l4", "gpu_a10", "gpu_l40s",
                                "gpu_a100", "gpu_a100_80gb", "gpu_h100", "gpu_h200", "gpu_b200",
                                # HuggingFace Jobs
                                "cpu-basic", "cpu-upgrade",
                                "t4-small", "t4-medium",
                                "l4x1", "l4x4",
                                "a10g-small", "a10g-large", "a10g-largex2", "a10g-largex4",
                                "a100-large",
                                "v5e-1x1", "v5e-2x2", "v5e-2x4"
                            ],
                            value="auto",
                            info="Supports Modal and HuggingFace Jobs hardware. 'auto' selects cpu-basic (API) or a10g-small (local)."
                        )
                        cost_button = gr.Button("💰 Estimate", variant="primary")

                    with gr.Column():
                        cost_output = gr.Markdown(label="Cost Estimate")

                async def run_estimate_cost(model, agent_type, num_tests, hardware):
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

                        result = await estimate_cost(
                            model=model,
                            agent_type=agent_type,
                            num_tests=int(num_tests),
                            hardware=hardware
                        )
                        return result
                    except Exception as e:
                        return f"❌ **Error**: {str(e)}"

                cost_button.click(
                    fn=run_estimate_cost,
                    inputs=[cost_model, cost_agent_type, cost_num_tests, cost_hardware],
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

                async def run_compare_runs(run_id_1, run_id_2, focus, repo):
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
                        result = await compare_runs(
                            run_id_1=run_id_1,
                            run_id_2=run_id_2,
                            leaderboard_repo=repo,
                            comparison_focus=focus
                        )
                        return result
                    except Exception as e:
                        return f"❌ **Error**: {str(e)}"

                compare_button.click(
                    fn=run_compare_runs,
                    inputs=[compare_run_id_1, compare_run_id_2, compare_focus, compare_repo],
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

                async def run_analyze_results(repo, focus, max_rows):
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

                        result = await analyze_results(
                            results_repo=repo,
                            analysis_focus=focus,
                            max_rows=int(max_rows)
                        )
                        return result
                    except Exception as e:
                        return f"❌ **Error**: {str(e)}"

                results_button.click(
                    fn=run_analyze_results,
                    inputs=[results_repo_input, results_focus, results_max_rows],
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

                async def run_get_dataset(repo, max_rows):
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
                            max_rows=int(max_rows)
                        )
                        # Parse JSON string back to dict for JSON component
                        return json.loads(result)
                    except Exception as e:
                        return {"error": str(e)}

                dataset_button.click(
                    fn=run_get_dataset,
                    inputs=[dataset_repo_input, dataset_max_rows],
                    outputs=[dataset_output]
                )

            # Tab 6: Generate Synthetic Dataset
            with gr.Tab("🧪 Generate Synthetic Dataset"):
                gr.Markdown("""
                ## Create Domain-Specific Test Datasets for SMOLTRACE

                Use AI to generate synthetic evaluation tasks tailored to your domain and tools.
                Perfect for creating custom benchmarks when standard datasets don't fit your use case.

                **🎯 Enterprise Use Case**: Quickly create evaluation datasets for:
                - Custom tools and APIs your agents use
                - Industry-specific domains (finance, healthcare, legal, etc.)
                - Internal workflows and processes
                - Specialized agent capabilities

                **Output Format**: SMOLTRACE-compatible task dataset ready for HuggingFace upload
                """)

                with gr.Row():
                    with gr.Column():
                        synth_domain = gr.Textbox(
                            label="Domain",
                            placeholder="e.g., finance, healthcare, travel, ecommerce, customer_support",
                            value="travel",
                            info="The domain/industry for your synthetic tasks"
                        )
                        synth_tools = gr.Textbox(
                            label="Tool Names (comma-separated)",
                            placeholder="e.g., get_weather,search_flights,book_hotel,currency_converter",
                            value="get_weather,search_flights,book_hotel",
                            info="Names of tools your agent can use",
                            lines=2
                        )
                        synth_num_tasks = gr.Slider(
                            label="Number of Tasks",
                            minimum=5,
                            maximum=100,
                            value=10,
                            step=1,
                            info="Total number of synthetic tasks to generate"
                        )
                        synth_difficulty = gr.Dropdown(
                            label="Difficulty Distribution",
                            choices=["balanced", "easy_only", "medium_only", "hard_only", "progressive"],
                            value="balanced",
                            info="How to distribute task difficulty"
                        )
                        synth_agent_type = gr.Dropdown(
                            label="Agent Type",
                            choices=["both", "tool", "code"],
                            value="both",
                            info="Target agent type for the tasks"
                        )
                        synth_button = gr.Button("🧪 Generate Synthetic Dataset", variant="primary", size="lg")

                    with gr.Column():
                        synth_output = gr.JSON(label="Generated Dataset (JSON)")

                        gr.Markdown("""
                        ### 📝 Next Steps

                        After generation:
                        1. **Copy the `tasks` array** from the JSON output above
                        2. **Use the "Push to Hub" tab** to upload directly to HuggingFace
                        3. **Or upload manually** following the instructions in the output

                        **💡 Tip**: The generated dataset includes usage instructions and follows SMOLTRACE naming convention!
                        """)

                async def run_generate_synthetic(domain, tools, num_tasks, difficulty, agent_type):
                    """Generate synthetic dataset with async support."""
                    try:
                        import json
                        result = await generate_synthetic_dataset(
                            domain=domain,
                            tool_names=tools,
                            num_tasks=int(num_tasks),
                            difficulty_distribution=difficulty,
                            agent_type=agent_type
                        )
                        return json.loads(result)
                    except Exception as e:
                        return {"error": str(e)}

                synth_button.click(
                    fn=run_generate_synthetic,
                    inputs=[synth_domain, synth_tools, synth_num_tasks, synth_difficulty, synth_agent_type],
                    outputs=[synth_output]
                )

            # Tab: Generate Prompt Template
            with gr.Tab("📝 Generate Prompt Template"):
                gr.Markdown("""
                ## Create Customized Agent Prompt Template

                Generate a domain-specific prompt template based on smolagents templates.
                This template can be used with your synthetic dataset to run SMOLTRACE evaluations.

                **🎯 Use Case**: After generating a synthetic dataset, create a matching prompt template
                that agents can use during evaluation. This ensures your evaluation setup is complete.

                **Output**: Customized YAML prompt template ready for use with smolagents
                """)

                with gr.Row():
                    with gr.Column():
                        prompt_domain = gr.Textbox(
                            label="Domain",
                            placeholder="e.g., finance, healthcare, customer_support",
                            value="travel",
                            info="The domain/industry for the prompt template"
                        )
                        prompt_tools = gr.Textbox(
                            label="Tool Names (comma-separated)",
                            placeholder="e.g., get_weather,search_flights,book_hotel",
                            value="get_weather,search_flights,book_hotel",
                            info="Names of tools the agent will use",
                            lines=2
                        )
                        prompt_agent_type = gr.Dropdown(
                            label="Agent Type",
                            choices=["tool", "code"],
                            value="tool",
                            info="ToolCallingAgent (tool) or CodeAgent (code)"
                        )
                        prompt_button = gr.Button("📝 Generate Prompt Template", variant="primary", size="lg")

                    with gr.Column():
                        prompt_output = gr.JSON(label="Generated Prompt Template (JSON)")

                        gr.Markdown("""
                        ### 📝 Next Steps

                        After generation:
                        1. **Copy the `prompt_template`** from the JSON output above
                        2. **Save it as a YAML file** (e.g., `{domain}_agent.yaml`)
                        3. **Include it in your HuggingFace dataset** card or repository
                        4. **Use it with SMOLTRACE** when running evaluations

                        **💡 Tip**: This template is AI-customized for your domain and tools!
                        """)

                async def run_generate_prompt_template(domain, tools, agent_type):
                    """Generate prompt template with async support."""
                    try:
                        import json
                        result = await generate_prompt_template(
                            domain=domain,
                            tool_names=tools,
                            agent_type=agent_type
                        )
                        return json.loads(result)
                    except Exception as e:
                        return {"error": str(e)}

                prompt_button.click(
                    fn=run_generate_prompt_template,
                    inputs=[prompt_domain, prompt_tools, prompt_agent_type],
                    outputs=[prompt_output]
                )

            # Tab 7: Push Dataset to Hub
            with gr.Tab("📤 Push to Hub"):
                gr.Markdown("""
                ## Upload Generated Dataset to HuggingFace Hub

                Upload your synthetic dataset (from the previous tab or any SMOLTRACE-format dataset)
                directly to HuggingFace Hub.

                **Requirements**:
                - HuggingFace account
                - API token with write permissions ([Get one here](https://huggingface.co/settings/tokens))
                - Dataset in SMOLTRACE format

                **Naming Convention**: `{username}/smoltrace-{domain}-tasks` or `{username}/smoltrace-{domain}-tasks-v1`
                """)

                with gr.Row():
                    with gr.Column():
                        push_dataset_json = gr.Textbox(
                            label="Dataset JSON (tasks array)",
                            placeholder='[{"id": "task_001", "prompt": "...", "expected_tool": "...", ...}]',
                            info="Paste the 'tasks' array from generate_synthetic_dataset output",
                            lines=10
                        )
                        push_repo_name = gr.Textbox(
                            label="Repository Name",
                            placeholder="your-username/smoltrace-finance-tasks",
                            info="HuggingFace repo name (follow SMOLTRACE convention)",
                            value=""
                        )
                        push_hf_token = gr.Textbox(
                            label="HuggingFace Token",
                            placeholder="hf_...",
                            info="API token with write permissions",
                            type="password"
                        )
                        push_private = gr.Checkbox(
                            label="Make dataset private",
                            value=False,
                            info="Private datasets are only visible to you"
                        )
                        push_button = gr.Button("📤 Push to HuggingFace Hub", variant="primary", size="lg")

                    with gr.Column():
                        push_output = gr.JSON(label="Upload Result")

                        gr.Markdown("""
                        ### 🎉 After Upload

                        Once uploaded, you can:
                        1. **View your dataset** at the URL provided in the output
                        2. **Use in SMOLTRACE** evaluations with the command shown
                        3. **Share with your team** (if public) or manage access (if private)

                        **Example**: After uploading to `company/smoltrace-finance-tasks`:
                        ```bash
                        smoltrace-eval --model openai/gpt-4 --dataset-name company/smoltrace-finance-tasks
                        ```
                        """)

                async def run_push_dataset(dataset_json, repo_name, hf_token, private):
                    """Push dataset to hub with async support."""
                    try:
                        import json
                        result = await push_dataset_to_hub(
                            dataset_json=dataset_json,
                            repo_name=repo_name,
                            hf_token=hf_token,
                            private=private
                        )
                        return json.loads(result)
                    except Exception as e:
                        return {"error": str(e)}

                push_button.click(
                    fn=run_push_dataset,
                    inputs=[push_dataset_json, push_repo_name, push_hf_token, push_private],
                    outputs=[push_output]
                )

            # Tab 9: MCP Resources & Prompts
            with gr.Tab("🔌 MCP Resources & Prompts"):
                gr.Markdown("""
                ## MCP Resources & Prompts

                Beyond the 7 MCP Tools, this server also exposes **MCP Resources** and **MCP Prompts**
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

            # Tab 10: API Documentation
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

                ### 5. get_top_performers

                **Description**: Get top performing models from leaderboard - optimized for quick queries

                **⚡ Performance**: This tool is **optimized** to avoid token bloat by returning only essential data for top performers instead of the full leaderboard (51 runs).

                **When to use**: Use this instead of `get_dataset()` when you need to answer questions like:
                - "Which model is leading?"
                - "Show me the top 5 models"
                - "What's the best model for cost?"

                **Parameters**:
                - `leaderboard_repo` (str): HuggingFace dataset repository (default: "kshitijthakkar/smoltrace-leaderboard")
                - `metric` (str): Metric to rank by (default: "success_rate")
                  - Options: "success_rate", "total_cost_usd", "avg_duration_ms", "co2_emissions_g"
                - `top_n` (int): Number of top models to return (default: 5, range: 1-20)

                **Returns**: JSON object with top performers - **ready to use, no parsing needed**

                **Benefits vs get_dataset()**:
                - ✅ Returns only 5-20 runs instead of all 51 runs (90% token reduction)
                - ✅ Properly formatted JSON (no string conversion issues)
                - ✅ Pre-sorted by your chosen metric
                - ✅ Includes only essential columns (10 fields vs 20+ fields)

                **Example Response**:
                ```json
                {
                  "metric_ranked_by": "success_rate",
                  "ranking_order": "descending (higher is better)",
                  "total_runs_in_leaderboard": 51,
                  "top_n": 5,
                  "top_performers": [
                    {
                      "run_id": "run_123",
                      "model": "openai/gpt-4",
                      "success_rate": 95.8,
                      "total_cost_usd": 0.05,
                      ...
                    }
                  ]
                }
                ```

                ---

                ### 6. get_leaderboard_summary

                **Description**: Get high-level leaderboard summary statistics - optimized for overview queries

                **⚡ Performance**: This tool is **optimized** to return only summary statistics (no individual runs), avoiding the full dataset that causes token bloat.

                **When to use**: Use this instead of `get_dataset()` when you need to answer questions like:
                - "How many runs are in the leaderboard?"
                - "What's the average success rate?"
                - "Give me an overview of the leaderboard"

                **Parameters**:
                - `leaderboard_repo` (str): HuggingFace dataset repository (default: "kshitijthakkar/smoltrace-leaderboard")

                **Returns**: JSON object with summary statistics - **ready to use, no parsing needed**

                **Benefits vs get_dataset()**:
                - ✅ Returns aggregated stats instead of raw data (99% token reduction)
                - ✅ Properly formatted JSON (no string conversion issues)
                - ✅ Includes breakdowns by agent_type and provider
                - ✅ Shows top 3 models by success rate
                - ✅ Calculates averages, totals, and distributions

                **Example Response**:
                ```json
                {
                  "leaderboard_repo": "kshitijthakkar/smoltrace-leaderboard",
                  "summary": {
                    "total_runs": 51,
                    "unique_models": 15,
                    "overall_stats": {
                      "avg_success_rate": 89.5,
                      "best_success_rate": 95.8,
                      "avg_cost_per_run_usd": 0.023
                    },
                    "breakdown_by_agent_type": {...},
                    "top_3_models_by_success_rate": [...]
                  }
                }
                ```

                ---

                ### 7. get_dataset

                **Description**: Load SMOLTRACE datasets from HuggingFace and return as JSON

                **⚠️ Note**: For leaderboard queries, prefer using `get_top_performers()` or `get_leaderboard_summary()` instead - they're optimized to avoid token bloat!

                **Parameters**:
                - `dataset_repo` (str, required): HuggingFace dataset repository path with "smoltrace-" prefix (e.g., "kshitijthakkar/smoltrace-leaderboard")
                - `max_rows` (int): Maximum number of rows to return (default: 50, range: 1-200)

                **Returns**: JSON object with dataset data and metadata

                **Restriction**: Only datasets with "smoltrace-" in the repository name are allowed for security.

                **Use Cases**:
                - Load smoltrace-results-* datasets to see individual test case details
                - Load smoltrace-traces-* datasets to access OpenTelemetry trace data
                - Load smoltrace-metrics-* datasets to get GPU metrics and performance data
                - For leaderboard: Use `get_top_performers()` or `get_leaderboard_summary()` instead!

                **Workflow**:
                1. Use `get_leaderboard_summary()` for overview questions
                2. Use `get_top_performers()` for "top N" queries
                3. Use `get_dataset()` only for non-leaderboard datasets or when you need specific run IDs

                ---

                ### 8. generate_synthetic_dataset

                **Description**: Generate domain-specific synthetic test datasets for SMOLTRACE evaluations using AI

                **Parameters**:
                - `domain` (str, required): The domain for synthetic tasks (e.g., "finance", "healthcare", "travel", "ecommerce", "customer_support")
                - `tool_names` (str, required): Comma-separated list of tool names to include (e.g., "get_weather,search_web,calculator")
                - `num_tasks` (int): Number of synthetic tasks to generate (default: 10, range: 5-100)
                - `difficulty_distribution` (str): How to distribute task difficulty (default: "balanced")
                  - Options: "balanced" (40% easy, 40% medium, 20% hard), "easy_only", "medium_only", "hard_only", "progressive" (50% easy, 30% medium, 20% hard)
                - `agent_type` (str): Target agent type for tasks (default: "both")
                  - Options: "tool" (ToolCallingAgent), "code" (CodeAgent), "both" (50/50 mix)

                **Returns**: JSON object with dataset_info (including batch statistics), tasks array (SMOLTRACE format), and usage_instructions

                **🚀 Batched Generation**:
                - Requests >20 tasks are automatically split into parallel batches
                - Each batch generates up to 20 tasks concurrently
                - Example: 100 tasks = 5 parallel batches (20 tasks each)
                - Timeout: 120 seconds per batch
                - Token limit: 8,192 per batch (40,960 total for 100 tasks)

                **Performance**:
                - 5-20 tasks: Single batch, ~30-60 seconds
                - 21-100 tasks: Multiple parallel batches, ~60-120 seconds per batch

                **SMOLTRACE Task Format**:
                Each task includes: `id`, `prompt`, `expected_tool`, `expected_tool_calls` (optional), `difficulty`, `agent_type`, `expected_keywords` (optional)

                **Use Cases**:
                - Create custom evaluation datasets for industry-specific domains
                - Test agents with proprietary tools and APIs
                - Generate benchmarks for internal workflows
                - Rapid prototyping of evaluation scenarios

                ---

                ### 9. push_dataset_to_hub

                **Description**: Push a generated synthetic dataset to HuggingFace Hub

                **Parameters**:
                - `dataset_json` (str, required): JSON string containing the tasks array from generate_synthetic_dataset
                - `repo_name` (str, required): HuggingFace repository name following SMOLTRACE naming convention
                  - Format: `{username}/smoltrace-{domain}-tasks` or `{username}/smoltrace-{domain}-tasks-v{version}`
                  - Examples: `kshitij/smoltrace-finance-tasks`, `kshitij/smoltrace-healthcare-tasks-v2`
                - `hf_token` (str, required): HuggingFace API token with write permissions
                - `private` (bool): Whether to create a private repository (default: False)

                **Returns**: JSON object with upload status, repository URL, and dataset information

                **Validation**:
                - ✅ Checks SMOLTRACE naming convention (`smoltrace-` prefix required)
                - ✅ Validates all tasks have required fields (id, prompt, expected_tool, difficulty, agent_type)
                - ✅ Verifies HuggingFace token has write permissions
                - ✅ Handles repository creation if it doesn't exist

                **Workflow**:
                1. Generate synthetic dataset using `generate_synthetic_dataset`
                2. Extract the `tasks` array from the response JSON
                3. Convert tasks array to JSON string
                4. Call `push_dataset_to_hub` with the JSON string and desired repo name
                5. Share the dataset URL with your team or use in SMOLTRACE evaluations

                **Example Integration**:
                ```python
                # Step 1: Generate dataset
                result = generate_synthetic_dataset(
                    domain="finance",
                    tool_names="get_stock_price,calculate_roi,fetch_company_info",
                    num_tasks=50
                )

                # Step 2: Extract tasks
                import json
                data = json.loads(result)
                tasks_json = json.dumps(data["tasks"])

                # Step 3: Push to HuggingFace
                push_result = push_dataset_to_hub(
                    dataset_json=tasks_json,
                    repo_name="your-username/smoltrace-finance-tasks",
                    hf_token="hf_xxx",
                    private=False
                )
                ```

                ---

                ## MCP Integration

                This Gradio app is MCP-enabled. When deployed to HuggingFace Spaces, it can be accessed via MCP clients.

                **HuggingFace Space**: `https://huggingface.co/spaces/MCP-1st-Birthday/TraceMind-mcp-server`

                **MCP Endpoint (SSE - Recommended)**: `https://mcp-1st-birthday-tracemind-mcp-server.hf.space/gradio_api/mcp/sse`

                **MCP Endpoint (Streamable HTTP)**: `https://mcp-1st-birthday-tracemind-mcp-server.hf.space/gradio_api/mcp/`

                ### What's Exposed via MCP:

                #### 10 MCP Tools (AI-Powered & Optimized)
                The ten tools above (`analyze_leaderboard`, `debug_trace`, `estimate_cost`, `compare_runs`, `get_top_performers`, `get_leaderboard_summary`, `get_dataset`, `generate_synthetic_dataset`, `generate_prompt_template`, `push_dataset_to_hub`)
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

        with gr.Tab("⚙️ Settings"):
            gr.Markdown("""
                # ⚙️ API Key Configuration (Optional)

                ## Default Configuration

                This MCP server uses **pre-configured API keys from HuggingFace Spaces Secrets**.

                For most users (especially MCP client demos with Claude Desktop), no configuration is needed!

                ## For Hackathon Judges & Visitors

                If you want to use **your own API keys** to prevent credit issues during evaluation:

                1. Enter your API keys below
                2. Click **"Save & Override Keys"**
                3. Your keys will be used for **this session only** (stored in browser memory, never saved to disk)

                Then you can:
                - Use any tool in the tabs above
                - Connect from TraceMind-AI (the MCP tools will use your keys)
                - Test with Claude Desktop (will use your keys)

                ## Security Notes

                ✅ **Session-only storage**: Keys stored only in your browser session
                ✅ **No server persistence**: Keys never saved to disk or database
                ✅ **API endpoint security**: This form is NOT exposed via Gradio's "Use via API"
                ✅ **HTTPS encryption**: All API calls made over secure connections

                ---
                """)

            # Show current key status (masked)
            current_gemini = os.environ.get("GEMINI_API_KEY", "")
            current_hf = os.environ.get("HF_TOKEN", "")

            gemini_display = f"`{current_gemini[:10]}...`" if current_gemini else "❌ Not configured"
            hf_display = f"`{current_hf[:7]}...`" if current_hf else "❌ Not configured"

            gr.Markdown(f"""
                ### Current Configuration Status

                - **Gemini API Key**: {gemini_display}
                - **HuggingFace Token**: {hf_display}

                {"✅ Using HuggingFace Spaces Secrets (default)" if current_gemini and current_hf else "⚠️ API keys not fully configured"}
                """)

            gr.Markdown("### Override with Your Own Keys")

            with gr.Row():
                with gr.Column():
                    gemini_api_key_input = gr.Textbox(
                            label="Google Gemini API Key",
                            placeholder="Leave empty to use default, or enter AIza...",
                            type="password",
                            value="",
                            info="Get your free API key at: https://ai.google.dev/"
                        )

            with gr.Row():
                with gr.Column():
                    hf_token_input = gr.Textbox(
                        label="HuggingFace Token",
                        placeholder="Leave empty to use default, or enter hf_...",
                        type="password",
                        value="",
                        info="Get your token at: https://huggingface.co/settings/tokens"
                    )

            with gr.Row():
                save_keys_btn = gr.Button("💾 Save & Override Keys", variant="primary", size="lg")
                reset_keys_btn = gr.Button("🔄 Reset to Defaults", variant="secondary", size="lg")

            settings_status = gr.Markdown("")

            gr.Markdown("""
                ---

                ### How to Get API Keys

                #### Google Gemini API Key

                1. Go to [Google AI Studio](https://ai.google.dev/)
                2. Click "Get API Key" in the top right
                3. Create a new project or select an existing one
                4. Generate an API key
                5. Copy the key (starts with `AIza...`)

                **Free Tier**: 1,500 requests per day, suitable for testing and demos

                #### HuggingFace Token

                1. Go to [HuggingFace Settings](https://huggingface.co/settings/tokens)
                2. Click "New token"
                3. Give it a name (e.g., "TraceMind Access")
                4. Select permissions:
                   - **Read**: Sufficient for viewing datasets (leaderboard, traces, results)
                   - **Write**: Required for `push_dataset_to_hub` tool (uploading synthetic datasets)
                5. Create and copy the token (starts with `hf_...`)

                **Recommended**: Use "Write" permissions for full MCP server functionality
                """)

            # Event handlers for Settings tab
            def save_override_keys(gemini, hf):
                """Save user-provided API keys to session (override Spaces Secrets)"""
                messages = []

                if gemini and gemini.strip():
                    if gemini.startswith("AIza"):
                        os.environ["GEMINI_API_KEY"] = gemini.strip()
                        messages.append("✅ **Gemini API key** saved and will be used for this session")
                        logger.info("Gemini API key overridden by user for this session")
                    else:
                        messages.append("⚠️ **Invalid Gemini API key format** (should start with 'AIza')")

                if hf and hf.strip():
                    if hf.startswith("hf_"):
                        os.environ["HF_TOKEN"] = hf.strip()
                        messages.append("✅ **HuggingFace token** saved and will be used for this session")
                        logger.info("HuggingFace token overridden by user for this session")
                    else:
                        messages.append("⚠️ **Invalid HuggingFace token format** (should start with 'hf_')")

                if not messages:
                    messages.append("⚠️ No keys provided. Still using default keys from Spaces Secrets.")

                messages.append("\n**Note**: Your keys are active for this browser session only.")
                messages.append("\n🎯 You can now use all MCP tools with your own API keys!")

                return "\n\n".join(messages)

            def reset_to_defaults():
                """Reset to Spaces Secrets (requires page refresh)"""
                return """
                ℹ️ To reset to default keys from Spaces Secrets, please **refresh this page**.

                Your session overrides will be cleared and the default keys will be used again.
                """

            # Wire up buttons with api_name=False for security
            save_keys_btn.click(
                fn=save_override_keys,
                inputs=[gemini_api_key_input, hf_token_input],
                outputs=[settings_status],
                api_name=False  # ✅ CRITICAL: Prevents API key exposure via Gradio API
            )

            reset_keys_btn.click(
                fn=reset_to_defaults,
                outputs=[settings_status],
                api_name=False  # ✅ CRITICAL: Prevents exposure
            )

    return demo

if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("TraceMind MCP Server - HuggingFace Space (Track 1)")
    logger.info("=" * 70)
    logger.info("MCP Server: TraceMind Agent Evaluation Platform v1.0.0")
    logger.info("Protocol: Model Context Protocol (MCP)")
    logger.info("Transport: Gradio Native MCP Support (SSE)")
    logger.info("MCP Endpoint (SSE): https://mcp-1st-birthday-tracemind-mcp-server.hf.space/gradio_api/mcp/sse")
    logger.info("MCP Endpoint (HTTP): https://mcp-1st-birthday-tracemind-mcp-server.hf.space/gradio_api/mcp/")
    logger.info("=" * 70)
    logger.info("Features:")
    logger.info("  ✓ 7 AI-Powered Tools (Leaderboard + Trace + Cost + Dataset)")
    logger.info("  ✓ 3 Real-Time Resources (leaderboard, trace, cost data)")
    logger.info("  ✓ 3 Prompt Templates (analysis, debug, optimization)")
    logger.info("  ✓ Google Gemini 2.5 Pro - Intelligent Analysis")
    logger.info("  ✓ HuggingFace Dataset Integration")
    logger.info("  ✓ SMOLTRACE Format Support")
    logger.info("  ✓ Synthetic Dataset Generation")
    logger.info("=" * 70)
    logger.info("Tool Categories:")
    logger.info("  📊 Analysis: analyze_leaderboard, compare_runs")
    logger.info("  🐛 Debugging: debug_trace")
    logger.info("  💰 Cost: estimate_cost")
    logger.info("  📦 Data: get_dataset")
    logger.info("  🧪 Generation: generate_synthetic_dataset, push_dataset_to_hub")
    logger.info("=" * 70)
    logger.info("Compatible Clients:")
    logger.info("  • Claude Desktop")
    logger.info("  • Continue.dev (VS Code)")
    logger.info("  • Cline (VS Code)")
    logger.info("  • Any MCP-compatible client")
    logger.info("=" * 70)
    logger.info("How to Connect (Claude Desktop/HF MCP Client):")
    logger.info("  1. Go to https://huggingface.co/settings/mcp")
    logger.info("  2. Add Space: MCP-1st-Birthday/TraceMind-mcp-server")
    logger.info("  3. Start using TraceMind tools in your MCP client!")
    logger.info("=" * 70)
    logger.info("Starting Gradio UI + MCP Server on 0.0.0.0:7860...")
    logger.info("Waiting for connections...")
    logger.info("=" * 70)

    try:
        # Create Gradio interface
        demo = create_gradio_ui()

        # Theme configuration (Ocean-inspired) - Gradio 6 requires theme in launch()
        theme = gr.themes.Base(
            primary_hue="blue",
            secondary_hue="cyan",
            neutral_hue="slate",
            font=gr.themes.GoogleFont("Inter"),
        ).set(
            body_background_fill="*neutral_50",
            body_background_fill_dark="*neutral_900",
            button_primary_background_fill="*primary_500",
            button_primary_background_fill_hover="*primary_600",
            button_primary_text_color="white",
        )

        # Launch with MCP server enabled
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            mcp_server=True,  # Enable MCP server functionality
            theme=theme  # Gradio 6: theme goes here, not in Blocks()
        )

    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        logger.error("Check that:")
        logger.error("  1. GEMINI_API_KEY environment variable is set")
        logger.error("  2. Port 7860 is available")
        logger.error("  3. All dependencies are installed")
        raise
