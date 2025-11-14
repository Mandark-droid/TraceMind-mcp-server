"""
MCP Tool Implementations for TraceMind

Implements:
- 5 MCP Tools: analyze_leaderboard, debug_trace, estimate_cost, compare_runs, get_dataset
- 3 MCP Resources: leaderboard data, trace data, cost data
- 3 MCP Prompts: analysis prompts, debug prompts, optimization prompts

With Gradio's native MCP support (mcp_server=True), these are automatically
exposed based on decorators (@gr.mcp.tool, @gr.mcp.resource, @gr.mcp.prompt),
docstrings, and type hints.
"""

import os
import json
from typing import Optional
from datasets import load_dataset
import pandas as pd
from datetime import datetime, timedelta
import gradio as gr

from gemini_client import GeminiClient


async def analyze_leaderboard(
    gemini_client: GeminiClient,
    leaderboard_repo: str = "kshitijthakkar/smoltrace-leaderboard",
    metric_focus: str = "overall",
    time_range: str = "last_week",
    top_n: int = 5,
    hf_token: Optional[str] = None
) -> str:
    """
    Analyze evaluation leaderboard and generate AI-powered insights.

    This tool loads agent evaluation data from HuggingFace datasets and uses
    Google Gemini 2.5 Pro to provide intelligent analysis of top performers,
    trends, cost/performance trade-offs, and actionable recommendations.

    Args:
        gemini_client (GeminiClient): Initialized Gemini client for AI analysis
        leaderboard_repo (str): HuggingFace dataset repository containing leaderboard data. Default: "kshitijthakkar/smoltrace-leaderboard"
        metric_focus (str): Primary metric to focus analysis on. Options: "overall", "accuracy", "cost", "latency", "co2". Default: "overall"
        time_range (str): Time range for analysis. Options: "last_week", "last_month", "all_time". Default: "last_week"
        top_n (int): Number of top models to highlight in analysis. Must be between 3 and 10. Default: 5
        hf_token (Optional[str]): HuggingFace token for dataset access. If None, uses HF_TOKEN environment variable.

    Returns:
        str: Markdown-formatted analysis with top performers, insights, trade-offs, and recommendations
    """
    try:
        # Load leaderboard data from HuggingFace
        print(f"Loading leaderboard from {leaderboard_repo}...")

        # Use user-provided token or fall back to environment variable
        token = hf_token if hf_token else os.getenv("HF_TOKEN")
        ds = load_dataset(leaderboard_repo, split="train", token=token)
        df = pd.DataFrame(ds)

        # Filter by time range
        if time_range != "all_time":
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            now = datetime.now()

            if time_range == "last_week":
                cutoff = now - timedelta(days=7)
            elif time_range == "last_month":
                cutoff = now - timedelta(days=30)

            df = df[df['timestamp'] >= cutoff]

        # Sort by metric
        metric_column_map = {
            "overall": "success_rate",
            "accuracy": "success_rate",
            "cost": "total_cost_usd",
            "latency": "avg_duration_ms",
            "co2": "co2_emissions_g"
        }

        sort_column = metric_column_map.get(metric_focus, "success_rate")
        ascending = metric_focus in ["cost", "latency", "co2"]  # Lower is better for these

        df_sorted = df.sort_values(sort_column, ascending=ascending)

        # Get top N
        top_models = df_sorted.head(top_n)

        # Prepare data summary for Gemini
        analysis_data = {
            "total_evaluations": len(df),
            "time_range": time_range,
            "metric_focus": metric_focus,
            "top_models": top_models[[
                "model", "agent_type", "provider",
                "success_rate", "total_cost_usd", "avg_duration_ms",
                "co2_emissions_g", "submitted_by"
            ]].to_dict('records'),
            "summary_stats": {
                "avg_success_rate": float(df['success_rate'].mean()),
                "avg_cost": float(df['total_cost_usd'].mean()),
                "avg_duration_ms": float(df['avg_duration_ms'].mean()),
                "total_co2_g": float(df['co2_emissions_g'].sum()),
                "models_tested": df['model'].nunique(),
                "unique_submitters": df['submitted_by'].nunique()
            }
        }

        # Get AI analysis from Gemini
        result = await gemini_client.analyze_with_context(
            data=analysis_data,
            analysis_type="leaderboard",
            specific_question=f"Focus on {metric_focus} performance. What are the key insights?"
        )

        return result

    except Exception as e:
        return f"❌ **Error analyzing leaderboard**: {str(e)}\n\nPlease check:\n- Repository name is correct\n- You have access to the dataset\n- HF_TOKEN is set correctly"


async def debug_trace(
    gemini_client: GeminiClient,
    trace_id: str,
    traces_repo: str,
    question: str = "Analyze this trace and explain what happened",
    hf_token: Optional[str] = None
) -> str:
    """
    Debug a specific agent execution trace using OpenTelemetry data.

    This tool analyzes OpenTelemetry trace data from agent executions and uses
    Google Gemini 2.5 Pro to answer specific questions about the execution flow,
    identify bottlenecks, and explain agent behavior.

    Args:
        gemini_client (GeminiClient): Initialized Gemini client for AI analysis
        trace_id (str): Unique identifier for the trace to analyze (e.g., "trace_abc123")
        traces_repo (str): HuggingFace dataset repository containing trace data (e.g., "username/agent-traces-model-timestamp")
        question (str): Specific question about the trace. Default: "Analyze this trace and explain what happened"
        hf_token (Optional[str]): HuggingFace token for dataset access. If None, uses HF_TOKEN environment variable.

    Returns:
        str: Markdown-formatted debug analysis with step-by-step breakdown, timing information, and answer to the question
    """
    try:
        # Load traces dataset
        print(f"Loading traces from {traces_repo}...")

        # Use user-provided token or fall back to environment variable
        token = hf_token if hf_token else os.getenv("HF_TOKEN")
        ds = load_dataset(traces_repo, split="train", token=token)
        df = pd.DataFrame(ds)

        # Find the specific trace
        trace_data = df[df['trace_id'] == trace_id]

        if len(trace_data) == 0:
            return f"❌ **Trace not found**: No trace with ID `{trace_id}` in repository `{traces_repo}`"

        trace_row = trace_data.iloc[0]

        # Parse spans (OpenTelemetry format)
        spans = trace_row['spans']
        if isinstance(spans, str):
            import json
            spans = json.loads(spans)

        # Helper function to handle different OTEL timestamp field formats
        def get_timestamp(span, field):
            """Get timestamp handling multiple OTEL formats"""
            # Try different field name variations
            for key in [field, f"{field}UnixNano", f"{field}_unix_nano", "timeUnixNano"]:
                if key in span:
                    return span[key]
            return 0

        # Build trace analysis data
        start_time = get_timestamp(spans[0], 'startTime')
        end_time = get_timestamp(spans[-1], 'endTime')

        trace_analysis = {
            "trace_id": trace_id,
            "run_id": trace_row.get('run_id', 'unknown'),
            "total_duration_ms": (end_time - start_time) / 1_000_000 if end_time > start_time else 0,
            "num_spans": len(spans),
            "spans": []
        }

        # Process each span
        for span in spans:
            span_start = get_timestamp(span, 'startTime')
            span_end = get_timestamp(span, 'endTime')

            span_info = {
                "name": span.get('name', 'Unknown'),
                "kind": span.get('kind', 'INTERNAL'),
                "duration_ms": (span_end - span_start) / 1_000_000 if span_end > span_start else 0,
                "attributes": span.get('attributes', {}),
                "status": span.get('status', {}).get('code', 'UNKNOWN')
            }
            trace_analysis["spans"].append(span_info)

        # Get AI analysis from Gemini
        result = await gemini_client.analyze_with_context(
            data=trace_analysis,
            analysis_type="trace",
            specific_question=question
        )

        return result

    except Exception as e:
        return f"❌ **Error debugging trace**: {str(e)}\n\nPlease check:\n- Trace ID is correct\n- Repository name is correct\n- You have access to the dataset"


async def estimate_cost(
    gemini_client: GeminiClient,
    model: str,
    agent_type: str,
    num_tests: int = 100,
    hardware: str = "auto"
) -> str:
    """
    Estimate the cost, duration, and CO2 emissions of running agent evaluations.

    This tool predicts costs before running evaluations by calculating LLM API costs,
    HuggingFace Jobs compute costs, and CO2 emissions. Uses Google Gemini 2.5 Pro
    to provide cost breakdown and optimization recommendations.

    Args:
        gemini_client (GeminiClient): Initialized Gemini client for AI analysis
        model (str): Model identifier in litellm format (e.g., "openai/gpt-4", "meta-llama/Llama-3.1-8B")
        agent_type (str): Type of agent capabilities to test. Options: "tool", "code", "both"
        num_tests (int): Number of test cases to run. Must be between 10 and 1000. Default: 100
        hardware (str): Hardware type for HuggingFace Jobs. Options: "auto", "cpu", "gpu_a10", "gpu_h200". Default: "auto"

    Returns:
        str: Markdown-formatted cost estimate with breakdown of LLM costs, HF Jobs costs, duration, CO2 emissions, and optimization tips
    """
    try:
        # Determine if API or local model
        is_api_model = any(provider in model.lower() for provider in ["openai", "anthropic", "google", "cohere"])

        # Auto-select hardware
        if hardware == "auto":
            hardware = "cpu" if is_api_model else "gpu_a10"

        # Cost data (simplified estimates)
        llm_costs = {
            "openai/gpt-4": {"input": 0.03, "output": 0.06},  # per 1K tokens
            "openai/gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
            "anthropic/claude-3-opus": {"input": 0.015, "output": 0.075},
            "anthropic/claude-3-sonnet": {"input": 0.003, "output": 0.015},
            "meta-llama/Llama-3.1-8B": {"input": 0, "output": 0},  # Local model
            "default": {"input": 0.001, "output": 0.002}
        }

        hf_jobs_costs = {
            "cpu": 0.60,  # per hour
            "gpu_a10": 1.10,  # per hour
            "gpu_h200": 4.50  # per hour
        }

        # Get model costs
        model_cost = llm_costs.get(model, llm_costs["default"])

        # Estimate token usage per test
        # Tool agent: ~200 tokens input, ~150 output
        # Code agent: ~300 tokens input, ~400 output
        # Both: ~400 tokens input, ~500 output
        token_estimates = {
            "tool": {"input": 200, "output": 150},
            "code": {"input": 300, "output": 400},
            "both": {"input": 400, "output": 500}
        }

        tokens_per_test = token_estimates[agent_type]

        # Calculate LLM costs
        llm_cost_per_test = (
            (tokens_per_test["input"] / 1000) * model_cost["input"] +
            (tokens_per_test["output"] / 1000) * model_cost["output"]
        )
        total_llm_cost = llm_cost_per_test * num_tests

        # Estimate duration (seconds per test)
        if is_api_model:
            duration_per_test = 3.0  # API models are fast
        else:
            duration_per_test = 8.0  # Local models slower but depends on GPU

        total_duration_hours = (duration_per_test * num_tests) / 3600

        # Calculate HF Jobs costs
        jobs_hourly_rate = hf_jobs_costs.get(hardware, hf_jobs_costs["cpu"])
        total_jobs_cost = total_duration_hours * jobs_hourly_rate

        # Estimate CO2 (rough estimates)
        co2_per_hour = {
            "cpu": 0.05,  # kg CO2
            "gpu_a10": 0.15,
            "gpu_h200": 0.30
        }

        total_co2_kg = total_duration_hours * co2_per_hour.get(hardware, 0.05)

        # Prepare estimate data
        estimate_data = {
            "model": model,
            "agent_type": agent_type,
            "num_tests": num_tests,
            "hardware": hardware,
            "is_api_model": is_api_model,
            "estimates": {
                "llm_cost_usd": round(total_llm_cost, 4),
                "llm_cost_per_test": round(llm_cost_per_test, 4),
                "jobs_cost_usd": round(total_jobs_cost, 4),
                "total_cost_usd": round(total_llm_cost + total_jobs_cost, 4),
                "duration_hours": round(total_duration_hours, 2),
                "duration_per_test_seconds": round(duration_per_test, 2),
                "co2_emissions_kg": round(total_co2_kg, 3),
                "tokens_per_test": tokens_per_test
            }
        }

        # Get AI analysis from Gemini
        result = await gemini_client.analyze_with_context(
            data=estimate_data,
            analysis_type="cost_estimate",
            specific_question="Provide cost breakdown and optimization recommendations"
        )

        return result

    except Exception as e:
        return f"❌ **Error estimating cost**: {str(e)}"


async def compare_runs(
    gemini_client: GeminiClient,
    run_id_1: str,
    run_id_2: str,
    leaderboard_repo: str = "kshitijthakkar/smoltrace-leaderboard",
    comparison_focus: str = "comprehensive",
    hf_token: Optional[str] = None
) -> str:
    """
    Compare two evaluation runs and generate AI-powered comparative analysis.

    This tool fetches data for two evaluation runs from the leaderboard and uses
    Google Gemini 2.5 Pro to provide intelligent comparison across multiple dimensions:
    success rate, cost efficiency, speed, environmental impact, and use case recommendations.

    Args:
        gemini_client (GeminiClient): Initialized Gemini client for AI analysis
        run_id_1 (str): First run ID to compare
        run_id_2 (str): Second run ID to compare
        leaderboard_repo (str): HuggingFace dataset repository containing leaderboard data. Default: "kshitijthakkar/smoltrace-leaderboard"
        comparison_focus (str): Focus area for comparison. Options: "comprehensive", "cost", "performance", "eco_friendly". Default: "comprehensive"
        hf_token (Optional[str]): HuggingFace token for dataset access. If None, uses HF_TOKEN environment variable.

    Returns:
        str: Markdown-formatted comparative analysis with winner for each category, trade-offs, and use case recommendations
    """
    try:
        # Load leaderboard data
        # Use user-provided token or fall back to environment variable
        token = hf_token if hf_token else os.getenv("HF_TOKEN")
        dataset = load_dataset(leaderboard_repo, split="train", token=token)
        df = pd.DataFrame(dataset)

        # Find the two runs
        run1 = df[df['run_id'] == run_id_1]
        run2 = df[df['run_id'] == run_id_2]

        if run1.empty:
            return f"❌ **Error**: Run ID '{run_id_1}' not found in leaderboard"
        if run2.empty:
            return f"❌ **Error**: Run ID '{run_id_2}' not found in leaderboard"

        run1_data = run1.iloc[0].to_dict()
        run2_data = run2.iloc[0].to_dict()

        # Build comparison context for Gemini
        comparison_data = {
            "run_1": {
                "run_id": run1_data.get('run_id'),
                "model": run1_data.get('model'),
                "agent_type": run1_data.get('agent_type'),
                "success_rate": run1_data.get('success_rate'),
                "total_tests": run1_data.get('total_tests'),
                "successful_tests": run1_data.get('successful_tests'),
                "avg_duration_ms": run1_data.get('avg_duration_ms'),
                "total_cost_usd": run1_data.get('total_cost_usd'),
                "avg_cost_per_test_usd": run1_data.get('avg_cost_per_test_usd'),
                "co2_emissions_g": run1_data.get('co2_emissions_g'),
                "gpu_utilization_avg": run1_data.get('gpu_utilization_avg'),
                "total_tokens": run1_data.get('total_tokens'),
                "provider": run1_data.get('provider'),
                "job_type": run1_data.get('job_type'),
                "timestamp": run1_data.get('timestamp')
            },
            "run_2": {
                "run_id": run2_data.get('run_id'),
                "model": run2_data.get('model'),
                "agent_type": run2_data.get('agent_type'),
                "success_rate": run2_data.get('success_rate'),
                "total_tests": run2_data.get('total_tests'),
                "successful_tests": run2_data.get('successful_tests'),
                "avg_duration_ms": run2_data.get('avg_duration_ms'),
                "total_cost_usd": run2_data.get('total_cost_usd'),
                "avg_cost_per_test_usd": run2_data.get('avg_cost_per_test_usd'),
                "co2_emissions_g": run2_data.get('co2_emissions_g'),
                "gpu_utilization_avg": run2_data.get('gpu_utilization_avg'),
                "total_tokens": run2_data.get('total_tokens'),
                "provider": run2_data.get('provider'),
                "job_type": run2_data.get('job_type'),
                "timestamp": run2_data.get('timestamp')
            },
            "comparison_focus": comparison_focus
        }

        # Create comparison prompt based on focus
        if comparison_focus == "comprehensive":
            prompt = f"""
You are analyzing a comparison between two agent evaluation runs. Provide a comprehensive analysis covering all aspects.

**Run 1 ({comparison_data['run_1']['model']}):**
{json.dumps(comparison_data['run_1'], indent=2)}

**Run 2 ({comparison_data['run_2']['model']}):**
{json.dumps(comparison_data['run_2'], indent=2)}

Please provide a detailed comparison in the following format:

## 📊 Head-to-Head Comparison

### 🎯 Accuracy Winner
[Which run has better success rate and by how much? Explain significance]

### ⚡ Speed Winner
[Which run is faster and by how much? Include average duration comparison]

### 💰 Cost Winner
[Which run is more cost-effective? Compare total cost AND cost per test]

### 🌱 Eco-Friendly Winner
[Which run has lower CO2 emissions? Calculate the difference]

### 🔧 GPU Efficiency Winner (if applicable)
[For GPU jobs, which has better utilization? Explain implications]

## 📈 Performance Summary

### Run 1 Strengths
- [List 3-4 key strengths]

### Run 2 Strengths
- [List 3-4 key strengths]

## 💡 Use Case Recommendations

### When to Choose Run 1 ({comparison_data['run_1']['model']})
[Specific scenarios where Run 1 is the better choice]

### When to Choose Run 2 ({comparison_data['run_2']['model']})
[Specific scenarios where Run 2 is the better choice]

## ⚖️ Overall Recommendation
[Based on the analysis, provide a balanced recommendation considering different priorities]

Be specific with numbers and percentages. Make the comparison actionable and insightful.
"""
        elif comparison_focus == "cost":
            prompt = f"""
Compare these two evaluation runs focusing specifically on cost efficiency:

**Run 1:** {json.dumps(comparison_data['run_1'], indent=2)}
**Run 2:** {json.dumps(comparison_data['run_2'], indent=2)}

Provide detailed cost analysis:
1. Which run has lower total cost and by what percentage?
2. Cost per test comparison - which is more efficient?
3. Calculate cost per successful test (accounting for failures)
4. Token usage efficiency - cost per 1000 tokens
5. ROI analysis - is higher cost justified by better accuracy?
6. Scaling implications - at 1000 tests, what would each cost?

Provide actionable cost optimization recommendations.
"""
        elif comparison_focus == "performance":
            prompt = f"""
Compare these two evaluation runs focusing on performance (speed + accuracy):

**Run 1:** {json.dumps(comparison_data['run_1'], indent=2)}
**Run 2:** {json.dumps(comparison_data['run_2'], indent=2)}

Analyze:
1. Success rate difference - statistical significance?
2. Speed comparison - average duration per test
3. Which delivers faster results without sacrificing accuracy?
4. Throughput analysis - tests per minute
5. Quality vs Speed trade-off assessment
6. GPU utilization efficiency (if applicable)

Recommend which run offers best performance for production workloads.
"""
        elif comparison_focus == "eco_friendly":
            prompt = f"""
Compare these two evaluation runs focusing on environmental impact:

**Run 1:** {json.dumps(comparison_data['run_1'], indent=2)}
**Run 2:** {json.dumps(comparison_data['run_2'], indent=2)}

Analyze:
1. CO2 emissions comparison - which is greener?
2. Emissions per test and per successful test
3. GPU vs API model environmental trade-offs
4. Energy efficiency based on duration and GPU utilization
5. Emissions reduction if scaled to 10,000 tests
6. Carbon offset cost comparison

Provide eco-conscious recommendations for sustainable AI deployment.
"""

        # Get AI analysis from Gemini
        analysis = await gemini_client.analyze_with_context(
            comparison_data,
            analysis_type="comparison",
            specific_question=prompt
        )

        return analysis

    except Exception as e:
        return f"❌ **Error comparing runs**: {str(e)}"


async def get_dataset(
    dataset_repo: str,
    max_rows: int = 50,
    hf_token: Optional[str] = None
) -> str:
    """
    Load SMOLTRACE datasets from HuggingFace and return as JSON.

    This tool loads datasets with the "smoltrace-" prefix and returns the raw data
    as JSON. Use this to access:
    - Leaderboard data (kshitijthakkar/smoltrace-leaderboard)
    - Results datasets (e.g., username/smoltrace-results-*)
    - Traces datasets (e.g., username/smoltrace-traces-*)
    - Metrics datasets (e.g., username/smoltrace-metrics-*)
    - Any other smoltrace-prefixed evaluation dataset

    If you don't know which dataset to load, first load the leaderboard to see
    the dataset references in the results_dataset, traces_dataset, metrics_dataset,
    and dataset_used fields.

    Args:
        dataset_repo (str): HuggingFace dataset repository path with "smoltrace-" prefix (e.g., "kshitijthakkar/smoltrace-leaderboard")
        max_rows (int): Maximum number of rows to return. Default: 50. Range: 1-200
        hf_token (Optional[str]): HuggingFace token for dataset access. If None, uses HF_TOKEN environment variable.

    Returns:
        str: JSON object with dataset data and metadata
    """
    try:
        # Validate dataset has smoltrace- prefix
        if "smoltrace-" not in dataset_repo:
            return json.dumps({
                "dataset_repo": dataset_repo,
                "error": "Only datasets with 'smoltrace-' prefix are allowed. Please use smoltrace-leaderboard or other smoltrace-* datasets.",
                "data": []
            }, indent=2)

        # Load dataset from HuggingFace
        # Use user-provided token or fall back to environment variable
        token = hf_token if hf_token else os.getenv("HF_TOKEN")
        dataset = load_dataset(dataset_repo, split="train", token=token)
        df = pd.DataFrame(dataset)

        if df.empty:
            return json.dumps({
                "dataset_repo": dataset_repo,
                "error": "Dataset is empty",
                "total_rows": 0,
                "data": []
            }, indent=2)

        # Get total row count before limiting
        total_rows = len(df)

        # Limit rows to avoid overwhelming the context
        max_rows = max(1, min(200, max_rows))

        # Sort by timestamp if available (newest first)
        if "timestamp" in df.columns:
            df = df.sort_values("timestamp", ascending=False)

        df_limited = df.head(max_rows)

        # Convert to list of dictionaries
        data = df_limited.to_dict(orient="records")

        # Build response with metadata
        result = {
            "dataset_repo": dataset_repo,
            "total_rows": total_rows,
            "rows_returned": len(data),
            "columns": list(df.columns),
            "data": data
        }

        return json.dumps(result, indent=2, default=str)

    except Exception as e:
        return json.dumps({
            "dataset_repo": dataset_repo,
            "error": f"Failed to load dataset: {str(e)}",
            "data": []
        }, indent=2)


# ============================================================================
# MCP RESOURCES - Expose data for retrieval by MCP clients
# ============================================================================

@gr.mcp.resource("leaderboard://{repo}")
def get_leaderboard_data(repo: str = "kshitijthakkar/smoltrace-leaderboard", hf_token: Optional[str] = None) -> str:
    """
    Get raw leaderboard data from HuggingFace dataset.

    This resource provides direct access to leaderboard data in JSON format,
    allowing MCP clients to retrieve and process evaluation results.

    Args:
        repo (str): HuggingFace dataset repository name. Default: "kshitijthakkar/smoltrace-leaderboard"
        hf_token (Optional[str]): HuggingFace token for dataset access. If None, uses HF_TOKEN environment variable.

    Returns:
        str: JSON string containing leaderboard data with all evaluation runs
    """
    try:
        # Use user-provided token or fall back to environment variable
        token = hf_token if hf_token else os.getenv("HF_TOKEN")
        ds = load_dataset(repo, split="train", token=token)
        df = pd.DataFrame(ds)

        # Convert to JSON with proper formatting
        data = df.to_dict('records')
        return json.dumps({
            "total_runs": len(data),
            "repository": repo,
            "data": data
        }, indent=2)

    except Exception as e:
        return json.dumps({
            "error": str(e),
            "repository": repo
        })


@gr.mcp.resource("trace://{trace_id}/{repo}")
def get_trace_data(trace_id: str, repo: str, hf_token: Optional[str] = None) -> str:
    """
    Get raw trace data for a specific trace ID from HuggingFace dataset.

    This resource provides direct access to OpenTelemetry trace data,
    allowing MCP clients to retrieve detailed execution information.

    Args:
        trace_id (str): Unique identifier for the trace (e.g., "trace_abc123")
        repo (str): HuggingFace dataset repository containing traces (e.g., "username/agent-traces-model")
        hf_token (Optional[str]): HuggingFace token for dataset access. If None, uses HF_TOKEN environment variable.

    Returns:
        str: JSON string containing trace data with all spans and attributes
    """
    try:
        # Use user-provided token or fall back to environment variable
        token = hf_token if hf_token else os.getenv("HF_TOKEN")
        ds = load_dataset(repo, split="train", token=token)
        df = pd.DataFrame(ds)

        # Find specific trace
        trace_data = df[df['trace_id'] == trace_id]

        if len(trace_data) == 0:
            return json.dumps({
                "error": f"Trace {trace_id} not found",
                "trace_id": trace_id,
                "repository": repo
            })

        trace_row = trace_data.iloc[0]

        # Parse spans if they're stored as string
        spans = trace_row['spans']
        if isinstance(spans, str):
            spans = json.loads(spans)

        return json.dumps({
            "trace_id": trace_id,
            "repository": repo,
            "run_id": trace_row.get('run_id', 'unknown'),
            "spans": spans
        }, indent=2)

    except Exception as e:
        return json.dumps({
            "error": str(e),
            "trace_id": trace_id,
            "repository": repo
        })


@gr.mcp.resource("cost://model/{model_name}")
def get_cost_data(model_name: str) -> str:
    """
    Get cost information for a specific model.

    This resource provides pricing data for LLM models and hardware configurations,
    helping users understand evaluation costs.

    Args:
        model_name (str): Model identifier (e.g., "openai/gpt-4", "meta-llama/Llama-3.1-8B")

    Returns:
        str: JSON string containing cost data for the model
    """
    # Cost database
    llm_costs = {
        "openai/gpt-4": {
            "input_per_1k_tokens": 0.03,
            "output_per_1k_tokens": 0.06,
            "type": "api",
            "provider": "openai"
        },
        "openai/gpt-3.5-turbo": {
            "input_per_1k_tokens": 0.0015,
            "output_per_1k_tokens": 0.002,
            "type": "api",
            "provider": "openai"
        },
        "anthropic/claude-3-opus": {
            "input_per_1k_tokens": 0.015,
            "output_per_1k_tokens": 0.075,
            "type": "api",
            "provider": "anthropic"
        },
        "anthropic/claude-3-sonnet": {
            "input_per_1k_tokens": 0.003,
            "output_per_1k_tokens": 0.015,
            "type": "api",
            "provider": "anthropic"
        },
        "meta-llama/Llama-3.1-8B": {
            "input_per_1k_tokens": 0,
            "output_per_1k_tokens": 0,
            "type": "local",
            "provider": "meta",
            "requires_gpu": True,
            "recommended_hardware": "gpu_a10"
        }
    }

    hardware_costs = {
        "cpu": {"hourly_rate_usd": 0.60, "type": "cpu"},
        "gpu_a10": {"hourly_rate_usd": 1.10, "type": "gpu", "model": "A10"},
        "gpu_h200": {"hourly_rate_usd": 4.50, "type": "gpu", "model": "H200"}
    }

    model_cost = llm_costs.get(model_name)

    if model_cost:
        return json.dumps({
            "model": model_name,
            "cost_data": model_cost,
            "hardware_options": hardware_costs,
            "currency": "USD"
        }, indent=2)
    else:
        return json.dumps({
            "model": model_name,
            "error": "Model not found in cost database",
            "available_models": list(llm_costs.keys()),
            "hardware_options": hardware_costs
        }, indent=2)


# ============================================================================
# MCP PROMPTS - Reusable prompt templates for common workflows
# ============================================================================

@gr.mcp.prompt()
def analysis_prompt(
    analysis_type: str = "leaderboard",
    focus_area: str = "overall",
    detail_level: str = "detailed"
) -> str:
    """
    Generate a prompt template for analyzing agent evaluation data.

    This prompt helps standardize analysis requests across different
    evaluation data types and focus areas.

    Args:
        analysis_type (str): Type of analysis. Options: "leaderboard", "trace", "cost". Default: "leaderboard"
        focus_area (str): What to focus on. Options: "overall", "performance", "cost", "efficiency". Default: "overall"
        detail_level (str): Level of detail. Options: "summary", "detailed", "comprehensive". Default: "detailed"

    Returns:
        str: Formatted prompt template for analysis
    """
    templates = {
        "leaderboard": {
            "overall": "Analyze the agent evaluation leaderboard data comprehensively. Identify top performers across all metrics (accuracy, cost, latency, CO2), explain trade-offs between different approaches, and provide actionable recommendations for model selection.",
            "performance": "Focus on performance metrics in the leaderboard. Compare success rates and accuracy across different models and agent types. Identify which configurations achieve the highest success rates and explain why.",
            "cost": "Analyze cost efficiency in the leaderboard. Compare costs across different models and identify the best cost-performance ratios. Recommend the most cost-effective configurations for different use cases.",
            "efficiency": "Evaluate efficiency metrics including latency, GPU utilization, and CO2 emissions. Identify the most efficient models and explain how to optimize for speed while maintaining quality."
        },
        "trace": {
            "overall": "Analyze this agent execution trace comprehensively. Explain the sequence of operations, identify any bottlenecks or inefficiencies, and suggest optimizations.",
            "performance": "Focus on performance aspects of this trace. Identify which steps took the most time, explain why, and suggest ways to improve execution speed.",
            "cost": "Analyze the cost implications of this trace execution. Break down token usage and API calls, calculate costs, and suggest ways to reduce expenses.",
            "efficiency": "Evaluate the efficiency of this trace. Identify redundant operations, suggest ways to optimize the execution flow, and recommend best practices."
        },
        "cost": {
            "overall": "Analyze the cost estimation comprehensively. Break down LLM API costs, infrastructure costs, and provide optimization recommendations.",
            "performance": "Focus on the cost-performance trade-off. Compare different hardware options and explain which provides the best value.",
            "cost": "Deep dive into cost breakdown. Explain each cost component in detail and provide specific recommendations for cost reduction.",
            "efficiency": "Analyze cost efficiency. Compare different model configurations and recommend the most cost-effective approach for the given use case."
        }
    }

    detail_prefixes = {
        "summary": "Provide a brief, high-level summary. ",
        "detailed": "Provide a detailed analysis with specific insights. ",
        "comprehensive": "Provide a comprehensive, in-depth analysis with detailed recommendations. "
    }

    prefix = detail_prefixes.get(detail_level, detail_prefixes["detailed"])
    template = templates.get(analysis_type, {}).get(focus_area, templates["leaderboard"]["overall"])

    return f"{prefix}{template}"


@gr.mcp.prompt()
def debug_prompt(
    debug_type: str = "error",
    context: str = "agent_execution"
) -> str:
    """
    Generate a prompt template for debugging agent traces.

    This prompt helps standardize debugging requests for different
    types of issues and contexts.

    Args:
        debug_type (str): Type of debugging. Options: "error", "performance", "behavior", "optimization". Default: "error"
        context (str): Execution context. Options: "agent_execution", "tool_calling", "llm_reasoning". Default: "agent_execution"

    Returns:
        str: Formatted prompt template for debugging
    """
    templates = {
        "error": {
            "agent_execution": "Debug this agent execution trace to identify why it failed. Analyze each step in the execution flow, identify where the error occurred, explain the root cause, and suggest how to fix it.",
            "tool_calling": "Debug this tool calling sequence. Identify which tool call failed or produced unexpected results, explain why it happened, and suggest corrections.",
            "llm_reasoning": "Debug the LLM reasoning in this trace. Analyze the prompts and responses, identify where the reasoning went wrong, and suggest improvements to the prompts or approach."
        },
        "performance": {
            "agent_execution": "Analyze this trace for performance issues. Identify bottlenecks, measure time spent in each component, and recommend optimizations to improve execution speed.",
            "tool_calling": "Analyze tool calling performance. Identify which tools are slow, explain why, and suggest ways to optimize tool execution or caching.",
            "llm_reasoning": "Analyze LLM reasoning efficiency. Identify unnecessary calls, redundant reasoning steps, and suggest ways to streamline the reasoning process."
        },
        "behavior": {
            "agent_execution": "Analyze the agent's behavior in this trace. Explain why the agent made certain decisions, whether the behavior is expected, and suggest improvements if needed.",
            "tool_calling": "Analyze tool selection behavior. Explain why certain tools were called, whether the choices were optimal, and suggest alternative approaches if applicable.",
            "llm_reasoning": "Analyze the LLM's reasoning patterns. Explain the logic flow, identify any unexpected reasoning, and suggest how to guide the model toward better decisions."
        },
        "optimization": {
            "agent_execution": "Analyze this trace for optimization opportunities. Identify redundant operations, suggest caching strategies, and recommend ways to reduce costs and improve efficiency.",
            "tool_calling": "Optimize tool usage in this trace. Suggest ways to reduce tool calls, batch operations, or use more efficient alternatives.",
            "llm_reasoning": "Optimize LLM usage. Suggest ways to reduce token usage, improve prompt efficiency, and achieve the same results with lower costs."
        }
    }

    template = templates.get(debug_type, {}).get(context, templates["error"]["agent_execution"])
    return template


@gr.mcp.prompt()
def optimization_prompt(
    optimization_goal: str = "cost",
    constraints: str = "maintain_quality"
) -> str:
    """
    Generate a prompt template for optimization recommendations.

    This prompt helps standardize optimization requests for different
    goals and constraints.

    Args:
        optimization_goal (str): What to optimize. Options: "cost", "speed", "quality", "efficiency". Default: "cost"
        constraints (str): Constraints to consider. Options: "maintain_quality", "maintain_speed", "no_constraints". Default: "maintain_quality"

    Returns:
        str: Formatted prompt template for optimization
    """
    templates = {
        "cost": {
            "maintain_quality": "Analyze this evaluation setup and recommend cost optimizations while maintaining quality. Consider cheaper models, optimized prompts, caching strategies, and hardware selection. Quantify potential savings.",
            "maintain_speed": "Recommend cost optimizations while maintaining execution speed. Consider model alternatives, batch processing, and infrastructure choices that reduce costs without adding latency.",
            "no_constraints": "Recommend aggressive cost optimizations. Identify all opportunities to reduce expenses, even if it means trade-offs in quality or speed. Prioritize maximum cost reduction."
        },
        "speed": {
            "maintain_quality": "Recommend speed optimizations while maintaining quality. Consider parallel execution, caching, faster models with similar accuracy, and infrastructure upgrades. Quantify potential speedups.",
            "maintain_cost": "Recommend speed optimizations within the current cost budget. Suggest configuration changes, caching strategies, and optimizations that don't increase expenses.",
            "no_constraints": "Recommend aggressive speed optimizations. Identify all opportunities to reduce latency, even if it increases costs. Prioritize maximum performance."
        },
        "quality": {
            "maintain_cost": "Recommend quality improvements within the current cost budget. Suggest better prompts, model configurations, and strategies that improve accuracy without increasing expenses.",
            "maintain_speed": "Recommend quality improvements while maintaining execution speed. Suggest prompt improvements, reasoning enhancements, and configurations that improve accuracy without adding latency.",
            "no_constraints": "Recommend quality improvements without budget constraints. Suggest the best models, optimal configurations, and strategies to maximize accuracy and success rates."
        },
        "efficiency": {
            "maintain_quality": "Recommend overall efficiency improvements. Optimize for the best cost-speed-quality balance. Identify waste, suggest streamlined processes, and provide holistic optimization strategies.",
            "maintain_cost": "Recommend efficiency improvements within budget. Focus on reducing waste, optimizing resource usage, and getting better results with the same cost.",
            "maintain_speed": "Recommend efficiency improvements maintaining speed. Reduce unnecessary operations, optimize resource usage, and improve output quality without adding latency."
        }
    }

    # Handle constraint variations
    if constraints == "maintain_quality" and optimization_goal == "speed":
        constraints = "maintain_quality"  # Use existing template
    elif constraints == "maintain_speed" and optimization_goal == "cost":
        constraints = "maintain_speed"  # Use existing template

    template = templates.get(optimization_goal, {}).get(constraints, templates["cost"]["maintain_quality"])
    return template
