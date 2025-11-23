"""
MCP Tool Implementations for TraceMind MCP Server

This module implements 13 MCP components (7 Tools + 3 Resources + 3 Prompts) for
AI-powered agent evaluation analysis.

With Gradio's native MCP support (mcp_server=True), these are automatically
exposed based on decorators (@gr.mcp.tool, @gr.mcp.resource, @gr.mcp.prompt),
docstrings, and type hints.

🛠️ Tools (7 AI-Powered):
    📊 analyze_leaderboard - Get AI insights from evaluation leaderboard data
    🐛 debug_trace - Debug agent execution traces with AI assistance
    💰 estimate_cost - Predict evaluation costs with AI recommendations
    ⚖️ compare_runs - Compare two evaluation runs with AI analysis
    📦 get_dataset - Load SMOLTRACE datasets as JSON for flexible analysis
    🧪 generate_synthetic_dataset - Create domain-specific test datasets
    📤 push_dataset_to_hub - Upload datasets to HuggingFace Hub

📦 Resources (3 Data Access):
    leaderboard://{repo} - Raw leaderboard data in JSON format
    trace://{trace_id}/{repo} - Raw OpenTelemetry trace data
    cost://model/{model_name} - Model pricing and hardware cost data

📝 Prompts (3 Templates):
    analysis_prompt - Standardized templates for analysis requests
    debug_prompt - Standardized templates for debugging scenarios
    optimization_prompt - Standardized templates for optimization goals

All AI analysis powered by Google Gemini 2.5 Flash.
Track 1: Building MCP Servers - Enterprise Category
"""

import os
import json
from typing import Optional
from datasets import load_dataset
import pandas as pd
from datetime import datetime, timedelta
import gradio as gr

from gemini_client import GeminiClient


@gr.mcp.tool()
async def analyze_leaderboard(
    leaderboard_repo: str = "kshitijthakkar/smoltrace-leaderboard",
    metric_focus: str = "overall",
    time_range: str = "last_week",
    top_n: int = 5
) -> str:
    """
    Answer questions about the leaderboard with AI-powered analysis and insights.

    USE THIS TOOL when you need to:
    - Answer questions like "Which model is leading?", "What's the best model for cost?"
    - Get intelligent insights about top performers and trends
    - Compare models and understand trade-offs
    - Get recommendations based on leaderboard data

    DO NOT use the leaderboard:// resource for questions - use this tool instead!
    The resource only returns raw JSON data without any analysis.

    This tool uses Google Gemini 2.5 Flash to provide intelligent analysis of
    agent evaluation results, including top performers, trends, cost/performance
    trade-offs, and actionable recommendations.

    **Security**: Requires GEMINI_API_KEY environment variable.
    **Note**: All SMOLTRACE datasets are public - no HF token required.

    Args:
        leaderboard_repo (str): HuggingFace dataset repository containing leaderboard data. Default: "kshitijthakkar/smoltrace-leaderboard"
        metric_focus (str): Primary metric to focus analysis on. Options: "overall", "accuracy", "cost", "latency", "co2". Default: "overall"
        time_range (str): Time range for analysis. Options: "last_week", "last_month", "all_time". Default: "last_week"
        top_n (int): Number of top models to highlight in analysis. Must be between 3 and 10. Default: 5

    Returns:
        str: Markdown-formatted analysis with top performers, insights, trade-offs, and recommendations
    """
    try:
        # Initialize Gemini client from environment variable only
        gemini_client = GeminiClient()
        # Load leaderboard data from HuggingFace (public dataset)
        print(f"Loading leaderboard from {leaderboard_repo}...")

        ds = load_dataset(leaderboard_repo, split="train")
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


@gr.mcp.tool()
async def debug_trace(
    trace_id: str,
    traces_repo: str,
    question: str = "Analyze this trace and explain what happened"
) -> str:
    """
    Answer questions about agent traces with AI-powered debugging and analysis.

    USE THIS TOOL when you need to:
    - Answer questions like "Why did this fail?", "What took the most time?", "Why was X called?"
    - Debug agent execution traces and understand what happened
    - Identify bottlenecks and performance issues
    - Get explanations about agent behavior

    DO NOT use the trace:// resource for questions - use this tool instead!
    The resource only returns raw OTEL JSON data without any analysis.

    This tool uses Google Gemini 2.5 Flash to analyze OpenTelemetry trace data and
    provide intelligent debugging insights, step-by-step breakdowns, and answers
    to specific questions about execution flow.

    Args:
        trace_id (str): Unique identifier for the trace to analyze (e.g., "trace_abc123")
        traces_repo (str): HuggingFace dataset repository containing trace data (e.g., "username/agent-traces-model-timestamp")
        question (str): Specific question about the trace. Default: "Analyze this trace and explain what happened"
    Returns:
        str: Markdown-formatted debug analysis with step-by-step breakdown, timing information, and answer to the question
    """
    try:
        # Initialize Gemini client with provided key or from environment
        gemini_client = GeminiClient()
        # Load traces dataset (public dataset)
        print(f"Loading traces from {traces_repo}...")

        ds = load_dataset(traces_repo, split="train")
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


@gr.mcp.tool()
async def estimate_cost(
    model: str,
    agent_type: str,
    num_tests: int = 100,
    hardware: str = "auto"
) -> str:
    """
    Answer questions about evaluation costs with AI-powered estimates and recommendations.

    USE THIS TOOL when you need to:
    - Answer questions like "How much will this cost?", "What's the cheapest option?"
    - Get cost predictions for running evaluations
    - Compare costs between different models or hardware
    - Get optimization recommendations to reduce costs

    DO NOT use the cost:// resource for estimates - use this tool instead!
    The resource only returns raw pricing tables without calculations.

    This tool uses Google Gemini 2.5 Flash to calculate LLM API costs, HuggingFace
    Jobs compute costs, CO2 emissions, and provide intelligent cost breakdowns with
    optimization recommendations.

    Args:
        model (str): Model identifier in litellm format (e.g., "openai/gpt-4", "meta-llama/Llama-3.1-8B")
        agent_type (str): Type of agent capabilities to test. Options: "tool", "code", "both"
        num_tests (int): Number of test cases to run. Must be between 10 and 1000. Default: 100
        hardware (str): Hardware type for compute. Supports Modal (gpu_t4, gpu_a10, gpu_h200, etc.) and HuggingFace Jobs (cpu-basic, t4-small, a10g-small, a100-large, etc.). Default: "auto"
    Returns:
        str: Markdown-formatted cost estimate with breakdown of LLM costs, compute costs, duration, CO2 emissions, and optimization tips
    """
    try:
        # Initialize Gemini client with provided key or from environment
        gemini_client = GeminiClient()

        # Fetch LLM pricing from genai_otel project
        import requests
        pricing_url = "https://raw.githubusercontent.com/Mandark-droid/genai_otel_instrument/refs/heads/main/genai_otel/llm_pricing.json"

        try:
            response = requests.get(pricing_url, timeout=5)
            response.raise_for_status()
            llm_pricing_db = response.json()
            print(f"[INFO] Loaded {len(llm_pricing_db)} models from pricing database")
        except Exception as e:
            print(f"[WARNING] Failed to load pricing database: {e}, using fallback")
            llm_pricing_db = {}

        # Determine if API or local model
        is_api_model = any(provider in model.lower() for provider in ["openai", "anthropic", "google", "cohere"])

        # Auto-select hardware
        if hardware == "auto":
            hardware = "cpu-basic" if is_api_model else "a10g-small"

        # Compute costs (per second) - Modal + HuggingFace Jobs
        compute_costs = {
            # Modal GPU Tasks (per second)
            "gpu_b200": 0.001736,      # Nvidia B200
            "gpu_h200": 0.001261,      # Nvidia H200
            "gpu_h100": 0.001097,      # Nvidia H100
            "gpu_a100_80gb": 0.000694, # Nvidia A100, 80 GB
            "gpu_a100": 0.000583,      # Nvidia A100, 40 GB
            "gpu_l40s": 0.000542,      # Nvidia L40S
            "gpu_a10": 0.000306,       # Nvidia A10
            "gpu_l4": 0.000222,        # Nvidia L4
            "gpu_t4": 0.000164,        # Nvidia T4
            # Modal CPU (per core)
            "cpu": 0.0000131,          # Physical core (2 vCPU equivalent)

            # HuggingFace Jobs (estimated per second based on typical hourly rates)
            # Note: HF Jobs pricing varies, these are estimates
            "cpu-basic": 0.0000167,    # ~$0.06/hour
            "cpu-upgrade": 0.0000278,  # ~$0.10/hour
            "t4-small": 0.000167,      # ~$0.60/hour
            "t4-medium": 0.000278,     # ~$1.00/hour
            "l4x1": 0.000250,          # ~$0.90/hour
            "l4x4": 0.001000,          # ~$3.60/hour
            "a10g-small": 0.000333,    # ~$1.20/hour
            "a10g-large": 0.000556,    # ~$2.00/hour
            "a10g-largex2": 0.001111,  # ~$4.00/hour
            "a10g-largex4": 0.002222,  # ~$8.00/hour
            "a100-large": 0.001389,    # ~$5.00/hour
            # TPU (estimated)
            "v5e-1x1": 0.000417,       # ~$1.50/hour
            "v5e-2x2": 0.001667,       # ~$6.00/hour
            "v5e-2x4": 0.003333        # ~$12.00/hour
        }

        # Get model costs from pricing database
        model_cost = None

        # Try exact match first
        if model in llm_pricing_db:
            model_cost = llm_pricing_db[model]
        else:
            # Try without provider prefix (e.g., "gpt-4" instead of "openai/gpt-4")
            model_name = model.split('/')[-1]
            for key in llm_pricing_db:
                if model_name in key or key in model_name:
                    model_cost = llm_pricing_db[key]
                    print(f"[INFO] Found pricing for {model} via fuzzy match: {key}")
                    break

        # Fallback to default if not found
        if model_cost is None:
            print(f"[WARNING] Model {model} not in pricing database, using default")
            if is_api_model:
                model_cost = {"input_cost_per_token": 0.000001, "output_cost_per_token": 0.000002}
            else:
                model_cost = {"input_cost_per_token": 0, "output_cost_per_token": 0}  # Local model

        # Estimate token usage per test (based on real data from kshitijthakkar/smoltrace-results-20251117_104845)
        # These are averages from actual agent evaluation runs
        # Input/output split estimated at 60/40 based on typical agent patterns
        # (agents have large context with system prompts, tool outputs, etc.)
        token_estimates = {
            "tool": {
                "input": 7577,    # 60% of 12,629 avg total tokens
                "output": 5052    # 40% of 12,629 avg total tokens
            },
            "code": {
                "input": 10321,   # 60% of 17,202 avg total tokens
                "output": 6881    # 40% of 17,202 avg total tokens
            },
            "both": {
                "input": 8900,    # Average of tool+code inputs
                "output": 5933    # Average of tool+code outputs
            }
        }

        tokens_per_test = token_estimates[agent_type]

        # Calculate LLM costs (pricing is per token, not per 1K tokens)
        llm_cost_per_test = (
            tokens_per_test["input"] * model_cost.get("input_cost_per_token", 0) +
            tokens_per_test["output"] * model_cost.get("output_cost_per_token", 0)
        )
        total_llm_cost = llm_cost_per_test * num_tests

        # Estimate duration (seconds per test)
        if is_api_model:
            duration_per_test = 3.0  # API models are fast
        else:
            duration_per_test = 8.0  # Local models slower but depends on GPU

        total_duration_seconds = duration_per_test * num_tests

        # Calculate compute costs (per second)
        compute_rate_per_sec = compute_costs.get(hardware, compute_costs.get("cpu-basic", 0.0000167))

        # For CPU-based hardware, estimate core usage (assume 2 cores for agent workload)
        # For GPU/TPU, direct cost
        if hardware in ["cpu", "cpu-basic", "cpu-upgrade"]:
            num_cores = 2  # Estimate 2 cores for typical agent workload
            total_compute_cost = total_duration_seconds * compute_rate_per_sec * num_cores
        else:
            total_compute_cost = total_duration_seconds * compute_rate_per_sec

        # Estimate CO2 (rough estimates in kg per hour)
        co2_per_hour = {
            # Modal
            "cpu": 0.05,
            "gpu_t4": 0.10,
            "gpu_l4": 0.12,
            "gpu_a10": 0.15,
            "gpu_l40s": 0.20,
            "gpu_a100": 0.25,
            "gpu_a100_80gb": 0.28,
            "gpu_h100": 0.30,
            "gpu_h200": 0.32,
            "gpu_b200": 0.35,
            # HuggingFace Jobs
            "cpu-basic": 0.03,
            "cpu-upgrade": 0.04,
            "t4-small": 0.08,
            "t4-medium": 0.10,
            "l4x1": 0.12,
            "l4x4": 0.48,
            "a10g-small": 0.13,
            "a10g-large": 0.15,
            "a10g-largex2": 0.30,
            "a10g-largex4": 0.60,
            "a100-large": 0.25,
            "v5e-1x1": 0.18,
            "v5e-2x2": 0.72,
            "v5e-2x4": 1.44
        }

        total_co2_kg = (total_duration_seconds / 3600) * co2_per_hour.get(hardware, 0.05)

        # Prepare estimate data
        estimate_data = {
            "model": model,
            "agent_type": agent_type,
            "num_tests": num_tests,
            "hardware": hardware,
            "is_api_model": is_api_model,
            "pricing_source": "genai_otel pricing database + Modal/HF Jobs compute costs",
            "estimates": {
                "llm_cost_usd": round(total_llm_cost, 6),
                "llm_cost_per_test": round(llm_cost_per_test, 6),
                "compute_cost_usd": round(total_compute_cost, 6),
                "total_cost_usd": round(total_llm_cost + total_compute_cost, 6),
                "duration_seconds": round(total_duration_seconds, 2),
                "duration_minutes": round(total_duration_seconds / 60, 2),
                "duration_per_test_seconds": round(duration_per_test, 2),
                "co2_emissions_kg": round(total_co2_kg, 4),
                "tokens_per_test": tokens_per_test,
                "compute_rate_per_second": compute_rate_per_sec
            },
            "model_pricing": {
                "input_cost_per_token": model_cost.get("input_cost_per_token", 0),
                "output_cost_per_token": model_cost.get("output_cost_per_token", 0),
                "found_in_database": model in llm_pricing_db
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


@gr.mcp.tool()
async def compare_runs(
    run_id_1: str,
    run_id_2: str,
    leaderboard_repo: str = "kshitijthakkar/smoltrace-leaderboard",
    comparison_focus: str = "comprehensive"
) -> str:
    """
    Compare two evaluation runs and generate AI-powered comparative analysis.

    This tool fetches data for two evaluation runs from the leaderboard and uses
    Google Gemini 2.5 Flash to provide intelligent comparison across multiple dimensions:
    success rate, cost efficiency, speed, environmental impact, and use case recommendations.

    Args:
        run_id_1 (str): First run ID to compare
        run_id_2 (str): Second run ID to compare
        leaderboard_repo (str): HuggingFace dataset repository containing leaderboard data. Default: "kshitijthakkar/smoltrace-leaderboard"
        comparison_focus (str): Focus area for comparison. Options: "comprehensive", "cost", "performance", "eco_friendly". Default: "comprehensive"
    Returns:
        str: Markdown-formatted comparative analysis with winner for each category, trade-offs, and use case recommendations
    """
    try:
        # Initialize Gemini client with provided key or from environment
        gemini_client = GeminiClient()

        # Load leaderboard data
        dataset = load_dataset(leaderboard_repo, split="train")
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


@gr.mcp.tool()
async def analyze_results(
    results_repo: str,
    analysis_focus: str = "comprehensive",
    max_rows: int = 100
) -> str:
    """
    Analyze detailed test results and provide optimization recommendations.

    USE THIS TOOL when you need to:
    - Understand why tests are failing and get recommendations
    - Identify performance bottlenecks in specific test cases
    - Find cost optimization opportunities
    - Get insights about tool usage patterns
    - Analyze which types of tasks work well vs poorly

    This tool analyzes individual test case results (not aggregate leaderboard data)
    and uses Google Gemini 2.5 Flash to provide actionable optimization recommendations.

    Args:
        results_repo (str): HuggingFace dataset repository containing results (e.g., "username/smoltrace-results-gpt4-20251114")
        analysis_focus (str): Focus area. Options: "failures", "performance", "cost", "comprehensive". Default: "comprehensive"
        max_rows (int): Maximum test cases to analyze. Default: 100. Range: 10-500
    Returns:
        str: Markdown-formatted analysis with failure patterns, performance insights, cost analysis, and optimization recommendations
    """
    try:
        # Initialize Gemini client
        gemini_client = GeminiClient()

        # Load results dataset
        print(f"Loading results from {results_repo}...")

        ds = load_dataset(results_repo, split="train")
        df = pd.DataFrame(ds)

        if df.empty:
            return "❌ **Error**: Results dataset is empty"

        # Limit rows
        max_rows = max(10, min(500, max_rows))
        df_sample = df.head(max_rows)

        # Calculate statistics
        total_tests = len(df_sample)
        successful = df_sample[df_sample['success'] == True]
        failed = df_sample[df_sample['success'] == False]

        success_rate = (len(successful) / total_tests * 100) if total_tests > 0 else 0

        # Analyze by category/difficulty
        category_stats = {}
        if 'category' in df_sample.columns:
            cat_agg = df_sample.groupby('category').agg({
                'success': ['count', 'sum', 'mean'],
                'execution_time_ms': 'mean',
                'cost_usd': 'sum'
            })
            # Flatten multi-index columns
            cat_agg.columns = ['_'.join(col).strip() for col in cat_agg.columns.values]
            category_stats = cat_agg.to_dict('index')

        difficulty_stats = {}
        if 'difficulty' in df_sample.columns:
            diff_agg = df_sample.groupby('difficulty').agg({
                'success': ['count', 'sum', 'mean'],
                'execution_time_ms': 'mean'
            })
            # Flatten multi-index columns
            diff_agg.columns = ['_'.join(col).strip() for col in diff_agg.columns.values]
            difficulty_stats = diff_agg.to_dict('index')

        # Find slowest tests
        slowest_tests = df_sample.nlargest(5, 'execution_time_ms')[
            ['task_id', 'prompt', 'execution_time_ms', 'success', 'cost_usd']
        ].to_dict('records')

        # Find most expensive tests
        if 'cost_usd' in df_sample.columns:
            most_expensive = df_sample.nlargest(5, 'cost_usd')[
                ['task_id', 'prompt', 'cost_usd', 'total_tokens', 'success']
            ].to_dict('records')
        else:
            most_expensive = []

        # Analyze failures
        failure_analysis = []
        if len(failed) > 0:
            # Define which columns to include in failure sample
            failure_columns = ['task_id', 'prompt']

            # Add optional columns if they exist
            optional_columns = ['error', 'error_type', 'tool_called', 'expected_tool']
            for col in optional_columns:
                if col in failed.columns:
                    failure_columns.append(col)

            # Get sample of failures with only existing columns
            failure_sample = failed.head(10)[failure_columns].to_dict('records')

            # Count error types if column exists
            if 'error_type' in failed.columns:
                error_type_counts = failed['error_type'].value_counts().to_dict()
            else:
                error_type_counts = {}

            failure_analysis = {
                "total_failures": len(failed),
                "failure_rate": (len(failed) / total_tests * 100),
                "error_type_counts": error_type_counts,
                "sample_failures": failure_sample
            }

        # Prepare data for Gemini analysis
        analysis_data = {
            "results_repo": results_repo,
            "total_tests_analyzed": total_tests,
            "overall_stats": {
                "success_rate": round(success_rate, 2),
                "successful_tests": len(successful),
                "failed_tests": len(failed),
                "avg_execution_time_ms": float(df_sample['execution_time_ms'].mean()),
                "total_cost_usd": float(df_sample['cost_usd'].sum()) if 'cost_usd' in df_sample.columns else 0,
                "avg_tokens_per_test": float(df_sample['total_tokens'].mean()) if 'total_tokens' in df_sample.columns else 0
            },
            "category_performance": category_stats,
            "difficulty_performance": difficulty_stats,
            "slowest_tests": slowest_tests,
            "most_expensive_tests": most_expensive,
            "failure_analysis": failure_analysis,
            "analysis_focus": analysis_focus
        }

        # Create focus-specific prompt
        focus_prompts = {
            "failures": "Focus specifically on failure patterns. Analyze why tests are failing, identify common error types, and provide actionable recommendations to improve success rate.",
            "performance": "Focus on performance optimization. Analyze execution times, identify bottlenecks, and recommend ways to speed up test execution.",
            "cost": "Focus on cost optimization. Analyze token usage and costs, identify expensive tests, and recommend ways to reduce evaluation costs.",
            "comprehensive": "Provide comprehensive analysis covering failures, performance, cost, and overall optimization opportunities."
        }

        specific_question = focus_prompts.get(analysis_focus, focus_prompts["comprehensive"])

        # Get AI analysis
        result = await gemini_client.analyze_with_context(
            data=analysis_data,
            analysis_type="results",
            specific_question=specific_question
        )

        return result

    except Exception as e:
        return f"❌ **Error analyzing results**: {str(e)}\n\nPlease check:\n- Repository name is correct (should be smoltrace-results-*)\n- You have access to the dataset\n- HF_TOKEN is set correctly"


@gr.mcp.tool()
async def get_top_performers(
    leaderboard_repo: str = "kshitijthakkar/smoltrace-leaderboard",
    metric: str = "success_rate",
    top_n: int = 5
) -> str:
    """
    Get top performing models from leaderboard - optimized for quick queries.

    **USE THIS TOOL** instead of get_dataset() when you need to answer questions like:
    - "Which model is leading?"
    - "Show me the top 5 models"
    - "What's the best model for cost?"

    This tool returns ONLY the essential data for top performers, avoiding the
    full 51-run dataset that causes token bloat. Returns properly formatted JSON
    that's ready to use without parsing.

    Args:
        leaderboard_repo (str): HuggingFace dataset repository. Default: "kshitijthakkar/smoltrace-leaderboard"
        metric (str): Metric to rank by. Options: "success_rate", "total_cost_usd", "avg_duration_ms", "co2_emissions_g". Default: "success_rate"
        top_n (int): Number of top models to return. Range: 1-20. Default: 5

    Returns:
        str: JSON object with top performers - ready to use, no parsing needed
    """
    try:
        # Load leaderboard dataset
        ds = load_dataset(leaderboard_repo, split="train")
        df = pd.DataFrame(ds)

        if df.empty:
            return json.dumps({
                "error": "Leaderboard dataset is empty",
                "top_performers": []
            }, indent=2)

        # Validate metric
        valid_metrics = ["success_rate", "total_cost_usd", "avg_duration_ms", "co2_emissions_g"]
        if metric not in valid_metrics:
            return json.dumps({
                "error": f"Invalid metric '{metric}'. Valid options: {valid_metrics}",
                "top_performers": []
            }, indent=2)

        # Limit top_n
        top_n = max(1, min(20, top_n))

        # Sort by metric (ascending for cost/latency/co2, descending for success_rate)
        ascending = metric in ["total_cost_usd", "avg_duration_ms", "co2_emissions_g"]
        df_sorted = df.sort_values(metric, ascending=ascending)

        # Get top N
        top_models = df_sorted.head(top_n)

        # Select only essential columns to minimize tokens
        essential_columns = [
            "run_id", "model", "agent_type", "provider",
            "success_rate", "total_cost_usd", "avg_duration_ms",
            "co2_emissions_g", "total_tests", "timestamp"
        ]

        # Filter to only columns that exist
        available_columns = [col for col in essential_columns if col in top_models.columns]
        top_models_filtered = top_models[available_columns]

        # CRITICAL FIX: Handle NaN/None properly
        top_models_filtered = top_models_filtered.where(pd.notnull(top_models_filtered), None)

        # Convert to dict
        top_performers_data = top_models_filtered.to_dict(orient="records")

        result = {
            "metric_ranked_by": metric,
            "ranking_order": "ascending (lower is better)" if ascending else "descending (higher is better)",
            "total_runs_in_leaderboard": len(df),
            "top_n": top_n,
            "top_performers": top_performers_data
        }

        return json.dumps(result, indent=2)

    except Exception as e:
        return json.dumps({
            "error": f"Failed to get top performers: {str(e)}",
            "top_performers": []
        }, indent=2)


@gr.mcp.tool()
async def get_leaderboard_summary(
    leaderboard_repo: str = "kshitijthakkar/smoltrace-leaderboard"
) -> str:
    """
    Get high-level leaderboard summary statistics - optimized for overview queries.

    **USE THIS TOOL** instead of get_dataset() when you need to answer questions like:
    - "How many runs are in the leaderboard?"
    - "What's the average success rate?"
    - "Give me an overview of the leaderboard"

    This tool returns ONLY summary statistics (no individual runs), avoiding the
    full dataset that causes token bloat. Returns properly formatted JSON that's
    ready to use without parsing.

    Args:
        leaderboard_repo (str): HuggingFace dataset repository. Default: "kshitijthakkar/smoltrace-leaderboard"

    Returns:
        str: JSON object with summary statistics - ready to use, no parsing needed
    """
    try:
        # Load leaderboard dataset
        ds = load_dataset(leaderboard_repo, split="train")
        df = pd.DataFrame(ds)

        if df.empty:
            return json.dumps({
                "error": "Leaderboard dataset is empty",
                "summary": {}
            }, indent=2)

        # Calculate summary statistics
        summary = {
            "total_runs": len(df),
            "unique_models": int(df['model'].nunique()) if 'model' in df.columns else 0,
            "unique_submitters": int(df['submitted_by'].nunique()) if 'submitted_by' in df.columns else 0,
            "overall_stats": {
                "avg_success_rate": float(df['success_rate'].mean()) if 'success_rate' in df.columns else None,
                "best_success_rate": float(df['success_rate'].max()) if 'success_rate' in df.columns else None,
                "worst_success_rate": float(df['success_rate'].min()) if 'success_rate' in df.columns else None,
                "avg_cost_per_run_usd": float(df['total_cost_usd'].mean()) if 'total_cost_usd' in df.columns else None,
                "avg_duration_ms": float(df['avg_duration_ms'].mean()) if 'avg_duration_ms' in df.columns else None,
                "total_co2_emissions_g": float(df['co2_emissions_g'].sum()) if 'co2_emissions_g' in df.columns else None
            },
            "breakdown_by_agent_type": {},
            "breakdown_by_provider": {},
            "top_3_models_by_success_rate": []
        }

        # Breakdown by agent type
        if 'agent_type' in df.columns and 'success_rate' in df.columns:
            agent_stats = df.groupby('agent_type').agg({
                'success_rate': 'mean',
                'run_id': 'count'
            }).to_dict()

            summary["breakdown_by_agent_type"] = {
                agent_type: {
                    "count": int(agent_stats['run_id'][agent_type]),
                    "avg_success_rate": float(agent_stats['success_rate'][agent_type])
                }
                for agent_type in agent_stats['run_id'].keys()
            }

        # Breakdown by provider
        if 'provider' in df.columns and 'success_rate' in df.columns:
            provider_stats = df.groupby('provider').agg({
                'success_rate': 'mean',
                'run_id': 'count'
            }).to_dict()

            summary["breakdown_by_provider"] = {
                provider: {
                    "count": int(provider_stats['run_id'][provider]),
                    "avg_success_rate": float(provider_stats['success_rate'][provider])
                }
                for provider in provider_stats['run_id'].keys()
            }

        # Top 3 models by success rate
        if 'success_rate' in df.columns and 'model' in df.columns:
            top_3 = df.nlargest(3, 'success_rate')[['model', 'success_rate', 'total_cost_usd', 'avg_duration_ms']]
            top_3 = top_3.where(pd.notnull(top_3), None)
            summary["top_3_models_by_success_rate"] = top_3.to_dict(orient="records")

        result = {
            "leaderboard_repo": leaderboard_repo,
            "summary": summary
        }

        return json.dumps(result, indent=2)

    except Exception as e:
        return json.dumps({
            "error": f"Failed to get leaderboard summary: {str(e)}",
            "summary": {}
        }, indent=2)


@gr.mcp.tool()
async def get_dataset(
    dataset_repo: str,
    max_rows: int = 50
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
        dataset = load_dataset(dataset_repo, split="train")
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

        # CRITICAL FIX: Replace NaN/None values with proper None before conversion
        # This ensures json.dumps() handles them correctly as null instead of "None" string
        df_limited = df_limited.where(pd.notnull(df_limited), None)

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

        # CRITICAL FIX: Remove default=str to ensure proper JSON serialization
        # Using default=str was converting None to string "None" causing agent parsing issues
        return json.dumps(result, indent=2)

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
def get_leaderboard_data(repo: str = "kshitijthakkar/smoltrace-leaderboard") -> str:
    """
    [RAW DATA ONLY] Get raw leaderboard data in JSON format - NO analysis or insights.

    ⚠️ DO NOT USE THIS for questions like "Which model is leading?" or "What's the best model?"
    Instead, use the analyze_leaderboard TOOL which provides AI-powered insights.

    This resource is ONLY for:
    - Getting raw JSON data when you need to process it yourself
    - Low-level data access for custom analysis
    - Direct dataset retrieval without AI interpretation

    For questions, insights, recommendations, or analysis → use analyze_leaderboard tool instead!

    **Note**: All SMOLTRACE datasets are public - no authentication required.

    Args:
        repo (str): HuggingFace dataset repository name. Default: "kshitijthakkar/smoltrace-leaderboard"
    Returns:
        str: Raw JSON string containing all evaluation runs without any analysis
    """
    try:        
        ds = load_dataset(repo, split="train")
        df = pd.DataFrame(ds)

        # Convert to JSON with proper formatting
        data = df.to_dict('records')
        return json.dumps({
            "total_runs": len(data),
            "repository": repo,
            "data": data
        }, indent=2, default=str)

    except Exception as e:
        return json.dumps({
            "error": str(e),
            "repository": repo
        }, indent=2, default=str)


@gr.mcp.resource("trace://{trace_id}/{repo}")
def get_trace_data(trace_id: str, repo: str) -> str:
    """
    [RAW DATA ONLY] Get raw OpenTelemetry trace data in JSON format - NO analysis.

    ⚠️ DO NOT USE THIS for questions like "Why did this fail?" or "What took the most time?"
    Instead, use the debug_trace TOOL which provides AI-powered debugging and insights.

    This resource is ONLY for:
    - Getting raw OTEL span data when you need to process it yourself
    - Low-level trace access for custom analysis
    - Direct dataset retrieval without AI interpretation

    For debugging, questions, or analysis → use debug_trace tool instead!

    **Note**: All SMOLTRACE datasets are public - no authentication required.

    Args:
        trace_id (str): Unique identifier for the trace (e.g., "trace_abc123")
        repo (str): HuggingFace dataset repository containing traces (e.g., "username/agent-traces-model")
    Returns:
        str: Raw JSON string containing OpenTelemetry spans without any analysis
    """
    try:        
        ds = load_dataset(repo, split="train")
        df = pd.DataFrame(ds)

        # Find specific trace
        trace_data = df[df['trace_id'] == trace_id]

        if len(trace_data) == 0:
            return json.dumps({
                "error": f"Trace {trace_id} not found",
                "trace_id": trace_id,
                "repository": repo
            }, indent=2, default=str)

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
        }, indent=2, default=str)

    except Exception as e:
        return json.dumps({
            "error": str(e),
            "trace_id": trace_id,
            "repository": repo
        }, indent=2, default=str)


@gr.mcp.resource("cost://model/{model_name}")
def get_cost_data(model_name: str) -> str:
    """
    [RAW DATA ONLY] Get raw pricing data for a model in JSON format - NO estimates or analysis.

    ⚠️ DO NOT USE THIS for questions like "How much will this cost?" or "What's the best value?"
    Instead, use the estimate_cost TOOL which provides AI-powered cost estimates and recommendations.

    This resource is ONLY for:
    - Getting raw pricing tables when you need to process them yourself
    - Looking up base rates for models and hardware
    - Direct price data retrieval without calculations

    For cost estimates, predictions, or recommendations → use estimate_cost tool instead!

    Args:
        model_name (str): Model identifier (e.g., "openai/gpt-4", "meta-llama/Llama-3.1-8B")

    Returns:
        str: Raw JSON string with pricing rates without any cost estimation
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
        }, indent=2, default=str)
    else:
        return json.dumps({
            "model": model_name,
            "error": "Model not found in cost database",
            "available_models": list(llm_costs.keys()),
            "hardware_options": hardware_costs
        }, indent=2, default=str)


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


# ========================================
# NEW TOOLS: Synthetic Dataset Generation
# ========================================

@gr.mcp.tool()
async def generate_synthetic_dataset(
    domain: str,
    tool_names: str,
    num_tasks: int = 10,
    difficulty_distribution: str = "balanced",
    agent_type: str = "both"
) -> str:
    """
    Generate domain-specific synthetic test datasets for SMOLTRACE evaluations using AI.

    This tool uses Google Gemini 2.5 Flash to create realistic, domain-specific evaluation
    tasks that follow the SMOLTRACE task dataset format. Perfect for creating custom
    benchmarks when standard datasets don't fit your use case.

    **🚀 Batched Generation for Scale**:
    - Requests >20 tasks are automatically split into parallel batches
    - Utilizes Gemini's large context window efficiently
    - Supports up to 100 tasks with 120s timeout per batch
    - Example: 100 tasks = 5 parallel batches (20 tasks each)

    **Enterprise Use Case**: Quickly create evaluation datasets for:
    - Custom tools and APIs your agents use
    - Industry-specific domains (finance, healthcare, legal, manufacturing, etc.)
    - Internal workflows and business processes
    - Specialized agent capabilities

    **Security**: Requires GEMINI_API_KEY environment variable.

    Args:
        domain (str): The domain for synthetic tasks (e.g., "finance", "healthcare", "travel", "ecommerce", "customer_support")
        tool_names (str): Comma-separated list of tool names to include (e.g., "get_weather,search_web,calculator")
        num_tasks (int): Number of synthetic tasks to generate. Must be between 5 and 100. Default: 10
                        - 5-20 tasks: Single batch (fast, ~30-60s)
                        - 21-100 tasks: Multiple parallel batches (slower, ~60-120s per batch)
        difficulty_distribution (str): How to distribute task difficulty. Options: "balanced" (40% easy, 40% medium, 20% hard), "easy_only", "medium_only", "hard_only", "progressive" (50% easy, 30% medium, 20% hard). Default: "balanced"
        agent_type (str): Target agent type for tasks. Options: "tool" (ToolCallingAgent), "code" (CodeAgent), "both" (50/50 mix). Default: "both"

    Returns:
        str: JSON-formatted response with dataset_info (including batch statistics), tasks array (SMOLTRACE format), and usage_instructions
    """
    try:
        # Initialize Gemini client
        gemini_client = GeminiClient()

        # Validate inputs
        if num_tasks < 5 or num_tasks > 100:
            return json.dumps({
                "error": "num_tasks must be between 5 and 100",
                "num_tasks_provided": num_tasks
            }, indent=2)

        # Parse tool names
        tools = [tool.strip() for tool in tool_names.split(",") if tool.strip()]
        if len(tools) == 0:
            return json.dumps({
                "error": "At least one tool name must be provided",
                "tool_names_provided": tool_names
            }, indent=2)

        # Calculate distributions
        difficulty_counts = _calculate_difficulty_distribution(num_tasks, difficulty_distribution)
        agent_type_counts = _calculate_agent_type_distribution(num_tasks, agent_type)

        # Create generation prompt
        generation_prompt = f"""You are an expert at creating synthetic evaluation datasets for AI agents.

Generate {num_tasks} synthetic test tasks for the **{domain}** domain following the SMOLTRACE task format.

**Available Tools**: {", ".join(tools)}

**Difficulty Distribution**:
- Easy ({difficulty_counts['easy']} tasks): Single tool call, straightforward input, clear expected output
- Medium ({difficulty_counts['medium']} tasks): Multiple tool calls OR complex input parsing OR conditional logic
- Hard ({difficulty_counts['hard']} tasks): Multiple tools, complex reasoning, edge cases, error handling

**Agent Type Distribution**:
- Tool Agent ({agent_type_counts['tool']} tasks): Uses ToolCallingAgent - declarative tool calling
- Code Agent ({agent_type_counts['code']} tasks): Uses CodeAgent - writes Python code with tools

**SMOLTRACE Task Format** (required structure):
```json
{{
  "id": "string - unique identifier like '{domain.lower()}_{{tool}}_{{number}}'",
  "prompt": "string - clear, specific task description",
  "expected_tool": "string - the tool name that should be used",
  "expected_tool_calls": "integer - how many times the tool should be called (optional, default 1)",
  "difficulty": "string - 'easy', 'medium', or 'hard'",
  "agent_type": "string - 'tool' or 'code'",
  "expected_keywords": "array of strings - keywords expected in response (optional)"
}}
```

**Generation Guidelines**:
1. **Domain Specificity**: Make tasks realistic and specific to the {domain} domain
2. **Tool Usage**: Ensure each task requires using one of: {", ".join(tools)}
3. **Prompt Quality**: Write clear, unambiguous prompts that an agent can execute
4. **Expected Keywords**: Include 2-4 expected keywords for validation (optional but recommended)
5. **Variety**: Vary the tasks to cover different aspects of the domain

**IMPORTANT**: Return ONLY a valid JSON array of tasks. No explanatory text, no markdown formatting, no code blocks. Just the raw JSON array starting with [ and ending with ].

Generate exactly {num_tasks} tasks:"""

        print(f"[GENERATE_SYNTHETIC_DATASET] Generating {num_tasks} tasks for domain '{domain}'...")
        print(f"[GENERATE_SYNTHETIC_DATASET] Tools: {', '.join(tools)}")

        # Import required modules
        import asyncio
        import google.generativeai as genai

        # Determine batching strategy
        # Gemini can handle ~20 tasks per call with 8192 token output limit
        TASKS_PER_BATCH = 20
        num_batches = (num_tasks + TASKS_PER_BATCH - 1) // TASKS_PER_BATCH  # Ceiling division

        if num_batches > 1:
            print(f"[GENERATE_SYNTHETIC_DATASET] Large request detected. Splitting into {num_batches} parallel batches...")

        # Create batch generation tasks
        async def generate_batch(batch_num: int, batch_size: int, batch_difficulty: dict, batch_agent_type: dict):
            """Generate a single batch of tasks"""
            batch_prompt = f"""You are an expert at creating synthetic evaluation datasets for AI agents.

Generate {batch_size} synthetic test tasks for the **{domain}** domain following the SMOLTRACE task format.

**Available Tools**: {", ".join(tools)}

**Difficulty Distribution for this batch**:
- Easy ({batch_difficulty['easy']} tasks): Single tool call, straightforward input, clear expected output
- Medium ({batch_difficulty['medium']} tasks): Multiple tool calls OR complex input parsing OR conditional logic
- Hard ({batch_difficulty['hard']} tasks): Multiple tools, complex reasoning, edge cases, error handling

**Agent Type Distribution for this batch**:
- Tool Agent ({batch_agent_type['tool']} tasks): Uses ToolCallingAgent - declarative tool calling
- Code Agent ({batch_agent_type['code']} tasks): Uses CodeAgent - writes Python code with tools

**SMOLTRACE Task Format** (required structure):
```json
{{
  "id": "string - unique identifier like '{domain.lower()}_{{tool}}_batch{batch_num}_{{number}}'",
  "prompt": "string - clear, specific task description",
  "expected_tool": "string - the tool name that should be used",
  "expected_tool_calls": "integer - how many times the tool should be called (optional, default 1)",
  "difficulty": "string - 'easy', 'medium', or 'hard'",
  "agent_type": "string - 'tool' or 'code'",
  "expected_keywords": "array of strings - keywords expected in response (optional)"
}}
```

**Generation Guidelines**:
1. **Domain Specificity**: Make tasks realistic and specific to the {domain} domain
2. **Tool Usage**: Ensure each task requires using one of: {", ".join(tools)}
3. **Prompt Quality**: Write clear, unambiguous prompts that an agent can execute
4. **Expected Keywords**: Include 2-4 expected keywords for validation (optional but recommended)
5. **Variety**: Vary the tasks to cover different aspects of the domain
6. **Unique IDs**: Include 'batch{batch_num}' in task IDs to ensure uniqueness across batches

**IMPORTANT**: Return ONLY a valid JSON array of tasks. No explanatory text, no markdown formatting, no code blocks. Just the raw JSON array starting with [ and ending with ].

Generate exactly {batch_size} tasks:"""

            generation_config = {
                "temperature": 0.8,  # Higher for creativity and diversity
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 8192,
            }

            try:
                response = await asyncio.wait_for(
                    gemini_client.model.generate_content_async(
                        batch_prompt,
                        generation_config=generation_config
                    ),
                    timeout=120.0  # 120 seconds per batch for larger datasets
                )
                return response.text, None
            except Exception as e:
                return None, str(e)

        # Split difficulty and agent type distributions across batches
        def split_distribution(total_counts: dict, num_batches: int, batch_num: int, remaining_tasks: int):
            """Split distribution counts across batches fairly"""
            batch_counts = {}
            for key, total in total_counts.items():
                # Calculate fair share for this batch
                base_share = total // num_batches
                extra = 1 if batch_num < (total % num_batches) else 0
                batch_counts[key] = min(base_share + extra, remaining_tasks)
            return batch_counts

        # Generate all batches in parallel
        batch_tasks = []
        remaining_tasks = num_tasks

        for batch_num in range(num_batches):
            batch_size = min(TASKS_PER_BATCH, remaining_tasks)

            # Calculate distributions for this batch
            batch_difficulty = split_distribution(difficulty_counts, num_batches, batch_num, batch_size)
            batch_agent_type = split_distribution(agent_type_counts, num_batches, batch_num, batch_size)

            batch_tasks.append(generate_batch(batch_num, batch_size, batch_difficulty, batch_agent_type))
            remaining_tasks -= batch_size

        print(f"[GENERATE_SYNTHETIC_DATASET] Executing {num_batches} parallel Gemini API calls...")

        # Execute all batches in parallel
        batch_results = await asyncio.gather(*batch_tasks)

        # Combine and validate results
        all_tasks = []
        errors = []

        for batch_num, (response_text, error) in enumerate(batch_results):
            if error:
                errors.append(f"Batch {batch_num} failed: {error}")
                continue

            try:
                # Clean response (remove markdown if present)
                cleaned_response = response_text.strip()
                if cleaned_response.startswith("```"):
                    import re
                    match = re.search(r'```(?:json)?\s*\n(.*?)\n```', cleaned_response, re.DOTALL)
                    if match:
                        cleaned_response = match.group(1)

                # Parse JSON
                batch_tasks_parsed = json.loads(cleaned_response)

                if not isinstance(batch_tasks_parsed, list):
                    errors.append(f"Batch {batch_num} did not return a JSON array")
                    continue

                all_tasks.extend(batch_tasks_parsed)

            except json.JSONDecodeError as e:
                errors.append(f"Batch {batch_num} JSON parsing failed: {str(e)}")

        # Check if we got enough tasks
        if len(all_tasks) == 0:
            return json.dumps({
                "error": "All batches failed to generate tasks",
                "batch_errors": errors,
                "suggestion": "Check GEMINI_API_KEY and try again"
            }, indent=2)

        if errors:
            print(f"[GENERATE_SYNTHETIC_DATASET] Warning: Some batches failed: {errors}")

        print(f"[GENERATE_SYNTHETIC_DATASET] Successfully generated {len(all_tasks)} tasks across {num_batches} batch(es)")

        # Validate required fields for all tasks
        synthetic_tasks = all_tasks
        required_fields = ["id", "prompt", "expected_tool", "difficulty", "agent_type"]
        for i, task in enumerate(synthetic_tasks):
            missing_fields = [field for field in required_fields if field not in task]
            if missing_fields:
                return json.dumps({
                    "error": f"Task {i} is missing required fields: {missing_fields}",
                    "task": task
                }, indent=2)

        # Return formatted dataset with metadata
        result = {
            "dataset_info": {
                "domain": domain,
                "tools": tools,
                "num_tasks_requested": num_tasks,
                "num_tasks_generated": len(synthetic_tasks),
                "num_batches": num_batches,
                "batches_succeeded": num_batches - len(errors),
                "batches_failed": len(errors) if errors else 0,
                "batch_errors": errors if errors else None,
                "difficulty_distribution": difficulty_counts,
                "agent_type_distribution": agent_type_counts,
                "generated_at": datetime.now().isoformat(),
                "smoltrace_naming_convention": f"{{username}}/smoltrace-{domain.lower()}-tasks",
                "warning": f"⚠️ {len(errors)} batch(es) failed. Generated {len(synthetic_tasks)}/{num_tasks} tasks." if errors else None
            },
            "tasks": synthetic_tasks,
            "usage_instructions": {
                "format": "SMOLTRACE task dataset format",
                "naming_convention": f"Follow SMOLTRACE naming: {{username}}/smoltrace-{domain.lower()}-tasks or {{username}}/smoltrace-{domain.lower()}-tasks-v1 for versioning",
                "how_to_upload": [
                    "Option 1: Use the push_dataset_to_hub tool in this MCP server",
                    "Option 2: Manual upload with Python code (see example_code below)"
                ],
                "example_code": f"""from datasets import Dataset

# Extract tasks from this response
tasks = result["tasks"]

# Create and push to HuggingFace (following SMOLTRACE naming convention)
dataset = Dataset.from_list(tasks)
dataset.push_to_hub("your-username/smoltrace-{domain.lower()}-tasks")

# Use in SMOLTRACE evaluation
# smoltrace-eval --model openai/gpt-4 --dataset-name your-username/smoltrace-{domain.lower()}-tasks"""
            }
        }

        return json.dumps(result, indent=2, default=str)

    except Exception as e:
        return json.dumps({
            "error": f"Failed to generate synthetic dataset: {str(e)}",
            "domain": domain,
            "tools": tool_names
        }, indent=2)


@gr.mcp.tool()
async def push_dataset_to_hub(
    dataset_json: str,
    repo_name: str,
    hf_token: str = None,
    private: bool = False,
    prompt_template: str = None
) -> str:
    """
    Push a generated synthetic dataset to HuggingFace Hub with optional prompt template.

    This tool uploads datasets created by generate_synthetic_dataset (or any SMOLTRACE-format
    dataset) to HuggingFace Hub, making them ready for use in SMOLTRACE evaluations.
    Optionally includes a customized prompt template in the dataset card.

    **Naming Convention**: Repo name should follow SMOLTRACE convention:
    - Format: {username}/smoltrace-{domain}-tasks or {username}/smoltrace-{domain}-tasks-v{version}
    - Examples: "mycompany/smoltrace-finance-tasks", "alice/smoltrace-healthcare-tasks-v2"

    **Security**: Requires valid HuggingFace token with write permissions. If not provided,
    will use HF_TOKEN from environment variables or Settings.

    Args:
        dataset_json (str): JSON string containing the tasks array (from generate_synthetic_dataset output, use the "tasks" field)
        repo_name (str): HuggingFace repository name following SMOLTRACE naming: {username}/smoltrace-{domain}-tasks
        hf_token (str): HuggingFace API token with write permissions (optional - uses HF_TOKEN env var if not provided)
        private (bool): Whether to create a private dataset. Default: False (public)
        prompt_template (str): Optional YAML prompt template to include in dataset card (from generate_prompt_template)

    Returns:
        str: JSON response with upload status, dataset URL, and next steps
    """
    try:
        import os
        from huggingface_hub import HfApi

        # Use provided token or fallback to environment variable
        token = hf_token or os.environ.get("HF_TOKEN")
        if not token:
            return json.dumps({
                "error": "HuggingFace token required",
                "message": "Please provide hf_token parameter or set HF_TOKEN environment variable in Settings",
                "get_token": "https://huggingface.co/settings/tokens"
            }, indent=2)

        # Validate repo name follows SMOLTRACE convention
        if "smoltrace-" not in repo_name and "-tasks" not in repo_name:
            return json.dumps({
                "warning": "Repository name doesn't follow SMOLTRACE naming convention",
                "expected_format": "{username}/smoltrace-{domain}-tasks or {username}/smoltrace-{domain}-tasks-v{version}",
                "your_repo_name": repo_name,
                "recommendation": "Consider renaming to follow the convention for consistency with SMOLTRACE ecosystem",
                "proceeding": "Continuing with upload..."
            }, indent=2)

        # Parse dataset JSON
        try:
            tasks = json.loads(dataset_json)
            if not isinstance(tasks, list):
                return json.dumps({
                    "error": "dataset_json must be a JSON array of tasks",
                    "type_received": str(type(tasks))
                }, indent=2)
        except json.JSONDecodeError as e:
            return json.dumps({
                "error": "Invalid JSON in dataset_json",
                "parse_error": str(e)
            }, indent=2)

        # Validate task structure
        required_fields = ["id", "prompt", "expected_tool", "difficulty", "agent_type"]
        for i, task in enumerate(tasks):
            missing_fields = [field for field in required_fields if field not in task]
            if missing_fields:
                return json.dumps({
                    "error": f"Task {i} is missing required SMOLTRACE fields: {missing_fields}",
                    "task": task
                }, indent=2)

        # Create dataset and push to hub
        from datasets import Dataset

        dataset = Dataset.from_list(tasks)

        print(f"[PUSH_DATASET_TO_HUB] Uploading {len(tasks)} tasks to {repo_name}...")

        # Push to hub
        dataset.push_to_hub(
            repo_name,
            token=token,
            private=private
        )

        # If prompt template provided, add it to the dataset card
        if prompt_template and prompt_template.strip():
            try:
                print(f"[PUSH_DATASET_TO_HUB] Adding prompt template to dataset card...")

                # Create enhanced README with prompt template
                readme_content = f"""---
tags:
- smoltrace
- synthetic-data
- agent-evaluation
- mcp-generated
license: mit
---

# SMOLTRACE Synthetic Dataset

This dataset was generated using the TraceMind MCP Server's synthetic data generation tools.

## Dataset Info

- **Tasks**: {len(tasks)}
- **Format**: SMOLTRACE evaluation format
- **Generated**: AI-powered synthetic task generation

## Usage with SMOLTRACE

```python
from datasets import load_dataset

# Load dataset
dataset = load_dataset("{repo_name}")

# Use with SMOLTRACE
# smoltrace-eval --model openai/gpt-4 --dataset-name {repo_name}
```

## Prompt Template

This dataset includes a customized agent prompt template optimized for the domain and tools used.

### Template File

Save the following as `prompt_template.yaml`:

```yaml
{prompt_template}
```

### Using the Template

```python
from smolagents import ToolCallingAgent  # or CodeAgent

agent = ToolCallingAgent(
    tools=[...],  # Your tools
    model="openai/gpt-4",
    system_prompt_path="prompt_template.yaml"
)
```

## Dataset Structure

Each task contains:
- `id`: Unique task identifier
- `prompt`: Task description
- `expected_tool`: Tool the agent should use
- `difficulty`: Task complexity (easy/medium/hard)
- `agent_type`: Type of agent (tool/code)

## Generated with TraceMind MCP Server

🔗 [TraceMind MCP Server](https://huggingface.co/spaces/MCP-1st-Birthday/TraceMind-mcp-server)

Part of the MCP's 1st Birthday Hackathon project.
"""

                # Upload README to dataset repository
                api = HfApi()
                api.upload_file(
                    path_or_fileobj=readme_content.encode('utf-8'),
                    path_in_repo="README.md",
                    repo_id=repo_name,
                    repo_type="dataset",
                    token=token
                )

                print(f"[PUSH_DATASET_TO_HUB] Prompt template added to dataset card successfully")

            except Exception as readme_error:
                print(f"[WARNING] Failed to add prompt template to README: {readme_error}")
                # Don't fail the whole operation if README update fails

        # Return success response
        result = {
            "status": "success",
            "message": f"Successfully uploaded {len(tasks)} tasks to HuggingFace Hub" + (" with prompt template" if prompt_template else ""),
            "dataset_info": {
                "repository": repo_name,
                "num_tasks": len(tasks),
                "visibility": "private" if private else "public",
                "dataset_url": f"https://huggingface.co/datasets/{repo_name}",
                "includes_prompt_template": bool(prompt_template)
            },
            "next_steps": {
                "view_dataset": f"https://huggingface.co/datasets/{repo_name}",
                "use_in_smoltrace": f"smoltrace-eval --model openai/gpt-4 --dataset-name {repo_name}",
                "use_prompt_template": "Check the README.md for the customized prompt template" if prompt_template else "Generate a prompt template using generate_prompt_template tool",
                "share_with_team": f"Team members can access at https://huggingface.co/datasets/{repo_name}" if not private else "Dataset is private - share access via HuggingFace settings"
            }
        }

        return json.dumps(result, indent=2)

    except ImportError:
        return json.dumps({
            "error": "Required packages not installed",
            "missing_packages": "datasets, huggingface_hub",
            "install_command": "pip install datasets huggingface_hub"
        }, indent=2)
    except Exception as e:
        return json.dumps({
            "error": f"Failed to push dataset to hub: {str(e)}",
            "repo_name": repo_name
        }, indent=2)


# Helper functions for synthetic dataset generation
def _calculate_difficulty_distribution(num_tasks: int, difficulty_distribution: str) -> dict:
    """Calculate how many tasks of each difficulty to generate."""
    if difficulty_distribution == "balanced":
        easy = int(num_tasks * 0.4)
        medium = int(num_tasks * 0.4)
        hard = num_tasks - easy - medium
    elif difficulty_distribution == "easy_only":
        easy, medium, hard = num_tasks, 0, 0
    elif difficulty_distribution == "medium_only":
        easy, medium, hard = 0, num_tasks, 0
    elif difficulty_distribution == "hard_only":
        easy, medium, hard = 0, 0, num_tasks
    elif difficulty_distribution == "progressive":
        easy = int(num_tasks * 0.5)
        medium = int(num_tasks * 0.3)
        hard = num_tasks - easy - medium
    else:
        # Default to balanced
        easy = int(num_tasks * 0.4)
        medium = int(num_tasks * 0.4)
        hard = num_tasks - easy - medium

    return {"easy": easy, "medium": medium, "hard": hard}


def _calculate_agent_type_distribution(num_tasks: int, agent_type: str) -> dict:
    """Calculate how many tasks for each agent type to generate."""
    if agent_type == "tool":
        return {"tool": num_tasks, "code": 0}
    elif agent_type == "code":
        return {"tool": 0, "code": num_tasks}
    elif agent_type == "both":
        tool_count = num_tasks // 2
        code_count = num_tasks - tool_count
        return {"tool": tool_count, "code": code_count}
    else:
        # Default to both
        tool_count = num_tasks // 2
        code_count = num_tasks - tool_count
        return {"tool": tool_count, "code": code_count}


@gr.mcp.tool()
async def generate_prompt_template(
    domain: str,
    tool_names: str,
    agent_type: str = "tool"
) -> str:
    """
    Generate customized smolagents prompt template for a specific domain and tool set.

    This tool fetches the base prompt template from smolagents GitHub repository and uses
    Gemini AI to adapt it for your specific domain and tools. The result is a ready-to-use
    prompt template that you can use with SMOLTRACE evaluations.

    **Use Case**: When you generate synthetic datasets with `generate_synthetic_dataset`,
    use this tool to create a matching prompt template that agents can use during evaluation.
    This ensures your evaluation setup is complete and ready to run.

    **Integration**: The generated prompt template can be included in your HuggingFace dataset
    card, making it easy for anyone to run evaluations with your dataset.

    Args:
        domain (str): The domain for the prompt template (e.g., "finance", "healthcare", "customer_support")
        tool_names (str): Comma-separated list of tool names (e.g., "get_stock_price,calculate_roi,fetch_company_info")
        agent_type (str): Agent type - "tool" for ToolCallingAgent or "code" for CodeAgent. Default: "tool"

    Returns:
        str: JSON response containing the customized YAML prompt template and metadata
    """
    try:
        import aiohttp

        # Initialize Gemini client
        gemini_client = GeminiClient()

        # Validate agent_type
        if agent_type not in ["tool", "code"]:
            return json.dumps({
                "error": "agent_type must be 'tool' or 'code'",
                "agent_type_provided": agent_type
            }, indent=2)

        # Parse tool names
        tools = [tool.strip() for tool in tool_names.split(",") if tool.strip()]
        if len(tools) == 0:
            return json.dumps({
                "error": "At least one tool name must be provided",
                "tool_names_provided": tool_names
            }, indent=2)

        # Determine which template to fetch
        if agent_type == "tool":
            template_url = "https://raw.githubusercontent.com/huggingface/smolagents/refs/heads/main/src/smolagents/prompts/toolcalling_agent.yaml"
            template_name = "ToolCallingAgent"
        else:  # code
            template_url = "https://raw.githubusercontent.com/huggingface/smolagents/refs/heads/main/src/smolagents/prompts/code_agent.yaml"
            template_name = "CodeAgent"

        # Fetch the base template from GitHub
        async with aiohttp.ClientSession() as session:
            async with session.get(template_url) as response:
                if response.status != 200:
                    return json.dumps({
                        "error": f"Failed to fetch template from GitHub (status {response.status})",
                        "template_url": template_url
                    }, indent=2)

                base_template = await response.text()

        # Create customization prompt for Gemini
        customization_prompt = f"""You are an expert at creating agent prompt templates for smolagents.

I have a base {template_name} prompt template and need to customize it for a specific domain and set of tools.

**Domain**: {domain}
**Tools Available**: {", ".join(tools)}
**Agent Type**: {template_name}

**Base Template**:
```yaml
{base_template}
```

**Your Task**:
1. Analyze the base template structure
2. Customize it for the {domain} domain
3. Integrate the provided tools ({", ".join(tools)}) into the template
4. Add domain-specific instructions and examples
5. Ensure the tool descriptions are clear and domain-relevant

**Customization Guidelines**:
- Keep the YAML structure intact
- Update the introduction/system message to be domain-specific
- Add clear descriptions for each tool in the context of the {domain} domain
- Include domain-specific examples where appropriate
- Maintain the same placeholder variables (e.g., {{{{tool_descriptions}}}}, {{{{tools}}}})
- Ensure the template is immediately usable with SMOLTRACE

**Output Format**: Return ONLY the customized YAML template. No explanations, no markdown code blocks, just the raw YAML content.

Start your response with the YAML content immediately."""

        # Call Gemini to customize the template
        generation_config = {
            "temperature": 0.3,  # Lower temperature for more consistent formatting
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 4096,
        }

        response = await gemini_client.model.generate_content_async(
            customization_prompt,
            generation_config=generation_config
        )
        customized_template = response.text

        # Clean up the response (remove any markdown formatting if present)
        customized_template = customized_template.strip()
        if customized_template.startswith("```yaml"):
            customized_template = customized_template.replace("```yaml\n", "").replace("```", "").strip()
        elif customized_template.startswith("```"):
            customized_template = customized_template.replace("```\n", "").replace("```", "").strip()

        # Return response with metadata
        return json.dumps({
            "template_info": {
                "domain": domain,
                "tools": tools,
                "agent_type": agent_type,
                "template_name": template_name,
                "base_template_url": template_url,
                "customization_method": "Google Gemini 2.5 Flash"
            },
            "prompt_template": customized_template,
            "usage_instructions": f"""
# How to Use This Prompt Template

## In SMOLTRACE Evaluations

1. Save this template to a file (e.g., `{domain}_{agent_type}_agent.yaml`)
2. Use it with SMOLTRACE:
   ```python
   from smolagents import {template_name}

   agent = {template_name}(
       tools=[...],  # Your tools: {", ".join(tools)}
       model="openai/gpt-4",  # Or your preferred model
       system_prompt_path="{domain}_{agent_type}_agent.yaml"
   )
   ```

## In HuggingFace Dataset Card

Add this template to your dataset's README.md:
```markdown
## Agent Prompt Template

This dataset was designed for the following agent configuration:

- **Agent Type**: {template_name}
- **Domain**: {domain}
- **Tools**: {", ".join(tools)}

### Prompt Template (YAML)

See the `prompt_template.yaml` file in this repository.
```

## Testing the Template

Use this template when evaluating with the synthetic dataset you generated.
The template is pre-configured for the {domain} domain and includes all necessary
tool descriptions and examples.
"""
        }, indent=2)

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return json.dumps({
            "error": f"Failed to generate prompt template: {str(e)}",
            "error_details": error_details
        }, indent=2)
