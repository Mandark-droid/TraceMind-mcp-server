# TraceMind MCP Server - Complete API Documentation

This document provides comprehensive API reference for all MCP components provided by TraceMind MCP Server.

## Table of Contents

- [MCP Tools (11)](#mcp-tools)
  - [AI-Powered Analysis Tools](#ai-powered-analysis-tools)
  - [Token-Optimized Tools](#token-optimized-tools)
  - [Data Management Tools](#data-management-tools)
- [MCP Resources (3)](#mcp-resources)
- [MCP Prompts (3)](#mcp-prompts)
- [Error Handling](#error-handling)
- [Best Practices](#best-practices)

---

## MCP Tools

### AI-Powered Analysis Tools

These tools use Google Gemini 2.5 Flash to provide intelligent, context-aware analysis of agent evaluation data.

#### 1. analyze_leaderboard

Analyzes evaluation leaderboard data from HuggingFace datasets and generates AI-powered insights.

**Parameters:**
- `leaderboard_repo` (str): HuggingFace dataset repository
  - Default: `"kshitijthakkar/smoltrace-leaderboard"`
  - Format: `"username/dataset-name"`
- `metric_focus` (str): Primary metric to analyze
  - Options: `"overall"`, `"accuracy"`, `"cost"`, `"latency"`, `"co2"`
  - Default: `"overall"`
- `time_range` (str): Time period to analyze
  - Options: `"last_week"`, `"last_month"`, `"all_time"`
  - Default: `"last_week"`
- `top_n` (int): Number of top models to highlight
  - Range: 1-20
  - Default: 5

**Returns:** String containing AI-generated analysis with:
- Top performers by selected metric
- Trade-off analysis (e.g., accuracy vs cost)
- Trend identification
- Actionable recommendations

**Example Use Case:**
Before choosing a model for production, get AI-powered insights on which configuration offers the best cost/performance for your requirements.

**Example Call:**
```python
result = await analyze_leaderboard(
    leaderboard_repo="kshitijthakkar/smoltrace-leaderboard",
    metric_focus="cost",
    time_range="last_week",
    top_n=5
)
```

**Example Response:**
```
Based on 247 evaluations in the past week:

Top Performers (Cost Focus):
1. meta-llama/Llama-3.1-8B: $0.002 per run, 93.4% accuracy
2. mistralai/Mistral-7B: $0.003 per run, 91.2% accuracy
3. openai/gpt-3.5-turbo: $0.008 per run, 94.1% accuracy

Trade-off Analysis:
- Llama-3.1 offers best cost/performance ratio at 25x cheaper than GPT-4
- GPT-4 leads in accuracy (95.8%) but costs $0.05 per run
- For production with 1M runs/month: Llama-3.1 saves $48,000 vs GPT-4

Recommendations:
- Cost-sensitive: Use Llama-3.1-8B (93% accuracy, minimal cost)
- Accuracy-critical: Use GPT-4 (96% accuracy, premium cost)
- Balanced: Use GPT-3.5-Turbo (94% accuracy, moderate cost)
```

---

#### 2. debug_trace

Analyzes OpenTelemetry trace data and answers specific questions about agent execution.

**Parameters:**
- `trace_dataset` (str): HuggingFace dataset containing traces
  - Format: `"username/smoltrace-traces-model"`
  - Must contain "smoltrace-" prefix
- `trace_id` (str): Specific trace ID to analyze
  - Format: `"trace_abc123"`
- `question` (str): Question about the trace
  - Examples: "Why was tool X called twice?", "Which step took the most time?"
- `include_metrics` (bool): Include GPU metrics in analysis
  - Default: `true`

**Returns:** String containing AI analysis of the trace with:
- Answer to the specific question
- Relevant span details
- Performance insights
- GPU metrics (if available and requested)

**Example Use Case:**
When an agent test fails, understand exactly what happened without manually parsing trace spans.

**Example Call:**
```python
result = await debug_trace(
    trace_dataset="kshitij/smoltrace-traces-gpt4",
    trace_id="trace_abc123",
    question="Why was the search tool called twice?",
    include_metrics=True
)
```

**Example Response:**
```
Based on trace analysis:

Answer:
The agent called the search_web tool twice due to an iterative reasoning pattern:

1. First call (span_003 at 14:23:19.000):
   - Query: "weather in Tokyo"
   - Duration: 890ms
   - Result: 5 results, oldest was 2 days old

2. Second call (span_005 at 14:23:21.200):
   - Query: "latest weather in Tokyo"
   - Duration: 1200ms
   - Modified reasoning: LLM determined first results were stale

Performance Impact:
- Added 2.09s to total execution time
- Cost increase: +$0.0003 (tokens for second reasoning step)
- This is normal behavior for tool-calling agents with iterative reasoning

GPU Metrics:
- N/A (API model, no GPU used)
```

---

#### 3. estimate_cost

Predicts costs, duration, and environmental impact before running evaluations.

**Parameters:**
- `model` (str, required): Model name to evaluate
  - Format: `"provider/model-name"` (e.g., `"openai/gpt-4"`, `"meta-llama/Llama-3.1-8B"`)
- `agent_type` (str): Type of agent evaluation
  - Options: `"tool"`, `"code"`, `"both"`
  - Default: `"both"`
- `num_tests` (int): Number of test cases
  - Range: 1-10000
  - Default: 100
- `hardware` (str): Hardware type
  - Options: `"auto"`, `"cpu"`, `"gpu_a10"`, `"gpu_h200"`
  - Default: `"auto"` (auto-selects based on model)

**Returns:** String containing cost estimate with:
- LLM API costs (for API models)
- HuggingFace Jobs compute costs (for local models)
- Estimated duration
- CO2 emissions estimate
- Hardware recommendations

**Example Use Case:**
Compare the cost of evaluating GPT-4 vs Llama-3.1 across 1000 tests before committing resources.

**Example Call:**
```python
result = await estimate_cost(
    model="openai/gpt-4",
    agent_type="both",
    num_tests=1000,
    hardware="auto"
)
```

**Example Response:**
```
Cost Estimate for openai/gpt-4:

LLM API Costs:
- Estimated tokens per test: 1,500
- Token cost: $0.03/1K input, $0.06/1K output
- Total LLM cost: $50.00 (1000 tests)

Compute Costs:
- Recommended hardware: cpu-basic (API model)
- HF Jobs cost: ~$0.05/hr
- Estimated duration: 45 minutes
- Total compute cost: $0.04

Total Cost: $50.04
Cost per test: $0.05
CO2 emissions: ~0.5g (API calls, minimal compute)

Recommendations:
- This is an API model, CPU hardware is sufficient
- For cost optimization, consider Llama-3.1-8B (25x cheaper)
- Estimated runtime: 45 minutes for 1000 tests
```

---

#### 4. compare_runs

Compares two evaluation runs with AI-powered analysis across multiple dimensions.

**Parameters:**
- `run_id_1` (str, required): First run ID from leaderboard
- `run_id_2` (str, required): Second run ID from leaderboard
- `leaderboard_repo` (str): Leaderboard dataset repository
  - Default: `"kshitijthakkar/smoltrace-leaderboard"`
- `focus` (str): Comparison focus area
  - Options:
    - `"comprehensive"`: All dimensions
    - `"cost"`: Cost efficiency and ROI
    - `"performance"`: Speed and accuracy trade-offs
    - `"eco_friendly"`: Environmental impact
  - Default: `"comprehensive"`

**Returns:** String containing AI comparison with:
- Success rate comparison with statistical significance
- Cost efficiency analysis
- Speed comparison
- Environmental impact (CO2 emissions)
- GPU efficiency (for GPU jobs)

**Example Use Case:**
After running evaluations with two different models, compare them head-to-head to determine which is better for production deployment.

**Example Call:**
```python
result = await compare_runs(
    run_id_1="run_abc123",
    run_id_2="run_def456",
    leaderboard_repo="kshitijthakkar/smoltrace-leaderboard",
    focus="cost"
)
```

**Example Response:**
```
Comparison: GPT-4 vs Llama-3.1-8B (Cost Focus)

Success Rates:
- GPT-4: 95.8% (96/100 tests)
- Llama-3.1: 93.4% (93/100 tests)
- Difference: +2.4% for GPT-4 (statistically significant, p<0.05)

Cost Efficiency:
- GPT-4: $0.05 per test, $0.052 per successful test
- Llama-3.1: $0.002 per test, $0.0021 per successful test
- Cost ratio: GPT-4 is 25x more expensive

ROI Analysis:
- For 1M evaluations/month:
  - GPT-4: $50,000/month, 958K successes
  - Llama-3.1: $2,000/month, 934K successes
- GPT-4 provides 24K more successes for $48K more cost
- Cost per additional success: $2.00

Recommendation (Cost Focus):
Use Llama-3.1-8B for cost-sensitive workloads where 93% accuracy is acceptable.
Switch to GPT-4 only for accuracy-critical tasks where the 2.4% improvement justifies 25x cost.
```

---

#### 5. analyze_results

Analyzes detailed test results and provides optimization recommendations.

**Parameters:**
- `results_repo` (str, required): HuggingFace dataset containing results
  - Format: `"username/smoltrace-results-model-timestamp"`
  - Must contain "smoltrace-results-" prefix
- `analysis_focus` (str): Focus area for analysis
  - Options: `"failures"`, `"performance"`, `"cost"`, `"comprehensive"`
  - Default: `"comprehensive"`
- `max_rows` (int): Maximum test cases to analyze
  - Range: 10-500
  - Default: 100

**Returns:** String containing AI analysis with:
- Failure patterns and root causes
- Performance bottlenecks in specific test cases
- Cost optimization opportunities
- Tool usage patterns
- Task-specific insights (which types work well vs poorly)
- Actionable optimization recommendations

**Example Use Case:**
After running an evaluation, analyze the detailed test results to understand why certain tests are failing and get specific recommendations for improving success rate.

**Example Call:**
```python
result = await analyze_results(
    results_repo="kshitij/smoltrace-results-gpt4-20251120",
    analysis_focus="failures",
    max_rows=100
)
```

**Example Response:**
```
Analysis of Test Results (100 tests analyzed)

Overall Statistics:
- Success Rate: 89% (89/100 tests passed)
- Average Duration: 3.2s per test
- Total Cost: $4.50 ($0.045 per test)

Failure Analysis (11 failures):
1. Tool Not Found (6 failures):
   - Test IDs: task_012, task_045, task_067, task_089, task_091, task_093
   - Pattern: All failed tests required the 'get_weather' tool
   - Root Cause: Tool definition missing or incorrect name
   - Fix: Ensure 'get_weather' tool is available in agent's tool list

2. Timeout (3 failures):
   - Test IDs: task_034, task_071, task_088
   - Pattern: Complex multi-step tasks with >5 tool calls
   - Root Cause: Exceeding 30s timeout limit
   - Fix: Increase timeout to 60s or simplify complex tasks

3. Incorrect Response (2 failures):
   - Test IDs: task_056, task_072
   - Pattern: Math calculation tasks
   - Root Cause: Model hallucinating numbers instead of using calculator tool
   - Fix: Update prompt to emphasize tool usage for calculations

Performance Insights:
- Fast tasks (<2s): 45 tests - Simple single-tool calls
- Slow tasks (>5s): 12 tests - Multi-step reasoning with 3+ tools
- Optimal duration: 2-3s for most tasks

Cost Optimization:
- High-cost tests: task_023 ($0.12) - Used 4K tokens
- Low-cost tests: task_087 ($0.008) - Used 180 tokens
- Recommendation: Optimize prompt to reduce token usage by 20%

Recommendations:
1. Add missing 'get_weather' tool → Fixes 6 failures
2. Increase timeout from 30s to 60s → Fixes 3 failures
3. Strengthen calculator tool instruction → Fixes 2 failures
4. Expected improvement: 89% → 100% success rate
```

---

### Token-Optimized Tools

These tools are specifically designed to minimize token usage when querying leaderboard data.

#### 6. get_top_performers

Get top N performing models from leaderboard with 90% token reduction.

**Performance Optimization:** Returns only top N models instead of loading the full leaderboard dataset (51 runs), resulting in **90% token reduction**.

**When to Use:** Perfect for queries like "Which model is leading?", "Show me the top 5 models".

**Parameters:**
- `leaderboard_repo` (str): HuggingFace dataset repository
  - Default: `"kshitijthakkar/smoltrace-leaderboard"`
- `metric` (str): Metric to rank by
  - Options: `"success_rate"`, `"total_cost_usd"`, `"avg_duration_ms"`, `"co2_emissions_g"`
  - Default: `"success_rate"`
- `top_n` (int): Number of top models to return
  - Range: 1-20
  - Default: 5

**Returns:** JSON string with:
- Metric used for ranking
- Ranking order (ascending/descending)
- Total runs in leaderboard
- Array of top performers with 10 essential fields

**Benefits:**
- ✅ Token Reduction: 90% fewer tokens vs full dataset
- ✅ Ready to Use: Properly formatted JSON
- ✅ Pre-Sorted: Already ranked by chosen metric
- ✅ Essential Data Only: 10 fields vs 20+ in full dataset

**Example Call:**
```python
result = await get_top_performers(
    leaderboard_repo="kshitijthakkar/smoltrace-leaderboard",
    metric="total_cost_usd",
    top_n=3
)
```

**Example Response:**
```json
{
  "metric": "total_cost_usd",
  "order": "ascending",
  "total_runs": 51,
  "top_performers": [
    {
      "run_id": "run_001",
      "model": "meta-llama/Llama-3.1-8B",
      "success_rate": 93.4,
      "total_cost_usd": 0.002,
      "avg_duration_ms": 2100,
      "agent_type": "both",
      "provider": "transformers",
      "submitted_by": "kshitij",
      "timestamp": "2025-11-20T10:30:00Z",
      "total_tests": 100
    },
    ...
  ]
}
```

---

#### 7. get_leaderboard_summary

Get high-level leaderboard statistics with 99% token reduction.

**Performance Optimization:** Returns only aggregated statistics instead of raw data, resulting in **99% token reduction**.

**When to Use:** Perfect for overview queries like "How many runs are in the leaderboard?", "What's the average success rate?".

**Parameters:**
- `leaderboard_repo` (str): HuggingFace dataset repository
  - Default: `"kshitijthakkar/smoltrace-leaderboard"`

**Returns:** JSON string with:
- Total runs count
- Unique models and submitters
- Overall statistics (avg/best/worst success rates, avg cost, avg duration, total CO2)
- Breakdown by agent type
- Breakdown by provider
- Top 3 models by success rate

**Benefits:**
- ✅ Extreme Token Reduction: 99% fewer tokens
- ✅ Ready to Use: Properly formatted JSON
- ✅ Comprehensive Stats: Averages, distributions, breakdowns
- ✅ Quick Insights: Perfect for overview questions

**Example Call:**
```python
result = await get_leaderboard_summary(
    leaderboard_repo="kshitijthakkar/smoltrace-leaderboard"
)
```

**Example Response:**
```json
{
  "total_runs": 51,
  "unique_models": 12,
  "unique_submitters": 3,
  "overall_stats": {
    "avg_success_rate": 89.2,
    "best_success_rate": 95.8,
    "worst_success_rate": 78.3,
    "avg_cost_usd": 0.012,
    "avg_duration_ms": 3200,
    "total_co2_g": 45.6
  },
  "by_agent_type": {
    "tool": {"count": 20, "avg_success_rate": 88.5},
    "code": {"count": 18, "avg_success_rate": 87.2},
    "both": {"count": 13, "avg_success_rate": 92.1}
  },
  "by_provider": {
    "litellm": {"count": 30, "avg_success_rate": 91.3},
    "transformers": {"count": 21, "avg_success_rate": 86.4}
  },
  "top_3_models": [
    {"model": "openai/gpt-4", "success_rate": 95.8},
    {"model": "anthropic/claude-3", "success_rate": 94.1},
    {"model": "meta-llama/Llama-3.1-8B", "success_rate": 93.4}
  ]
}
```

---

### Data Management Tools

#### 8. get_dataset

Loads SMOLTRACE datasets from HuggingFace and returns raw data as JSON.

**⚠️ Important:** For leaderboard queries, prefer using `get_top_performers()` or `get_leaderboard_summary()` to avoid token bloat!

**Security Restriction:** Only datasets with "smoltrace-" in the repository name are allowed.

**Parameters:**
- `dataset_repo` (str, required): HuggingFace dataset repository
  - Must contain "smoltrace-" prefix
  - Format: `"username/smoltrace-type-model"`
- `split` (str): Dataset split to load
  - Default: `"train"`
- `limit` (int): Maximum rows to return
  - Range: 1-200
  - Default: 100

**Returns:** JSON string with:
- Total rows in dataset
- List of column names
- Array of data rows (up to `limit`)

**Primary Use Cases:**
- Load `smoltrace-results-*` datasets for test case details
- Load `smoltrace-traces-*` datasets for OpenTelemetry data
- Load `smoltrace-metrics-*` datasets for GPU metrics
- **NOT recommended** for leaderboard queries (use optimized tools)

**Example Call:**
```python
result = await get_dataset(
    dataset_repo="kshitij/smoltrace-results-gpt4",
    split="train",
    limit=50
)
```

---

#### 9. generate_synthetic_dataset

Creates domain-specific test datasets for SMOLTRACE evaluations using AI.

**Parameters:**
- `domain` (str, required): Domain for tasks
  - Examples: "e-commerce", "customer service", "finance", "healthcare"
- `tools` (list[str], required): Available tools
  - Example: `["search_web", "get_weather", "calculator"]`
- `num_tasks` (int): Number of tasks to generate
  - Range: 1-100
  - Default: 20
- `difficulty_distribution` (str): Task difficulty mix
  - Options: `"balanced"`, `"easy_only"`, `"medium_only"`, `"hard_only"`, `"progressive"`
  - Default: `"balanced"`
- `agent_type` (str): Target agent type
  - Options: `"tool"`, `"code"`, `"both"`
  - Default: `"both"`

**Returns:** JSON string with:
- `dataset_info`: Metadata (domain, tools, counts, timestamp)
- `tasks`: Array of SMOLTRACE-formatted tasks
- `usage_instructions`: Guide for HuggingFace upload and SMOLTRACE usage

**SMOLTRACE Task Format:**
```json
{
  "id": "unique_identifier",
  "prompt": "Clear, specific task for the agent",
  "expected_tool": "tool_name",
  "expected_tool_calls": 1,
  "difficulty": "easy|medium|hard",
  "agent_type": "tool|code",
  "expected_keywords": ["keyword1", "keyword2"]
}
```

**Difficulty Calibration:**
- **Easy** (40%): Single tool call, straightforward input
- **Medium** (40%): Multiple tool calls OR complex input parsing
- **Hard** (20%): Multiple tools, complex reasoning, edge cases

**Enterprise Use Cases:**
- Custom Tools: Benchmark proprietary APIs
- Industry-Specific: Generate tasks for finance, healthcare, legal
- Internal Workflows: Test company-specific processes

**Example Call:**
```python
result = await generate_synthetic_dataset(
    domain="customer service",
    tools=["search_knowledge_base", "create_ticket", "send_email"],
    num_tasks=50,
    difficulty_distribution="balanced",
    agent_type="tool"
)
```

---

#### 10. push_dataset_to_hub

Upload generated datasets to HuggingFace Hub with proper formatting.

**Parameters:**
- `dataset_name` (str, required): Repository name on HuggingFace
  - Format: `"username/my-dataset"`
- `data` (str or list, required): Dataset content
  - Can be JSON string or list of dictionaries
- `description` (str): Dataset description for card
  - Default: Auto-generated
- `private` (bool): Make dataset private
  - Default: `False`

**Returns:** Success message with dataset URL

**Example Workflow:**
1. Generate synthetic dataset with `generate_synthetic_dataset`
2. Review and modify tasks if needed
3. Upload to HuggingFace with `push_dataset_to_hub`
4. Use in SMOLTRACE evaluations or share with team

**Example Call:**
```python
result = await push_dataset_to_hub(
    dataset_name="kshitij/my-custom-evaluation",
    data=generated_tasks,
    description="Custom evaluation dataset for e-commerce agents",
    private=False
)
```

---

#### 11. generate_prompt_template

Generate customized smolagents prompt template for a specific domain and tool set.

**Parameters:**
- `domain` (str, required): Domain for the prompt template
  - Examples: `"finance"`, `"healthcare"`, `"customer_support"`, `"e-commerce"`
- `tool_names` (str, required): Comma-separated list of tool names
  - Format: `"tool1,tool2,tool3"`
  - Example: `"get_stock_price,calculate_roi,fetch_company_info"`
- `agent_type` (str): Agent type
  - Options: `"tool"` (ToolCallingAgent), `"code"` (CodeAgent)
  - Default: `"tool"`

**Returns:** JSON response containing:
- Customized YAML prompt template
- Metadata (domain, tools, agent_type, timestamp)
- Usage instructions

**Use Case:**
When you generate synthetic datasets with `generate_synthetic_dataset`, use this tool to create a matching prompt template that agents can use during evaluation. This ensures your evaluation setup is complete and ready to run.

**Integration:**
The generated prompt template can be included in your HuggingFace dataset card, making it easy for anyone to run evaluations with your dataset.

**Example Call:**
```python
result = await generate_prompt_template(
    domain="customer_support",
    tool_names="search_knowledge_base,create_ticket,send_email,escalate_to_human",
    agent_type="tool"
)
```

**Example Response:**
```json
{
  "prompt_template": "---\nname: customer_support_agent\ndescription: An AI agent for customer support tasks...\n\ninstructions: |-\n  You are a helpful customer support agent...\n  \n  Available tools:\n  - search_knowledge_base: Search the knowledge base...\n  - create_ticket: Create a support ticket...\n  ...",
  "metadata": {
    "domain": "customer_support",
    "tools": ["search_knowledge_base", "create_ticket", "send_email", "escalate_to_human"],
    "agent_type": "tool",
    "base_template": "ToolCallingAgent",
    "timestamp": "2025-11-21T10:30:00Z"
  },
  "usage_instructions": "1. Save the prompt_template to a file (e.g., customer_support_prompt.yaml)\n2. Use with SMOLTRACE: smoltrace-eval --model your-model --prompt-file customer_support_prompt.yaml\n3. Or include in your dataset card for easy evaluation"
}
```

---

## MCP Resources

Resources provide direct data access without AI analysis. Access via URI scheme.

### 1. leaderboard://{repo}

Direct access to raw leaderboard data in JSON format.

**URI Format:**
```
leaderboard://username/dataset-name
```

**Example:**
```
GET leaderboard://kshitijthakkar/smoltrace-leaderboard
```

**Returns:** JSON array with all evaluation runs, including:
- run_id, model, agent_type, provider
- success_rate, total_tests, successful_tests, failed_tests
- avg_duration_ms, total_tokens, total_cost_usd, co2_emissions_g
- results_dataset, traces_dataset, metrics_dataset (references)
- timestamp, submitted_by, hf_job_id

---

### 2. trace://{trace_id}/{repo}

Direct access to trace data with OpenTelemetry spans.

**URI Format:**
```
trace://trace_id/username/dataset-name
```

**Example:**
```
GET trace://trace_abc123/kshitij/agent-traces-gpt4
```

**Returns:** JSON with:
- traceId
- spans array (spanId, parentSpanId, name, kind, startTime, endTime, attributes, status)

---

### 3. cost://model/{model_name}

Model pricing and hardware cost information.

**URI Format:**
```
cost://model/provider/model-name
```

**Example:**
```
GET cost://model/openai/gpt-4
```

**Returns:** JSON with:
- Model pricing (input/output token costs)
- Recommended hardware tier
- Estimated compute costs
- CO2 emissions per 1K tokens

---

## MCP Prompts

Prompts provide reusable templates for standardized interactions.

### 1. analysis_prompt

Templates for different analysis types.

**Parameters:**
- `analysis_type` (str): Type of analysis
  - Options: `"leaderboard"`, `"cost"`, `"performance"`, `"trace"`
- `focus_area` (str): Specific focus
  - Options: `"overall"`, `"cost"`, `"accuracy"`, `"speed"`, `"eco"`
- `detail_level` (str): Level of detail
  - Options: `"summary"`, `"detailed"`, `"comprehensive"`

**Returns:** Formatted prompt string for use with AI tools

**Example:**
```python
prompt = analysis_prompt(
    analysis_type="leaderboard",
    focus_area="cost",
    detail_level="detailed"
)
# Returns: "Provide a detailed analysis of cost efficiency in the leaderboard..."
```

---

### 2. debug_prompt

Templates for debugging scenarios.

**Parameters:**
- `debug_type` (str): Type of debugging
  - Options: `"failure"`, `"performance"`, `"tool_calling"`, `"reasoning"`
- `context` (str): Additional context
  - Options: `"test_failure"`, `"timeout"`, `"unexpected_tool"`, `"reasoning_loop"`

**Returns:** Formatted prompt string

**Example:**
```python
prompt = debug_prompt(
    debug_type="performance",
    context="tool_calling"
)
# Returns: "Analyze tool calling performance. Identify which tools are slow..."
```

---

### 3. optimization_prompt

Templates for optimization goals.

**Parameters:**
- `optimization_goal` (str): Optimization target
  - Options: `"cost"`, `"speed"`, `"accuracy"`, `"co2"`
- `constraints` (str): Constraints to respect
  - Options: `"maintain_quality"`, `"no_accuracy_loss"`, `"budget_limit"`, `"time_limit"`

**Returns:** Formatted prompt string

**Example:**
```python
prompt = optimization_prompt(
    optimization_goal="cost",
    constraints="maintain_quality"
)
# Returns: "Analyze this evaluation setup and recommend cost optimizations..."
```

---

## Error Handling

### Common Error Responses

**Invalid Dataset Repository:**
```json
{
  "error": "Dataset must contain 'smoltrace-' prefix for security",
  "provided": "username/invalid-dataset"
}
```

**Dataset Not Found:**
```json
{
  "error": "Dataset not found on HuggingFace",
  "repository": "username/smoltrace-nonexistent"
}
```

**API Rate Limit:**
```json
{
  "error": "Gemini API rate limit exceeded",
  "retry_after": 60
}
```

**Invalid Parameters:**
```json
{
  "error": "Invalid parameter value",
  "parameter": "top_n",
  "value": 50,
  "allowed_range": "1-20"
}
```

---

## Best Practices

### 1. Token Optimization

**DO:**
- Use `get_top_performers()` for "top N" queries (90% token reduction)
- Use `get_leaderboard_summary()` for overview queries (99% token reduction)
- Set appropriate `limit` when using `get_dataset()`

**DON'T:**
- Use `get_dataset()` for leaderboard queries (loads all 51 runs)
- Request more data than needed
- Ignore token optimization tools

### 2. AI Tool Usage

**DO:**
- Use AI tools (`analyze_leaderboard`, `debug_trace`) for complex analysis
- Provide specific questions to `debug_trace` for focused answers
- Use `focus` parameter in `compare_runs` for targeted comparisons

**DON'T:**
- Use AI tools for simple data retrieval (use resources instead)
- Make vague requests (be specific for better results)

### 3. Dataset Security

**DO:**
- Only use datasets with "smoltrace-" prefix
- Verify dataset exists before requesting
- Use public datasets or authenticate for private ones

**DON'T:**
- Try to access arbitrary HuggingFace datasets
- Share private dataset URLs without authentication

### 4. Cost Management

**DO:**
- Use `estimate_cost` before running large evaluations
- Compare cost estimates across different models
- Consider token-optimized tools to reduce API costs

**DON'T:**
- Skip cost estimation for expensive operations
- Ignore hardware recommendations
- Overlook CO2 emissions in decision-making

---

## Support

For issues or questions:
- 📧 GitHub Issues: [TraceMind-mcp-server/issues](https://github.com/Mandark-droid/TraceMind-mcp-server/issues)
- 💬 HF Discord: `#agents-mcp-hackathon-winter25`
- 🏷️ Tag: `building-mcp-track-enterprise`
