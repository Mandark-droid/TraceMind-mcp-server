"""
Gemini Client for TraceMind MCP Server

Handles all interactions with Google Gemini 2.5 Flash API
"""

import os
import google.generativeai as genai
from typing import Optional, Dict, Any, List
import json

class GeminiClient:
    """Client for Google Gemini API"""

    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-2.5-flash-lite"):
        """
        Initialize Gemini client

        Args:
            api_key: Gemini API key (defaults to GEMINI_API_KEY env var)
            model_name: Model to use (default: gemini-2.5-flash-lite, can also use gemini-2.5-flash)
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")

        # Configure API
        genai.configure(api_key=self.api_key)

        # Initialize model
        self.model = genai.GenerativeModel(model_name)

        # Generation config for consistent outputs
        # Reduced max_output_tokens for faster responses on HF Spaces
        self.generation_config = {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 4096,  # Reduced from 8192 for faster responses
        }

        # Request timeout (30 seconds for HF Spaces compatibility)
        self.request_timeout = 30

    async def analyze_with_context(
        self,
        data: Dict[str, Any],
        analysis_type: str,
        specific_question: Optional[str] = None
    ) -> str:
        """
        Analyze data with Gemini, providing context about the analysis type

        Args:
            data: Data to analyze (will be converted to JSON)
            analysis_type: Type of analysis ("leaderboard", "trace", "cost_estimate")
            specific_question: Optional specific question to answer

        Returns:
            Markdown-formatted analysis
        """

        # Build prompt based on analysis type
        if analysis_type == "leaderboard":
            system_prompt = """You are an expert AI agent performance analyst.

You are analyzing evaluation leaderboard data from agent benchmarks. Your task is to:
1. Identify top performers across key metrics (accuracy, cost, latency, CO2)
2. Explain trade-offs between different approaches (API vs local models, GPU types)
3. Identify trends and patterns
4. Provide actionable recommendations

Focus on insights that would help developers choose the right agent configuration for their use case.

Format your response in clear markdown with sections for:
- **Top Performers**
- **Key Insights**
- **Trade-offs**
- **Recommendations**
"""

        elif analysis_type == "trace":
            system_prompt = """You are an expert agent debugging specialist.

You are analyzing OpenTelemetry trace data from agent execution. Your task is to:
1. Understand the sequence of operations (LLM calls, tool calls, etc.)
2. Identify performance bottlenecks or inefficiencies
3. Explain why certain decisions were made
4. Answer the specific question asked

Focus on providing clear explanations that help developers understand agent behavior.

Format your response in clear markdown with relevant code snippets and timing information.
"""

        elif analysis_type == "cost_estimate":
            system_prompt = """You are an expert in LLM cost optimization and cloud resource estimation.

You are estimating the cost of running agent evaluations. Your task is to:
1. Calculate LLM API costs based on token usage patterns
2. Estimate HuggingFace Jobs compute costs
3. Predict CO2 emissions
4. Provide cost optimization recommendations

Focus on giving accurate estimates with clear breakdowns.

Format your response in clear markdown with cost breakdowns and optimization tips.
"""

        else:
            system_prompt = "You are a helpful AI assistant analyzing agent evaluation data."

        # Build user prompt
        data_json = json.dumps(data, indent=2)

        user_prompt = f"{system_prompt}\n\n**Data to analyze:**\n```json\n{data_json}\n```\n\n"

        if specific_question:
            user_prompt += f"**Specific question:** {specific_question}\n\n"

        user_prompt += "Provide your analysis:"

        # Generate response with timeout handling
        try:
            import asyncio

            # Add timeout to prevent hanging on HF Spaces
            response = await asyncio.wait_for(
                self.model.generate_content_async(
                    user_prompt,
                    generation_config=self.generation_config
                ),
                timeout=self.request_timeout
            )

            return response.text

        except asyncio.TimeoutError:
            return "⏱️ **Analysis timed out**. The request took too long. Try analyzing a smaller dataset or simplifying the query."
        except Exception as e:
            return f"❌ **Error generating analysis**: {str(e)}"

    async def generate_summary(
        self,
        text: str,
        max_words: int = 100
    ) -> str:
        """
        Generate a concise summary of text

        Args:
            text: Text to summarize
            max_words: Maximum words in summary

        Returns:
            Summary text
        """
        prompt = f"Summarize the following in {max_words} words or less:\n\n{text}"

        try:
            response = await self.model.generate_content_async(prompt)
            return response.text
        except Exception as e:
            return f"Error generating summary: {str(e)}"

    async def answer_question(
        self,
        context: str,
        question: str
    ) -> str:
        """
        Answer a question given context

        Args:
            context: Context information
            question: Question to answer

        Returns:
            Answer
        """
        prompt = f"""Based on the following context, answer the question.

**Context:**
{context}

**Question:** {question}

**Answer:**"""

        try:
            response = await self.model.generate_content_async(
                prompt,
                generation_config=self.generation_config
            )
            return response.text
        except Exception as e:
            return f"Error answering question: {str(e)}"
