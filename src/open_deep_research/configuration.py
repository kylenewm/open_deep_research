"""Configuration management for the Open Deep Research system."""

import os
from enum import Enum
from typing import Any, List, Optional

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field


class SearchAPI(Enum):
    """Enumeration of available search API providers."""
    
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    TAVILY = "tavily"
    NONE = "none"

class MCPConfig(BaseModel):
    """Configuration for Model Context Protocol (MCP) servers."""
    
    url: Optional[str] = Field(
        default=None,
        optional=True,
    )
    """The URL of the MCP server"""
    tools: Optional[List[str]] = Field(
        default=None,
        optional=True,
    )
    """The tools to make available to the LLM"""
    auth_required: Optional[bool] = Field(
        default=False,
        optional=True,
    )
    """Whether the MCP server requires authentication"""

class Configuration(BaseModel):
    """Main configuration class for the Deep Research agent."""
    
    # Test Mode - Reduces iterations and costs for faster testing
    test_mode: bool = Field(
        default=False,
        metadata={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": False,
                "description": "Enable test mode for faster iteration. Limits: max_researcher_iterations=2, max_react_tool_calls=3, max_concurrent_research_units=2"
            }
        }
    )
    
    # General Configuration
    max_structured_output_retries: int = Field(
        default=3,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 3,
                "min": 1,
                "max": 10,
                "description": "Maximum number of retries for structured output calls from models"
            }
        }
    )
    allow_clarification: bool = Field(
        default=True,
        metadata={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": True,
                "description": "Whether to allow the researcher to ask the user clarifying questions before starting research"
            }
        }
    )
    max_concurrent_research_units: int = Field(
        default=5,
        metadata={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 5,
                "min": 1,
                "max": 20,
                "step": 1,
                "description": "Maximum number of research units to run concurrently. This will allow the researcher to use multiple sub-agents to conduct research. Note: with more concurrency, you may run into rate limits."
            }
        }
    )
    # Research Configuration
    search_api: SearchAPI = Field(
        default=SearchAPI.TAVILY,
        metadata={
            "x_oap_ui_config": {
                "type": "select",
                "default": "tavily",
                "description": "Search API to use for research. NOTE: Make sure your Researcher Model supports the selected search API.",
                "options": [
                    {"label": "Tavily", "value": SearchAPI.TAVILY.value},
                    {"label": "OpenAI Native Web Search", "value": SearchAPI.OPENAI.value},
                    {"label": "Anthropic Native Web Search", "value": SearchAPI.ANTHROPIC.value},
                    {"label": "None", "value": SearchAPI.NONE.value}
                ]
            }
        }
    )
    max_researcher_iterations: int = Field(
        default=6,
        metadata={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 6,
                "min": 1,
                "max": 10,
                "step": 1,
                "description": "Maximum number of research iterations for the Research Supervisor. This is the number of times the Research Supervisor will reflect on the research and ask follow-up questions."
            }
        }
    )
    max_react_tool_calls: int = Field(
        default=10,
        metadata={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 10,
                "min": 1,
                "max": 30,
                "step": 1,
                "description": "Maximum number of tool calling iterations to make in a single researcher step."
            }
        }
    )
    # Model Configuration
    summarization_model: str = Field(
        default="openai:gpt-4.1-mini",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "openai:gpt-4.1-mini",
                "description": "Model for summarizing research results from Tavily search results"
            }
        }
    )
    summarization_model_max_tokens: int = Field(
        default=8192,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 8192,
                "description": "Maximum output tokens for summarization model"
            }
        }
    )
    max_content_length: int = Field(
        default=50000,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 50000,
                "min": 1000,
                "max": 200000,
                "description": "Maximum character length for webpage content before summarization"
            }
        }
    )
    research_model: str = Field(
        default="openai:gpt-4.1",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "openai:gpt-4.1",
                "description": "Model for conducting research. NOTE: Make sure your Researcher Model supports the selected search API."
            }
        }
    )
    research_model_max_tokens: int = Field(
        default=10000,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 10000,
                "description": "Maximum output tokens for research model"
            }
        }
    )
    compression_model: str = Field(
        default="openai:gpt-4.1",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "openai:gpt-4.1",
                "description": "Model for compressing research findings from sub-agents. NOTE: Make sure your Compression Model supports the selected search API."
            }
        }
    )
    compression_model_max_tokens: int = Field(
        default=8192,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 8192,
                "description": "Maximum output tokens for compression model"
            }
        }
    )
    final_report_model: str = Field(
        default="openai:gpt-4.1",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "openai:gpt-4.1",
                "description": "Model for writing the final report from all research findings"
            }
        }
    )
    final_report_model_max_tokens: int = Field(
        default=10000,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 10000,
                "description": "Maximum output tokens for final report model"
            }
        }
    )
    # MCP server configuration
    mcp_config: Optional[MCPConfig] = Field(
        default=None,
        optional=True,
        metadata={
            "x_oap_ui_config": {
                "type": "mcp",
                "description": "MCP server configuration"
            }
        }
    )
    mcp_prompt: Optional[str] = Field(
        default=None,
        optional=True,
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "description": "Any additional instructions to pass along to the Agent regarding the MCP tools that are available to it."
            }
        }
    )
    
    # Council Configuration - Multi-model verification for research briefs
    use_council: bool = Field(
        default=True,
        metadata={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": True,
                "description": "Enable multi-model council verification for research briefs before execution"
            }
        }
    )
    council_models: List[str] = Field(
        default=["openai:gpt-4.1", "anthropic:claude-sonnet-4-5-20250929"],
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "openai:gpt-4.1,anthropic:claude-sonnet-4-5-20250929",
                "description": "Comma-separated list of models for the council (e.g., openai:gpt-4.1,anthropic:claude-sonnet-4-5-20250929)"
            }
        }
    )
    council_min_consensus: float = Field(
        default=0.7,
        metadata={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 0.7,
                "min": 0.1,
                "max": 1.0,
                "step": 0.1,
                "description": "Minimum consensus score required for approval (0.0-1.0). Higher = stricter."
            }
        }
    )
    council_max_revisions: int = Field(
        default=3,
        metadata={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 3,
                "min": 0,
                "max": 5,
                "step": 1,
                "description": "Maximum revision attempts before forcing proceed. 0 = no revisions allowed."
            }
        }
    )
    
    # Council 2: Fact-check findings after research
    use_findings_council: bool = Field(
        default=True,
        metadata={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": True,
                "description": "Enable council fact-checking of research findings before final report. Catches hallucinations and fabrications."
            }
        }
    )
    findings_max_revisions: int = Field(
        default=2,
        metadata={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 2,
                "min": 0,
                "max": 3,
                "step": 1,
                "description": "Maximum revision attempts for findings fact-check. 0 = no revisions."
            }
        }
    )
    
    # Human Review Mode - Council as Advisor, Human as Authority
    review_mode: str = Field(
        default="none",
        metadata={
            "x_oap_ui_config": {
                "type": "select",
                "default": "none",
                "description": "Human review checkpoints. 'none'=fully automated, 'brief'=review brief before research, 'full'=review brief AND final report",
                "options": [
                    {"label": "None (Fully Automated)", "value": "none"},
                    {"label": "Brief Only (Review research plan)", "value": "brief"},
                    {"label": "Full (Review brief + report)", "value": "full"}
                ]
            }
        }
    )
    
    # Brief Context Injection Configuration
    # Pre-searches to inject recent context into brief generation
    enable_brief_context: bool = Field(
        default=True,
        metadata={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": True,
                "description": "Enable Tavily pre-search to inject recent context into brief generation. Improves brief specificity."
            }
        }
    )
    brief_context_max_queries: int = Field(
        default=3,
        metadata={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 3,
                "min": 1,
                "max": 5,
                "step": 1,
                "description": "Number of exploratory search queries to generate for context gathering."
            }
        }
    )
    brief_context_max_results: int = Field(
        default=5,
        metadata={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 5,
                "min": 3,
                "max": 10,
                "step": 1,
                "description": "Maximum results per search query for context gathering."
            }
        }
    )
    brief_context_days: int = Field(
        default=90,
        metadata={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 90,
                "min": 30,
                "max": 180,
                "step": 30,
                "description": "Only search for content from the last N days. 90 = 3 months, 180 = 6 months."
            }
        }
    )
    brief_context_include_news: bool = Field(
        default=True,
        metadata={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": True,
                "description": "Also search news sources for recent developments (in addition to general search)."
            }
        }
    )
    
    # Claim Verification Configuration
    use_claim_verification: bool = Field(
        default=False,  # Off by default until stable
        metadata={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": False,
                "description": "Enable claim-level verification health check after report generation. Adds ~60 seconds and ~$0.45 to run."
            }
        }
    )
    
    max_claims_to_verify: int = Field(
        default=25,
        metadata={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 25,
                "min": 5,
                "max": 50,
                "step": 5,
                "description": "Maximum claims to verify (cost control). Each claim costs ~$0.02."
            }
        }
    )
    
    verification_confidence_threshold: float = Field(
        default=0.8,
        metadata={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 0.8,
                "min": 0.5,
                "max": 1.0,
                "step": 0.1,
                "description": "Claims below this confidence are flagged as warnings."
            }
        }
    )

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """Create a Configuration instance from a RunnableConfig."""
        configurable = config.get("configurable", {}) if config else {}
        field_names = list(cls.model_fields.keys())
        values: dict[str, Any] = {
            field_name: os.environ.get(field_name.upper(), configurable.get(field_name))
            for field_name in field_names
        }
        return cls(**{k: v for k, v in values.items() if v is not None})

    def get_effective_max_researcher_iterations(self) -> int:
        """Get max researcher iterations, reduced in test mode."""
        return 2 if self.test_mode else self.max_researcher_iterations

    def get_effective_max_react_tool_calls(self) -> int:
        """Get max tool calls, reduced in test mode."""
        return 3 if self.test_mode else self.max_react_tool_calls

    def get_effective_max_concurrent_research_units(self) -> int:
        """Get max concurrent units, reduced in test mode."""
        return 2 if self.test_mode else self.max_concurrent_research_units

    class Config:
        """Pydantic configuration."""
        
        arbitrary_types_allowed = True