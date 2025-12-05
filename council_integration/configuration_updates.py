"""
Configuration updates for LLM Council.

INSTRUCTIONS:
Add these fields to your existing Configuration class in configuration.py
"""

# ============================================================================
# ADD THESE IMPORTS at the top of configuration.py
# ============================================================================

# from typing import List  # If not already imported
# from dataclasses import field  # If not already imported


# ============================================================================
# ADD THESE FIELDS to your Configuration class
# ============================================================================

"""
Add these inside your @dataclass Configuration class:

    # ===== COUNCIL SETTINGS =====
    
    # Enable council (set False to skip verification entirely)
    use_council: bool = True
    
    # Models to use in the council (need API keys for each)
    council_models: list = field(default_factory=lambda: [
        "anthropic:claude-sonnet-4-20250514",
        "openai:gpt-4.1",
    ])
    
    # Minimum agreement to auto-approve (0.0 to 1.0)
    # 0.7 = 70% of weighted votes must approve
    council_min_consensus: float = 0.7
    
    # Maximum revision attempts before giving up
    council_max_revisions: int = 3
"""


# ============================================================================
# FULL EXAMPLE - What your Configuration class should look like
# ============================================================================

"""
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum

class SearchAPI(Enum):
    TAVILY = "tavily"
    PERPLEXITY = "perplexity"
    # ... other APIs ...

@dataclass(kw_only=True)
class Configuration:
    # ===== EXISTING FIELDS (keep these) =====
    report_structure: str = DEFAULT_REPORT_STRUCTURE
    search_api: SearchAPI = SearchAPI.TAVILY
    search_api_config: Optional[Dict[str, Any]] = None
    number_of_queries: int = 2
    max_search_depth: int = 2
    
    # Model settings
    planner_provider: str = "openai"
    planner_model: str = "o3-mini"
    writer_provider: str = "openai"
    writer_model: str = "o3-mini"
    
    # ... other existing fields ...
    
    # ===== NEW COUNCIL FIELDS (add these) =====
    
    use_council: bool = True
    
    council_models: List[str] = field(default_factory=lambda: [
        "anthropic:claude-sonnet-4-20250514",
        "openai:gpt-4.1",
    ])
    
    council_min_consensus: float = 0.7
    
    council_max_revisions: int = 3
    
    # ===== END NEW FIELDS =====
    
    @classmethod
    def from_runnable_config(cls, config: Optional[RunnableConfig] = None) -> "Configuration":
        # ... existing implementation ...
"""
