"""
Graph.py updates for LLM Council integration.

INSTRUCTIONS:
Make these changes to your existing graph.py file.
"""

# ============================================================================
# STEP 1: ADD IMPORT
# ============================================================================

"""
Near the top of graph.py, with your other imports, add:

from open_deep_research.council import council_feedback
"""


# ============================================================================
# STEP 2: REPLACE human_feedback FUNCTION
# ============================================================================

"""
Find the existing human_feedback function (around line 100-150) and REPLACE it with this:
"""

async def human_feedback(state, config):
    """
    Automated feedback using LLM council.
    
    Replaces the original human-in-the-loop with multi-model verification.
    Falls back to auto-approve if council is disabled.
    """
    from langgraph.types import Command
    from langgraph.constants import Send
    from open_deep_research.configuration import Configuration
    
    # Check if council is enabled
    configurable = Configuration.from_runnable_config(config)
    use_council = getattr(configurable, 'use_council', True)
    
    topic = state["topic"]
    sections = state["sections"]
    
    if use_council:
        # Use the council for verification
        from open_deep_research.council import council_feedback
        return await council_feedback(state, config)
    else:
        # Auto-approve if council disabled (no human, no council)
        print("Council disabled - auto-approving plan")
        return Command(goto=[
            Send("build_section_with_web_research", {
                "topic": topic,
                "section": s,
                "search_iterations": 0
            })
            for s in sections
            if s.research
        ])


# ============================================================================
# STEP 3: UPDATE STATE (Optional but recommended)
# ============================================================================

"""
In state.py, add these optional fields to ReportState for tracking:

class ReportState(TypedDict):
    # ... existing fields ...
    
    # Council tracking (optional)
    council_revision_count: int  # Track revision attempts
    council_verdict: dict        # Store last verdict for debugging
"""


# ============================================================================
# THAT'S IT! 
# ============================================================================

"""
Summary of changes:
1. Add council.py to src/open_deep_research/
2. Add council fields to Configuration class
3. Replace human_feedback function in graph.py

The node name stays "human_feedback" so you don't need to change any edges.
"""
