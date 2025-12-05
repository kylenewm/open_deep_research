"""
LLM Council for automated verification - replaces human-in-the-loop.

Drop this file into: src/open_deep_research/council.py
"""

import asyncio
from typing import List, Literal, Optional, Dict, Any
from dataclasses import dataclass, field
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage


# ============================================================================
# Data Models
# ============================================================================

class PlanReview(BaseModel):
    """Structured review of a research plan."""
    decision: Literal["approve", "reject", "revise"] = Field(
        description="approve=good to go, reject=fundamentally flawed, revise=needs changes"
    )
    confidence: float = Field(ge=0, le=1, description="How confident are you? 0-1")
    strengths: List[str] = Field(description="What's good about this plan?")
    weaknesses: List[str] = Field(description="What's wrong or missing?")
    suggested_changes: List[str] = Field(description="Specific changes to make")
    reasoning: str = Field(description="Overall reasoning for your decision")


class CouncilVote(BaseModel):
    """A single council member's vote."""
    model_name: str
    decision: Literal["approve", "reject", "revise"]
    confidence: float = Field(ge=0, le=1)
    reasoning: str
    suggested_changes: Optional[str] = None


class CouncilVerdict(BaseModel):
    """The council's collective decision."""
    decision: Literal["approve", "reject", "revise"]
    consensus_score: float = Field(ge=0, le=1)
    votes: List[CouncilVote]
    synthesized_feedback: str
    requires_revision: bool


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class CouncilConfig:
    """Configuration for the LLM council."""
    
    # Models in the council
    models: List[str] = field(default_factory=lambda: [
        "anthropic:claude-sonnet-4-20250514",
        "openai:gpt-4.1",
    ])
    
    # Consensus thresholds
    min_consensus_for_approve: float = 0.7
    min_confidence_threshold: float = 0.6
    
    # Behavior
    max_revision_rounds: int = 3
    require_unanimous_for_reject: bool = True
    
    # Synthesis model
    synthesis_model: str = "anthropic:claude-sonnet-4-20250514"


# ============================================================================
# Review Prompt
# ============================================================================

REVIEW_PROMPT = """You are a senior research advisor reviewing a research plan.
Your job is to evaluate whether this plan will produce a high-quality, comprehensive report.

EVALUATION CRITERIA:
1. COVERAGE: Does the plan cover all important aspects of the topic?
2. STRUCTURE: Is the organization logical? No overlapping sections?
3. FEASIBILITY: Can each section be researched with web searches?
4. BALANCE: Are different perspectives given appropriate weight?
5. GAPS: What important aspects are missing?

DECISION GUIDELINES:
- APPROVE: Plan is solid, maybe minor issues but good to proceed
- REVISE: Plan has fixable issues that should be addressed first  
- REJECT: Plan is fundamentally flawed or off-topic

Be constructive. If you suggest revisions, be specific about what to change."""


# ============================================================================
# Core Functions
# ============================================================================

async def get_single_vote(
    model_name: str,
    plan_content: str,
    topic: str
) -> CouncilVote:
    """Get one council member's vote."""
    
    llm = init_chat_model(model=model_name)
    
    prompt = f"""{REVIEW_PROMPT}

RESEARCH TOPIC:
{topic}

PROPOSED RESEARCH PLAN:
{plan_content}

Review this plan and provide your assessment."""

    try:
        structured_llm = llm.with_structured_output(PlanReview)
        review: PlanReview = await structured_llm.ainvoke([HumanMessage(content=prompt)])
        
        return CouncilVote(
            model_name=model_name,
            decision=review.decision,
            confidence=review.confidence,
            reasoning=review.reasoning,
            suggested_changes="\n".join(review.suggested_changes) if review.suggested_changes else None
        )
    except Exception as e:
        # Fallback on error
        return CouncilVote(
            model_name=model_name,
            decision="revise",
            confidence=0.3,
            reasoning=f"Error getting review: {str(e)}",
            suggested_changes=None
        )


def format_plan_for_review(sections: List[Any]) -> str:
    """Format sections into readable text."""
    
    lines = ["RESEARCH PLAN", "=" * 50, ""]
    
    for i, section in enumerate(sections, 1):
        lines.append(f"Section {i}: {section.name}")
        lines.append(f"  Description: {section.description}")
        lines.append(f"  Requires Research: {'Yes' if section.research else 'No'}")
        lines.append("")
    
    return "\n".join(lines)


def calculate_verdict(votes: List[CouncilVote], config: CouncilConfig) -> CouncilVerdict:
    """Calculate collective decision from votes."""
    
    weighted_votes = {"approve": 0.0, "revise": 0.0, "reject": 0.0}
    total_weight = 0.0
    
    for vote in votes:
        weight = vote.confidence if vote.confidence >= config.min_confidence_threshold else vote.confidence * 0.5
        weighted_votes[vote.decision] += weight
        total_weight += weight
    
    # Normalize
    if total_weight > 0:
        for key in weighted_votes:
            weighted_votes[key] /= total_weight
    
    # Determine decision
    approve_score = weighted_votes["approve"]
    all_reject = all(v.decision == "reject" for v in votes)
    
    if approve_score >= config.min_consensus_for_approve:
        decision = "approve"
        consensus_score = approve_score
    elif all_reject and config.require_unanimous_for_reject:
        decision = "reject"
        consensus_score = weighted_votes["reject"]
    else:
        decision = "revise"
        consensus_score = 1.0 - approve_score
    
    return CouncilVerdict(
        decision=decision,
        consensus_score=consensus_score,
        votes=votes,
        synthesized_feedback="",
        requires_revision=(decision == "revise")
    )


async def synthesize_feedback(votes: List[CouncilVote], synthesis_model: str) -> str:
    """Combine all feedback into actionable revision instructions."""
    
    llm = init_chat_model(model=synthesis_model)
    
    feedback_parts = []
    for vote in votes:
        feedback_parts.append(f"""
{vote.model_name} ({vote.decision}, confidence: {vote.confidence:.0%}):
Reasoning: {vote.reasoning}
Suggested Changes: {vote.suggested_changes or 'None'}
""")
    
    prompt = f"""Synthesize this feedback into clear revision instructions.

REVIEWER FEEDBACK:
{"".join(feedback_parts)}

Create a concise, actionable list of what needs to change.
Focus on specific items. Prioritize by importance."""

    response = await llm.ainvoke([HumanMessage(content=prompt)])
    return response.content


async def council_vote_on_plan(
    topic: str,
    sections: List[Any],
    config: CouncilConfig
) -> CouncilVerdict:
    """Have the council vote on a research plan."""
    
    plan_content = format_plan_for_review(sections)
    
    # Get votes in parallel
    vote_tasks = [
        get_single_vote(model, plan_content, topic)
        for model in config.models
    ]
    votes = await asyncio.gather(*vote_tasks)
    
    # Calculate verdict
    verdict = calculate_verdict(votes, config)
    
    # Synthesize feedback if revision needed
    if verdict.decision != "approve":
        verdict.synthesized_feedback = await synthesize_feedback(votes, config.synthesis_model)
    
    return verdict


# ============================================================================
# Main Integration Function - Drop-in replacement for human_feedback
# ============================================================================

async def council_feedback(state: Dict[str, Any], config: Dict[str, Any]):
    """
    Drop-in replacement for human_feedback node.
    
    Usage in graph.py:
        from open_deep_research.council import council_feedback
        builder.add_node("human_feedback", council_feedback)
    """
    from langgraph.types import Command
    from langgraph.constants import Send
    
    topic = state["topic"]
    sections = state["sections"]
    
    # Build council config from runnable config
    council_config = CouncilConfig()
    if "configurable" in config:
        cfg = config["configurable"]
        if "council_models" in cfg:
            council_config.models = cfg["council_models"]
        if "council_min_consensus" in cfg:
            council_config.min_consensus_for_approve = cfg["council_min_consensus"]
        if "council_max_revisions" in cfg:
            council_config.max_revision_rounds = cfg["council_max_revisions"]
    
    # Get verdict
    verdict = await council_vote_on_plan(topic, sections, council_config)
    
    # Track revisions
    revision_count = state.get("council_revision_count", 0)
    
    # Log decision
    print(f"\n{'='*50}")
    print(f"COUNCIL DECISION: {verdict.decision.upper()}")
    print(f"Consensus: {verdict.consensus_score:.0%}")
    for vote in verdict.votes:
        print(f"  {vote.model_name}: {vote.decision} ({vote.confidence:.0%})")
    print(f"{'='*50}\n")
    
    if verdict.decision == "approve":
        # Proceed to research
        return Command(goto=[
            Send("build_section_with_web_research", {
                "topic": topic,
                "section": s,
                "search_iterations": 0
            })
            for s in sections
            if s.research
        ])
    
    elif verdict.decision == "reject" or revision_count >= council_config.max_revision_rounds:
        # End with error
        error_msg = f"Research plan rejected by council after {revision_count} attempts.\n\nFeedback:\n{verdict.synthesized_feedback}"
        return Command(
            goto="__end__",
            update={
                "final_report": error_msg,
                "council_verdict": verdict.model_dump() if hasattr(verdict, 'model_dump') else str(verdict)
            }
        )
    
    else:
        # Revise
        return Command(
            goto="generate_report_plan",
            update={
                "feedback_on_report_plan": [verdict.synthesized_feedback],
                "council_revision_count": revision_count + 1
            }
        )
