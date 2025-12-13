"""
LLM Council for automated verification of research briefs.

Multi-model voting system that validates research briefs before execution,
replacing human-in-the-loop with automated consensus-based approval.
"""

import asyncio
import logging
from typing import List, Literal, Optional, Dict, Any
from dataclasses import dataclass, field
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from open_deep_research.utils import get_api_key_for_model

logger = logging.getLogger(__name__)


# ============================================================================
# Data Models
# ============================================================================

class BriefReview(BaseModel):
    """Structured review of a research brief."""
    
    decision: Literal["approve", "reject", "revise"] = Field(
        description="approve=good to go, reject=fundamentally flawed, revise=needs changes"
    )
    confidence: float = Field(
        ge=0, le=1, 
        description="How confident are you in this assessment? 0-1"
    )
    strengths: List[str] = Field(
        description="What's good about this research brief?"
    )
    weaknesses: List[str] = Field(
        description="What's wrong, vague, or missing?"
    )
    suggested_changes: List[str] = Field(
        description="Specific changes to improve the brief"
    )
    reasoning: str = Field(
        description="Overall reasoning for your decision"
    )


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
    
    # Models in the council (GPT-4.1 + Claude Sonnet 4.5 dated model by default)
    models: List[str] = field(default_factory=lambda: [
        "openai:gpt-4.1",
        "anthropic:claude-sonnet-4-5-20250929",
    ])
    
    # Consensus thresholds
    min_consensus_for_approve: float = 0.7
    min_confidence_threshold: float = 0.6
    
    # Behavior
    max_revision_rounds: int = 3
    require_unanimous_for_reject: bool = True
    
    # Synthesis model (uses first council model by default)
    synthesis_model: Optional[str] = None
    
    def get_synthesis_model(self) -> str:
        """Get the model to use for synthesizing feedback."""
        return self.synthesis_model or self.models[0]


# ============================================================================
# Review Prompt
# ============================================================================

BRIEF_REVIEW_PROMPT = """You are a senior research advisor reviewing a research brief.
Your job is to evaluate whether this brief will guide effective, comprehensive research.

EVALUATION CRITERIA:
1. CLARITY: Is the research question specific and unambiguous?
2. SCOPE: Is the scope appropriate - not too broad, not too narrow?
3. FEASIBILITY: Can this be researched with web searches and available tools?
4. COMPLETENESS: Does it include all necessary context and constraints?
5. ACTIONABILITY: Can a researcher immediately start working from this brief?

DECISION GUIDELINES:
- APPROVE: Brief is clear, specific, and actionable - proceed with research
- REVISE: Brief has fixable issues (vague terms, missing context, unclear scope)
- REJECT: Brief is fundamentally flawed, off-topic, or impossible to research

Be constructive. If you suggest revisions, be specific about what to change."""


# ============================================================================
# Core Functions
# ============================================================================

async def get_single_vote(
    model_name: str,
    brief: str,
    runnable_config: Optional[RunnableConfig] = None,
) -> CouncilVote:
    """Get one council member's vote on a research brief.
    
    Args:
        model_name: The model identifier (e.g., "openai:gpt-4.1", "anthropic:claude-sonnet-4")
        brief: The research brief to evaluate
        runnable_config: Runtime configuration for API key access
        
    Returns:
        CouncilVote with the model's decision and reasoning
    """
    # Get API key for this model
    api_key = get_api_key_for_model(model_name, runnable_config) if runnable_config else None
    
    # Initialize model with LangSmith tracing tag and API key
    llm = init_chat_model(
        model=model_name,
        api_key=api_key,
        tags=["langsmith:council_vote"]
    )
    
    prompt = f"""{BRIEF_REVIEW_PROMPT}

RESEARCH BRIEF TO EVALUATE:
{brief}

Review this brief and provide your assessment."""

    try:
        structured_llm = llm.with_structured_output(BriefReview)
        review: BriefReview = await structured_llm.ainvoke([HumanMessage(content=prompt)])
        
        return CouncilVote(
            model_name=model_name,
            decision=review.decision,
            confidence=review.confidence,
            reasoning=review.reasoning,
            suggested_changes="\n".join(review.suggested_changes) if review.suggested_changes else None
        )
    except Exception as e:
        # Fallback on error - default to revise with low confidence
        logger.warning(f"Council vote failed for {model_name}: {e}")
        return CouncilVote(
            model_name=model_name,
            decision="revise",
            confidence=0.3,
            reasoning=f"Error getting review: {str(e)}",
            suggested_changes=None
        )


def calculate_verdict(votes: List[CouncilVote], config: CouncilConfig) -> CouncilVerdict:
    """Calculate collective decision from individual votes.
    
    Uses confidence-weighted voting where low-confidence votes carry less weight.
    
    Args:
        votes: List of individual council member votes
        config: Council configuration with thresholds
        
    Returns:
        CouncilVerdict with the collective decision
    """
    weighted_votes = {"approve": 0.0, "revise": 0.0, "reject": 0.0}
    total_weight = 0.0
    
    for vote in votes:
        # Reduce weight for low-confidence votes
        weight = vote.confidence if vote.confidence >= config.min_confidence_threshold else vote.confidence * 0.5
        weighted_votes[vote.decision] += weight
        total_weight += weight
    
    # Normalize to percentages
    if total_weight > 0:
        for key in weighted_votes:
            weighted_votes[key] /= total_weight
    
    # Determine decision based on thresholds
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


async def synthesize_feedback(
    votes: List[CouncilVote], 
    synthesis_model: str,
    runnable_config: Optional[RunnableConfig] = None
) -> str:
    """Combine all council feedback into actionable revision instructions.
    
    Args:
        votes: List of council votes with feedback
        synthesis_model: Model to use for synthesis
        runnable_config: Runtime configuration for API key access
        
    Returns:
        Synthesized, prioritized feedback string
    """
    api_key = get_api_key_for_model(synthesis_model, runnable_config) if runnable_config else None
    
    llm = init_chat_model(
        model=synthesis_model,
        api_key=api_key,
        tags=["langsmith:council_synthesis"]
    )
    
    feedback_parts = []
    for vote in votes:
        feedback_parts.append(f"""
{vote.model_name} ({vote.decision}, confidence: {vote.confidence:.0%}):
Reasoning: {vote.reasoning}
Suggested Changes: {vote.suggested_changes or 'None'}
""")
    
    prompt = f"""Synthesize this feedback into clear revision instructions for improving the research brief.

REVIEWER FEEDBACK:
{"".join(feedback_parts)}

Create a concise, actionable list of what needs to change.
Focus on specific items. Prioritize by importance.
Keep it brief - no more than 5 bullet points."""

    response = await llm.ainvoke([HumanMessage(content=prompt)])
    return response.content


async def council_vote_on_brief(
    brief: str,
    config: CouncilConfig,
    runnable_config: Optional[RunnableConfig] = None
) -> CouncilVerdict:
    """Have the council vote on a research brief.
    
    Queries all council models in parallel, then calculates consensus.
    
    Args:
        brief: The research brief to evaluate
        config: Council configuration
        runnable_config: Runtime configuration for API key access
        
    Returns:
        CouncilVerdict with decision and feedback
    """
    # Get votes from all council members in parallel
    vote_tasks = [
        get_single_vote(model, brief, runnable_config)
        for model in config.models
    ]
    votes = await asyncio.gather(*vote_tasks)
    
    # Calculate collective verdict
    verdict = calculate_verdict(list(votes), config)
    
    # Synthesize feedback if revision or rejection needed
    if verdict.decision != "approve":
        verdict.synthesized_feedback = await synthesize_feedback(
            list(votes), 
            config.get_synthesis_model(),
            runnable_config
        )
    
    return verdict


def log_council_decision(verdict: CouncilVerdict) -> None:
    """Log the council's decision with formatting for visibility.
    
    Args:
        verdict: The council's verdict to log
    """
    print(f"\n{'='*60}")
    print(f"COUNCIL DECISION: {verdict.decision.upper()}")
    print(f"Consensus Score: {verdict.consensus_score:.0%}")
    print(f"{'-'*60}")
    for vote in verdict.votes:
        status_icon = "✓" if vote.decision == "approve" else ("✗" if vote.decision == "reject" else "~")
        print(f"  {status_icon} {vote.model_name}: {vote.decision} ({vote.confidence:.0%})")
        if vote.suggested_changes:
            # Show first line of suggestions
            first_change = vote.suggested_changes.split('\n')[0][:60]
            print(f"    → {first_change}...")
    if verdict.synthesized_feedback:
        print(f"{'-'*60}")
        print("Synthesized Feedback:")
        for line in verdict.synthesized_feedback.split('\n')[:5]:
            print(f"  {line}")
    print(f"{'='*60}\n")

