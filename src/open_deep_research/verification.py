"""
Claim-level verification for research reports.
Uses council-based verification against preserved source documents.

Matching Strategy (v2):
- Embedding similarity for semantic understanding
- Entity boost for specific names/numbers/terms
- Multi-source verification (top 3 sources per claim)
- Keyword fallback for edge cases
"""

import asyncio
import logging
import os
import re
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import OpenAIEmbeddings
from pydantic import BaseModel, Field

from open_deep_research.configuration import Configuration
from open_deep_research.state import (
    ClaimVerification,
    SourceRecord,
    VerificationResult,
    VerificationSummary,
)
from open_deep_research.utils import get_api_key_for_model


##########################
# Embedding Utilities
##########################

# Initialize embeddings lazily (reuse across calls)
_embeddings = None


def get_embeddings():
    """Get or initialize OpenAI embeddings model."""
    global _embeddings
    if _embeddings is None:
        _embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return _embeddings


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    a_arr, b_arr = np.array(a), np.array(b)
    dot_product = np.dot(a_arr, b_arr)
    norm_a = np.linalg.norm(a_arr)
    norm_b = np.linalg.norm(b_arr)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot_product / (norm_a * norm_b))


##########################
# Entity Extraction
##########################

# Known financial/AI entity patterns for better matching
KNOWN_ENTITIES = {
    'bamchatgpt', 'turion', 'alphasense', 'chatgpt', 'gpt-4', 'gpt-4o',
    'citadel', 'balyasny', 'point72', 'bridgewater', 'two sigma', 'd.e. shaw',
    'man group', 'viking global', 'tiger global', 'coatue'
}


def extract_entities(text: str) -> set:
    """Extract entity names, numbers, percentages, and specific terms from text.
    
    Returns a set of entities for matching between claims and sources.
    """
    entities = set()
    
    # Company/person names (capitalized multi-word phrases)
    names = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', text)
    entities.update(name.lower() for name in names)
    
    # Single capitalized words (for acronyms, proper nouns)
    caps = re.findall(r'\b[A-Z][a-z]+\b', text)
    entities.update(word.lower() for word in caps if len(word) > 2)
    
    # Acronyms (2+ uppercase letters)
    acronyms = re.findall(r'\b[A-Z]{2,}\b', text)
    entities.update(acr.lower() for acr in acronyms)
    
    # Numbers and percentages (important for fact-checking)
    numbers = re.findall(r'\b\d+(?:\.\d+)?%?\b', text)
    entities.update(numbers)
    
    # Dollar amounts
    dollars = re.findall(r'\$[\d,]+(?:\.\d+)?(?:\s*(?:billion|million|B|M))?\b', text, re.I)
    entities.update(d.lower().replace(',', '') for d in dollars)
    
    # Known AI/finance tools (case-insensitive)
    for known in KNOWN_ENTITIES:
        if known in text.lower():
            entities.add(known)
    
    # Specific tool patterns
    tools = re.findall(r'\b(?:BAMChatGPT|Turion|AlphaSense|ChatGPT|GPT-\d+[a-z]?)\b', text, re.I)
    entities.update(t.lower() for t in tools)
    
    return entities


# Pydantic models for structured LLM output
class ExtractedClaims(BaseModel):
    """LLM output for claim extraction."""
    claims: List[str] = Field(description="List of verifiable factual claims")


class VerificationVote(BaseModel):
    """Single model's verification vote."""
    status: str = Field(description="SUPPORTED, PARTIALLY_SUPPORTED, UNSUPPORTED, or UNCERTAIN")
    confidence: float = Field(description="Confidence score 0.0 to 1.0")
    evidence: str = Field(description="Quote from source supporting/contradicting claim, max 150 chars")


# Prompts
CLAIM_EXTRACTION_PROMPT = """Extract all verifiable factual claims from this research report.

Rules:
- Only extract FACTUAL claims (not opinions, speculation, or hedged statements)
- Each claim should be a standalone, complete sentence
- Include claims with specific: numbers, dates, percentages, names, statistics
- Do NOT include vague statements like "many experts believe" or "research suggests"
- Do NOT include the same claim twice (even if rephrased)
- Maximum 30 claims

Report:
{report}
"""

VERIFICATION_PROMPT = """You are a fact-checker. Determine if this SOURCE supports the CLAIM.

CLAIM: {claim}

SOURCE PASSAGE:
{passage}

SOURCE URL: {source_url}

Rules:
- SUPPORTED: Source directly states or strongly implies the claim (confidence >= 0.8)
- PARTIALLY_SUPPORTED: Source supports part of the claim, or claim is a reasonable inference (confidence 0.5-0.8)
- UNSUPPORTED: Source contradicts the claim OR provides no evidence for it (confidence < 0.3)
- UNCERTAIN: Cannot determine from this source alone (confidence 0.3-0.5)

Be STRICT. If the source doesn't explicitly support the claim, mark UNSUPPORTED or UNCERTAIN.
"""


async def extract_claims(report: str, llm) -> List[str]:
    """Extract verifiable factual claims from a research report.
    
    Args:
        report: The final research report text
        llm: Configured LLM with structured output
        
    Returns:
        List of claim strings (max 30)
    """
    try:
        prompt = CLAIM_EXTRACTION_PROMPT.format(report=report[:15000])  # Limit input size
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        
        claims = response.claims if hasattr(response, 'claims') else []
        logging.info(f"[VERIFY] Extracted {len(claims)} claims from report")
        return claims[:30]  # Hard cap at 30
        
    except Exception as e:
        logging.error(f"[VERIFY] Claim extraction failed: {e}")
        return []


def extract_best_paragraph(claim: str, content: str, max_length: int = 1500) -> str:
    """Extract the most relevant paragraph from content for a claim.
    
    Uses keyword overlap to find the best paragraph within already-matched content.
    """
    if not content:
        return ""
    
    # Simple keyword extraction for paragraph matching
    claim_words = set(
        word.lower().strip('.,!?":;()[]') 
        for word in claim.split() 
        if len(word) > 3
    )
    
    paragraphs = content.split('\n\n')
    best_para = ""
    best_score = 0
    
    for para in paragraphs:
        if len(para) < 50:
            continue
        para_lower = para.lower()
        score = sum(1 for word in claim_words if word in para_lower)
        if score > best_score:
            best_score = score
            best_para = para
    
    # Fallback to first chunk if no good paragraph
    if not best_para:
        best_para = content[:max_length]
    
    return best_para[:max_length]


async def find_relevant_passages(
    claim: str, 
    sources: List[SourceRecord],
    top_k: int = 3,
    max_snippet_length: int = 1500
) -> List[Tuple[SourceRecord, str, float]]:
    """Find top-k most relevant sources using embedding similarity + entity boost.
    
    Strategy 3: Combines semantic similarity with entity matching for precision.
    Returns multiple sources to check, improving recall for specific claims.
    
    Args:
        claim: The claim to find sources for
        sources: List of source records to search
        top_k: Number of top sources to return
        max_snippet_length: Maximum length of returned snippets
        
    Returns:
        List of (source, passage, score) tuples, sorted by score descending
    """
    if not sources:
        return []
    
    try:
        embeddings = get_embeddings()
        claim_embedding = await embeddings.aembed_query(claim)
        claim_entities = extract_entities(claim)
        
        scored_sources = []
        
        for source in sources:
            content = source.get("content", "")
            title = source.get("title", "")
            if not content:
                continue
            
            # Embed source (title + first 2000 chars for better coverage)
            source_text = f"{title} {content[:2000]}"
            source_embedding = await embeddings.aembed_query(source_text)
            
            # Base embedding similarity score
            embed_score = cosine_similarity(claim_embedding, source_embedding)
            
            # Entity overlap bonus (0.0 to 0.3)
            source_entities = extract_entities(title + " " + content)
            entity_overlap = len(claim_entities & source_entities)
            entity_bonus = min(0.3, entity_overlap * 0.05)  # 0.05 per entity, max 0.3
            
            final_score = embed_score + entity_bonus
            snippet = extract_best_paragraph(claim, content, max_snippet_length)
            
            scored_sources.append((source, snippet, final_score))
        
        # Sort by score descending
        scored_sources.sort(key=lambda x: x[2], reverse=True)
        top_sources = scored_sources[:top_k]
        
        # If top scores are too low, try keyword fallback
        if top_sources and top_sources[0][2] < 0.4:
            keyword_results = _find_relevant_passages_keyword(claim, sources, top_k, max_snippet_length)
            # Merge: take best from each approach
            if keyword_results and keyword_results[0][2] > top_sources[0][2]:
                logging.debug(f"[VERIFY] Keyword fallback improved match for: {claim[:40]}...")
                # Combine and re-sort
                combined = {s.get('url'): (s, p, sc) for s, p, sc in top_sources}
                for s, p, sc in keyword_results:
                    url = s.get('url')
                    if url not in combined or sc > combined[url][2]:
                        combined[url] = (s, p, sc)
                top_sources = sorted(combined.values(), key=lambda x: x[2], reverse=True)[:top_k]
        
        if top_sources:
            logging.debug(
                f"[VERIFY] Top source: {top_sources[0][0].get('title', 'Unknown')[:30]}... "
                f"(score: {top_sources[0][2]:.2f})"
            )
        
        return top_sources
        
    except Exception as e:
        logging.error(f"[VERIFY] Embedding search failed: {e}. Using keyword fallback.")
        return _find_relevant_passages_keyword(claim, sources, top_k, max_snippet_length)


# Keep old function for backwards compatibility
async def find_relevant_passage(
    claim: str, 
    sources: List[SourceRecord],
    max_snippet_length: int = 1500
) -> Tuple[Optional[SourceRecord], str, float]:
    """Find the most relevant source (backwards compatible wrapper)."""
    results = await find_relevant_passages(claim, sources, top_k=1, max_snippet_length=max_snippet_length)
    if results:
        return results[0]
    return None, "", 0.0


def _find_relevant_passages_keyword(
    claim: str, 
    sources: List[SourceRecord],
    top_k: int = 3,
    max_snippet_length: int = 1500
) -> List[Tuple[SourceRecord, str, float]]:
    """Keyword-based source matching fallback. Returns top-k results."""
    if not sources:
        return []
    
    stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                 'should', 'may', 'might', 'must', 'shall', 'can', 'to', 'of', 'in',
                 'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through',
                 'during', 'before', 'after', 'above', 'below', 'between', 'under',
                 'and', 'but', 'or', 'nor', 'so', 'yet', 'both', 'either', 'neither',
                 'not', 'only', 'own', 'same', 'than', 'too', 'very', 'just', 'that',
                 'this', 'these', 'those', 'it', 'its'}
    
    claim_words = set(
        word.lower().strip('.,!?":;()[]\'') 
        for word in claim.split() 
        if word.lower() not in stopwords and len(word) > 2
    )
    
    scored_sources = []
    
    for source in sources:
        content = source.get("content", "")
        if not content:
            continue
        
        content_lower = content.lower()
        score = sum(1 for word in claim_words if word in content_lower)
        
        title = source.get("title", "").lower()
        title_bonus = sum(2 for word in claim_words if word in title)
        score += title_bonus
        
        if score > 0:
            # Normalize keyword score to 0-1 range
            normalized_score = min(1.0, score / 10.0)
            snippet = extract_best_paragraph(claim, content, max_snippet_length)
            scored_sources.append((source, snippet, normalized_score))
    
    # Sort by score descending and return top-k
    scored_sources.sort(key=lambda x: x[2], reverse=True)
    return scored_sources[:top_k]


# Backwards compatible wrapper
def _find_relevant_passage_keyword(
    claim: str, 
    sources: List[SourceRecord],
    max_snippet_length: int = 1500
) -> Tuple[Optional[SourceRecord], str, float]:
    """Single-result keyword fallback (backwards compatible)."""
    results = _find_relevant_passages_keyword(claim, sources, top_k=1, max_snippet_length=max_snippet_length)
    if results:
        return results[0]
    return None, "", 0.0


async def verify_single_claim(
    claim: str,
    passage: str,
    source: Optional[SourceRecord],
    llm,
    claim_id: str
) -> ClaimVerification:
    """Verify a single claim against a source passage using LLM.
    
    Args:
        claim: The claim text to verify
        passage: Relevant passage from source
        source: The source record (for URL/title)
        llm: Configured LLM with structured output
        claim_id: Unique identifier for this claim
        
    Returns:
        ClaimVerification result
    """
    if not passage or not source:
        return ClaimVerification(
            claim_id=claim_id,
            claim_text=claim,
            status="UNCERTAIN",
            confidence=0.0,
            source_url=None,
            source_title=None,
            evidence_snippet="No matching source found"
        )
    
    try:
        prompt = VERIFICATION_PROMPT.format(
            claim=claim,
            passage=passage,
            source_url=source.get("url", "Unknown")
        )
        
        response = await asyncio.wait_for(
            llm.ainvoke([HumanMessage(content=prompt)]),
            timeout=30.0
        )
        
        return ClaimVerification(
            claim_id=claim_id,
            claim_text=claim,
            status=response.status,
            confidence=response.confidence,
            source_url=source.get("url"),
            source_title=source.get("title"),
            evidence_snippet=response.evidence[:200] if response.evidence else None
        )
        
    except asyncio.TimeoutError:
        logging.warning(f"[VERIFY] Timeout verifying claim {claim_id}")
        return ClaimVerification(
            claim_id=claim_id,
            claim_text=claim,
            status="UNCERTAIN",
            confidence=0.0,
            source_url=source.get("url") if source else None,
            source_title=source.get("title") if source else None,
            evidence_snippet="Verification timed out"
        )
    except Exception as e:
        logging.error(f"[VERIFY] Error verifying claim {claim_id}: {e}")
        return ClaimVerification(
            claim_id=claim_id,
            claim_text=claim,
            status="UNCERTAIN",
            confidence=0.0,
            source_url=source.get("url") if source else None,
            source_title=source.get("title") if source else None,
            evidence_snippet=f"Verification error: {str(e)[:50]}"
        )


async def verify_claim_against_multiple_sources(
    claim: str,
    sources_with_passages: List[Tuple[SourceRecord, str, float]],
    llm,
    claim_id: str
) -> ClaimVerification:
    """Verify a claim against multiple candidate sources, return best result.
    
    Checks each source in order until strong support is found, or returns
    the best result if no strong support exists.
    
    Args:
        claim: The claim text to verify
        sources_with_passages: List of (source, passage, score) tuples
        llm: Configured LLM with structured output
        claim_id: Unique identifier for this claim
        
    Returns:
        ClaimVerification with the best result found
    """
    if not sources_with_passages:
        return ClaimVerification(
            claim_id=claim_id,
            claim_text=claim,
            status="UNCERTAIN",
            confidence=0.0,
            source_url=None,
            source_title=None,
            evidence_snippet="No matching source found"
        )
    
    best_result = None
    
    for source, passage, _score in sources_with_passages:
        result = await verify_single_claim(claim, passage, source, llm, claim_id)
        
        # If we find strong support, return immediately (early exit)
        if result["status"] == "SUPPORTED" and result["confidence"] >= 0.8:
            logging.debug(f"[VERIFY] Found strong support for {claim_id} in source: {source.get('title', '')[:30]}")
            return result
        
        # Track best result so far (prefer higher confidence, then SUPPORTED > PARTIAL > others)
        if best_result is None:
            best_result = result
        elif result["confidence"] > best_result["confidence"]:
            best_result = result
        elif result["confidence"] == best_result["confidence"]:
            # Prefer SUPPORTED/PARTIAL over UNSUPPORTED/UNCERTAIN at same confidence
            status_priority = {"SUPPORTED": 4, "PARTIALLY_SUPPORTED": 3, "UNCERTAIN": 2, "UNSUPPORTED": 1}
            if status_priority.get(result["status"], 0) > status_priority.get(best_result["status"], 0):
                best_result = result
    
    return best_result


async def verify_report(
    final_report: str,
    sources: List[SourceRecord],
    config: RunnableConfig
) -> VerificationResult:
    """Main verification function: extract claims, match sources, verify each.
    
    Args:
        final_report: The generated research report
        sources: List of preserved source records
        config: Runtime configuration
        
    Returns:
        VerificationResult with summary and per-claim details
    """
    configurable = Configuration.from_runnable_config(config)
    
    # Initialize LLMs for extraction and verification
    api_key = get_api_key_for_model(configurable.research_model, config)
    
    extraction_llm = init_chat_model(
        model=configurable.research_model,
        api_key=api_key,
        tags=["langsmith:nostream", "verification:extraction"]
    ).with_structured_output(ExtractedClaims)
    
    verification_llm = init_chat_model(
        model=configurable.research_model,
        api_key=api_key,
        tags=["langsmith:nostream", "verification:verify"]
    ).with_structured_output(VerificationVote)
    
    # Step 1: Extract claims
    claims = await extract_claims(final_report, extraction_llm)
    
    if not claims:
        logging.warning("[VERIFY] No claims extracted, returning empty result")
        return VerificationResult(
            summary=VerificationSummary(
                total_claims=0,
                supported=0,
                partially_supported=0,
                unsupported=0,
                uncertain=0,
                overall_confidence=0.0,
                verified_at=datetime.now().isoformat(),
                warnings=["No verifiable claims found in report"]
            ),
            claims=[]
        )
    
    # Step 2: Limit claims to configured max
    max_claims = getattr(configurable, 'max_claims_to_verify', 25)
    claims = claims[:max_claims]
    
    # Track data issues for logging
    data_issues = []
    
    # Check sources for data issues
    for source in sources:
        if not source.get("content"):
            data_issues.append(f"Empty content: {source.get('url', 'unknown')[:50]}")
        elif len(source.get("content", "")) < 100:
            data_issues.append(f"Truncated content (<100 chars): {source.get('url', 'unknown')[:50]}")
    
    # Step 3: Find top-k relevant sources for each claim (using embeddings + entity boost)
    claim_sources_map = []
    for claim in claims:
        top_sources = await find_relevant_passages(claim, sources, top_k=3)
        claim_sources_map.append((claim, top_sources))
        
        # Log low similarity matches as data issues
        if not top_sources or top_sources[0][2] < 0.4:
            best_sim = top_sources[0][2] if top_sources else 0.0
            data_issues.append(f"No good source match (sim={best_sim:.2f}): {claim[:50]}...")
    
    # Log data issues
    if data_issues:
        logging.warning(f"[VERIFY] Data issues detected ({len(data_issues)}):")
        for issue in data_issues[:5]:
            logging.warning(f"  - {issue}")
    
    # Step 4: Verify each claim against multiple sources (parallel, batched)
    verifications: List[ClaimVerification] = []
    batch_size = 3  # Smaller batch since each claim checks multiple sources
    
    for i in range(0, len(claim_sources_map), batch_size):
        batch = claim_sources_map[i:i+batch_size]
        tasks = [
            verify_claim_against_multiple_sources(
                claim=claim,
                sources_with_passages=top_sources,
                llm=verification_llm,
                claim_id=f"claim_{i+j+1:03d}"
            )
            for j, (claim, top_sources) in enumerate(batch)
        ]
        batch_results = await asyncio.gather(*tasks)
        verifications.extend(batch_results)
    
    # Step 5: Calculate summary statistics
    status_counts = {"SUPPORTED": 0, "PARTIALLY_SUPPORTED": 0, "UNSUPPORTED": 0, "UNCERTAIN": 0}
    total_confidence = 0.0
    warnings = []
    
    # Get confidence threshold from config
    confidence_threshold = getattr(configurable, 'verification_confidence_threshold', 0.8)
    
    for v in verifications:
        status_counts[v["status"]] = status_counts.get(v["status"], 0) + 1
        total_confidence += v["confidence"]
        
        # Flag low confidence claims as warnings
        if v["confidence"] < confidence_threshold:
            warnings.append(
                f"[{v['status']}] ({v['confidence']:.0%}) {v['claim_text'][:60]}..."
            )
    
    summary = VerificationSummary(
        total_claims=len(verifications),
        supported=status_counts["SUPPORTED"],
        partially_supported=status_counts["PARTIALLY_SUPPORTED"],
        unsupported=status_counts["UNSUPPORTED"],
        uncertain=status_counts["UNCERTAIN"],
        overall_confidence=total_confidence / len(verifications) if verifications else 0.0,
        verified_at=datetime.now().isoformat(),
        warnings=warnings[:10]  # Cap warnings at 10
    )
    
    # Add data issues to summary for visibility
    if data_issues:
        summary["data_issues"] = data_issues[:10]
    
    logging.info(
        f"[VERIFY] Complete: {summary['supported']}/{summary['total_claims']} supported, "
        f"{len(warnings)} warnings, {len(data_issues)} data issues"
    )
    
    return VerificationResult(summary=summary, claims=verifications)

