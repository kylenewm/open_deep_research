"""
Verification Gate Pipeline.

The verification gate sits BETWEEN extraction and the fact store.
Facts that fail verification do NOT enter the store.
This is the critical anti-hallucination mechanism.

Bifurcated verification:
- Text facts: Verify sentence_string exists in source (whitespace-normalized)
- Table facts: Verify cell coordinates exist and value matches with scale
"""
from __future__ import annotations

import logging
import re
from io import StringIO
from typing import List, Optional, Tuple

import pandas as pd

from open_deep_research.models import (
    DocumentSnapshot,
    ExtractedTable,
    Fact,
    Section,
)
from open_deep_research.extraction import (
    extract_facts_from_text,
    extract_facts_from_table,
)
from open_deep_research.numeric_verification import (
    normalize_financial_value,
    verify_numeric_fact,
)
from open_deep_research.tables import extract_tables_from_section


logger = logging.getLogger(__name__)


# =============================================================================
# Whitespace Normalization
# =============================================================================


def normalize_for_comparison(text: str) -> str:
    """Normalize text for substring comparison.
    
    Handles whitespace differences between:
    - What the LLM extracts: "Revenue was $10B"
    - What the HTML contains: "Revenue  was\n$10B"
    
    Both normalize to: "revenue was $10b"
    
    Args:
        text: The text to normalize
        
    Returns:
        Normalized lowercase text with collapsed whitespace
    """
    if not text:
        return ""
    
    text = text.lower()
    text = text.replace('\xa0', ' ')  # &nbsp; unicode (non-breaking space)
    text = text.replace('&nbsp;', ' ')  # HTML entity
    text = text.replace('\n', ' ')
    text = text.replace('\r', ' ')
    text = text.replace('\t', ' ')
    text = re.sub(r'\s+', ' ', text)  # collapse multiple spaces
    return text.strip()


# =============================================================================
# Number Extraction from Text
# =============================================================================


def extract_number_from_text(text: str) -> Optional[float]:
    """Extract the first financial number from text.
    
    Args:
        text: Text that may contain a financial number
        
    Returns:
        Normalized float value, or None if no number found
    """
    if not text:
        return None
    
    patterns = [
        r'\$[\d,]+\.?\d*\s*[BMKTbmkt]?(?:illion|ousand)?',  # $10.5B, $10,500 million
        r'[\d,]+\.?\d*\s*(?:billion|million|thousand)',  # 10 billion
        r'\([\d,]+\.?\d*\)',  # (500) accounting notation
        r'[\d,]+\.?\d*',  # plain numbers
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                return normalize_financial_value(match.group())
            except Exception:
                continue
    
    return None


# =============================================================================
# Text Fact Verification
# =============================================================================


def verify_text_fact(fact: Fact, source_text: str) -> Fact:
    """Verify a text-based fact.
    
    For source_format == "html_text" only.
    
    Verification steps:
    1. Whitespace-safe sentence verification (sentence_string must exist in source)
    2. Numeric verification if value is present
    
    Args:
        fact: The Fact to verify (must have source_format == "html_text")
        source_text: The original source text the fact was extracted from
        
    Returns:
        The fact with updated verification_status
        
    Raises:
        ValueError: If fact is not a text fact
    """
    if fact.source_format != "html_text":
        raise ValueError("Use verify_table_fact for table facts")
    
    # Step 1: Whitespace-safe sentence verification
    if not fact.location.sentence_string:
        fact.verification_status = "mismatch"
        return fact
    
    normalized_sentence = normalize_for_comparison(fact.location.sentence_string)
    normalized_source = normalize_for_comparison(source_text)
    
    if normalized_sentence not in normalized_source:
        fact.verification_status = "mismatch"
        return fact
    
    # Step 2: Numeric verification if value present
    if fact.value is not None:
        extracted_value = extract_number_from_text(fact.location.sentence_string)
        if extracted_value is not None:
            # Compare extracted value from sentence with fact value
            status = verify_numeric_fact(fact.value, extracted_value)
            fact.verification_status = status
        else:
            # Sentence exists but no numeric value to verify
            fact.verification_status = "exact_match"
    else:
        # Non-numeric fact, sentence verification passed
        fact.verification_status = "exact_match"
    
    return fact


# =============================================================================
# Table Fact Verification
# =============================================================================


def verify_table_fact(fact: Fact, table: ExtractedTable) -> Fact:
    """Verify a table-based fact.
    
    For source_format == "html_table" only.
    
    Verification steps:
    1. Verify cell coordinates exist in the table
    2. Extract value from table cell
    3. Numeric comparison with scale
    
    Args:
        fact: The Fact to verify (must have source_format == "html_table")
        table: The ExtractedTable containing the fact
        
    Returns:
        The fact with updated verification_status
        
    Raises:
        ValueError: If fact is not a table fact
    """
    if fact.source_format != "html_table":
        raise ValueError("Use verify_text_fact for text facts")
    
    # Parse table DataFrame
    try:
        df = pd.read_json(StringIO(table.dataframe_json))
    except Exception as e:
        logger.warning(f"Failed to parse table DataFrame: {e}")
        fact.verification_status = "mismatch"
        return fact
    
    # Step 1: Verify coordinates exist
    row_idx = fact.location.row_index
    col_idx = fact.location.column_index
    
    if row_idx is None or col_idx is None:
        fact.verification_status = "mismatch"
        return fact
    
    if row_idx < 0 or row_idx >= len(df) or col_idx < 0 or col_idx >= len(df.columns):
        fact.verification_status = "mismatch"
        return fact
    
    # Step 2: Extract value from table cell
    cell_value = str(df.iloc[row_idx, col_idx])
    
    # Step 3: Numeric comparison with scale
    if fact.value is not None:
        # Use table scale for source value comparison
        # The claim value is already normalized (from extraction)
        # The cell value needs the table scale applied
        status = verify_numeric_fact(
            claim_value=fact.value,
            source_value=cell_value,
            claim_scale=None,  # fact.value is already in base units
            source_scale=table.scale  # Apply table scale to cell value
        )
        fact.verification_status = status
    else:
        # Non-numeric fact, coordinate verification passed
        fact.verification_status = "exact_match"
    
    return fact


# =============================================================================
# Deduplication
# =============================================================================


def deduplicate_facts(facts: List[Fact]) -> List[Fact]:
    """Remove duplicate facts based on (entity, metric, period, value).
    
    Keeps the first occurrence of each unique fact.
    
    Args:
        facts: List of facts to deduplicate
        
    Returns:
        List of unique facts
    """
    seen = set()
    unique = []
    
    for fact in facts:
        # Create deduplication key
        key = (
            fact.entity.upper(),
            fact.metric.lower(),
            fact.period,
            fact.value
        )
        
        if key not in seen:
            seen.add(key)
            unique.append(fact)
    
    return unique


# =============================================================================
# Main Processing Function
# =============================================================================


def process_extracted_facts(
    facts: List[Fact],
    source_text: str,
    tables: List[ExtractedTable]
) -> Tuple[List[Fact], List[Fact]]:
    """Verify each fact using appropriate method and separate into verified/rejected.
    
    Routes each fact to the appropriate verifier:
    - html_text facts → verify_text_fact
    - html_table facts → verify_table_fact
    
    Args:
        facts: List of extracted facts to verify
        source_text: The source text for text fact verification
        tables: List of ExtractedTable objects for table fact verification
        
    Returns:
        Tuple of (verified_facts, rejected_facts)
        - verified_facts: Facts with exact_match or approximate_match status (deduplicated)
        - rejected_facts: Facts with mismatch status
    """
    verified = []
    rejected = []
    
    # Index tables by table_index for quick lookup
    table_by_index = {t.table_index: t for t in tables}
    
    for fact in facts:
        if fact.source_format == "html_text":
            verified_fact = verify_text_fact(fact, source_text)
        elif fact.source_format == "html_table":
            table = table_by_index.get(fact.location.table_index)
            if table is None:
                fact.verification_status = "mismatch"
                verified_fact = fact
                logger.warning(
                    f"Table not found for fact: table_index={fact.location.table_index}, "
                    f"metric={fact.metric}"
                )
            else:
                verified_fact = verify_table_fact(fact, table)
        else:
            fact.verification_status = "mismatch"
            verified_fact = fact
            logger.warning(f"Unknown source_format: {fact.source_format}")
        
        if verified_fact.verification_status in ("exact_match", "approximate_match"):
            verified.append(verified_fact)
            logger.info(
                f"Verified ({fact.source_format}): {fact.metric} = {fact.value} "
                f"[{verified_fact.verification_status}]"
            )
        else:
            rejected.append(verified_fact)
            logger.warning(
                f"Rejected ({fact.source_format}): {fact.metric} = {fact.value} "
                f"[{verified_fact.verification_status}]"
            )
    
    # Deduplicate verified facts
    return deduplicate_facts(verified), rejected


# =============================================================================
# Full Extraction Pipeline
# =============================================================================


def extraction_pipeline(
    section: Section,
    entity: str,
    doc_snapshot: DocumentSnapshot
) -> Tuple[List[Fact], List[Fact]]:
    """Full pipeline for a section: extract, verify, and separate facts.
    
    This is the main entry point for processing a section.
    
    Args:
        section: The Section to process
        entity: Company ticker
        doc_snapshot: Document snapshot for metadata
        
    Returns:
        Tuple of (verified_facts, rejected_facts)
    """
    all_facts = []
    
    # Extract tables from section
    tables = extract_tables_from_section(section)
    
    # Extract facts from paragraphs
    for para in section.paragraphs:
        text_facts = extract_facts_from_text(
            text=para.text,
            entity=entity,
            doc_snapshot=doc_snapshot,
            section_id=section.section_id,
            paragraph_index=para.index
        )
        all_facts.extend(text_facts)
    
    # Extract facts from tables
    for table in tables:
        table_facts = extract_facts_from_table(
            table=table,
            entity=entity,
            doc_snapshot=doc_snapshot
        )
        all_facts.extend(table_facts)
    
    # Build source text from paragraphs
    source_text = "\n".join(para.text for para in section.paragraphs)
    
    # Process through verification gate
    verified, rejected = process_extracted_facts(all_facts, source_text, tables)
    
    logger.info(
        f"Section {section.section_id}: "
        f"{len(verified)} verified, {len(rejected)} rejected, "
        f"{len(tables)} tables"
    )
    
    return verified, rejected

