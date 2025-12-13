"""
Fact extraction from SEC filings.

Bifurcated extraction for text vs tables:
- Text facts: Include sentence_string, use paragraph_index
- Table facts: Use table coordinates, no sentences

The extraction outputs Fact objects matching the schema in models.py EXACTLY.
"""
from __future__ import annotations

import json
import logging
import os
import random
import re
import time
import uuid
from io import StringIO
from typing import List, Optional

import pandas as pd
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from open_deep_research.models import (
    DocumentSnapshot,
    ExtractedTable,
    Fact,
    Location,
)


# =============================================================================
# Prompt Templates
# =============================================================================


TEXT_EXTRACTION_PROMPT = '''
Extract financial facts from the following text about {entity}.

TEXT:
{text}

RULES:
1. Only extract facts that are EXPLICITLY stated in the text
2. Do NOT infer, calculate, or guess any values
3. Each fact must include sentence_string which is an EXACT QUOTE from the text
4. If no financial facts are present, return an empty list []

OUTPUT FORMAT (JSON array):
[
  {{
    "metric": "string - name of the metric (e.g., 'revenue', 'net_income', 'eps')",
    "value": number or null,
    "unit": "string - USD, shares, percent, etc.",
    "period": "string - e.g., 'Q3 FY2025', 'FY2024'",
    "period_end_date": "string - EXACT date from text if available, e.g., '2024-10-27'",
    "sentence_string": "string - EXACT quote from text containing this fact"
  }}
]

Return ONLY valid JSON. No explanations.
'''


TABLE_EXTRACTION_PROMPT = '''
Extract financial facts from the following table data for {entity}.

TABLE SCALE: {scale_info}

TABLE DATA:
{table_as_string}

COLUMN HEADERS: {headers}

RULES:
1. Do NOT generate sentences - tables do not have sentences
2. Return the EXACT row label from the first column
3. Return the EXACT column header for the period
4. The value is the RAW number from the cell (scale will be applied separately)
5. Only extract rows that represent financial metrics (Revenue, Net Income, etc.)
6. Skip header rows, total rows, or empty rows

OUTPUT FORMAT (JSON array):
[
  {{
    "metric": "string - the row label (e.g., 'Data Center', 'Total revenue')",
    "value": number or null,
    "unit": "string - USD (assume unless stated otherwise)",
    "period_end_date": "string - the EXACT column header date (e.g., 'Oct 27, 2024')",
    "row_label": "string - EXACT text from first column",
    "column_label": "string - EXACT column header",
    "row_index": number - 0-based row index,
    "column_index": number - 0-based column index
  }}
]

Return ONLY valid JSON. No explanations.
'''


# =============================================================================
# LLM Integration
# =============================================================================

# Retry configuration
MAX_RETRIES = 2  # 2 retries = 3 total attempts
INITIAL_BACKOFF = 1.0  # seconds
BACKOFF_MULTIPLIER = 2.0


def call_llm(prompt: str) -> str:
    """Call the LLM for fact extraction with retry logic.
    
    Uses Anthropic Claude with exponential backoff for rate limits and server errors.
    
    Args:
        prompt: The extraction prompt
        
    Returns:
        LLM response text
        
    Raises:
        Exception: If all retries exhausted
    """
    try:
        from anthropic import Anthropic, RateLimitError, APIStatusError
    except ImportError:
        logging.error("Anthropic package not installed. Please install with: pip install anthropic")
        raise
    
    client = Anthropic()  # Uses ANTHROPIC_API_KEY env var
    last_exception = None
    
    for attempt in range(MAX_RETRIES + 1):
        try:
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
            
        except RateLimitError as e:
            last_exception = e
            if attempt < MAX_RETRIES:
                delay = INITIAL_BACKOFF * (BACKOFF_MULTIPLIER ** attempt) + random.uniform(0, 0.5)
                logging.warning(f"Rate limited. Retry {attempt + 1}/{MAX_RETRIES} in {delay:.1f}s")
                time.sleep(delay)
            else:
                logging.error(f"Rate limit exceeded after {MAX_RETRIES} retries")
                raise
                
        except APIStatusError as e:
            # Retry on 5xx server errors only
            if e.status_code >= 500:
                last_exception = e
                if attempt < MAX_RETRIES:
                    delay = INITIAL_BACKOFF * (BACKOFF_MULTIPLIER ** attempt) + random.uniform(0, 0.5)
                    logging.warning(f"Server error ({e.status_code}). Retry {attempt + 1}/{MAX_RETRIES} in {delay:.1f}s")
                    time.sleep(delay)
                else:
                    logging.error(f"Server error after {MAX_RETRIES} retries")
                    raise
            else:
                # 4xx errors (except rate limit) should not retry
                logging.error(f"LLM API error ({e.status_code}): {e}")
                raise
                
        except Exception as e:
            logging.error(f"LLM call failed: {e}")
            raise
    
    # Should not reach here
    if last_exception:
        raise last_exception
    raise Exception("Unexpected error in LLM retry loop")


# =============================================================================
# Prompt Creation Functions
# =============================================================================


def create_text_extraction_prompt(text: str, entity: str) -> str:
    """Create the prompt for LLM extraction from TEXT.
    
    Args:
        text: The paragraph text to extract facts from
        entity: The company ticker or name
        
    Returns:
        Formatted prompt string
    """
    return TEXT_EXTRACTION_PROMPT.format(
        entity=entity,
        text=text
    )


def create_table_extraction_prompt(
    table_df: pd.DataFrame,
    entity: str,
    table_scale: Optional[str] = None
) -> str:
    """Create the prompt for LLM extraction from TABLES.
    
    Args:
        table_df: The table as a pandas DataFrame
        entity: The company ticker or name
        table_scale: Scale from table header (e.g., "millions")
        
    Returns:
        Formatted prompt string
    """
    # Format scale info
    if table_scale:
        scale_info = f"Numbers are in {table_scale} (e.g., 100 = 100 {table_scale})"
    else:
        scale_info = "No scale specified - numbers are as shown"
    
    # Convert DataFrame to string representation
    table_as_string = table_df.to_string()
    
    # Get column headers
    headers = ", ".join(str(col) for col in table_df.columns.tolist())
    
    return TABLE_EXTRACTION_PROMPT.format(
        entity=entity,
        scale_info=scale_info,
        table_as_string=table_as_string,
        headers=headers
    )


# =============================================================================
# Response Parsing
# =============================================================================


def parse_llm_response(response: str) -> List[dict]:
    """Parse LLM JSON response.
    
    Handles:
    - Plain JSON arrays
    - Markdown code blocks (```json ... ```)
    - Empty arrays
    
    Args:
        response: Raw LLM response string
        
    Returns:
        List of dicts that can be converted to Fact objects.
        Returns empty list on parse failure.
    """
    if not response or not response.strip():
        return []
    
    # Try to extract JSON from markdown code blocks
    text = response.strip()
    
    # Pattern: ```json\n...\n``` or ```\n...\n```
    code_block_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
    match = re.search(code_block_pattern, text, re.DOTALL)
    if match:
        text = match.group(1).strip()
    
    # Try to parse as JSON
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
        else:
            logging.warning(f"LLM response is not a list: {type(result)}")
            return []
    except json.JSONDecodeError as e:
        logging.warning(f"Failed to parse LLM response as JSON: {e}")
        logging.debug(f"Response was: {text[:500]}")
        return []


# =============================================================================
# Validation Functions
# =============================================================================


def validate_text_fact(fact: Fact, source_text: str) -> bool:
    """Verify the sentence_string is an exact substring of source.
    
    Args:
        fact: The extracted Fact object
        source_text: The original text the fact was extracted from
        
    Returns:
        True if valid (sentence_string found in source), False otherwise
    """
    if fact.source_format != "html_text":
        return True  # Not a text fact, skip validation
    
    if not fact.location.sentence_string:
        return False  # Text facts must have sentence_string
    
    return fact.location.sentence_string in source_text


def validate_table_fact(fact: Fact, table: ExtractedTable) -> bool:
    """Verify the cell coordinates exist in the table.
    
    Args:
        fact: The extracted Fact object
        table: The ExtractedTable the fact was extracted from
        
    Returns:
        True if valid (coordinates within bounds), False otherwise
    """
    if fact.source_format != "html_table":
        return True  # Not a table fact, skip validation
    
    if fact.location.row_index is None or fact.location.column_index is None:
        return False  # Table facts must have coordinates
    
    # Check bounds
    return (
        0 <= fact.location.row_index < table.row_count and
        0 <= fact.location.column_index < table.column_count
    )


# =============================================================================
# Text Fact Extraction
# =============================================================================


def extract_facts_from_text(
    text: str,
    entity: str,
    doc_snapshot: DocumentSnapshot,
    section_id: str,
    paragraph_index: int
) -> List[Fact]:
    """Extract financial facts from text using LLM.
    
    Args:
        text: The paragraph text to extract facts from
        entity: The company ticker
        doc_snapshot: The document snapshot for metadata
        section_id: The section ID (e.g., "Item7")
        paragraph_index: 0-based paragraph index within section
        
    Returns:
        List of Fact objects with verification_status = "unverified"
    """
    if not text or not text.strip():
        return []
    
    # Create prompt and call LLM
    prompt = create_text_extraction_prompt(text, entity)
    
    try:
        response = call_llm(prompt)
    except Exception as e:
        logging.error(f"LLM extraction failed: {e}")
        return []
    
    # Parse response
    raw_facts = parse_llm_response(response)
    
    if not raw_facts:
        return []
    
    # Convert to Fact objects
    facts = []
    for raw in raw_facts:
        try:
            fact = _create_text_fact(
                raw,
                entity=entity,
                doc_snapshot=doc_snapshot,
                section_id=section_id,
                paragraph_index=paragraph_index
            )
            
            # Validate: sentence_string must be in source text
            if validate_text_fact(fact, text):
                facts.append(fact)
            else:
                logging.warning(
                    f"Rejected fact: sentence_string not found in source text. "
                    f"metric={raw.get('metric')}"
                )
        except Exception as e:
            logging.warning(f"Failed to create text fact: {e}, raw={raw}")
            continue
    
    return facts


def _create_text_fact(
    raw: dict,
    entity: str,
    doc_snapshot: DocumentSnapshot,
    section_id: str,
    paragraph_index: int
) -> Fact:
    """Create a Fact object from raw LLM extraction for text.
    
    Args:
        raw: Raw extraction dict from LLM
        entity: Company ticker
        doc_snapshot: Document snapshot for metadata
        section_id: Section ID
        paragraph_index: Paragraph index within section
        
    Returns:
        Fact object
    """
    return Fact(
        fact_id=str(uuid.uuid4()),
        entity=entity,
        metric=raw.get("metric") or "",
        value=raw.get("value"),
        unit=raw.get("unit") or "USD",
        period=raw.get("period") or "",
        period_end_date=raw.get("period_end_date") or "",
        location=Location(
            cik=doc_snapshot.cik,
            doc_date=doc_snapshot.retrieved_at.strftime("%Y-%m-%d"),
            doc_type=doc_snapshot.doc_type,
            section_id=section_id,
            paragraph_index=paragraph_index,
            sentence_string=raw.get("sentence_string"),
            # Table fields are None for text facts
            table_index=None,
            row_index=None,
            column_index=None,
            row_label=None,
            column_label=None,
        ),
        source_format="html_text",
        extracted_scale=None,  # Text facts don't have scale
        doc_hash=doc_snapshot.content_hash,
        snapshot_id=doc_snapshot.snapshot_id,
        verification_status="unverified",
    )


# =============================================================================
# Table Fact Extraction
# =============================================================================


def extract_facts_from_table(
    table: ExtractedTable,
    entity: str,
    doc_snapshot: DocumentSnapshot
) -> List[Fact]:
    """Extract financial facts from a table using LLM.
    
    Does NOT generate sentences for table data - tables don't have sentences.
    Uses table coordinates (row_index, column_index, row_label, column_label).
    
    Args:
        table: The ExtractedTable to extract facts from
        entity: The company ticker
        doc_snapshot: The document snapshot for metadata
        
    Returns:
        List of Fact objects with verification_status = "unverified"
    """
    # Convert table to DataFrame
    try:
        df = pd.read_json(StringIO(table.dataframe_json))
    except Exception as e:
        logging.error(f"Failed to parse table DataFrame: {e}")
        return []
    
    if df.empty:
        return []
    
    # Create prompt and call LLM
    prompt = create_table_extraction_prompt(df, entity, table.scale)
    
    try:
        response = call_llm(prompt)
    except Exception as e:
        logging.error(f"LLM extraction failed: {e}")
        return []
    
    # Parse response
    raw_facts = parse_llm_response(response)
    
    if not raw_facts:
        return []
    
    # Convert to Fact objects
    facts = []
    for raw in raw_facts:
        try:
            fact = _create_table_fact(
                raw,
                entity=entity,
                doc_snapshot=doc_snapshot,
                table=table
            )
            
            # Validate: coordinates must be within table bounds
            if validate_table_fact(fact, table):
                facts.append(fact)
            else:
                logging.warning(
                    f"Rejected fact: coordinates out of bounds. "
                    f"metric={raw.get('metric')}, "
                    f"row={raw.get('row_index')}, col={raw.get('column_index')}"
                )
        except Exception as e:
            logging.warning(f"Failed to create table fact: {e}, raw={raw}")
            continue
    
    return facts


def _create_table_fact(
    raw: dict,
    entity: str,
    doc_snapshot: DocumentSnapshot,
    table: ExtractedTable
) -> Fact:
    """Create a Fact object from raw LLM extraction for table.
    
    Args:
        raw: Raw extraction dict from LLM
        entity: Company ticker
        doc_snapshot: Document snapshot for metadata
        table: The ExtractedTable source
        
    Returns:
        Fact object
    """
    return Fact(
        fact_id=str(uuid.uuid4()),
        entity=entity,
        metric=raw.get("metric") or "",
        value=raw.get("value"),
        unit=raw.get("unit") or "USD",
        period="",  # Tables use period_end_date from column header
        period_end_date=raw.get("period_end_date") or "",
        location=Location(
            cik=doc_snapshot.cik,
            doc_date=doc_snapshot.retrieved_at.strftime("%Y-%m-%d"),
            doc_type=doc_snapshot.doc_type,
            section_id=table.section_id or "",
            # Text fields are None for table facts
            paragraph_index=None,
            sentence_string=None,  # Tables do NOT have sentences
            # Table coordinates
            table_index=table.table_index,
            row_index=raw.get("row_index"),
            column_index=raw.get("column_index"),
            row_label=raw.get("row_label"),
            column_label=raw.get("column_label"),
        ),
        source_format="html_table",
        extracted_scale=table.scale,  # Table scale from header
        doc_hash=doc_snapshot.content_hash,
        snapshot_id=doc_snapshot.snapshot_id,
        verification_status="unverified",
    )


# =============================================================================
# Batch Extraction Helpers
# =============================================================================


def extract_all_facts_from_section(
    section_html: str,
    paragraphs: List[dict],
    tables: List[ExtractedTable],
    entity: str,
    doc_snapshot: DocumentSnapshot,
    section_id: str
) -> List[Fact]:
    """Extract all facts from a section (both text and tables).
    
    Args:
        section_html: Raw HTML of the section
        paragraphs: List of paragraph dicts with 'text' and 'index'
        tables: List of ExtractedTable objects from the section
        entity: Company ticker
        doc_snapshot: Document snapshot for metadata
        section_id: Section ID (e.g., "Item7")
        
    Returns:
        Combined list of all extracted facts
    """
    all_facts = []
    
    # Extract from paragraphs
    for para in paragraphs:
        text = para.get("text", "")
        index = para.get("index", 0)
        
        facts = extract_facts_from_text(
            text=text,
            entity=entity,
            doc_snapshot=doc_snapshot,
            section_id=section_id,
            paragraph_index=index
        )
        all_facts.extend(facts)
    
    # Extract from tables
    for table in tables:
        facts = extract_facts_from_table(
            table=table,
            entity=entity,
            doc_snapshot=doc_snapshot
        )
        all_facts.extend(facts)
    
    return all_facts

