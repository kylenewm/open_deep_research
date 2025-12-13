"""
Pydantic models for financial fact extraction and verification.
"""
from __future__ import annotations

from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel


class EntityInfo(BaseModel):
    """Information about a company/entity resolved from ticker or name."""
    
    ticker: str
    company_name: str
    cik: str  # 10-digit zero-padded string
    fiscal_year_end: Optional[str] = None  # Month name: "January", "December", etc.


class DocumentSnapshot(BaseModel):
    """Immutable snapshot of an SEC filing document."""
    
    snapshot_id: str  # UUID
    url: str
    cik: str
    doc_type: str  # 10-K, 10-Q, 8-K
    retrieved_at: datetime
    content_hash: str  # SHA-256 of raw_html
    raw_html: str


class Location(BaseModel):
    """Identifies the exact location of a fact within an SEC filing."""
    
    cik: str
    doc_date: str
    doc_type: str  # 10-K, 10-Q, 8-K
    section_id: str  # Item7, Item1A, etc.
    
    # For TEXT facts (from paragraphs)
    paragraph_index: Optional[int] = None
    sentence_string: Optional[str] = None
    
    # For TABLE facts (from cells)
    table_index: Optional[int] = None
    row_index: Optional[int] = None
    column_index: Optional[int] = None
    row_label: Optional[str] = None      # e.g., "Data Center"
    column_label: Optional[str] = None   # e.g., "Oct 27, 2024"


class FactContext(BaseModel):
    """Additional context about a financial fact."""
    
    yoy_change: Optional[str] = None
    vs_guidance: Optional[str] = None


class Fact(BaseModel):
    """A single financial fact extracted from an SEC filing."""
    
    fact_id: str
    entity: str  # ticker
    metric: str
    value: Optional[float] = None
    unit: str
    period: str
    period_end_date: str  # EXACT date from source, e.g., "2024-10-27"
    location: Location
    source_format: Literal["html_text", "html_table"]
    
    # For table facts: the scale defined in the table header
    # e.g., "millions", "thousands", "billions"
    extracted_scale: Optional[str] = None
    
    doc_hash: str
    snapshot_id: str
    verification_status: str  # exact_match, approximate_match, mismatch, unverified
    negative_evidence: Optional[str] = None
    context: Optional[FactContext] = None


# =============================================================================
# Parsing Models
# =============================================================================


class Paragraph(BaseModel):
    """A paragraph extracted from a filing section."""
    
    index: int  # 0-based index within section
    text: str   # Clean text content (whitespace normalized)
    html: str   # Original HTML for reference


class Section(BaseModel):
    """A section of an SEC filing (e.g., Item 7, Item 1A)."""
    
    section_id: str       # Normalized: "Item7", "Item1A"
    title: str            # Full title text if available
    paragraphs: List[Paragraph]
    raw_html: str         # Original HTML of entire section


class CoverPageMetadata(BaseModel):
    """Metadata extracted from the filing cover page."""
    
    fiscal_period_end_date: Optional[str] = None  # e.g., "January 28, 2024"
    fiscal_period_type: Optional[str] = None      # "annual" or "quarterly"
    company_name: Optional[str] = None


class ParsedFiling(BaseModel):
    """A fully parsed SEC filing."""
    
    cik: str
    company_name: Optional[str] = None
    filing_type: Optional[str] = None
    filing_date: Optional[str] = None
    cover_page: CoverPageMetadata
    sections: List[Section]
    raw_html: str         # Original full document HTML


# =============================================================================
# Table Models
# =============================================================================


class TableCell(BaseModel):
    """A single cell extracted from a financial table."""
    
    value: str              # Raw string value from cell
    row_label: str          # Label from first column (e.g., "Data Center")
    column_label: str       # Header of the column (e.g., "Oct 27, 2024")
    row_index: int          # 0-based row index
    column_index: int       # 0-based column index
    effective_scale: Optional[str] = None  # Scale to apply for THIS cell (None for per-share)


class ExtractedTable(BaseModel):
    """A table extracted from an SEC filing."""
    
    table_index: int        # Index within document or section
    section_id: Optional[str] = None  # Which section it came from
    html: str               # Original table HTML
    headers: List[str]      # Column headers (flattened if MultiIndex)
    row_count: int          # Number of data rows
    column_count: int       # Number of columns
    scale: Optional[str] = None  # Base scale from header: "millions", "thousands", etc.
    has_per_share_exception: bool = False  # True if "except per share" in header
    
    # Store DataFrame as JSON for serialization
    dataframe_json: str
    
    model_config = {"arbitrary_types_allowed": True}

