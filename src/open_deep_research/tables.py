"""
Table extraction from SEC filings.

Extracts structured financial tables from SEC filing HTML,
handling scale indicators, per-share exceptions, and MultiIndex headers.
"""
from __future__ import annotations

import re
from datetime import datetime
from io import StringIO
from typing import List, Optional, Tuple

import pandas as pd
from bs4 import BeautifulSoup

from open_deep_research.models import ExtractedTable, Section, TableCell


# =============================================================================
# Scale and Unit Extraction
# =============================================================================


def extract_table_scale(table_html: str) -> Tuple[Optional[str], bool]:
    """Extract the scale/unit from table header or caption.
    
    Returns:
        tuple: (scale, has_per_share_exception)
        - scale: "millions", "thousands", "billions", or None
        - has_per_share_exception: True if header says "except per share"
    
    Looks for patterns like:
    - "(in millions)"
    - "(in thousands)"
    - "($ in millions)"
    - "(in millions, except per share data)"
    - "Amounts in thousands"
    """
    html_lower = table_html.lower()
    
    # Check for per-share exception
    has_per_share_exception = bool(re.search(
        r'except\s+(?:per\s+share|eps)',
        html_lower
    ))
    
    # Extract scale
    scale_patterns = [
        (r'\(\s*(?:\$\s*)?in\s+millions', 'millions'),
        (r'\(\s*(?:\$\s*)?in\s+thousands', 'thousands'),
        (r'\(\s*(?:\$\s*)?in\s+billions', 'billions'),
        (r'amounts?\s+in\s+millions', 'millions'),
        (r'amounts?\s+in\s+thousands', 'thousands'),
        (r'amounts?\s+in\s+billions', 'billions'),
    ]
    
    scale = None
    for pattern, scale_name in scale_patterns:
        if re.search(pattern, html_lower):
            scale = scale_name
            break
    
    return scale, has_per_share_exception


# =============================================================================
# Per-Share Detection
# =============================================================================


def is_per_share_metric(row_label: str) -> bool:
    """Check if a row represents a per-share metric.
    
    Per-share metrics should NOT have table scale applied.
    
    Examples that return True:
    - "Earnings per share"
    - "EPS"
    - "Diluted EPS"
    - "Basic earnings per share"
    - "Net income per share"
    - "Dividends per share"
    """
    label_lower = row_label.lower()
    
    per_share_patterns = [
        r'per\s+share',
        r'\beps\b',
        r'per\s+common\s+share',
        r'per\s+diluted\s+share',
        r'dividend.*per',
    ]
    
    for pattern in per_share_patterns:
        if re.search(pattern, label_lower):
            return True
    
    return False


def get_effective_scale(
    table_scale: Optional[str],
    has_per_share_exception: bool,
    row_label: str
) -> Optional[str]:
    """Get the effective scale to apply for a specific row.
    
    If the table header says "except per share" AND the row is a 
    per-share metric, return None (no scale).
    
    Args:
        table_scale: The scale from table header ("millions", etc.)
        has_per_share_exception: Whether header had "except per share"
        row_label: The label of the row being extracted
    
    Returns:
        The scale to apply, or None for no scaling
    """
    if table_scale is None:
        return None
    
    # If header says "except per share" and this is a per-share row
    if has_per_share_exception and is_per_share_metric(row_label):
        return None
    
    # Even without explicit exception, be safe with per-share metrics
    if is_per_share_metric(row_label):
        return None
    
    return table_scale


# =============================================================================
# MultiIndex Handling
# =============================================================================


def flatten_multiindex_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten MultiIndex columns into single-level strings.
    
    Converts:
        ('Three Months Ended', 'Oct 27, 2024')
    To:
        'Three Months Ended Oct 27, 2024'
    """
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            ' '.join(str(c) for c in col if str(c) != 'nan' and pd.notna(c)).strip()
            for col in df.columns.values
        ]
    return df


# =============================================================================
# Table Extraction
# =============================================================================


def extract_tables_from_html(
    html_content: str,
    section_id: Optional[str] = None
) -> List[ExtractedTable]:
    """Extract all tables from HTML content.
    
    Args:
        html_content: Raw HTML content
        section_id: Optional section ID for tracking
    
    Returns:
        List of ExtractedTable objects
    """
    tables = []
    soup = BeautifulSoup(html_content, 'lxml')
    
    # Find all table elements
    table_elements = soup.find_all('table')
    
    for idx, table_elem in enumerate(table_elements):
        try:
            table_html = str(table_elem)
            
            # Extract scale and per-share exception
            scale, has_per_share_exception = extract_table_scale(table_html)
            
            # Parse table with pandas
            dfs = pd.read_html(StringIO(table_html))
            if not dfs:
                continue
            
            df = dfs[0]
            
            # Skip very small tables (likely formatting)
            if df.shape[0] < 2 or df.shape[1] < 2:
                continue
            
            # Flatten MultiIndex columns
            df = flatten_multiindex_columns(df)
            
            # Clean up column names
            df.columns = [str(col).strip() for col in df.columns]
            
            # Create ExtractedTable
            headers = df.columns.tolist()
            
            extracted = ExtractedTable(
                table_index=idx,
                section_id=section_id,
                html=table_html,
                headers=headers,
                row_count=len(df),
                column_count=len(df.columns),
                scale=scale,
                has_per_share_exception=has_per_share_exception,
                dataframe_json=df.to_json(),
            )
            
            tables.append(extracted)
            
        except Exception:
            # Some tables may fail to parse - skip them
            continue
    
    return tables


def extract_tables_from_section(section: Section) -> List[ExtractedTable]:
    """Extract tables from a specific section.
    
    Args:
        section: Section object with raw_html
    
    Returns:
        List of ExtractedTable objects with section_id set
    """
    return extract_tables_from_html(section.raw_html, section_id=section.section_id)


# =============================================================================
# Table Search and Value Extraction
# =============================================================================


def find_table_by_metric(
    tables: List[ExtractedTable],
    metric_keywords: List[str]
) -> Optional[ExtractedTable]:
    """Search tables for one containing specific metrics.
    
    Args:
        tables: List of ExtractedTable objects
        metric_keywords: Keywords to search for in row labels
    
    Returns:
        First matching table or None
    """
    for table in tables:
        try:
            df = pd.read_json(StringIO(table.dataframe_json))
            
            # Check first column for row labels
            if df.shape[1] < 1:
                continue
            
            first_col = df.iloc[:, 0].astype(str).str.lower()
            
            for keyword in metric_keywords:
                keyword_lower = keyword.lower()
                if first_col.str.contains(keyword_lower, regex=False).any():
                    return table
                    
        except Exception:
            continue
    
    return None


def extract_value_from_table(
    table: ExtractedTable,
    row_pattern: str,
    column_index: int = -1
) -> Optional[TableCell]:
    """Extract a specific cell value from a table.
    
    Args:
        table: ExtractedTable object
        row_pattern: Regex pattern to match row label (case-insensitive)
        column_index: Which column to extract (-1 = last column)
    
    Returns:
        TableCell with value and location info, or None if not found
    """
    try:
        df = pd.read_json(StringIO(table.dataframe_json))
        
        if df.empty or df.shape[1] < 2:
            return None
        
        # Get first column as row labels
        row_labels = df.iloc[:, 0].astype(str)
        
        # Find matching row
        pattern = re.compile(row_pattern, re.IGNORECASE)
        matching_rows = []
        
        for i, label in enumerate(row_labels):
            if pattern.search(label):
                matching_rows.append((i, label))
        
        if not matching_rows:
            return None
        
        # Use first match
        row_idx, row_label = matching_rows[0]
        
        # Handle column index
        if column_index < 0:
            column_index = df.shape[1] + column_index
        
        if column_index < 0 or column_index >= df.shape[1]:
            return None
        
        # Get value
        value = str(df.iloc[row_idx, column_index])
        
        # Get column label
        column_label = str(df.columns[column_index])
        
        # Calculate effective scale
        effective_scale = get_effective_scale(
            table.scale,
            table.has_per_share_exception,
            row_label
        )
        
        return TableCell(
            value=value,
            row_label=row_label,
            column_label=column_label,
            row_index=row_idx,
            column_index=column_index,
            effective_scale=effective_scale,
        )
        
    except Exception:
        return None


# =============================================================================
# Table Type Identification
# =============================================================================


def identify_table_type(table: ExtractedTable) -> Optional[str]:
    """Identify the type of financial table.
    
    Returns one of:
    - "income_statement"
    - "balance_sheet"
    - "cash_flow"
    - "segment_revenue"
    - "unknown"
    - None (if cannot determine)
    """
    try:
        df = pd.read_json(StringIO(table.dataframe_json))
        
        if df.empty or df.shape[1] < 1:
            return None
        
        # Get all text from first column (row labels)
        labels_text = ' '.join(df.iloc[:, 0].astype(str)).lower()
        
        # Also check table HTML for captions/headers
        html_lower = table.html.lower()
        combined_text = labels_text + ' ' + html_lower
        
        # Income statement indicators
        income_patterns = [
            'revenue', 'net income', 'operating income', 'gross profit',
            'cost of revenue', 'operating expense', 'earnings per share',
        ]
        income_matches = sum(1 for p in income_patterns if p in combined_text)
        
        # Balance sheet indicators
        balance_patterns = [
            'total assets', 'total liabilities', 'stockholders\' equity',
            'cash and cash equivalents', 'accounts receivable', 'inventory',
            'property and equipment', 'accounts payable',
        ]
        balance_matches = sum(1 for p in balance_patterns if p in combined_text)
        
        # Cash flow indicators
        cashflow_patterns = [
            'cash flows from operating', 'cash flows from investing',
            'cash flows from financing', 'depreciation', 'amortization',
            'change in working capital', 'capital expenditures',
        ]
        cashflow_matches = sum(1 for p in cashflow_patterns if p in combined_text)
        
        # Segment revenue indicators
        segment_patterns = [
            'data center', 'gaming', 'professional visualization',
            'automotive', 'segment', 'by market', 'by region',
        ]
        segment_matches = sum(1 for p in segment_patterns if p in combined_text)
        
        # Determine type based on matches
        scores = {
            'income_statement': income_matches,
            'balance_sheet': balance_matches,
            'cash_flow': cashflow_matches,
            'segment_revenue': segment_matches,
        }
        
        max_score = max(scores.values())
        if max_score < 2:
            return 'unknown'
        
        for table_type, score in scores.items():
            if score == max_score:
                return table_type
        
        return 'unknown'
        
    except Exception:
        return None


# =============================================================================
# Column Date Extraction
# =============================================================================


# Month name mappings
MONTH_NAMES = {
    'jan': 1, 'january': 1,
    'feb': 2, 'february': 2,
    'mar': 3, 'march': 3,
    'apr': 4, 'april': 4,
    'may': 5,
    'jun': 6, 'june': 6,
    'jul': 7, 'july': 7,
    'aug': 8, 'august': 8,
    'sep': 9, 'sept': 9, 'september': 9,
    'oct': 10, 'october': 10,
    'nov': 11, 'november': 11,
    'dec': 12, 'december': 12,
}


def get_column_date(table: ExtractedTable, column_index: int) -> Optional[str]:
    """Extract the date from a column header.
    
    Args:
        table: ExtractedTable object
        column_index: Which column to get date from
    
    Returns:
        ISO format date string (YYYY-MM-DD) or None
    """
    if column_index < 0:
        column_index = len(table.headers) + column_index
    
    if column_index < 0 or column_index >= len(table.headers):
        return None
    
    header = table.headers[column_index]
    
    return parse_date_from_text(header)


def parse_date_from_text(text: str) -> Optional[str]:
    """Parse a date from text like 'Oct 27, 2024' or 'October 27, 2024'.
    
    Returns ISO format YYYY-MM-DD or None.
    """
    # Pattern: Month Day, Year (e.g., "Oct 27, 2024" or "October 27, 2024")
    pattern = r'([A-Za-z]+)\s+(\d{1,2}),?\s+(\d{4})'
    match = re.search(pattern, text)
    
    if match:
        month_str = match.group(1).lower()
        day = int(match.group(2))
        year = int(match.group(3))
        
        month = MONTH_NAMES.get(month_str)
        if month:
            try:
                date = datetime(year, month, day)
                return date.strftime('%Y-%m-%d')
            except ValueError:
                pass
    
    # Pattern: YYYY-MM-DD
    iso_pattern = r'(\d{4})-(\d{2})-(\d{2})'
    match = re.search(iso_pattern, text)
    if match:
        return match.group(0)
    
    return None


# =============================================================================
# Utility Functions for ExtractedTable
# =============================================================================


def table_to_dataframe(table: ExtractedTable) -> pd.DataFrame:
    """Reconstruct DataFrame from ExtractedTable."""
    return pd.read_json(StringIO(table.dataframe_json))


def dataframe_to_extracted_table(
    df: pd.DataFrame,
    table_index: int,
    section_id: Optional[str],
    html: str,
    scale: Optional[str] = None,
    has_per_share_exception: bool = False
) -> ExtractedTable:
    """Create ExtractedTable from a DataFrame."""
    headers = [str(col) for col in df.columns.tolist()]
    return ExtractedTable(
        table_index=table_index,
        section_id=section_id,
        html=html,
        headers=headers,
        row_count=len(df),
        column_count=len(df.columns),
        scale=scale,
        has_per_share_exception=has_per_share_exception,
        dataframe_json=df.to_json(),
    )

