"""
HTML parsing and section chunking for SEC filings.

Extracts structured content from SEC filing HTML documents,
including section identification, paragraph extraction, and
cover page metadata.
"""
from __future__ import annotations

import re
from typing import List, Optional

from bs4 import BeautifulSoup, NavigableString, Tag

from open_deep_research.models import (
    CoverPageMetadata,
    Paragraph,
    ParsedFiling,
    Section,
)


# =============================================================================
# Text Cleaning
# =============================================================================


def clean_text(text: str) -> str:
    """Normalize text for consistent storage and comparison.
    
    This is CRITICAL - the verification gate will compare extracted
    sentences against source text, and whitespace differences cause
    false rejections.
    
    - Replace &nbsp; with space
    - Replace newlines with space
    - Replace tabs with space
    - Collapse multiple spaces to single space
    - Strip leading/trailing whitespace
    """
    text = text.replace('\xa0', ' ')  # &nbsp; as unicode
    text = text.replace('&nbsp;', ' ')  # &nbsp; as string
    text = text.replace('\n', ' ')
    text = text.replace('\r', ' ')
    text = text.replace('\t', ' ')
    text = re.sub(r'\s+', ' ', text)  # collapse multiple spaces
    return text.strip()


# =============================================================================
# TOC Detection
# =============================================================================


def is_toc_line(text: str) -> bool:
    """Detect if this is a Table of Contents line, not an actual section header.
    
    TOC lines typically have:
    - Dots/periods followed by page numbers: "Item 7 .......... 45"
    - Just a page number at the end: "Item 7    45"
    - The word "page" nearby
    """
    # Pattern: dots followed by number (page reference)
    if re.search(r'\.{3,}\s*\d+\s*$', text):
        return True
    # Pattern: ends with standalone number after multiple spaces (page number)
    if re.search(r'\s{2,}\d+\s*$', text):
        return True
    # Pattern: contains "page"
    if re.search(r'\bpage\b', text, re.IGNORECASE):
        return True
    return False


# =============================================================================
# Header Matching
# =============================================================================


def normalize_header_text(text: str) -> str:
    """Normalize text for header matching."""
    text = text.replace('\xa0', ' ')
    text = text.replace('&nbsp;', ' ')
    text = re.sub(r'\s+', ' ', text)
    text = text.strip().lower()
    return text


def extract_section_id(text: str) -> Optional[str]:
    """Extract normalized section ID from header text.
    
    Returns: "Item7", "Item1A", etc. or None if not a section header.
    """
    # First, check if this is a TOC line
    if is_toc_line(text):
        return None
    
    normalized = normalize_header_text(text)
    # Pattern matches: item 1, item 1a, item 1., item 1a., etc.
    # Must be at start of string to avoid matching "Item 7" in body text
    pattern = re.compile(r'^item\s*(\d+)([a-z])?[\.\:\s\-–—]?', re.IGNORECASE)
    match = pattern.match(normalized)
    if match:
        num = match.group(1)
        letter = (match.group(2) or '').upper()
        return f"Item{num}{letter}"
    return None


# =============================================================================
# Cover Page Metadata Extraction
# =============================================================================


def extract_cover_page_metadata(html_content: str) -> CoverPageMetadata:
    """Extract metadata from the filing cover page.
    
    Looks for:
    - Fiscal year end date: "For the fiscal year ended January 28, 2024"
    - Fiscal quarter: "For the quarterly period ended October 27, 2024"
    - Company name
    
    Returns CoverPageMetadata with extracted values.
    """
    soup = BeautifulSoup(html_content, 'lxml')
    metadata = CoverPageMetadata()
    
    # Look for fiscal year/quarter patterns in first ~20% of document
    # For short documents (like tests), use the whole text
    full_text = soup.get_text()
    text_length = max(len(full_text) // 5, len(full_text))  # Use full text if short
    text = full_text[:text_length] if len(full_text) > 5000 else full_text
    
    # Pattern: "For the fiscal year ended [DATE]"
    annual_match = re.search(
        r'for the (?:fiscal )?year ended\s+([A-Za-z]+\s+\d{1,2},?\s+\d{4})',
        text, re.IGNORECASE
    )
    if annual_match:
        metadata.fiscal_period_end_date = annual_match.group(1)
        metadata.fiscal_period_type = 'annual'
    
    # Pattern: "For the quarterly period ended [DATE]"
    quarterly_match = re.search(
        r'for the (?:fiscal )?(?:quarterly |three[- ]month )?period ended\s+([A-Za-z]+\s+\d{1,2},?\s+\d{4})',
        text, re.IGNORECASE
    )
    if quarterly_match:
        metadata.fiscal_period_end_date = quarterly_match.group(1)
        metadata.fiscal_period_type = 'quarterly'
    
    # Try to extract company name from the beginning
    # Usually appears prominently at the top
    company_patterns = [
        r'([A-Z][A-Z\s,\.]+(?:INC|CORP|CORPORATION|LLC|LTD|CO|COMPANY)\.?)',
        r'NVIDIA\s+CORPORATION',
    ]
    for pattern in company_patterns:
        company_match = re.search(pattern, text[:2000], re.IGNORECASE)
        if company_match:
            metadata.company_name = clean_text(company_match.group(0))
            break
    
    return metadata


# =============================================================================
# Section Chunking
# =============================================================================


# Valid SEC filing section IDs
VALID_SECTION_IDS = {
    "Item1", "Item1A", "Item1B", "Item1C", "Item2", "Item3", "Item4", "Item5",
    "Item6", "Item7", "Item7A", "Item8", "Item9", "Item9A", "Item9B",
    "Item10", "Item11", "Item12", "Item13", "Item14", "Item15", "Item16"
}


def _find_section_headers(soup: BeautifulSoup) -> List[tuple]:
    """Find all section header elements in the document.
    
    Returns list of (element, section_id, title) tuples.
    """
    headers = []
    
    # Look for elements that might contain section headers
    # SEC filings often use various elements for headers
    candidate_elements = soup.find_all(['p', 'div', 'span', 'b', 'font', 'h1', 'h2', 'h3', 'h4', 'td'])
    
    for elem in candidate_elements:
        # Get direct text content
        text = elem.get_text(strip=True)
        if not text:
            continue
        
        # Skip very long text (not a header)
        if len(text) > 200:
            continue
        
        section_id = extract_section_id(text)
        if section_id and section_id in VALID_SECTION_IDS:
            # Extract full title if available
            title = clean_text(text)
            headers.append((elem, section_id, title))
    
    return headers


def _deduplicate_headers(headers: List[tuple]) -> List[tuple]:
    """Remove duplicate/nested header matches.
    
    When we find "Item 7" in both a <p> and its child <b>,
    keep only one occurrence.
    """
    if not headers:
        return headers
    
    # Sort by position in document (element order)
    # Then deduplicate by section_id if they're close together
    seen_ids = {}
    result = []
    
    for elem, section_id, title in headers:
        # If we've seen this section_id recently, skip
        # (within last 3 elements to handle nesting)
        if section_id in seen_ids:
            prev_idx = seen_ids[section_id]
            if len(result) - prev_idx < 3:
                continue
        
        seen_ids[section_id] = len(result)
        result.append((elem, section_id, title))
    
    return result


def chunk_by_section(html_content: str) -> List[Section]:
    """Split document into sections based on Item headers.
    
    Uses regex to find section headers directly in HTML, then slices the HTML 
    string to extract content between headers. This is more reliable than DOM 
    traversal for deeply nested SEC filing structures.
    
    Sections to identify: Item 1 through Item 16 (including A/B variants).
    Each section contains all content until the next Item header.
    TOC entries are filtered out.
    """
    # Pattern to match Item headers in HTML
    # Matches patterns like: >Item 7<, >ITEM 7.<, >Item 7 -<, etc.
    # The > ensures we're matching after a tag, avoiding matches in attributes
    header_pattern = re.compile(
        r'>[\s\n]*'  # After a tag close, optional whitespace
        r'((?:ITEM|Item)\s*(\d+)([A-Za-z])?)'  # Item N or Item NA
        r'[\.\:\s\-–—]*'  # Optional punctuation
        r'([^<]{0,150}?)'  # Optional title text (non-greedy, up to 150 chars)
        r'[\s\n]*<',  # Before next tag
        re.IGNORECASE | re.DOTALL
    )
    
    # Find all matches with positions
    matches = []
    for match in header_pattern.finditer(html_content):
        full_header = match.group(1)
        num = match.group(2)
        letter = (match.group(3) or '').upper()
        title_text = match.group(4).strip() if match.group(4) else ''
        section_id = f"Item{num}{letter}"
        
        if section_id not in VALID_SECTION_IDS:
            continue
        
        # Check if this is a TOC line by examining surrounding context
        start_ctx = max(0, match.start() - 100)
        end_ctx = min(len(html_content), match.end() + 100)
        context = html_content[start_ctx:end_ctx]
        
        # Extract just text from context for TOC check
        context_text = re.sub(r'<[^>]+>', ' ', context)
        if is_toc_line(context_text):
            continue
        
        # Build title
        title = clean_text(full_header + ' ' + title_text) if title_text else clean_text(full_header)
        
        matches.append({
            'section_id': section_id,
            'start': match.start(),
            'title': title,
        })
    
    if not matches:
        return []
    
    # Deduplicate: for each section_id, keep only the LAST occurrence
    # (TOC appears first, actual content appears later in document)
    seen = {}
    for m in matches:
        seen[m['section_id']] = m  # Later occurrences overwrite earlier
    
    # Sort by position to maintain document order
    deduped = sorted(seen.values(), key=lambda x: x['start'])
    
    # Build sections by slicing HTML
    sections = []
    for i, m in enumerate(deduped):
        start_pos = m['start']
        end_pos = deduped[i + 1]['start'] if i + 1 < len(deduped) else len(html_content)
        
        section_html = html_content[start_pos:end_pos]
        paragraphs = extract_paragraphs(section_html)
        
        sections.append(Section(
            section_id=m['section_id'],
            title=m['title'],
            paragraphs=paragraphs,
            raw_html=section_html,
        ))
    
    return sections


# =============================================================================
# Paragraph Extraction
# =============================================================================


def extract_paragraphs(section_html: str) -> List[Paragraph]:
    """Extract paragraphs from a section's HTML.
    
    - Track paragraph index within section (0-based)
    - Use clean_text() on all paragraph text
    - Skip empty paragraphs or navigation elements
    """
    soup = BeautifulSoup(section_html, 'lxml')
    paragraphs = []
    index = 0
    
    # Find paragraph-like elements
    for elem in soup.find_all(['p', 'div']):
        # Get text content
        text = elem.get_text()
        cleaned = clean_text(text)
        
        # Skip empty or very short paragraphs
        if len(cleaned) < 10:
            continue
        
        # Skip navigation elements
        if _is_navigation_element(elem, cleaned):
            continue
        
        # Get original HTML
        html = str(elem)
        
        paragraphs.append(Paragraph(
            index=index,
            text=cleaned,
            html=html,
        ))
        index += 1
    
    return paragraphs


def _is_navigation_element(elem: Tag, text: str) -> bool:
    """Check if this element is a navigation/boilerplate element."""
    # Skip table of contents links
    if 'table of contents' in text.lower():
        return True
    # Skip page numbers
    if re.match(r'^\d+$', text):
        return True
    # Skip "Back to top" type links
    if 'back to' in text.lower():
        return True
    return False


# =============================================================================
# Main Parsing Function
# =============================================================================


def parse_filing_html(html_content: str, cik: str = "") -> ParsedFiling:
    """Parse an SEC filing HTML document.
    
    Args:
        html_content: Raw HTML content of the filing
        cik: CIK number (optional, for metadata)
    
    Returns:
        ParsedFiling with extracted structure and content
    """
    # Extract cover page metadata
    cover_page = extract_cover_page_metadata(html_content)
    
    # Extract sections
    sections = chunk_by_section(html_content)
    
    return ParsedFiling(
        cik=cik,
        company_name=cover_page.company_name,
        filing_type=None,  # Could be extracted from metadata
        filing_date=None,  # Could be extracted from metadata
        cover_page=cover_page,
        sections=sections,
        raw_html=html_content,
    )

