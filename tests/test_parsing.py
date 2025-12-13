"""
Tests for HTML parsing and section chunking.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from open_deep_research.parsing import (
    chunk_by_section,
    clean_text,
    extract_cover_page_metadata,
    extract_paragraphs,
    extract_section_id,
    is_toc_line,
    normalize_header_text,
    parse_filing_html,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_10k_html():
    """Load sample NVIDIA 10-K HTML if available."""
    fixture_path = Path(__file__).parent.parent / "fixtures" / "nvidia_10k_sample.html"
    if fixture_path.exists():
        return fixture_path.read_text(encoding="utf-8", errors="replace")
    return None


@pytest.fixture
def minimal_html():
    """Minimal HTML for basic testing."""
    return """
    <!DOCTYPE html>
    <html>
    <head><title>Test Filing</title></head>
    <body>
        <p>For the fiscal year ended January 28, 2024</p>
        <p>NVIDIA CORPORATION</p>
        
        <p>Table of Contents</p>
        <p>Item 1 .......... 5</p>
        <p>Item 7 .......... 45</p>
        <p>Item 8 .......... 67</p>
        
        <p><b>Item 1. Business</b></p>
        <p>We are a computing company. We design GPUs.</p>
        <p>Our products include gaming and data center solutions.</p>
        
        <p><b>Item 7. Management's Discussion and Analysis</b></p>
        <p>Revenue increased significantly in fiscal 2024.</p>
        <p>Data center revenue was the primary driver of growth.</p>
        <p>We expect continued demand for AI infrastructure.</p>
        
        <p><b>Item 8. Financial Statements</b></p>
        <p>See the consolidated financial statements.</p>
    </body>
    </html>
    """


# =============================================================================
# clean_text Tests
# =============================================================================


class TestCleanText:
    """Tests for text cleaning function."""
    
    def test_nbsp_replaced(self):
        """&nbsp; should be replaced with space."""
        assert clean_text("foo&nbsp;bar") == "foo bar"
    
    def test_unicode_nbsp_replaced(self):
        """Unicode non-breaking space should be replaced."""
        assert clean_text("foo\xa0bar") == "foo bar"
    
    def test_multiple_spaces_collapse(self):
        """Multiple spaces should collapse to single space."""
        assert clean_text("foo  bar") == "foo bar"
        assert clean_text("foo    bar") == "foo bar"
    
    def test_newlines_become_spaces(self):
        """Newlines should become spaces."""
        assert clean_text("foo\nbar") == "foo bar"
        assert clean_text("foo\r\nbar") == "foo bar"
    
    def test_tabs_become_spaces(self):
        """Tabs should become spaces."""
        assert clean_text("foo\tbar") == "foo bar"
    
    def test_combination(self):
        """Combination of whitespace issues should be handled."""
        assert clean_text("foo&nbsp;\n  bar") == "foo bar"
        assert clean_text("  foo\t\n\xa0 bar  ") == "foo bar"
    
    def test_preserves_normal_text(self):
        """Normal text should be preserved."""
        assert clean_text("Hello world") == "Hello world"


# =============================================================================
# TOC Detection Tests
# =============================================================================


class TestTocDetection:
    """Tests for Table of Contents line detection."""
    
    def test_dots_with_page_number(self):
        """Lines with dots followed by page number are TOC."""
        assert is_toc_line("Item 7 .......... 45") is True
        assert is_toc_line("Item 1A .... 12") is True
    
    def test_spaces_with_page_number(self):
        """Lines with spaces followed by page number are TOC."""
        assert is_toc_line("Item 7    45") is True
        assert is_toc_line("Item 1A      12") is True
    
    def test_page_keyword(self):
        """Lines containing 'page' are TOC."""
        assert is_toc_line("Item 7 (see page 45)") is True
        assert is_toc_line("See Page 10") is True
    
    def test_actual_headers_not_toc(self):
        """Actual section headers should NOT be detected as TOC."""
        assert is_toc_line("Item 7. Management's Discussion") is False
        assert is_toc_line("ITEM 7") is False
        assert is_toc_line("Item 1A. Risk Factors") is False
        assert is_toc_line("Item 7") is False


# =============================================================================
# Header Normalization Tests
# =============================================================================


class TestHeaderNormalization:
    """Tests for header text normalization."""
    
    def test_basic_normalization(self):
        """Basic text should be normalized."""
        assert normalize_header_text("  Item 7  ") == "item 7"
    
    def test_nbsp_handled(self):
        """&nbsp; should be normalized."""
        assert normalize_header_text("Item\xa07") == "item 7"
        assert normalize_header_text("Item&nbsp;7") == "item 7"


# =============================================================================
# Section ID Extraction Tests
# =============================================================================


class TestExtractSectionId:
    """Tests for section ID extraction from headers."""
    
    def test_item_7(self):
        """'Item 7' extracts to 'Item7'."""
        assert extract_section_id("Item 7") == "Item7"
    
    def test_item_7_uppercase(self):
        """'ITEM 7' extracts to 'Item7'."""
        assert extract_section_id("ITEM 7") == "Item7"
    
    def test_item_7_with_period(self):
        """'Item 7.' extracts to 'Item7'."""
        assert extract_section_id("Item 7.") == "Item7"
    
    def test_item_7_with_colon(self):
        """'Item 7:' extracts to 'Item7'."""
        assert extract_section_id("Item 7:") == "Item7"
    
    def test_item_7a(self):
        """'Item 7A' extracts to 'Item7A'."""
        assert extract_section_id("Item 7A") == "Item7A"
        assert extract_section_id("Item 7a") == "Item7A"
    
    def test_item_1a_with_period(self):
        """'Item 1A.' extracts to 'Item1A'."""
        assert extract_section_id("Item 1A.") == "Item1A"
    
    def test_item_7_with_title(self):
        """'ITEM 7. MANAGEMENT'S DISCUSSION' extracts to 'Item7'."""
        assert extract_section_id("ITEM 7. MANAGEMENT'S DISCUSSION") == "Item7"
    
    def test_item_nbsp(self):
        """'Item&nbsp;7' extracts to 'Item7'."""
        assert extract_section_id("Item\xa07") == "Item7"
    
    def test_regular_text_returns_none(self):
        """Regular text should return None."""
        assert extract_section_id("Revenue was strong") is None
        assert extract_section_id("The company reported") is None
    
    def test_toc_line_returns_none(self):
        """TOC lines should return None (filtered out)."""
        assert extract_section_id("Item 7 .......... 45") is None
        assert extract_section_id("Item 1A    12") is None
    
    def test_item_with_dash(self):
        """'Item 7 -' or 'Item 7 –' should extract."""
        assert extract_section_id("Item 7 - MD&A") == "Item7"
        assert extract_section_id("Item 7 – Discussion") == "Item7"
    
    def test_item_with_em_dash(self):
        """'Item 7—' should extract."""
        assert extract_section_id("Item 7—Analysis") == "Item7"


# =============================================================================
# Cover Page Metadata Tests
# =============================================================================


class TestCoverPageMetadata:
    """Tests for cover page metadata extraction."""
    
    def test_fiscal_year_extraction(self):
        """Should extract fiscal year end date."""
        html = "<html><body><p>For the fiscal year ended January 28, 2024</p></body></html>"
        metadata = extract_cover_page_metadata(html)
        assert metadata.fiscal_period_end_date == "January 28, 2024"
        assert metadata.fiscal_period_type == "annual"
    
    def test_quarterly_period_extraction(self):
        """Should extract quarterly period end date."""
        html = "<html><body><p>For the quarterly period ended October 27, 2024</p></body></html>"
        metadata = extract_cover_page_metadata(html)
        assert metadata.fiscal_period_end_date == "October 27, 2024"
        assert metadata.fiscal_period_type == "quarterly"
    
    def test_three_month_period_extraction(self):
        """Should extract three-month period end date."""
        html = "<html><body><p>For the three-month period ended July 28, 2024</p></body></html>"
        metadata = extract_cover_page_metadata(html)
        assert metadata.fiscal_period_end_date == "July 28, 2024"
        assert metadata.fiscal_period_type == "quarterly"
    
    def test_company_name_extraction(self):
        """Should extract company name."""
        html = "<html><body><p>NVIDIA CORPORATION</p><p>10-K</p></body></html>"
        metadata = extract_cover_page_metadata(html)
        assert metadata.company_name is not None
        assert "NVIDIA" in metadata.company_name


# =============================================================================
# Section Chunking Tests
# =============================================================================


class TestSectionChunking:
    """Tests for section chunking."""
    
    def test_identifies_item_7(self, minimal_html):
        """Should correctly identify Item 7 (MD&A)."""
        sections = chunk_by_section(minimal_html)
        section_ids = [s.section_id for s in sections]
        assert "Item7" in section_ids
    
    def test_identifies_item_8(self, minimal_html):
        """Should correctly identify Item 8 (Financial Statements)."""
        sections = chunk_by_section(minimal_html)
        section_ids = [s.section_id for s in sections]
        assert "Item8" in section_ids
    
    def test_identifies_item_1(self, minimal_html):
        """Should correctly identify Item 1."""
        sections = chunk_by_section(minimal_html)
        section_ids = [s.section_id for s in sections]
        assert "Item1" in section_ids
    
    def test_section_id_normalized(self, minimal_html):
        """Section IDs should be normalized ('Item7' not 'item 7')."""
        sections = chunk_by_section(minimal_html)
        for section in sections:
            # Check format: starts with "Item", followed by number, optional letter
            assert section.section_id.startswith("Item")
            assert section.section_id[4:].isalnum()
            # No spaces
            assert " " not in section.section_id
    
    def test_toc_not_matched_as_section(self, minimal_html):
        """TOC 'Item 7' should NOT be matched as a section start."""
        sections = chunk_by_section(minimal_html)
        # We should have exactly 3 sections (Item 1, 7, 8), not 6 (including TOC entries)
        assert len(sections) <= 4  # Allow for some flexibility
        # More specifically, check we don't have duplicate Item7
        item7_count = sum(1 for s in sections if s.section_id == "Item7")
        assert item7_count == 1
    
    def test_section_contains_content(self, minimal_html):
        """Sections should contain content until next section."""
        sections = chunk_by_section(minimal_html)
        item7 = next((s for s in sections if s.section_id == "Item7"), None)
        assert item7 is not None
        # Item 7 should have paragraphs
        assert len(item7.paragraphs) > 0


# =============================================================================
# Paragraph Extraction Tests
# =============================================================================


class TestParagraphExtraction:
    """Tests for paragraph extraction."""
    
    def test_paragraphs_indexed_correctly(self):
        """Paragraphs should be indexed 0, 1, 2, ..."""
        html = """
        <div>
            <p>First paragraph with enough text to pass length check.</p>
            <p>Second paragraph with enough text to pass length check.</p>
            <p>Third paragraph with enough text to pass length check.</p>
        </div>
        """
        paragraphs = extract_paragraphs(html)
        indices = [p.index for p in paragraphs]
        assert indices == list(range(len(paragraphs)))
    
    def test_empty_paragraphs_excluded(self):
        """Empty paragraphs should be excluded."""
        html = """
        <div>
            <p>Real paragraph with sufficient content here.</p>
            <p></p>
            <p>   </p>
            <p>Another real paragraph with content.</p>
        </div>
        """
        paragraphs = extract_paragraphs(html)
        for p in paragraphs:
            assert len(p.text) >= 10
    
    def test_paragraph_text_is_clean(self):
        """Paragraph text should have no excessive whitespace."""
        html = """
        <div>
            <p>This   has    multiple     spaces and needs cleaning.</p>
            <p>This\nhas\nnewlines\nand needs to be cleaned properly.</p>
        </div>
        """
        paragraphs = extract_paragraphs(html)
        for p in paragraphs:
            # No multiple spaces
            assert "  " not in p.text
            # No newlines
            assert "\n" not in p.text


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""
    
    def test_missing_sections_handled(self):
        """Should handle filings with missing sections."""
        html = """
        <html><body>
            <p><b>Item 7. Management's Discussion</b></p>
            <p>Some discussion content here.</p>
        </body></html>
        """
        sections = chunk_by_section(html)
        # Should have at least Item 7
        assert any(s.section_id == "Item7" for s in sections)
    
    def test_item_in_body_text_not_matched(self):
        """'Item 7' in body text should not be matched incorrectly."""
        html = """
        <html><body>
            <p><b>Item 7. Management's Discussion</b></p>
            <p>As discussed in Item 7 above, revenue increased.</p>
            <p><b>Item 8. Financial Statements</b></p>
            <p>See Item 7 for related discussion about revenue matters.</p>
        </body></html>
        """
        sections = chunk_by_section(html)
        # Should only have 2 sections, not more
        assert len(sections) == 2


# =============================================================================
# Integration Test with Real Fixture
# =============================================================================


class TestWithRealFixture:
    """Integration tests using real NVIDIA 10-K fixture if available."""
    
    def test_parses_real_10k(self, sample_10k_html):
        """Should parse real 10-K without errors."""
        if sample_10k_html is None:
            pytest.skip("No fixture file available - run fixture setup first")
        
        parsed = parse_filing_html(sample_10k_html, cik="0001045810")
        
        # Should have sections
        assert len(parsed.sections) > 0
        
        # Should have cover page metadata
        assert parsed.cover_page is not None
    
    def test_finds_key_sections(self, sample_10k_html):
        """Should find key sections in real 10-K."""
        if sample_10k_html is None:
            pytest.skip("No fixture file available - run fixture setup first")
        
        sections = chunk_by_section(sample_10k_html)
        section_ids = [s.section_id for s in sections]
        
        # Key sections that should exist in a 10-K
        expected_sections = ["Item1", "Item1A", "Item7", "Item8"]
        for expected in expected_sections:
            assert expected in section_ids, f"Missing expected section: {expected}"
    
    def test_extracts_fiscal_period(self, sample_10k_html):
        """Should extract fiscal period from real 10-K."""
        if sample_10k_html is None:
            pytest.skip("No fixture file available - run fixture setup first")
        
        metadata = extract_cover_page_metadata(sample_10k_html)
        
        # Should have fiscal period info
        assert metadata.fiscal_period_end_date is not None
        assert metadata.fiscal_period_type in ["annual", "quarterly", None]

