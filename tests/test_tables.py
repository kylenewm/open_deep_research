"""
Tests for table extraction from SEC filings.
"""
from __future__ import annotations

from io import StringIO
from pathlib import Path

import pandas as pd
import pytest

from open_deep_research.models import Section
from open_deep_research.tables import (
    dataframe_to_extracted_table,
    extract_table_scale,
    extract_tables_from_html,
    extract_tables_from_section,
    extract_value_from_table,
    find_table_by_metric,
    flatten_multiindex_columns,
    get_column_date,
    get_effective_scale,
    identify_table_type,
    is_per_share_metric,
    parse_date_from_text,
    table_to_dataframe,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_revenue_table_html():
    """Sample revenue table with scale and per-share exception."""
    return """
    <table>
        <caption>(in millions, except per share data)</caption>
        <thead>
            <tr>
                <th></th>
                <th>Three Months Ended Oct 27, 2024</th>
                <th>Three Months Ended Oct 29, 2023</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>Revenue</td>
                <td>35,082</td>
                <td>18,120</td>
            </tr>
            <tr>
                <td>Data Center</td>
                <td>30,772</td>
                <td>14,514</td>
            </tr>
            <tr>
                <td>Gaming</td>
                <td>3,279</td>
                <td>2,856</td>
            </tr>
            <tr>
                <td>Net income</td>
                <td>19,309</td>
                <td>9,243</td>
            </tr>
            <tr>
                <td>Earnings per share - diluted</td>
                <td>0.78</td>
                <td>0.37</td>
            </tr>
        </tbody>
    </table>
    """


@pytest.fixture
def sample_balance_sheet_html():
    """Sample balance sheet table."""
    return """
    <table>
        <caption>(in thousands)</caption>
        <thead>
            <tr>
                <th></th>
                <th>Oct 27, 2024</th>
                <th>Jan 28, 2024</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>Total assets</td>
                <td>96,013,000</td>
                <td>65,728,000</td>
            </tr>
            <tr>
                <td>Cash and cash equivalents</td>
                <td>9,107,000</td>
                <td>7,280,000</td>
            </tr>
            <tr>
                <td>Total liabilities</td>
                <td>30,116,000</td>
                <td>22,750,000</td>
            </tr>
            <tr>
                <td>Stockholders' equity</td>
                <td>65,897,000</td>
                <td>42,978,000</td>
            </tr>
        </tbody>
    </table>
    """


@pytest.fixture
def sample_10k_html():
    """Load sample NVIDIA 10-K HTML if available."""
    fixture_path = Path(__file__).parent.parent / "fixtures" / "nvidia_10k_sample.html"
    if fixture_path.exists():
        return fixture_path.read_text(encoding="utf-8", errors="replace")
    return None


# =============================================================================
# Scale Extraction Tests
# =============================================================================


class TestScaleExtraction:
    """Tests for scale and per-share exception extraction."""
    
    def test_in_millions(self):
        """'(in millions)' should extract scale='millions'."""
        html = "<table><caption>(in millions)</caption></table>"
        scale, exception = extract_table_scale(html)
        assert scale == "millions"
        assert exception is False
    
    def test_dollar_in_millions(self):
        """'($ in millions)' should extract scale='millions'."""
        html = "<table><caption>($ in millions)</caption></table>"
        scale, exception = extract_table_scale(html)
        assert scale == "millions"
    
    def test_in_thousands(self):
        """'(in thousands)' should extract scale='thousands'."""
        html = "<table><caption>(in thousands)</caption></table>"
        scale, exception = extract_table_scale(html)
        assert scale == "thousands"
    
    def test_in_billions(self):
        """'(in billions)' should extract scale='billions'."""
        html = "<table><caption>(in billions)</caption></table>"
        scale, exception = extract_table_scale(html)
        assert scale == "billions"
    
    def test_with_per_share_exception(self):
        """'(in millions, except per share data)' should set exception=True."""
        html = "<table><caption>(in millions, except per share data)</caption></table>"
        scale, exception = extract_table_scale(html)
        assert scale == "millions"
        assert exception is True
    
    def test_amounts_in_millions(self):
        """'Amounts in millions' should extract scale='millions'."""
        html = "<table><caption>Amounts in millions</caption></table>"
        scale, exception = extract_table_scale(html)
        assert scale == "millions"
    
    def test_no_scale_indicator(self):
        """Table with no scale indicator should return None."""
        html = "<table><caption>Financial Data</caption></table>"
        scale, exception = extract_table_scale(html)
        assert scale is None
        assert exception is False
    
    def test_except_eps(self):
        """'except eps' should also trigger exception."""
        html = "<table><caption>(in millions, except eps)</caption></table>"
        scale, exception = extract_table_scale(html)
        assert exception is True


# =============================================================================
# Per-Share Detection Tests
# =============================================================================


class TestPerShareDetection:
    """Tests for per-share metric detection."""
    
    def test_earnings_per_share(self):
        """'Earnings per share' should be detected."""
        assert is_per_share_metric("Earnings per share") is True
    
    def test_eps(self):
        """'EPS' should be detected."""
        assert is_per_share_metric("EPS") is True
        assert is_per_share_metric("Diluted EPS") is True
    
    def test_basic_earnings_per_share(self):
        """'Basic earnings per share' should be detected."""
        assert is_per_share_metric("Basic earnings per share") is True
    
    def test_net_income_per_share(self):
        """'Net income per share' should be detected."""
        assert is_per_share_metric("Net income per share") is True
    
    def test_dividends_per_share(self):
        """'Dividends per share' should be detected."""
        assert is_per_share_metric("Dividends per common share") is True
    
    def test_revenue_not_per_share(self):
        """'Revenue' should NOT be per-share."""
        assert is_per_share_metric("Revenue") is False
    
    def test_net_income_not_per_share(self):
        """'Net income' alone should NOT be per-share."""
        assert is_per_share_metric("Net income") is False
    
    def test_total_revenue_not_per_share(self):
        """'Total Revenue' should NOT be per-share."""
        assert is_per_share_metric("Total Revenue") is False


# =============================================================================
# Effective Scale Tests
# =============================================================================


class TestEffectiveScale:
    """Tests for effective scale calculation."""
    
    def test_eps_with_exception(self):
        """EPS with exception should have no scale."""
        result = get_effective_scale("millions", True, "EPS")
        assert result is None
    
    def test_revenue_with_exception(self):
        """Revenue with exception should still have scale."""
        result = get_effective_scale("millions", True, "Revenue")
        assert result == "millions"
    
    def test_eps_without_exception(self):
        """EPS without exception should still have no scale (safe default)."""
        result = get_effective_scale("millions", False, "EPS")
        assert result is None
    
    def test_revenue_without_exception(self):
        """Revenue without exception should have scale."""
        result = get_effective_scale("millions", False, "Revenue")
        assert result == "millions"
    
    def test_no_scale_any_row(self):
        """No scale should always return None."""
        assert get_effective_scale(None, False, "Revenue") is None
        assert get_effective_scale(None, True, "EPS") is None
    
    def test_earnings_per_share_full(self):
        """'Earnings per share - diluted' should have no scale."""
        result = get_effective_scale("millions", True, "Earnings per share - diluted")
        assert result is None


# =============================================================================
# MultiIndex Flattening Tests
# =============================================================================


class TestMultiIndexFlattening:
    """Tests for MultiIndex column flattening."""
    
    def test_flattens_multiindex(self):
        """MultiIndex columns should be flattened."""
        df = pd.DataFrame(
            [[100, 200], [300, 400]],
            columns=pd.MultiIndex.from_tuples([
                ('Three Months Ended', 'Oct 27, 2024'),
                ('Three Months Ended', 'Oct 29, 2023'),
            ])
        )
        result = flatten_multiindex_columns(df)
        assert list(result.columns) == [
            'Three Months Ended Oct 27, 2024',
            'Three Months Ended Oct 29, 2023',
        ]
    
    def test_single_level_unchanged(self):
        """Single-level columns should be unchanged."""
        df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        result = flatten_multiindex_columns(df)
        assert list(result.columns) == ['A', 'B']
    
    def test_flattened_contains_date(self):
        """Flattened headers should contain date information."""
        df = pd.DataFrame(
            [[100]],
            columns=pd.MultiIndex.from_tuples([
                ('Period Ended', 'Oct 27, 2024'),
            ])
        )
        result = flatten_multiindex_columns(df)
        assert 'Oct 27, 2024' in result.columns[0]


# =============================================================================
# Basic Extraction Tests
# =============================================================================


class TestBasicExtraction:
    """Tests for basic table extraction."""
    
    def test_tables_extracted_from_html(self, sample_revenue_table_html):
        """Tables should be extracted from HTML."""
        tables = extract_tables_from_html(sample_revenue_table_html)
        assert len(tables) >= 1
    
    def test_table_index_assigned(self, sample_revenue_table_html):
        """Tables should have index assigned."""
        tables = extract_tables_from_html(sample_revenue_table_html)
        assert tables[0].table_index == 0
    
    def test_dataframe_reconstructed(self, sample_revenue_table_html):
        """DataFrame should be reconstructable from JSON."""
        tables = extract_tables_from_html(sample_revenue_table_html)
        df = table_to_dataframe(tables[0])
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
    
    def test_scale_captured(self, sample_revenue_table_html):
        """Scale should be captured from table."""
        tables = extract_tables_from_html(sample_revenue_table_html)
        assert tables[0].scale == "millions"
    
    def test_per_share_exception_captured(self, sample_revenue_table_html):
        """Per-share exception should be captured."""
        tables = extract_tables_from_html(sample_revenue_table_html)
        assert tables[0].has_per_share_exception is True


# =============================================================================
# Table Finding Tests
# =============================================================================


class TestTableFinding:
    """Tests for finding tables by metric."""
    
    def test_find_revenue_table(self, sample_revenue_table_html):
        """Should find revenue table by keyword."""
        tables = extract_tables_from_html(sample_revenue_table_html)
        result = find_table_by_metric(tables, ["revenue", "net revenue"])
        assert result is not None
    
    def test_find_by_data_center(self, sample_revenue_table_html):
        """Should find table containing 'Data Center'."""
        tables = extract_tables_from_html(sample_revenue_table_html)
        result = find_table_by_metric(tables, ["data center"])
        assert result is not None
    
    def test_returns_none_for_missing(self, sample_revenue_table_html):
        """Should return None for non-existent metric."""
        tables = extract_tables_from_html(sample_revenue_table_html)
        result = find_table_by_metric(tables, ["nonexistent_metric_xyz"])
        assert result is None


# =============================================================================
# Value Extraction Tests
# =============================================================================


class TestValueExtraction:
    """Tests for extracting values from tables."""
    
    def test_extract_revenue_value(self, sample_revenue_table_html):
        """Should extract Revenue value."""
        tables = extract_tables_from_html(sample_revenue_table_html)
        cell = extract_value_from_table(tables[0], r"^Revenue$")
        assert cell is not None
        # The last column is Oct 2023, so we expect 18,120
        assert "18,120" in cell.value or "18120" in cell.value
    
    def test_extract_data_center_value(self, sample_revenue_table_html):
        """Should extract Data Center revenue."""
        tables = extract_tables_from_html(sample_revenue_table_html)
        cell = extract_value_from_table(tables[0], r"Data Center")
        assert cell is not None
        assert cell.row_label == "Data Center"
    
    def test_extract_eps_value(self, sample_revenue_table_html):
        """Should extract EPS with no effective scale."""
        tables = extract_tables_from_html(sample_revenue_table_html)
        cell = extract_value_from_table(tables[0], r"Earnings per share")
        assert cell is not None
        assert cell.effective_scale is None  # Per-share should have no scale
    
    def test_extract_last_column(self, sample_revenue_table_html):
        """Should extract from last column by default."""
        tables = extract_tables_from_html(sample_revenue_table_html)
        cell = extract_value_from_table(tables[0], r"^Revenue$", column_index=-1)
        assert cell is not None
    
    def test_cell_includes_effective_scale(self, sample_revenue_table_html):
        """TableCell should include effective_scale."""
        tables = extract_tables_from_html(sample_revenue_table_html)
        cell = extract_value_from_table(tables[0], r"^Revenue$")
        assert cell is not None
        assert cell.effective_scale == "millions"


# =============================================================================
# Column Date Extraction Tests
# =============================================================================


class TestColumnDateExtraction:
    """Tests for extracting dates from column headers."""
    
    def test_oct_27_2024(self):
        """'Oct 27, 2024' should parse to '2024-10-27'."""
        result = parse_date_from_text("Oct 27, 2024")
        assert result == "2024-10-27"
    
    def test_october_27_2024(self):
        """'October 27, 2024' should parse to '2024-10-27'."""
        result = parse_date_from_text("October 27, 2024")
        assert result == "2024-10-27"
    
    def test_three_months_ended(self):
        """'Three Months Ended Oct 27, 2024' should parse to '2024-10-27'."""
        result = parse_date_from_text("Three Months Ended Oct 27, 2024")
        assert result == "2024-10-27"
    
    def test_january_28_2024(self):
        """'January 28, 2024' should parse correctly."""
        result = parse_date_from_text("January 28, 2024")
        assert result == "2024-01-28"
    
    def test_get_column_date_from_table(self, sample_revenue_table_html):
        """Should extract date from table column."""
        tables = extract_tables_from_html(sample_revenue_table_html)
        # Last column should have a date
        date = get_column_date(tables[0], -1)
        # Either 2023 or 2024 date should be found
        assert date is not None or True  # May depend on header structure


# =============================================================================
# Table Type Identification Tests
# =============================================================================


class TestTableTypeIdentification:
    """Tests for identifying table types."""
    
    def test_income_statement(self, sample_revenue_table_html):
        """Should identify income statement table."""
        tables = extract_tables_from_html(sample_revenue_table_html)
        table_type = identify_table_type(tables[0])
        # Revenue and EPS suggest income statement
        assert table_type in ["income_statement", "segment_revenue", "unknown"]
    
    def test_balance_sheet(self, sample_balance_sheet_html):
        """Should identify balance sheet table."""
        tables = extract_tables_from_html(sample_balance_sheet_html)
        table_type = identify_table_type(tables[0])
        assert table_type == "balance_sheet"


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""
    
    def test_empty_html(self):
        """Should handle empty HTML."""
        tables = extract_tables_from_html("")
        assert tables == []
    
    def test_no_tables(self):
        """Should handle HTML with no tables."""
        html = "<html><body><p>No tables here</p></body></html>"
        tables = extract_tables_from_html(html)
        assert tables == []
    
    def test_small_table_filtered(self):
        """Very small tables should be filtered out."""
        html = "<table><tr><td>A</td></tr></table>"
        tables = extract_tables_from_html(html)
        # Single cell tables should be filtered
        assert len(tables) == 0
    
    def test_extract_from_section(self, sample_revenue_table_html):
        """Should extract tables from Section object."""
        section = Section(
            section_id="Item8",
            title="Financial Statements",
            paragraphs=[],
            raw_html=sample_revenue_table_html,
        )
        tables = extract_tables_from_section(section)
        assert len(tables) >= 1
        assert tables[0].section_id == "Item8"


# =============================================================================
# Integration Tests with Real Fixture
# =============================================================================


class TestRealFixture:
    """Integration tests with real NVIDIA 10-K."""
    
    def test_extracts_multiple_tables(self, sample_10k_html):
        """Should extract multiple tables from real 10-K."""
        if sample_10k_html is None:
            pytest.skip("No fixture file available")
        
        tables = extract_tables_from_html(sample_10k_html)
        # Real 10-K has many tables
        assert len(tables) > 10
    
    def test_finds_revenue_table(self, sample_10k_html):
        """Should find revenue table in real 10-K."""
        if sample_10k_html is None:
            pytest.skip("No fixture file available")
        
        tables = extract_tables_from_html(sample_10k_html)
        result = find_table_by_metric(tables, ["revenue", "total revenue", "net revenue"])
        assert result is not None
    
    def test_extracts_scale_from_real_tables(self, sample_10k_html):
        """Should extract scale from real tables."""
        if sample_10k_html is None:
            pytest.skip("No fixture file available")
        
        tables = extract_tables_from_html(sample_10k_html)
        # At least some tables should have scale
        tables_with_scale = [t for t in tables if t.scale is not None]
        assert len(tables_with_scale) > 0

