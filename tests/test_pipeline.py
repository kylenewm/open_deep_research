"""
Tests for the verification gate pipeline.

Tests cover:
- Whitespace normalization
- Text fact verification
- Table fact verification
- Bifurcation (routing to correct verifier)
- Deduplication
- Scale-aware numeric comparison
"""
from __future__ import annotations

import json
from datetime import datetime

import pandas as pd
import pytest

from open_deep_research.models import (
    DocumentSnapshot,
    ExtractedTable,
    Fact,
    Location,
)
from open_deep_research.pipeline import (
    deduplicate_facts,
    extract_number_from_text,
    normalize_for_comparison,
    process_extracted_facts,
    verify_table_fact,
    verify_text_fact,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_doc_snapshot() -> DocumentSnapshot:
    """Create a sample DocumentSnapshot for testing."""
    return DocumentSnapshot(
        snapshot_id="test-snapshot-123",
        url="https://www.sec.gov/test/filing.htm",
        cik="0001234567",
        doc_type="10-K",
        retrieved_at=datetime(2024, 10, 27),
        content_hash="abc123hash",
        raw_html="<html><body>Test</body></html>",
    )


def create_text_fact(
    metric: str,
    value: float | None,
    sentence_string: str,
    entity: str = "NVDA",
) -> Fact:
    """Helper to create a text-based fact for testing."""
    return Fact(
        fact_id="test-fact-001",
        entity=entity,
        metric=metric,
        value=value,
        unit="USD",
        period="Q3 FY2025",
        period_end_date="2024-10-27",
        location=Location(
            cik="0001234567",
            doc_date="2024-10-27",
            doc_type="10-K",
            section_id="Item7",
            paragraph_index=0,
            sentence_string=sentence_string,
        ),
        source_format="html_text",
        doc_hash="abc123",
        snapshot_id="test-snapshot",
        verification_status="unverified",
    )


def create_table_fact(
    metric: str,
    value: float | None,
    row_index: int,
    column_index: int,
    table_index: int = 0,
    entity: str = "NVDA",
) -> Fact:
    """Helper to create a table-based fact for testing."""
    return Fact(
        fact_id="test-fact-002",
        entity=entity,
        metric=metric,
        value=value,
        unit="USD",
        period="Q3 FY2025",
        period_end_date="2024-10-27",
        location=Location(
            cik="0001234567",
            doc_date="2024-10-27",
            doc_type="10-K",
            section_id="Item7",
            table_index=table_index,
            row_index=row_index,
            column_index=column_index,
            row_label="Data Center",
            column_label="Oct 27, 2024",
        ),
        source_format="html_table",
        extracted_scale="millions",
        doc_hash="abc123",
        snapshot_id="test-snapshot",
        verification_status="unverified",
    )


def create_test_table(
    data: dict,
    scale: str | None = "millions",
    table_index: int = 0,
) -> ExtractedTable:
    """Helper to create an ExtractedTable for testing."""
    df = pd.DataFrame(data)
    return ExtractedTable(
        table_index=table_index,
        section_id="Item7",
        html="<table>...</table>",
        headers=df.columns.tolist(),
        row_count=len(df),
        column_count=len(df.columns),
        scale=scale,
        has_per_share_exception=False,
        dataframe_json=df.to_json(),
    )


# =============================================================================
# Whitespace Normalization Tests
# =============================================================================


class TestWhitespaceNormalization:
    """Tests for normalize_for_comparison function."""

    def test_basic_normalization(self):
        """Collapsed spaces and lowercase."""
        result = normalize_for_comparison("Revenue  Was  HIGH")
        assert result == "revenue was high"

    def test_newline_handling(self):
        """Newlines should be converted to spaces."""
        result = normalize_for_comparison("Revenue\nwas\n$10B")
        assert result == "revenue was $10b"

    def test_nbsp_unicode_handling(self):
        """Unicode non-breaking space should be converted."""
        result = normalize_for_comparison("Revenue\xa0was $10B")
        assert result == "revenue was $10b"

    def test_nbsp_html_entity_handling(self):
        """HTML &nbsp; entity should be converted."""
        result = normalize_for_comparison("Revenue&nbsp;was $10B")
        assert result == "revenue was $10b"

    def test_tabs_and_carriage_returns(self):
        """Tabs and carriage returns should be handled."""
        result = normalize_for_comparison("Revenue\twas\r$10B")
        assert result == "revenue was $10b"

    def test_multiple_whitespace_collapsed(self):
        """Multiple consecutive spaces should collapse to one."""
        result = normalize_for_comparison("Revenue    was      $10B")
        assert result == "revenue was $10b"

    def test_match_across_formats(self):
        """Different whitespace formats should normalize to same result."""
        # What LLM extracts
        llm_extract = "Revenue was $10B"
        # What HTML contains
        html_contains = "Revenue  was\n$10B"
        
        assert normalize_for_comparison(llm_extract) == normalize_for_comparison(html_contains)

    def test_empty_string(self):
        """Empty string should return empty string."""
        assert normalize_for_comparison("") == ""

    def test_none_handling(self):
        """None should return empty string."""
        assert normalize_for_comparison(None) == ""


# =============================================================================
# Text Fact Verification Tests
# =============================================================================


class TestTextFactVerification:
    """Tests for verify_text_fact function."""

    def test_valid_sentence_passes(self):
        """Sentence that exists in source should pass."""
        fact = create_text_fact(
            metric="revenue",
            value=35000000000.0,
            sentence_string="Revenue was $35 billion for the quarter."
        )
        source_text = "In Q3 FY2025, Revenue was $35 billion for the quarter. This exceeded expectations."
        
        result = verify_text_fact(fact, source_text)
        
        assert result.verification_status in ("exact_match", "approximate_match")

    def test_hallucinated_sentence_rejected(self):
        """Sentence that doesn't exist in source should be rejected."""
        fact = create_text_fact(
            metric="revenue",
            value=35000000000.0,
            sentence_string="Revenue hit a record $100 billion."  # Hallucinated!
        )
        source_text = "In Q3 FY2025, Revenue was $35 billion for the quarter."
        
        result = verify_text_fact(fact, source_text)
        
        assert result.verification_status == "mismatch"

    def test_whitespace_differences_tolerated(self):
        """Whitespace differences between LLM and source should be tolerated."""
        fact = create_text_fact(
            metric="revenue",
            value=10000000000.0,
            sentence_string="Revenue was $10B"  # Clean whitespace
        )
        source_text = "Revenue  was\n$10B"  # Different whitespace
        
        result = verify_text_fact(fact, source_text)
        
        assert result.verification_status in ("exact_match", "approximate_match")

    def test_missing_sentence_string_rejected(self):
        """Fact without sentence_string should be rejected."""
        fact = create_text_fact(
            metric="revenue",
            value=10000000000.0,
            sentence_string=None  # Missing!
        )
        source_text = "Revenue was $10B"
        
        result = verify_text_fact(fact, source_text)
        
        assert result.verification_status == "mismatch"

    def test_numeric_value_verified(self):
        """Numeric value in sentence should be verified against fact value."""
        fact = create_text_fact(
            metric="revenue",
            value=10500000000.0,  # $10.5B
            sentence_string="Revenue was $10.5 billion"
        )
        source_text = "Revenue was $10.5 billion for Q3."
        
        result = verify_text_fact(fact, source_text)
        
        assert result.verification_status in ("exact_match", "approximate_match")

    def test_non_numeric_fact_passes(self):
        """Non-numeric fact should pass if sentence exists."""
        fact = create_text_fact(
            metric="risk_factor",
            value=None,  # No numeric value
            sentence_string="Competition remains intense."
        )
        source_text = "Competition remains intense. This affects margins."
        
        result = verify_text_fact(fact, source_text)
        
        assert result.verification_status == "exact_match"

    def test_wrong_verifier_raises(self):
        """Using text verifier on table fact should raise ValueError."""
        fact = create_table_fact(
            metric="revenue",
            value=10000000000.0,
            row_index=0,
            column_index=1
        )
        
        with pytest.raises(ValueError, match="table facts"):
            verify_text_fact(fact, "some source text")


# =============================================================================
# Table Fact Verification Tests
# =============================================================================


class TestTableFactVerification:
    """Tests for verify_table_fact function."""

    def test_valid_coordinates_pass(self):
        """Valid coordinates that match value should pass."""
        table = create_test_table(
            data={
                "Segment": ["Data Center", "Gaming"],
                "Oct 27, 2024": ["14,514", "3,279"],
            },
            scale="millions"
        )
        
        fact = create_table_fact(
            metric="Data Center",
            value=14514000000.0,  # $14.514B in base units
            row_index=0,
            column_index=1,
        )
        
        result = verify_table_fact(fact, table)
        
        assert result.verification_status in ("exact_match", "approximate_match")

    def test_out_of_bounds_row_rejected(self):
        """Row index out of bounds should be rejected."""
        table = create_test_table(
            data={
                "Segment": ["Data Center", "Gaming"],
                "Oct 27, 2024": ["14,514", "3,279"],
            }
        )
        
        fact = create_table_fact(
            metric="Data Center",
            value=14514000000.0,
            row_index=5,  # Out of bounds!
            column_index=1,
        )
        
        result = verify_table_fact(fact, table)
        
        assert result.verification_status == "mismatch"

    def test_out_of_bounds_column_rejected(self):
        """Column index out of bounds should be rejected."""
        table = create_test_table(
            data={
                "Segment": ["Data Center"],
                "Oct 27, 2024": ["14,514"],
            }
        )
        
        fact = create_table_fact(
            metric="Data Center",
            value=14514000000.0,
            row_index=0,
            column_index=10,  # Out of bounds!
        )
        
        result = verify_table_fact(fact, table)
        
        assert result.verification_status == "mismatch"

    def test_missing_coordinates_rejected(self):
        """Fact without coordinates should be rejected."""
        table = create_test_table(
            data={
                "Segment": ["Data Center"],
                "Oct 27, 2024": ["14,514"],
            }
        )
        
        fact = create_table_fact(
            metric="Data Center",
            value=14514000000.0,
            row_index=0,
            column_index=1,
        )
        # Manually set coordinates to None
        fact.location.row_index = None
        fact.location.column_index = None
        
        result = verify_table_fact(fact, table)
        
        assert result.verification_status == "mismatch"

    def test_scale_millions_pass(self):
        """Scale-aware comparison: millions scale should work correctly."""
        table = create_test_table(
            data={
                "Segment": ["Data Center"],
                "Oct 27, 2024": ["14,514"],  # 14,514 in millions
            },
            scale="millions"
        )
        
        fact = create_table_fact(
            metric="Data Center",
            value=14514000000.0,  # Already in base units (14.514 billion)
            row_index=0,
            column_index=1,
        )
        
        result = verify_table_fact(fact, table)
        
        assert result.verification_status in ("exact_match", "approximate_match")

    def test_scale_thousands_mismatch(self):
        """Scale-aware comparison: wrong scale should cause mismatch."""
        table = create_test_table(
            data={
                "Segment": ["Data Center"],
                "Oct 27, 2024": ["14,514"],  # 14,514 in thousands = 14,514,000
            },
            scale="thousands"  # Wrong scale for the value
        )
        
        fact = create_table_fact(
            metric="Data Center",
            value=14514000000.0,  # 14.514 billion
            row_index=0,
            column_index=1,
        )
        
        result = verify_table_fact(fact, table)
        
        # 14,514,000 != 14,514,000,000, should be mismatch
        assert result.verification_status == "mismatch"

    def test_no_scale_applied(self):
        """Table without scale should use raw values."""
        table = create_test_table(
            data={
                "Metric": ["EPS"],
                "Value": ["2.75"],
            },
            scale=None  # No scale - per share data
        )
        
        fact = create_table_fact(
            metric="EPS",
            value=2.75,  # Raw value
            row_index=0,
            column_index=1,
        )
        
        result = verify_table_fact(fact, table)
        
        assert result.verification_status in ("exact_match", "approximate_match")

    def test_wrong_verifier_raises(self):
        """Using table verifier on text fact should raise ValueError."""
        fact = create_text_fact(
            metric="revenue",
            value=10000000000.0,
            sentence_string="Revenue was $10B"
        )
        
        table = create_test_table(
            data={"A": [1]},
            scale=None
        )
        
        with pytest.raises(ValueError, match="text facts"):
            verify_table_fact(fact, table)


# =============================================================================
# Bifurcation Tests
# =============================================================================


class TestBifurcation:
    """Tests for routing facts to correct verifiers."""

    def test_text_fact_routed_to_text_verifier(self):
        """Text facts should be processed by text verifier."""
        fact = create_text_fact(
            metric="revenue",
            value=10000000000.0,
            sentence_string="Revenue was $10 billion"
        )
        source_text = "Revenue was $10 billion for the quarter."
        tables = []
        
        verified, rejected = process_extracted_facts([fact], source_text, tables)
        
        # Should be verified (not rejected)
        assert len(verified) == 1
        assert len(rejected) == 0
        assert verified[0].verification_status in ("exact_match", "approximate_match")

    def test_table_fact_routed_to_table_verifier(self):
        """Table facts should be processed by table verifier."""
        table = create_test_table(
            data={
                "Segment": ["Data Center"],
                "Oct 27, 2024": ["14,514"],
            },
            scale="millions"
        )
        
        fact = create_table_fact(
            metric="Data Center",
            value=14514000000.0,
            row_index=0,
            column_index=1,
            table_index=0,
        )
        
        verified, rejected = process_extracted_facts([fact], "", [table])
        
        assert len(verified) == 1
        assert len(rejected) == 0
        assert verified[0].verification_status in ("exact_match", "approximate_match")

    def test_mixed_facts_routed_correctly(self):
        """Mixed text and table facts should be routed to correct verifiers."""
        # Text fact
        text_fact = create_text_fact(
            metric="guidance",
            value=None,
            sentence_string="Management expects continued growth."
        )
        
        # Table fact
        table = create_test_table(
            data={
                "Segment": ["Gaming"],
                "Oct 27, 2024": ["3,279"],
            },
            scale="millions"
        )
        
        table_fact = create_table_fact(
            metric="Gaming",
            value=3279000000.0,
            row_index=0,
            column_index=1,
            table_index=0,
        )
        
        source_text = "Management expects continued growth. This is positive."
        
        verified, rejected = process_extracted_facts(
            [text_fact, table_fact],
            source_text,
            [table]
        )
        
        assert len(verified) == 2
        assert len(rejected) == 0

    def test_missing_table_causes_rejection(self):
        """Table fact with no matching table should be rejected."""
        fact = create_table_fact(
            metric="Data Center",
            value=14514000000.0,
            row_index=0,
            column_index=1,
            table_index=99,  # Table doesn't exist!
        )
        
        tables = []  # Empty tables list
        
        verified, rejected = process_extracted_facts([fact], "", tables)
        
        assert len(verified) == 0
        assert len(rejected) == 1
        assert rejected[0].verification_status == "mismatch"


# =============================================================================
# Deduplication Tests
# =============================================================================


class TestDeduplication:
    """Tests for deduplicate_facts function."""

    def test_duplicate_removed(self):
        """Duplicate (entity, metric, period, value) should keep first."""
        fact1 = create_text_fact(
            metric="revenue",
            value=10000000000.0,
            sentence_string="Revenue was $10B"
        )
        fact2 = create_text_fact(
            metric="revenue",
            value=10000000000.0,  # Same value
            sentence_string="Revenue totaled $10B"  # Different sentence
        )
        
        result = deduplicate_facts([fact1, fact2])
        
        assert len(result) == 1
        assert result[0].location.sentence_string == "Revenue was $10B"  # First one kept

    def test_different_values_not_deduplicated(self):
        """Different values should NOT be deduplicated."""
        fact1 = create_text_fact(
            metric="revenue",
            value=10000000000.0,  # $10B
            sentence_string="Revenue was $10B"
        )
        fact2 = create_text_fact(
            metric="revenue",
            value=15000000000.0,  # $15B - different!
            sentence_string="Revenue was $15B"
        )
        
        result = deduplicate_facts([fact1, fact2])
        
        assert len(result) == 2

    def test_different_metrics_not_deduplicated(self):
        """Different metrics should NOT be deduplicated."""
        fact1 = create_text_fact(
            metric="revenue",
            value=10000000000.0,
            sentence_string="Revenue was $10B"
        )
        fact2 = create_text_fact(
            metric="net_income",  # Different metric
            value=10000000000.0,
            sentence_string="Net income was $10B"
        )
        
        result = deduplicate_facts([fact1, fact2])
        
        assert len(result) == 2

    def test_case_insensitive_entity(self):
        """Entity comparison should be case-insensitive."""
        fact1 = create_text_fact(
            metric="revenue",
            value=10000000000.0,
            sentence_string="Revenue was $10B",
            entity="NVDA"
        )
        fact2 = create_text_fact(
            metric="revenue",
            value=10000000000.0,
            sentence_string="Revenue totaled $10B",
            entity="nvda"  # lowercase
        )
        
        result = deduplicate_facts([fact1, fact2])
        
        assert len(result) == 1

    def test_case_insensitive_metric(self):
        """Metric comparison should be case-insensitive."""
        fact1 = create_text_fact(
            metric="Revenue",
            value=10000000000.0,
            sentence_string="Revenue was $10B"
        )
        fact2 = create_text_fact(
            metric="revenue",  # lowercase
            value=10000000000.0,
            sentence_string="Revenue totaled $10B"
        )
        
        result = deduplicate_facts([fact1, fact2])
        
        assert len(result) == 1

    def test_empty_list(self):
        """Empty list should return empty list."""
        result = deduplicate_facts([])
        assert result == []


# =============================================================================
# Number Extraction Tests
# =============================================================================


class TestNumberExtraction:
    """Tests for extract_number_from_text function."""

    def test_dollar_billion(self):
        """Extract $10.5B format."""
        result = extract_number_from_text("Revenue was $10.5B")
        assert result == pytest.approx(10500000000.0)

    def test_number_with_word_billion(self):
        """Extract '10 billion' format."""
        result = extract_number_from_text("Revenue was 10 billion")
        assert result == pytest.approx(10000000000.0)

    def test_dollar_with_commas(self):
        """Extract $10,500 format."""
        result = extract_number_from_text("The amount was $10,500")
        assert result == pytest.approx(10500.0)

    def test_accounting_negative(self):
        """Extract (500) accounting notation."""
        result = extract_number_from_text("Loss was (500)")
        assert result == pytest.approx(-500.0)

    def test_no_number(self):
        """Text without number should return None."""
        result = extract_number_from_text("No numbers here")
        assert result is None

    def test_empty_string(self):
        """Empty string should return None."""
        result = extract_number_from_text("")
        assert result is None

    def test_none_input(self):
        """None input should return None."""
        result = extract_number_from_text(None)
        assert result is None

