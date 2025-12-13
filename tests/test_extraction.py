"""
Tests for fact extraction module.

Covers:
- Prompt generation tests
- Response parsing tests
- Text fact creation tests
- Table fact creation tests
- Integration tests (with mocked LLM)
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime
from io import StringIO
from typing import List
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from open_deep_research.extraction import (
    create_table_extraction_prompt,
    create_text_extraction_prompt,
    extract_facts_from_table,
    extract_facts_from_text,
    parse_llm_response,
    validate_table_fact,
    validate_text_fact,
)
from open_deep_research.models import (
    DocumentSnapshot,
    ExtractedTable,
    Fact,
    Location,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_doc_snapshot() -> DocumentSnapshot:
    """Create a sample DocumentSnapshot for testing."""
    return DocumentSnapshot(
        snapshot_id=str(uuid.uuid4()),
        url="https://www.sec.gov/Archives/edgar/data/1045810/000104581024000065/nvda-20241027.htm",
        cik="0001045810",
        doc_type="10-Q",
        retrieved_at=datetime(2024, 11, 15, 10, 30, 0),
        content_hash="abc123def456",
        raw_html="<html>...</html>",
    )


@pytest.fixture
def sample_table() -> ExtractedTable:
    """Create a sample ExtractedTable for testing."""
    df = pd.DataFrame({
        "Metric": ["Revenue", "Data Center", "Gaming", "Net Income"],
        "Oct 27, 2024": [35082, 30770, 3279, 19309],
        "Oct 29, 2023": [18120, 14514, 2856, 9243],
    })
    return ExtractedTable(
        table_index=0,
        section_id="Item7",
        html="<table>...</table>",
        headers=["Metric", "Oct 27, 2024", "Oct 29, 2023"],
        row_count=4,
        column_count=3,
        scale="millions",
        has_per_share_exception=False,
        dataframe_json=df.to_json(),
    )


@pytest.fixture
def sample_text() -> str:
    """Sample paragraph text with financial facts."""
    return (
        "NVIDIA Corporation reported record revenue of $35.1 billion for the third quarter "
        "of fiscal 2025, ended October 27, 2024. Data Center revenue was $30.8 billion, up 112% "
        "year over year. The company's Gaming revenue reached $3.3 billion."
    )


# =============================================================================
# Prompt Generation Tests
# =============================================================================


class TestTextExtractionPrompt:
    """Tests for create_text_extraction_prompt."""
    
    def test_prompt_includes_entity_name(self):
        """Test that text prompt includes the entity name."""
        prompt = create_text_extraction_prompt(
            text="Sample financial text",
            entity="NVDA"
        )
        assert "NVDA" in prompt
    
    def test_prompt_includes_input_text(self):
        """Test that text prompt includes the input text."""
        text = "NVIDIA reported revenue of $35 billion."
        prompt = create_text_extraction_prompt(text=text, entity="NVDA")
        assert text in prompt
    
    def test_prompt_includes_extraction_rules(self):
        """Test that prompt includes key extraction rules."""
        prompt = create_text_extraction_prompt(
            text="Sample text",
            entity="NVDA"
        )
        # Should mention sentence_string requirement
        assert "sentence_string" in prompt
        assert "EXACT" in prompt
        # Should mention not to infer
        assert "infer" in prompt.lower() or "guess" in prompt.lower()


class TestTableExtractionPrompt:
    """Tests for create_table_extraction_prompt."""
    
    def test_prompt_includes_scale_information(self):
        """Test that table prompt includes scale information."""
        df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        prompt = create_table_extraction_prompt(
            table_df=df,
            entity="NVDA",
            table_scale="millions"
        )
        assert "millions" in prompt
    
    def test_prompt_includes_column_headers(self):
        """Test that table prompt includes column headers."""
        df = pd.DataFrame({
            "Metric": ["Revenue"],
            "Oct 27, 2024": [35082]
        })
        prompt = create_table_extraction_prompt(
            table_df=df,
            entity="NVDA",
            table_scale=None
        )
        assert "Metric" in prompt
        assert "Oct 27, 2024" in prompt
    
    def test_prompt_without_scale(self):
        """Test that prompt handles no scale gracefully."""
        df = pd.DataFrame({"A": [1]})
        prompt = create_table_extraction_prompt(
            table_df=df,
            entity="NVDA",
            table_scale=None
        )
        assert "No scale specified" in prompt
    
    def test_prompt_no_sentence_generation(self):
        """Test that prompt instructs not to generate sentences."""
        df = pd.DataFrame({"A": [1]})
        prompt = create_table_extraction_prompt(
            table_df=df,
            entity="NVDA",
            table_scale=None
        )
        assert "NOT generate sentences" in prompt or "Do NOT generate sentences" in prompt


# =============================================================================
# Response Parsing Tests
# =============================================================================


class TestParseLLMResponse:
    """Tests for parse_llm_response."""
    
    def test_parse_valid_json_array(self):
        """Test parsing valid JSON array."""
        response = '[{"metric": "revenue", "value": 35082}]'
        result = parse_llm_response(response)
        assert len(result) == 1
        assert result[0]["metric"] == "revenue"
        assert result[0]["value"] == 35082
    
    def test_parse_json_with_markdown_code_blocks(self):
        """Test parsing JSON with markdown code blocks."""
        response = '''```json
[{"metric": "revenue", "value": 35082}]
```'''
        result = parse_llm_response(response)
        assert len(result) == 1
        assert result[0]["metric"] == "revenue"
    
    def test_parse_json_with_plain_code_blocks(self):
        """Test parsing JSON with plain code blocks (no 'json' tag)."""
        response = '''```
[{"metric": "revenue", "value": 35082}]
```'''
        result = parse_llm_response(response)
        assert len(result) == 1
        assert result[0]["metric"] == "revenue"
    
    def test_parse_empty_array(self):
        """Test parsing empty array []."""
        response = "[]"
        result = parse_llm_response(response)
        assert result == []
    
    def test_handle_malformed_json(self):
        """Test handling malformed JSON (should not crash)."""
        response = '[{"metric": "revenue", "value": ]'  # Invalid JSON
        result = parse_llm_response(response)
        assert result == []
    
    def test_handle_empty_response(self):
        """Test handling empty response."""
        result = parse_llm_response("")
        assert result == []
    
    def test_handle_non_list_response(self):
        """Test handling non-list JSON response."""
        response = '{"metric": "revenue"}'  # Object, not array
        result = parse_llm_response(response)
        assert result == []
    
    def test_parse_multiline_json(self):
        """Test parsing properly formatted multiline JSON."""
        response = '''[
  {
    "metric": "revenue",
    "value": 35082,
    "unit": "USD"
  },
  {
    "metric": "net_income",
    "value": 19309,
    "unit": "USD"
  }
]'''
        result = parse_llm_response(response)
        assert len(result) == 2
        assert result[0]["metric"] == "revenue"
        assert result[1]["metric"] == "net_income"


# =============================================================================
# Text Fact Creation Tests
# =============================================================================


class TestTextFactCreation:
    """Tests for text fact creation and validation."""
    
    def test_fact_id_is_uuid_format(self, sample_doc_snapshot):
        """Test that fact_id is generated (UUID format)."""
        with patch("open_deep_research.extraction.call_llm") as mock_llm:
            mock_llm.return_value = json.dumps([{
                "metric": "revenue",
                "value": 35100,
                "unit": "USD",
                "period": "Q3 FY2025",
                "period_end_date": "2024-10-27",
                "sentence_string": "NVIDIA Corporation reported record revenue of $35.1 billion"
            }])
            
            facts = extract_facts_from_text(
                text="NVIDIA Corporation reported record revenue of $35.1 billion for Q3 FY2025.",
                entity="NVDA",
                doc_snapshot=sample_doc_snapshot,
                section_id="Item7",
                paragraph_index=0
            )
            
            assert len(facts) == 1
            # Verify UUID format
            uuid_obj = uuid.UUID(facts[0].fact_id)
            assert str(uuid_obj) == facts[0].fact_id
    
    def test_verification_status_is_unverified(self, sample_doc_snapshot):
        """Test that verification_status is 'unverified'."""
        with patch("open_deep_research.extraction.call_llm") as mock_llm:
            mock_llm.return_value = json.dumps([{
                "metric": "revenue",
                "value": 35100,
                "unit": "USD",
                "period": "Q3 FY2025",
                "period_end_date": "2024-10-27",
                "sentence_string": "NVIDIA reported revenue"
            }])
            
            facts = extract_facts_from_text(
                text="NVIDIA reported revenue of $35.1 billion.",
                entity="NVDA",
                doc_snapshot=sample_doc_snapshot,
                section_id="Item7",
                paragraph_index=0
            )
            
            assert len(facts) == 1
            assert facts[0].verification_status == "unverified"
    
    def test_source_format_is_html_text(self, sample_doc_snapshot):
        """Test that source_format is 'html_text'."""
        with patch("open_deep_research.extraction.call_llm") as mock_llm:
            mock_llm.return_value = json.dumps([{
                "metric": "revenue",
                "value": 35100,
                "unit": "USD",
                "period": "Q3 FY2025",
                "period_end_date": "2024-10-27",
                "sentence_string": "NVIDIA reported revenue"
            }])
            
            facts = extract_facts_from_text(
                text="NVIDIA reported revenue of $35.1 billion.",
                entity="NVDA",
                doc_snapshot=sample_doc_snapshot,
                section_id="Item7",
                paragraph_index=0
            )
            
            assert len(facts) == 1
            assert facts[0].source_format == "html_text"
    
    def test_location_sentence_string_is_populated(self, sample_doc_snapshot):
        """Test that location.sentence_string is populated."""
        sentence = "NVIDIA reported record revenue of $35.1 billion"
        with patch("open_deep_research.extraction.call_llm") as mock_llm:
            mock_llm.return_value = json.dumps([{
                "metric": "revenue",
                "value": 35100,
                "unit": "USD",
                "period": "Q3 FY2025",
                "period_end_date": "2024-10-27",
                "sentence_string": sentence
            }])
            
            facts = extract_facts_from_text(
                text=f"{sentence} for Q3 FY2025.",
                entity="NVDA",
                doc_snapshot=sample_doc_snapshot,
                section_id="Item7",
                paragraph_index=0
            )
            
            assert len(facts) == 1
            assert facts[0].location.sentence_string == sentence
    
    def test_location_table_index_is_none(self, sample_doc_snapshot):
        """Test that location.table_index is None for text facts."""
        with patch("open_deep_research.extraction.call_llm") as mock_llm:
            mock_llm.return_value = json.dumps([{
                "metric": "revenue",
                "value": 35100,
                "unit": "USD",
                "period": "Q3 FY2025",
                "period_end_date": "2024-10-27",
                "sentence_string": "NVIDIA reported"
            }])
            
            facts = extract_facts_from_text(
                text="NVIDIA reported $35.1 billion revenue.",
                entity="NVDA",
                doc_snapshot=sample_doc_snapshot,
                section_id="Item7",
                paragraph_index=0
            )
            
            assert len(facts) == 1
            assert facts[0].location.table_index is None
            assert facts[0].location.row_index is None
            assert facts[0].location.column_index is None


# =============================================================================
# Table Fact Creation Tests
# =============================================================================


class TestTableFactCreation:
    """Tests for table fact creation and validation."""
    
    def test_source_format_is_html_table(self, sample_doc_snapshot, sample_table):
        """Test that source_format is 'html_table'."""
        with patch("open_deep_research.extraction.call_llm") as mock_llm:
            mock_llm.return_value = json.dumps([{
                "metric": "Revenue",
                "value": 35082,
                "unit": "USD",
                "period_end_date": "Oct 27, 2024",
                "row_label": "Revenue",
                "column_label": "Oct 27, 2024",
                "row_index": 0,
                "column_index": 1
            }])
            
            facts = extract_facts_from_table(
                table=sample_table,
                entity="NVDA",
                doc_snapshot=sample_doc_snapshot
            )
            
            assert len(facts) == 1
            assert facts[0].source_format == "html_table"
    
    def test_location_table_coordinates_populated(self, sample_doc_snapshot, sample_table):
        """Test that location has table_index, row_index, column_index populated."""
        with patch("open_deep_research.extraction.call_llm") as mock_llm:
            mock_llm.return_value = json.dumps([{
                "metric": "Data Center",
                "value": 30770,
                "unit": "USD",
                "period_end_date": "Oct 27, 2024",
                "row_label": "Data Center",
                "column_label": "Oct 27, 2024",
                "row_index": 1,
                "column_index": 1
            }])
            
            facts = extract_facts_from_table(
                table=sample_table,
                entity="NVDA",
                doc_snapshot=sample_doc_snapshot
            )
            
            assert len(facts) == 1
            assert facts[0].location.table_index == 0
            assert facts[0].location.row_index == 1
            assert facts[0].location.column_index == 1
    
    def test_location_sentence_string_is_none(self, sample_doc_snapshot, sample_table):
        """Test that location.sentence_string is None for table facts."""
        with patch("open_deep_research.extraction.call_llm") as mock_llm:
            mock_llm.return_value = json.dumps([{
                "metric": "Revenue",
                "value": 35082,
                "unit": "USD",
                "period_end_date": "Oct 27, 2024",
                "row_label": "Revenue",
                "column_label": "Oct 27, 2024",
                "row_index": 0,
                "column_index": 1
            }])
            
            facts = extract_facts_from_table(
                table=sample_table,
                entity="NVDA",
                doc_snapshot=sample_doc_snapshot
            )
            
            assert len(facts) == 1
            assert facts[0].location.sentence_string is None
    
    def test_extracted_scale_matches_table_scale(self, sample_doc_snapshot, sample_table):
        """Test that extracted_scale matches table.scale."""
        with patch("open_deep_research.extraction.call_llm") as mock_llm:
            mock_llm.return_value = json.dumps([{
                "metric": "Revenue",
                "value": 35082,
                "unit": "USD",
                "period_end_date": "Oct 27, 2024",
                "row_label": "Revenue",
                "column_label": "Oct 27, 2024",
                "row_index": 0,
                "column_index": 1
            }])
            
            facts = extract_facts_from_table(
                table=sample_table,
                entity="NVDA",
                doc_snapshot=sample_doc_snapshot
            )
            
            assert len(facts) == 1
            assert facts[0].extracted_scale == "millions"
    
    def test_location_row_and_column_labels_populated(self, sample_doc_snapshot, sample_table):
        """Test that location.row_label and column_label are populated."""
        with patch("open_deep_research.extraction.call_llm") as mock_llm:
            mock_llm.return_value = json.dumps([{
                "metric": "Gaming",
                "value": 3279,
                "unit": "USD",
                "period_end_date": "Oct 27, 2024",
                "row_label": "Gaming",
                "column_label": "Oct 27, 2024",
                "row_index": 2,
                "column_index": 1
            }])
            
            facts = extract_facts_from_table(
                table=sample_table,
                entity="NVDA",
                doc_snapshot=sample_doc_snapshot
            )
            
            assert len(facts) == 1
            assert facts[0].location.row_label == "Gaming"
            assert facts[0].location.column_label == "Oct 27, 2024"


# =============================================================================
# Validation Tests
# =============================================================================


class TestValidateTextFact:
    """Tests for validate_text_fact."""
    
    def test_valid_text_fact(self, sample_doc_snapshot):
        """Test validation passes when sentence_string is in source."""
        source_text = "NVIDIA reported revenue of $35 billion for Q3 FY2025."
        
        fact = Fact(
            fact_id=str(uuid.uuid4()),
            entity="NVDA",
            metric="revenue",
            value=35000,
            unit="USD",
            period="Q3 FY2025",
            period_end_date="2024-10-27",
            location=Location(
                cik="0001045810",
                doc_date="2024-11-15",
                doc_type="10-Q",
                section_id="Item7",
                paragraph_index=0,
                sentence_string="NVIDIA reported revenue of $35 billion",
            ),
            source_format="html_text",
            doc_hash="abc123",
            snapshot_id=sample_doc_snapshot.snapshot_id,
            verification_status="unverified",
        )
        
        assert validate_text_fact(fact, source_text) is True
    
    def test_invalid_text_fact_sentence_not_found(self, sample_doc_snapshot):
        """Test validation fails when sentence_string not in source."""
        source_text = "NVIDIA reported revenue of $35 billion."
        
        fact = Fact(
            fact_id=str(uuid.uuid4()),
            entity="NVDA",
            metric="revenue",
            value=35000,
            unit="USD",
            period="Q3 FY2025",
            period_end_date="2024-10-27",
            location=Location(
                cik="0001045810",
                doc_date="2024-11-15",
                doc_type="10-Q",
                section_id="Item7",
                paragraph_index=0,
                sentence_string="AMD reported revenue of $35 billion",  # Wrong!
            ),
            source_format="html_text",
            doc_hash="abc123",
            snapshot_id=sample_doc_snapshot.snapshot_id,
            verification_status="unverified",
        )
        
        assert validate_text_fact(fact, source_text) is False
    
    def test_table_fact_skips_text_validation(self, sample_doc_snapshot):
        """Test that table facts skip text validation."""
        fact = Fact(
            fact_id=str(uuid.uuid4()),
            entity="NVDA",
            metric="revenue",
            value=35082,
            unit="USD",
            period="",
            period_end_date="Oct 27, 2024",
            location=Location(
                cik="0001045810",
                doc_date="2024-11-15",
                doc_type="10-Q",
                section_id="Item7",
                table_index=0,
                row_index=0,
                column_index=1,
            ),
            source_format="html_table",  # Table fact
            doc_hash="abc123",
            snapshot_id=sample_doc_snapshot.snapshot_id,
            verification_status="unverified",
        )
        
        # Should return True (skip validation) for table facts
        assert validate_text_fact(fact, "any source text") is True


class TestValidateTableFact:
    """Tests for validate_table_fact."""
    
    def test_valid_table_fact(self, sample_table):
        """Test validation passes when coordinates are within bounds."""
        fact = Fact(
            fact_id=str(uuid.uuid4()),
            entity="NVDA",
            metric="Revenue",
            value=35082,
            unit="USD",
            period="",
            period_end_date="Oct 27, 2024",
            location=Location(
                cik="0001045810",
                doc_date="2024-11-15",
                doc_type="10-Q",
                section_id="Item7",
                table_index=0,
                row_index=0,  # Valid: 0-3
                column_index=1,  # Valid: 0-2
                row_label="Revenue",
                column_label="Oct 27, 2024",
            ),
            source_format="html_table",
            extracted_scale="millions",
            doc_hash="abc123",
            snapshot_id="test-snapshot",
            verification_status="unverified",
        )
        
        assert validate_table_fact(fact, sample_table) is True
    
    def test_invalid_table_fact_row_out_of_bounds(self, sample_table):
        """Test validation fails when row_index out of bounds."""
        fact = Fact(
            fact_id=str(uuid.uuid4()),
            entity="NVDA",
            metric="Revenue",
            value=35082,
            unit="USD",
            period="",
            period_end_date="Oct 27, 2024",
            location=Location(
                cik="0001045810",
                doc_date="2024-11-15",
                doc_type="10-Q",
                section_id="Item7",
                table_index=0,
                row_index=10,  # Out of bounds! (max is 3)
                column_index=1,
            ),
            source_format="html_table",
            extracted_scale="millions",
            doc_hash="abc123",
            snapshot_id="test-snapshot",
            verification_status="unverified",
        )
        
        assert validate_table_fact(fact, sample_table) is False
    
    def test_invalid_table_fact_column_out_of_bounds(self, sample_table):
        """Test validation fails when column_index out of bounds."""
        fact = Fact(
            fact_id=str(uuid.uuid4()),
            entity="NVDA",
            metric="Revenue",
            value=35082,
            unit="USD",
            period="",
            period_end_date="Oct 27, 2024",
            location=Location(
                cik="0001045810",
                doc_date="2024-11-15",
                doc_type="10-Q",
                section_id="Item7",
                table_index=0,
                row_index=0,
                column_index=5,  # Out of bounds! (max is 2)
            ),
            source_format="html_table",
            extracted_scale="millions",
            doc_hash="abc123",
            snapshot_id="test-snapshot",
            verification_status="unverified",
        )
        
        assert validate_table_fact(fact, sample_table) is False
    
    def test_text_fact_skips_table_validation(self, sample_table):
        """Test that text facts skip table validation."""
        fact = Fact(
            fact_id=str(uuid.uuid4()),
            entity="NVDA",
            metric="revenue",
            value=35100,
            unit="USD",
            period="Q3 FY2025",
            period_end_date="2024-10-27",
            location=Location(
                cik="0001045810",
                doc_date="2024-11-15",
                doc_type="10-Q",
                section_id="Item7",
                paragraph_index=0,
                sentence_string="NVIDIA reported revenue",
            ),
            source_format="html_text",  # Text fact
            doc_hash="abc123",
            snapshot_id="test-snapshot",
            verification_status="unverified",
        )
        
        # Should return True (skip validation) for text facts
        assert validate_table_fact(fact, sample_table) is True


# =============================================================================
# Integration Tests (with mocked LLM)
# =============================================================================


class TestTextExtractionIntegration:
    """Integration tests for text extraction."""
    
    def test_extract_facts_from_sample_paragraph(self, sample_doc_snapshot, sample_text):
        """Test text extraction from sample paragraph with known facts."""
        with patch("open_deep_research.extraction.call_llm") as mock_llm:
            # Mock LLM returns facts matching the sample text
            mock_llm.return_value = json.dumps([
                {
                    "metric": "revenue",
                    "value": 35100000000,
                    "unit": "USD",
                    "period": "Q3 FY2025",
                    "period_end_date": "2024-10-27",
                    "sentence_string": "NVIDIA Corporation reported record revenue of $35.1 billion"
                },
                {
                    "metric": "Data Center revenue",
                    "value": 30800000000,
                    "unit": "USD",
                    "period": "Q3 FY2025",
                    "period_end_date": "2024-10-27",
                    "sentence_string": "Data Center revenue was $30.8 billion"
                }
            ])
            
            facts = extract_facts_from_text(
                text=sample_text,
                entity="NVDA",
                doc_snapshot=sample_doc_snapshot,
                section_id="Item7",
                paragraph_index=0
            )
            
            assert len(facts) == 2
            
            # Verify first fact
            assert facts[0].metric == "revenue"
            assert facts[0].value == 35100000000
            assert facts[0].source_format == "html_text"
            assert facts[0].verification_status == "unverified"
            
            # Verify sentence_string is present in source
            for fact in facts:
                assert fact.location.sentence_string in sample_text
    
    def test_sentence_string_validation_rejects_hallucinated_quotes(
        self, sample_doc_snapshot
    ):
        """Test that facts with hallucinated sentence_strings are rejected."""
        source_text = "NVIDIA reported revenue of $35 billion."
        
        with patch("open_deep_research.extraction.call_llm") as mock_llm:
            # LLM hallucinates a quote not in the source
            mock_llm.return_value = json.dumps([{
                "metric": "revenue",
                "value": 35000000000,
                "unit": "USD",
                "period": "Q3 FY2025",
                "period_end_date": "2024-10-27",
                "sentence_string": "NVIDIA achieved record-breaking revenue of $35 billion"  # Not in source!
            }])
            
            facts = extract_facts_from_text(
                text=source_text,
                entity="NVDA",
                doc_snapshot=sample_doc_snapshot,
                section_id="Item7",
                paragraph_index=0
            )
            
            # Should be rejected because sentence_string not found in source
            assert len(facts) == 0
    
    def test_missing_information_returns_empty_list(self, sample_doc_snapshot):
        """Test that missing information returns empty list."""
        text_without_facts = "This is a general discussion about market conditions."
        
        with patch("open_deep_research.extraction.call_llm") as mock_llm:
            mock_llm.return_value = "[]"
            
            facts = extract_facts_from_text(
                text=text_without_facts,
                entity="NVDA",
                doc_snapshot=sample_doc_snapshot,
                section_id="Item7",
                paragraph_index=0
            )
            
            assert facts == []


class TestTableExtractionIntegration:
    """Integration tests for table extraction."""
    
    def test_extract_facts_from_sample_table(self, sample_doc_snapshot, sample_table):
        """Test table extraction from sample table."""
        with patch("open_deep_research.extraction.call_llm") as mock_llm:
            mock_llm.return_value = json.dumps([
                {
                    "metric": "Revenue",
                    "value": 35082,
                    "unit": "USD",
                    "period_end_date": "Oct 27, 2024",
                    "row_label": "Revenue",
                    "column_label": "Oct 27, 2024",
                    "row_index": 0,
                    "column_index": 1
                },
                {
                    "metric": "Data Center",
                    "value": 30770,
                    "unit": "USD",
                    "period_end_date": "Oct 27, 2024",
                    "row_label": "Data Center",
                    "column_label": "Oct 27, 2024",
                    "row_index": 1,
                    "column_index": 1
                }
            ])
            
            facts = extract_facts_from_table(
                table=sample_table,
                entity="NVDA",
                doc_snapshot=sample_doc_snapshot
            )
            
            assert len(facts) == 2
            
            # Verify first fact
            assert facts[0].metric == "Revenue"
            assert facts[0].value == 35082
            assert facts[0].source_format == "html_table"
            assert facts[0].extracted_scale == "millions"
            
            # Verify location
            assert facts[0].location.table_index == 0
            assert facts[0].location.row_label == "Revenue"
            assert facts[0].location.column_label == "Oct 27, 2024"
    
    def test_row_label_matches_actual_table_data(
        self, sample_doc_snapshot, sample_table
    ):
        """Test that row_label matches actual table data."""
        with patch("open_deep_research.extraction.call_llm") as mock_llm:
            mock_llm.return_value = json.dumps([{
                "metric": "Gaming",
                "value": 3279,
                "unit": "USD",
                "period_end_date": "Oct 27, 2024",
                "row_label": "Gaming",
                "column_label": "Oct 27, 2024",
                "row_index": 2,
                "column_index": 1
            }])
            
            facts = extract_facts_from_table(
                table=sample_table,
                entity="NVDA",
                doc_snapshot=sample_doc_snapshot
            )
            
            assert len(facts) == 1
            assert facts[0].location.row_label == "Gaming"
            
            # Verify the row_label exists in the actual table
            df = pd.read_json(StringIO(sample_table.dataframe_json))
            first_col = df.iloc[:, 0].tolist()
            assert facts[0].location.row_label in first_col
    
    def test_invalid_coordinates_rejected(self, sample_doc_snapshot, sample_table):
        """Test that facts with invalid coordinates are rejected."""
        with patch("open_deep_research.extraction.call_llm") as mock_llm:
            mock_llm.return_value = json.dumps([{
                "metric": "Invalid",
                "value": 99999,
                "unit": "USD",
                "period_end_date": "Oct 27, 2024",
                "row_label": "Invalid",
                "column_label": "Oct 27, 2024",
                "row_index": 100,  # Way out of bounds
                "column_index": 100
            }])
            
            facts = extract_facts_from_table(
                table=sample_table,
                entity="NVDA",
                doc_snapshot=sample_doc_snapshot
            )
            
            # Should be rejected because coordinates out of bounds
            assert len(facts) == 0


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_empty_text_returns_empty_list(self, sample_doc_snapshot):
        """Test that empty text returns empty list without calling LLM."""
        with patch("open_deep_research.extraction.call_llm") as mock_llm:
            facts = extract_facts_from_text(
                text="",
                entity="NVDA",
                doc_snapshot=sample_doc_snapshot,
                section_id="Item7",
                paragraph_index=0
            )
            
            assert facts == []
            mock_llm.assert_not_called()
    
    def test_empty_table_returns_empty_list(self, sample_doc_snapshot):
        """Test that empty table returns empty list."""
        empty_df = pd.DataFrame()
        empty_table = ExtractedTable(
            table_index=0,
            section_id="Item7",
            html="<table></table>",
            headers=[],
            row_count=0,
            column_count=0,
            scale=None,
            has_per_share_exception=False,
            dataframe_json=empty_df.to_json(),
        )
        
        with patch("open_deep_research.extraction.call_llm") as mock_llm:
            facts = extract_facts_from_table(
                table=empty_table,
                entity="NVDA",
                doc_snapshot=sample_doc_snapshot
            )
            
            assert facts == []
            mock_llm.assert_not_called()
    
    def test_llm_error_returns_empty_list(self, sample_doc_snapshot):
        """Test that LLM errors return empty list without crashing."""
        with patch("open_deep_research.extraction.call_llm") as mock_llm:
            mock_llm.side_effect = Exception("LLM API error")
            
            facts = extract_facts_from_text(
                text="NVIDIA reported revenue of $35 billion.",
                entity="NVDA",
                doc_snapshot=sample_doc_snapshot,
                section_id="Item7",
                paragraph_index=0
            )
            
            assert facts == []
    
    def test_null_value_handled(self, sample_doc_snapshot, sample_table):
        """Test that null values are handled properly."""
        with patch("open_deep_research.extraction.call_llm") as mock_llm:
            mock_llm.return_value = json.dumps([{
                "metric": "Revenue",
                "value": None,  # Null value
                "unit": "USD",
                "period_end_date": "Oct 27, 2024",
                "row_label": "Revenue",
                "column_label": "Oct 27, 2024",
                "row_index": 0,
                "column_index": 1
            }])
            
            facts = extract_facts_from_table(
                table=sample_table,
                entity="NVDA",
                doc_snapshot=sample_doc_snapshot
            )
            
            assert len(facts) == 1
            assert facts[0].value is None

