"""
Tests for SEC EDGAR ingestion module.
"""
from __future__ import annotations

import hashlib
import os
import re
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from unittest import mock

import pytest

from open_deep_research.ingestion import (
    DEFAULT_CACHE_DIR,
    NVIDIA_CIK,
    _find_primary_html,
    _parse_user_agent,
    create_document_snapshot,
    download_filing,
    get_filing_html,
    get_nvidia_filing,
    get_sec_user_agent,
)


# =============================================================================
# Configuration Tests
# =============================================================================


class TestSecUserAgentConfig:
    """Tests for SEC User-Agent configuration."""
    
    def test_missing_user_agent_raises_value_error(self):
        """Missing SEC_USER_AGENT should raise ValueError with helpful message."""
        with mock.patch.dict(os.environ, {}, clear=True):
            # Ensure SEC_USER_AGENT is not set
            if "SEC_USER_AGENT" in os.environ:
                del os.environ["SEC_USER_AGENT"]
            
            with mock.patch.dict(os.environ, {}, clear=True):
                with pytest.raises(ValueError) as exc_info:
                    get_sec_user_agent()
                
                error_msg = str(exc_info.value)
                assert "SEC_USER_AGENT" in error_msg
                assert "environment variable required" in error_msg
                assert "export SEC_USER_AGENT" in error_msg
    
    def test_valid_user_agent_is_accepted(self):
        """Valid SEC_USER_AGENT should be returned."""
        test_agent = "AcmeResearch analyst@acmecorp.io"
        with mock.patch.dict(os.environ, {"SEC_USER_AGENT": test_agent}):
            result = get_sec_user_agent()
            assert result == test_agent
    
    def test_parse_user_agent_splits_correctly(self):
        """User agent should be parsed into company name and email."""
        company, email = _parse_user_agent("MyCompany admin@test.com")
        assert company == "MyCompany"
        assert email == "admin@test.com"
    
    def test_parse_user_agent_multi_word_company(self):
        """Multi-word company names should be preserved."""
        company, email = _parse_user_agent("My Research Company admin@test.com")
        assert company == "My Research Company"
        assert email == "admin@test.com"
    
    def test_parse_user_agent_single_value(self):
        """Single value should be used for both fields."""
        company, email = _parse_user_agent("test@example.com")
        assert company == "test@example.com"
        assert email == "test@example.com"


# =============================================================================
# Snapshot Tests (Unit Tests - No Network)
# =============================================================================


class TestDocumentSnapshot:
    """Tests for document snapshot creation."""
    
    def test_snapshot_id_is_uuid_format(self):
        """Snapshot ID should be a valid UUID."""
        snapshot = create_document_snapshot(
            html_content="<html><body>Test</body></html>",
            url="https://example.com/test",
            cik="0001234567",
            doc_type="10-K",
        )
        
        # Verify it's a valid UUID
        parsed_uuid = uuid.UUID(snapshot.snapshot_id)
        assert str(parsed_uuid) == snapshot.snapshot_id
    
    def test_content_hash_is_sha256(self):
        """Content hash should be a 64-character hex SHA-256."""
        snapshot = create_document_snapshot(
            html_content="<html><body>Test</body></html>",
            url="https://example.com/test",
            cik="0001234567",
            doc_type="10-K",
        )
        
        # SHA-256 produces 64 hex characters
        assert len(snapshot.content_hash) == 64
        # Verify it's all hex
        assert re.match(r"^[0-9a-f]{64}$", snapshot.content_hash)
    
    def test_content_hash_is_deterministic(self):
        """Same content should produce same hash."""
        html_content = "<html><body>Deterministic Test</body></html>"
        
        snapshot1 = create_document_snapshot(
            html_content=html_content,
            url="https://example.com/test1",
            cik="0001234567",
            doc_type="10-K",
        )
        
        snapshot2 = create_document_snapshot(
            html_content=html_content,
            url="https://example.com/test2",  # Different URL
            cik="0001234567",
            doc_type="10-K",
        )
        
        assert snapshot1.content_hash == snapshot2.content_hash
    
    def test_different_content_produces_different_hash(self):
        """Different content should produce different hashes."""
        snapshot1 = create_document_snapshot(
            html_content="<html>Content A</html>",
            url="https://example.com/test",
            cik="0001234567",
            doc_type="10-K",
        )
        
        snapshot2 = create_document_snapshot(
            html_content="<html>Content B</html>",
            url="https://example.com/test",
            cik="0001234567",
            doc_type="10-K",
        )
        
        assert snapshot1.content_hash != snapshot2.content_hash
    
    def test_retrieved_at_is_set_to_current_time(self):
        """Retrieved_at should be approximately the current time."""
        before = datetime.now()
        
        snapshot = create_document_snapshot(
            html_content="<html><body>Test</body></html>",
            url="https://example.com/test",
            cik="0001234567",
            doc_type="10-K",
        )
        
        after = datetime.now()
        
        assert before <= snapshot.retrieved_at <= after
    
    def test_snapshot_preserves_metadata(self):
        """Snapshot should preserve all provided metadata."""
        snapshot = create_document_snapshot(
            html_content="<html>Test</html>",
            url="https://sec.gov/filing/12345",
            cik="0001045810",
            doc_type="10-Q",
        )
        
        assert snapshot.url == "https://sec.gov/filing/12345"
        assert snapshot.cik == "0001045810"
        assert snapshot.doc_type == "10-Q"
        assert snapshot.raw_html == "<html>Test</html>"
    
    def test_content_hash_matches_manual_calculation(self):
        """Content hash should match manual SHA-256 calculation."""
        html_content = "<html><body>Verify Hash</body></html>"
        
        expected_hash = hashlib.sha256(html_content.encode("utf-8")).hexdigest()
        
        snapshot = create_document_snapshot(
            html_content=html_content,
            url="https://example.com/test",
            cik="0001234567",
            doc_type="10-K",
        )
        
        assert snapshot.content_hash == expected_hash


class TestFindPrimaryHtml:
    """Tests for finding primary HTML document in filing directory."""
    
    def test_finds_largest_html_file(self):
        """Should return the largest HTML file as primary document."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            
            # Create several HTML files of different sizes
            small_file = tmppath / "exhibit.htm"
            small_file.write_text("<html>Small</html>")
            
            large_file = tmppath / "primary-document.htm"
            large_file.write_text("<html>" + "X" * 1000 + "</html>")
            
            medium_file = tmppath / "other.htm"
            medium_file.write_text("<html>" + "Y" * 100 + "</html>")
            
            result = _find_primary_html(tmppath)
            assert result == large_file
    
    def test_handles_html_extension(self):
        """Should find files with .html extension too."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            
            html_file = tmppath / "document.html"
            html_file.write_text("<html>Content</html>")
            
            result = _find_primary_html(tmppath)
            assert result == html_file
    
    def test_raises_when_no_html_files(self):
        """Should raise FileNotFoundError if no HTML files found."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            
            # Create non-HTML file
            txt_file = tmppath / "readme.txt"
            txt_file.write_text("Not HTML")
            
            with pytest.raises(FileNotFoundError) as exc_info:
                _find_primary_html(tmppath)
            
            assert "No HTML files found" in str(exc_info.value)


# =============================================================================
# Integration Tests (Real Network Requests)
# =============================================================================


@pytest.mark.integration
class TestDownloadIntegration:
    """Integration tests that make real SEC EDGAR requests.
    
    Run with: pytest -m integration
    Requires: SEC_USER_AGENT environment variable
    """
    
    @pytest.fixture(autouse=True)
    def check_env(self):
        """Skip if SEC_USER_AGENT is not configured."""
        if not os.environ.get("SEC_USER_AGENT"):
            pytest.skip("SEC_USER_AGENT not set - skipping integration tests")
    
    def test_download_nvidia_10k(self, tmp_path):
        """Test downloading NVIDIA 10-K filing."""
        filing_dir = download_filing(NVIDIA_CIK, "10-K", str(tmp_path))
        
        assert filing_dir.exists()
        assert filing_dir.is_dir()
        
        # Should contain primary-document.html (with download_details=True)
        primary_doc = filing_dir / "primary-document.html"
        assert primary_doc.exists(), f"Expected primary-document.html in {filing_dir}"
    
    def test_download_nvidia_10q(self, tmp_path):
        """Test downloading NVIDIA 10-Q filing."""
        filing_dir = download_filing(NVIDIA_CIK, "10-Q", str(tmp_path))
        
        assert filing_dir.exists()
        assert filing_dir.is_dir()
        
        # Should contain primary-document.html (with download_details=True)
        primary_doc = filing_dir / "primary-document.html"
        assert primary_doc.exists(), f"Expected primary-document.html in {filing_dir}"
    
    def test_get_filing_html_returns_valid_html(self, tmp_path):
        """Test that HTML content is properly retrieved."""
        html = get_filing_html(NVIDIA_CIK, "10-K", cache_dir=str(tmp_path))
        
        # Should contain HTML markers
        assert "<html" in html.lower() or "<!doctype" in html.lower()
        
        # Should be substantial content
        assert len(html) > 10000  # Real 10-K filings are large
    
    def test_filing_is_cached(self, tmp_path):
        """Test that filings are cached after first download."""
        cache_dir = str(tmp_path)
        
        # First download
        html1 = get_filing_html(NVIDIA_CIK, "10-K", cache_dir=cache_dir)
        
        # Check cache file exists
        cached_file = tmp_path / f"{NVIDIA_CIK}_10-K_latest.html"
        assert cached_file.exists()
        
        # Second call should use cache (we can't easily test this without
        # mocking, but we can verify same content is returned)
        html2 = get_filing_html(NVIDIA_CIK, "10-K", cache_dir=cache_dir)
        
        assert html1 == html2
    
    def test_get_nvidia_filing_returns_snapshot(self, tmp_path):
        """Test the convenience function returns a valid snapshot."""
        # Use a custom cache dir for isolation
        with mock.patch("open_deep_research.ingestion.DEFAULT_CACHE_DIR", str(tmp_path)):
            snapshot = get_nvidia_filing("10-K")
        
        # Verify snapshot properties
        assert snapshot.cik == NVIDIA_CIK
        assert snapshot.doc_type == "10-K"
        assert len(snapshot.content_hash) == 64
        assert "<html" in snapshot.raw_html.lower() or "<!doctype" in snapshot.raw_html.lower()


# =============================================================================
# pytest configuration
# =============================================================================


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests (require network)"
    )

