"""
Tests for entity resolution module.
"""
import pytest
from unittest.mock import patch

from open_deep_research.entities import (
    pad_cik,
    load_ticker_mapping,
    clear_cache,
    resolve_entity,
    get_cik,
    get_fiscal_year_end,
    FISCAL_YEAR_ENDS,
)
from open_deep_research.models import EntityInfo


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def clear_ticker_cache():
    """Clear cache before and after each test."""
    clear_cache()
    yield
    clear_cache()


# =============================================================================
# CIK Padding Tests
# =============================================================================


class TestPadCik:
    """Tests for pad_cik function."""
    
    def test_pad_cik_from_int(self):
        """Test padding integer CIK."""
        assert pad_cik(320193) == "0000320193"
    
    def test_pad_cik_from_string(self):
        """Test padding string CIK."""
        assert pad_cik("320193") == "0000320193"
    
    def test_pad_cik_already_padded(self):
        """Test already 10-digit CIK remains unchanged."""
        assert pad_cik("0000320193") == "0000320193"
    
    def test_pad_cik_short(self):
        """Test short CIK gets fully padded."""
        assert pad_cik("1234") == "0000001234"
    
    def test_pad_cik_single_digit(self):
        """Test single digit CIK."""
        assert pad_cik(1) == "0000000001"
    
    def test_pad_cik_with_leading_zeros_string(self):
        """Test string with leading zeros."""
        assert pad_cik("0001234") == "0000001234"
    
    def test_pad_cik_large_number(self):
        """Test large CIK (10 digits)."""
        assert pad_cik(1234567890) == "1234567890"
    
    def test_pad_cik_nvidia(self):
        """Test NVIDIA's CIK."""
        assert pad_cik(1045810) == "0001045810"
    
    def test_pad_cik_apple(self):
        """Test Apple's CIK."""
        assert pad_cik(320193) == "0000320193"


# =============================================================================
# Data Loading Tests
# =============================================================================


class TestLoadTickerMapping:
    """Tests for load_ticker_mapping function."""
    
    def test_load_data_successfully(self):
        """Test that data loads successfully."""
        mapping = load_ticker_mapping()
        
        assert "by_ticker" in mapping
        assert "by_name" in mapping
        assert len(mapping["by_ticker"]) > 0
        assert len(mapping["by_name"]) > 0
    
    def test_cache_works(self):
        """Test that cache prevents reloading."""
        # First load
        mapping1 = load_ticker_mapping()
        # Second load should use cache
        mapping2 = load_ticker_mapping()
        
        # Should be the exact same object (from cache)
        assert mapping1 is mapping2
    
    def test_mapping_contains_nvidia(self):
        """Test that mapping includes NVIDIA."""
        mapping = load_ticker_mapping()
        
        assert "NVDA" in mapping["by_ticker"]
        assert mapping["by_ticker"]["NVDA"]["cik"] == 1045810
    
    def test_mapping_contains_apple(self):
        """Test that mapping includes Apple."""
        mapping = load_ticker_mapping()
        
        assert "AAPL" in mapping["by_ticker"]
        assert mapping["by_ticker"]["AAPL"]["cik"] == 320193
    
    def test_file_not_found_raises(self):
        """Test that missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError) as exc_info:
            load_ticker_mapping("/nonexistent/path/company_tickers.json")
        
        assert "company_tickers.json not found" in str(exc_info.value)


# =============================================================================
# Entity Resolution Tests
# =============================================================================


class TestResolveEntity:
    """Tests for resolve_entity function."""
    
    def test_exact_ticker_match(self):
        """Test exact ticker match."""
        entity = resolve_entity("NVDA")
        
        assert entity is not None
        assert entity.ticker == "NVDA"
        assert "NVIDIA" in entity.company_name.upper()
        assert entity.cik == "0001045810"
    
    def test_ticker_case_insensitivity(self):
        """Test ticker matching is case-insensitive."""
        entity_lower = resolve_entity("nvda")
        entity_upper = resolve_entity("NVDA")
        entity_mixed = resolve_entity("NvDa")
        
        assert entity_lower is not None
        assert entity_upper is not None
        assert entity_mixed is not None
        
        assert entity_lower.cik == entity_upper.cik == entity_mixed.cik
    
    def test_company_name_exact_match(self):
        """Test exact company name match."""
        entity = resolve_entity("Apple Inc.")
        
        assert entity is not None
        assert entity.ticker == "AAPL"
    
    def test_company_name_substring_match(self):
        """Test company name substring matching."""
        entity = resolve_entity("nvidia")
        
        assert entity is not None
        assert entity.ticker == "NVDA"
    
    def test_company_name_case_insensitivity(self):
        """Test company name matching is case-insensitive."""
        entity = resolve_entity("NVIDIA")
        
        assert entity is not None
        assert entity.ticker == "NVDA"
    
    def test_unknown_ticker_returns_none(self):
        """Test unknown ticker returns None."""
        entity = resolve_entity("XXXXX")
        
        assert entity is None
    
    def test_unknown_company_returns_none(self):
        """Test unknown company returns None."""
        entity = resolve_entity("FakeCompanyThatDoesNotExist12345")
        
        assert entity is None
    
    def test_empty_query_returns_none(self):
        """Test empty query returns None."""
        assert resolve_entity("") is None
        assert resolve_entity("   ") is None
    
    def test_cik_is_padded_in_result(self):
        """Test that CIK in result is properly padded."""
        entity = resolve_entity("AAPL")
        
        assert entity is not None
        assert len(entity.cik) == 10
        assert entity.cik == "0000320193"
    
    def test_returns_entity_info_type(self):
        """Test that result is EntityInfo instance."""
        entity = resolve_entity("MSFT")
        
        assert entity is not None
        assert isinstance(entity, EntityInfo)
    
    def test_fiscal_year_end_included_when_known(self):
        """Test that fiscal_year_end is populated for known tickers."""
        entity = resolve_entity("NVDA")
        
        assert entity is not None
        assert entity.fiscal_year_end == "January"
    
    def test_fiscal_year_end_none_when_unknown(self):
        """Test that fiscal_year_end is None for unknown tickers."""
        # Find a ticker that exists but isn't in FISCAL_YEAR_ENDS
        entity = resolve_entity("WMT")  # Walmart
        
        if entity is not None:
            assert entity.fiscal_year_end is None


# =============================================================================
# get_cik Convenience Function Tests
# =============================================================================


class TestGetCik:
    """Tests for get_cik convenience function."""
    
    def test_get_cik_returns_padded_cik(self):
        """Test get_cik returns padded CIK."""
        cik = get_cik("AAPL")
        
        assert cik == "0000320193"
    
    def test_get_cik_unknown_returns_none(self):
        """Test get_cik returns None for unknown ticker."""
        cik = get_cik("XXXXX")
        
        assert cik is None
    
    def test_get_cik_case_insensitive(self):
        """Test get_cik is case-insensitive."""
        assert get_cik("nvda") == get_cik("NVDA")


# =============================================================================
# Fiscal Year End Tests
# =============================================================================


class TestGetFiscalYearEnd:
    """Tests for get_fiscal_year_end function."""
    
    def test_nvda_fiscal_year_end(self):
        """Test NVIDIA fiscal year end is January."""
        assert get_fiscal_year_end("NVDA") == "January"
    
    def test_aapl_fiscal_year_end(self):
        """Test Apple fiscal year end is September."""
        assert get_fiscal_year_end("AAPL") == "September"
    
    def test_msft_fiscal_year_end(self):
        """Test Microsoft fiscal year end is June."""
        assert get_fiscal_year_end("MSFT") == "June"
    
    def test_googl_fiscal_year_end(self):
        """Test Alphabet fiscal year end is December."""
        assert get_fiscal_year_end("GOOGL") == "December"
    
    def test_case_insensitivity(self):
        """Test fiscal year end lookup is case-insensitive."""
        assert get_fiscal_year_end("nvda") == "January"
        assert get_fiscal_year_end("NvDa") == "January"
    
    def test_unknown_ticker_raises_value_error(self):
        """Test unknown ticker raises ValueError, NOT returns None."""
        with pytest.raises(ValueError) as exc_info:
            get_fiscal_year_end("XXXXX")
        
        # Check error message is helpful
        assert "Fiscal year end unknown" in str(exc_info.value)
        assert "XXXXX" in str(exc_info.value)
    
    def test_error_message_includes_known_tickers(self):
        """Test error message includes list of known tickers."""
        with pytest.raises(ValueError) as exc_info:
            get_fiscal_year_end("UNKNOWN")
        
        error_msg = str(exc_info.value)
        assert "Known tickers:" in error_msg
        # Check at least some known tickers are listed
        assert "NVDA" in error_msg
        assert "AAPL" in error_msg
    
    def test_all_known_tickers_have_fiscal_year(self):
        """Test all tickers in FISCAL_YEAR_ENDS work."""
        for ticker in FISCAL_YEAR_ENDS:
            fy_end = get_fiscal_year_end(ticker)
            assert fy_end is not None
            assert isinstance(fy_end, str)


# =============================================================================
# FISCAL_YEAR_ENDS Data Validation
# =============================================================================


class TestFiscalYearEndsData:
    """Tests for FISCAL_YEAR_ENDS constant."""
    
    def test_contains_expected_tickers(self):
        """Test that all expected tickers are present."""
        expected_tickers = ["NVDA", "AAPL", "MSFT", "GOOGL", "GOOG", 
                          "AMZN", "META", "TSLA", "AMD", "INTC"]
        
        for ticker in expected_tickers:
            assert ticker in FISCAL_YEAR_ENDS, f"Missing ticker: {ticker}"
    
    def test_values_are_valid_months(self):
        """Test that all values are valid month names."""
        valid_months = [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December"
        ]
        
        for ticker, month in FISCAL_YEAR_ENDS.items():
            assert month in valid_months, f"Invalid month '{month}' for {ticker}"


# =============================================================================
# Integration Tests
# =============================================================================


class TestEntityIntegration:
    """Integration tests using real data."""
    
    def test_resolve_multiple_tickers(self):
        """Test resolving multiple major tickers."""
        tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
        
        for ticker in tickers:
            entity = resolve_entity(ticker)
            assert entity is not None, f"Failed to resolve {ticker}"
            assert entity.ticker == ticker
            assert len(entity.cik) == 10
    
    def test_microsoft_complete_info(self):
        """Test complete info for Microsoft."""
        entity = resolve_entity("MSFT")
        
        assert entity is not None
        assert entity.ticker == "MSFT"
        assert entity.cik == "0000789019"
        assert "MICROSOFT" in entity.company_name.upper()
        assert entity.fiscal_year_end == "June"
    
    def test_google_both_tickers(self):
        """Test both GOOGL and GOOG resolve correctly."""
        googl = resolve_entity("GOOGL")
        goog = resolve_entity("GOOG")
        
        assert googl is not None
        assert goog is not None
        
        # Both should have December fiscal year end
        assert get_fiscal_year_end("GOOGL") == "December"
        assert get_fiscal_year_end("GOOG") == "December"

