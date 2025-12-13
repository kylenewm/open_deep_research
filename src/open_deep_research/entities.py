"""
Entity resolution for SEC filings.

Maps tickers and company names to CIKs using SEC's company_tickers.json.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Optional, Union

from .models import EntityInfo


# =============================================================================
# Module-level cache
# =============================================================================

_TICKER_CACHE: Optional[Dict] = None


# =============================================================================
# Fiscal year ends (CRITICAL: Do not silently default)
# =============================================================================

FISCAL_YEAR_ENDS: Dict[str, str] = {
    "NVDA": "January",
    "AAPL": "September",
    "MSFT": "June",
    "GOOGL": "December",
    "GOOG": "December",
    "AMZN": "December",
    "META": "December",
    "TSLA": "December",
    "AMD": "December",
    "INTC": "December",
}


# =============================================================================
# Core functions
# =============================================================================


def pad_cik(cik: Union[int, str]) -> str:
    """
    Convert CIK to 10-digit zero-padded string.
    
    SEC EDGAR URLs require 10-digit zero-padded CIKs.
    
    Args:
        cik: The CIK as an integer or string
        
    Returns:
        10-digit zero-padded CIK string
        
    Examples:
        >>> pad_cik(320193)
        '0000320193'
        >>> pad_cik("320193")
        '0000320193'
        >>> pad_cik("0000320193")
        '0000320193'
    """
    # Convert to string and strip any whitespace
    cik_str = str(cik).strip()
    
    # Remove any leading zeros and re-pad to exactly 10 digits
    cik_int = int(cik_str)
    return f"{cik_int:010d}"


def load_ticker_mapping(data_path: Optional[str] = None) -> Dict:
    """
    Load and parse company_tickers.json from SEC.
    
    Creates lookup structures for both ticker and company name.
    Results are cached in memory after first load.
    
    Args:
        data_path: Path to company_tickers.json. 
                   Defaults to data/company_tickers.json relative to package.
                   
    Returns:
        Dict with 'by_ticker' and 'by_name' lookups.
        
    Structure:
        {
            'by_ticker': {'AAPL': {'cik': 320193, 'title': 'Apple Inc.'}, ...},
            'by_name': {'apple inc.': {'cik': 320193, 'ticker': 'AAPL', 'title': 'Apple Inc.'}, ...}
        }
    """
    global _TICKER_CACHE
    
    if _TICKER_CACHE is not None:
        return _TICKER_CACHE
    
    # Determine data path
    if data_path is None:
        # Default to data/company_tickers.json relative to open_deep_research root
        package_dir = Path(__file__).parent.parent.parent  # src/open_deep_research -> open_deep_research
        data_path = package_dir / "data" / "company_tickers.json"
    else:
        data_path = Path(data_path)
    
    if not data_path.exists():
        raise FileNotFoundError(
            f"company_tickers.json not found at {data_path}. "
            "Download from: https://www.sec.gov/files/company_tickers.json"
        )
    
    with open(data_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    # Build lookup structures
    by_ticker: Dict[str, Dict] = {}
    by_name: Dict[str, Dict] = {}
    
    for entry in raw_data.values():
        ticker = entry.get("ticker", "").upper()
        cik = entry.get("cik_str")
        title = entry.get("title", "")
        
        if ticker and cik:
            by_ticker[ticker] = {
                "cik": cik,
                "title": title,
            }
            
            # Index by lowercase company name for case-insensitive search
            name_key = title.lower()
            by_name[name_key] = {
                "cik": cik,
                "ticker": ticker,
                "title": title,
            }
    
    _TICKER_CACHE = {
        "by_ticker": by_ticker,
        "by_name": by_name,
    }
    
    return _TICKER_CACHE


def clear_cache() -> None:
    """Clear the module-level ticker cache. Useful for testing."""
    global _TICKER_CACHE
    _TICKER_CACHE = None


def resolve_entity(query: str) -> Optional[EntityInfo]:
    """
    Resolve a ticker or company name to EntityInfo.
    
    Args:
        query: A ticker symbol (e.g., "NVDA") or company name (e.g., "NVIDIA")
        
    Returns:
        EntityInfo if found, None otherwise
        
    Resolution order:
        1. Exact ticker match (case-insensitive)
        2. Company name substring match (case-insensitive)
    """
    if not query or not query.strip():
        return None
        
    query = query.strip()
    mapping = load_ticker_mapping()
    
    # Try exact ticker match first (case-insensitive)
    ticker_upper = query.upper()
    if ticker_upper in mapping["by_ticker"]:
        entry = mapping["by_ticker"][ticker_upper]
        return EntityInfo(
            ticker=ticker_upper,
            company_name=entry["title"],
            cik=pad_cik(entry["cik"]),
            fiscal_year_end=FISCAL_YEAR_ENDS.get(ticker_upper),
        )
    
    # Try exact company name match
    query_lower = query.lower()
    if query_lower in mapping["by_name"]:
        entry = mapping["by_name"][query_lower]
        ticker = entry["ticker"]
        return EntityInfo(
            ticker=ticker,
            company_name=entry["title"],
            cik=pad_cik(entry["cik"]),
            fiscal_year_end=FISCAL_YEAR_ENDS.get(ticker),
        )
    
    # Try substring match on company names
    for name_key, entry in mapping["by_name"].items():
        if query_lower in name_key:
            ticker = entry["ticker"]
            return EntityInfo(
                ticker=ticker,
                company_name=entry["title"],
                cik=pad_cik(entry["cik"]),
                fiscal_year_end=FISCAL_YEAR_ENDS.get(ticker),
            )
    
    return None


def get_cik(ticker: str) -> Optional[str]:
    """
    Convenience function to get just the padded CIK for a ticker.
    
    Args:
        ticker: Stock ticker symbol (e.g., "AAPL")
        
    Returns:
        10-digit zero-padded CIK string, or None if ticker not found
    """
    entity = resolve_entity(ticker)
    if entity:
        return entity.cik
    return None


def get_fiscal_year_end(ticker: str) -> str:
    """
    Get fiscal year end month for a ticker.
    
    CRITICAL: This function explicitly fails for unknown tickers rather than
    silently defaulting to calendar year. Silent defaults cause period 
    mismatch hallucinations.
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        Month name (e.g., "January", "September")
        
    Raises:
        ValueError: If ticker is not in the hardcoded fiscal year map.
    """
    ticker_upper = ticker.upper()
    if ticker_upper not in FISCAL_YEAR_ENDS:
        raise ValueError(
            f"Fiscal year end unknown for ticker '{ticker}'. "
            f"Known tickers: {list(FISCAL_YEAR_ENDS.keys())}. "
            f"Add this ticker to FISCAL_YEAR_ENDS in entities.py before proceeding."
        )
    return FISCAL_YEAR_ENDS[ticker_upper]

