"""
SEC EDGAR ingestion for downloading and caching SEC filings.

Uses sec-edgar-downloader to fetch filings and creates immutable
document snapshots with content hashes for verification.
"""
from __future__ import annotations

import hashlib
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from sec_edgar_downloader import Downloader
from urllib3.util.retry import Retry
import requests

from open_deep_research.models import DocumentSnapshot
from open_deep_research.entities import resolve_entity, pad_cik

# Load environment variables from .env file
load_dotenv()


# Rate limiting delay (seconds) between SEC requests
# Using 0.2s (5 req/sec) for safety margin - SEC limit is 10 req/sec
SEC_RATE_LIMIT_DELAY = 0.2

# Default cache directory
DEFAULT_CACHE_DIR = ".cache/sec_filings"

# Hardcoded NVIDIA CIK (will be made dynamic in Step 5)
NVIDIA_CIK = "0001045810"


def get_sec_user_agent() -> str:
    """
    Get the SEC User-Agent from environment variable.
    
    The SEC requires a User-Agent string to identify requesters.
    Without this, requests will fail with 403 Forbidden.
    
    Returns:
        User-Agent string in format "AppName email@domain.com"
    
    Raises:
        ValueError: If SEC_USER_AGENT environment variable is not set or is a placeholder
    """
    user_agent = os.environ.get("SEC_USER_AGENT")
    if not user_agent:
        raise ValueError(
            "SEC_USER_AGENT environment variable required.\n"
            "The SEC requires identification for all requests.\n"
            "Set it to 'YourAppName your-email@domain.com'\n"
            "Example: export SEC_USER_AGENT='ResearchAgent admin@mycompany.com'"
        )
    
    # Additional safety checks for placeholder values
    # These are common copy-paste mistakes that will get you flagged/banned
    ua_lower = user_agent.lower()
    placeholder_patterns = [
        "your.real.email",
        "placeholder", 
        "example.com",
        "your-email",
        "youremail",
        "your_email",
        "admin@mycompany.com",  # The example from error message
    ]
    
    if any(placeholder in ua_lower for placeholder in placeholder_patterns):
        raise ValueError(
            f"SEC_USER_AGENT appears to be a placeholder: '{user_agent}'.\n"
            "DANGER: Using placeholder User-Agents can get your IP banned.\n"
            "Set it to your actual app name and real email address."
        )
    
    return user_agent


def get_requests_session() -> requests.Session:
    """
    Creates a requests session with automatic retries for rate limits and server errors.
    
    Uses exponential backoff: sleeps 1s, 2s, 4s between retries.
    Handles HTTP 429 (Rate Limit) and 5xx server errors.
    
    Returns:
        Configured requests.Session with retry logic
        
    Note:
        sec-edgar-downloader handles its own HTTP requests internally,
        so this session is for future direct SEC API calls.
    """
    session = requests.Session()
    
    # Retry strategy:
    # - Total retries: 3
    # - Backoff factor: 1 (sleeps 1s, 2s, 4s)
    # - Status codes: 429 (Rate Limit), 500/502/503/504 (Server Errors)
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"]
    )
    
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    
    return session


def _parse_user_agent(user_agent: str) -> tuple[str, str]:
    """
    Parse User-Agent into company name and email.
    
    sec-edgar-downloader requires separate company name and email.
    
    Args:
        user_agent: Full user agent string "CompanyName email@domain.com"
        
    Returns:
        Tuple of (company_name, email)
    """
    parts = user_agent.strip().split()
    if len(parts) < 2:
        # If only one part, use it as both
        return user_agent.strip(), user_agent.strip()
    
    # Last part is typically the email
    email = parts[-1]
    company_name = " ".join(parts[:-1])
    return company_name, email


def download_filing(
    cik: str,
    filing_type: str,
    output_dir: str,
    limit: int = 1,
) -> Path:
    """
    Download an SEC filing for a given CIK.
    
    Args:
        cik: The CIK number (e.g., "0001045810" for NVIDIA)
        filing_type: Type of filing ("10-K", "10-Q", "8-K")
        output_dir: Directory to save downloaded files
        limit: Number of most recent filings to download (default 1)
    
    Returns:
        Path to the directory containing downloaded filing files
        
    Raises:
        ValueError: If SEC_USER_AGENT is not set
        FileNotFoundError: If no filings were downloaded
    """
    user_agent = get_sec_user_agent()
    company_name, email = _parse_user_agent(user_agent)
    
    # Create downloader with required identification
    dl = Downloader(company_name, email, output_dir)
    
    # Rate limiting
    time.sleep(SEC_RATE_LIMIT_DELAY)
    
    # Download the filing with download_details=True to get HTML documents
    dl.get(filing_type, cik, limit=limit, download_details=True)
    
    # Find the downloaded directory
    # sec-edgar-downloader creates: output_dir/sec-edgar-filings/CIK/FILING_TYPE/
    filing_dir = Path(output_dir) / "sec-edgar-filings" / cik / filing_type
    
    if not filing_dir.exists():
        raise FileNotFoundError(
            f"No {filing_type} filings found for CIK {cik}. "
            f"Expected directory: {filing_dir}"
        )
    
    # Get the most recent filing directory (subdirectories are dated)
    subdirs = sorted(filing_dir.iterdir(), reverse=True)
    if not subdirs:
        raise FileNotFoundError(f"No filing subdirectories in {filing_dir}")
    
    return subdirs[0]


def _find_primary_html(filing_dir: Path) -> Path:
    """
    Find the primary HTML document in a filing directory.
    
    The sec-edgar-downloader downloads multiple files. We want the
    main filing document. With download_details=True, this is
    typically named 'primary-document.html'.
    
    Args:
        filing_dir: Directory containing downloaded filing files
        
    Returns:
        Path to the primary HTML file
        
    Raises:
        FileNotFoundError: If no HTML files found
    """
    # First check for the standard primary document name
    primary_doc = filing_dir / "primary-document.html"
    if primary_doc.exists():
        return primary_doc
    
    # Fallback: find HTML files
    html_files = list(filing_dir.glob("*.htm")) + list(filing_dir.glob("*.html"))
    
    if not html_files:
        raise FileNotFoundError(f"No HTML files found in {filing_dir}")
    
    # Primary document is usually the largest HTML file
    # (contains the full filing, not exhibits or graphics)
    primary = max(html_files, key=lambda f: f.stat().st_size)
    return primary


def get_filing_html(
    cik: str,
    filing_type: str,
    cache_dir: Optional[str] = None,
) -> str:
    """
    Get the HTML content of an SEC filing, using cache if available.
    
    Args:
        cik: The CIK number (e.g., "0001045810" for NVIDIA)
        filing_type: Type of filing ("10-K", "10-Q", "8-K")
        cache_dir: Directory for caching (default: .cache/sec_filings/)
    
    Returns:
        Raw HTML content as string
        
    Raises:
        ValueError: If SEC_USER_AGENT is not set
        FileNotFoundError: If no filings found
    """
    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR
    
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    
    # Check for cached HTML file
    cached_file = cache_path / f"{cik}_{filing_type}_latest.html"
    if cached_file.exists():
        return cached_file.read_text(encoding="utf-8")
    
    # Download the filing
    filing_dir = download_filing(cik, filing_type, cache_dir)
    
    # Find and read the primary HTML document
    html_path = _find_primary_html(filing_dir)
    html_content = html_path.read_text(encoding="utf-8", errors="replace")
    
    # Cache the HTML content
    cached_file.write_text(html_content, encoding="utf-8")
    
    return html_content


def create_document_snapshot(
    html_content: str,
    url: str,
    cik: str,
    doc_type: str,
) -> DocumentSnapshot:
    """
    Create an immutable snapshot of a document.
    
    Args:
        html_content: Raw HTML content of the document
        url: URL or identifier for the document source
        cik: The CIK number
        doc_type: Type of document ("10-K", "10-Q", "8-K")
    
    Returns:
        DocumentSnapshot with unique ID and content hash
    """
    # Generate unique snapshot ID
    snapshot_id = str(uuid.uuid4())
    
    # Compute SHA-256 hash of content
    content_hash = hashlib.sha256(html_content.encode("utf-8")).hexdigest()
    
    # Record current timestamp
    retrieved_at = datetime.now()
    
    return DocumentSnapshot(
        snapshot_id=snapshot_id,
        url=url,
        cik=cik,
        doc_type=doc_type,
        retrieved_at=retrieved_at,
        content_hash=content_hash,
        raw_html=html_content,
    )


def get_nvidia_filing(filing_type: str = "10-K") -> DocumentSnapshot:
    """
    Convenience function to get NVIDIA filing as a snapshot.
    
    Args:
        filing_type: Type of filing ("10-K", "10-Q", "8-K")
    
    Returns:
        DocumentSnapshot of the latest NVIDIA filing
    """
    html_content = get_filing_html(NVIDIA_CIK, filing_type)
    
    url = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={NVIDIA_CIK}&type={filing_type}"
    
    return create_document_snapshot(
        html_content=html_content,
        url=url,
        cik=NVIDIA_CIK,
        doc_type=filing_type,
    )


def download_filing_by_ticker(
    ticker: str,
    filing_type: str,
    output_dir: Optional[str] = None,
) -> Path:
    """
    Download an SEC filing by ticker symbol.
    
    Resolves the ticker to a CIK and downloads the filing.
    
    Args:
        ticker: Stock ticker symbol (e.g., "NVDA", "AAPL")
        filing_type: Type of filing ("10-K", "10-Q", "8-K")
        output_dir: Directory to save downloaded files (default: cache dir)
    
    Returns:
        Path to the directory containing downloaded filing files
        
    Raises:
        ValueError: If ticker cannot be resolved to a CIK
    """
    # Resolve ticker to entity info
    entity = resolve_entity(ticker)
    if entity is None:
        raise ValueError(
            f"Could not resolve ticker '{ticker}' to a CIK. "
            f"Ensure the ticker is valid and exists in SEC's company_tickers.json"
        )
    
    if output_dir is None:
        output_dir = DEFAULT_CACHE_DIR
    
    # Download using the resolved CIK
    return download_filing(entity.cik, filing_type, output_dir)


def get_filing_by_ticker(
    ticker: str,
    filing_type: str,
    cache_dir: Optional[str] = None,
) -> DocumentSnapshot:
    """
    Get an SEC filing by ticker symbol as a DocumentSnapshot.
    
    Resolves the ticker to a CIK, downloads the filing, and creates a snapshot.
    
    Args:
        ticker: Stock ticker symbol (e.g., "NVDA", "AAPL")
        filing_type: Type of filing ("10-K", "10-Q", "8-K")
        cache_dir: Directory for caching (default: .cache/sec_filings/)
    
    Returns:
        DocumentSnapshot of the filing
        
    Raises:
        ValueError: If ticker cannot be resolved to a CIK
    """
    # Resolve ticker to entity info
    entity = resolve_entity(ticker)
    if entity is None:
        raise ValueError(
            f"Could not resolve ticker '{ticker}' to a CIK. "
            f"Ensure the ticker is valid and exists in SEC's company_tickers.json"
        )
    
    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR
    
    # Get HTML content using the resolved CIK
    html_content = get_filing_html(entity.cik, filing_type, cache_dir)
    
    url = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={entity.cik}&type={filing_type}"
    
    return create_document_snapshot(
        html_content=html_content,
        url=url,
        cik=entity.cik,
        doc_type=filing_type,
    )

