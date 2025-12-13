"""
Numeric verification for financial facts.

Handles normalization and comparison of financial values with various
formats, scales, and notations (including accounting dashes).
"""
from __future__ import annotations

import re
from typing import Literal, Optional, Union


# Scale multipliers
SCALE_MULTIPLIERS = {
    # Trillion
    "t": 1_000_000_000_000,
    "trillion": 1_000_000_000_000,
    # Billion
    "b": 1_000_000_000,
    "bn": 1_000_000_000,
    "billion": 1_000_000_000,
    "billions": 1_000_000_000,
    # Million
    "m": 1_000_000,
    "mm": 1_000_000,
    "million": 1_000_000,
    "millions": 1_000_000,
    # Thousand
    "k": 1_000,
    "thousand": 1_000,
    "thousands": 1_000,
}

# Accounting dash patterns (represent zero)
ACCOUNTING_DASHES = {"-", "—", "–"}  # hyphen, em dash, en dash


def normalize_financial_value(value: Union[str, float, int], scale: Optional[str] = None) -> float:
    """
    Normalize a financial value to base units (single dollars/units).
    
    Args:
        value: The value to normalize. Can be:
            - int/float: applied scale if provided, returned as float
            - str: parsed for numeric value and scale indicators
        scale: Optional scale from table header (e.g., "millions", "thousands").
               Only applied if value is a raw number without embedded scale.
    
    Returns:
        Normalized value in base units (e.g., $10.5B -> 10500000000.0)
    
    Examples:
        >>> normalize_financial_value(10500000000)
        10500000000.0
        >>> normalize_financial_value("$10.5B")
        10500000000.0
        >>> normalize_financial_value("14,514", scale="millions")
        14514000000.0
        >>> normalize_financial_value("-")  # accounting dash
        0.0
    """
    # Handle numeric passthrough
    if isinstance(value, (int, float)):
        multiplier = 1.0
        if scale:
            scale_key = scale.lower().strip()
            multiplier = SCALE_MULTIPLIERS.get(scale_key, 1.0)
        return float(value) * multiplier
    
    # Handle string input
    if not isinstance(value, str):
        raise TypeError(f"Expected str, int, or float, got {type(value)}")
    
    # Strip whitespace
    text = value.strip()
    
    # Handle empty string
    if not text:
        return 0.0
    
    # Handle accounting dashes (zero values)
    # Check for standalone dashes or dashes with currency symbols/parens
    cleaned = re.sub(r'[\$€£\s\(\)]', '', text)
    if cleaned in ACCOUNTING_DASHES or cleaned == "":
        return 0.0
    
    # Determine if value is negative
    is_negative = False
    
    # Check for parentheses notation: (500) or ($500) means negative
    if text.startswith("(") and text.endswith(")"):
        is_negative = True
        text = text[1:-1].strip()
    # Check for leading minus sign
    elif text.startswith("-") and len(text) > 1 and text[1:].strip()[0].isdigit():
        is_negative = True
        text = text[1:].strip()
    
    # Remove currency symbols
    text = re.sub(r'[\$€£]', '', text)
    
    # Extract numeric part and scale from string
    # Pattern: optional negative, digits with commas/decimals, optional scale suffix
    pattern = r'(-?)[\s]*([\d,]+(?:\.\d+)?)\s*([a-zA-Z]*)'
    
    # Find all potential numeric matches
    matches = re.findall(pattern, text)
    
    if not matches:
        # Try to handle cases like "Revenue of $10,500 MM"
        # Extract just the number portion
        num_pattern = r'[\d,]+(?:\.\d+)?'
        num_match = re.search(num_pattern, text)
        if num_match:
            number_str = num_match.group().replace(',', '')
            number = float(number_str)
            
            # Look for scale in the rest of the text
            text_lower = text.lower()
            detected_multiplier = 1.0
            for scale_key, mult in SCALE_MULTIPLIERS.items():
                if scale_key in text_lower:
                    detected_multiplier = mult
                    break
            
            # Apply external scale only if no scale detected in string
            if detected_multiplier == 1.0 and scale:
                detected_multiplier = SCALE_MULTIPLIERS.get(scale.lower().strip(), 1.0)
            
            result = number * detected_multiplier
            return -result if is_negative else result
        return 0.0
    
    # Use the first match (usually the main number)
    sign, number_str, scale_suffix = matches[0]
    
    # Handle sign from regex
    if sign == "-":
        is_negative = True
    
    # Parse the number
    number_str = number_str.replace(',', '')
    try:
        number = float(number_str)
    except ValueError:
        return 0.0
    
    # Determine the scale multiplier
    multiplier = 1.0
    
    # First check for scale in the matched suffix
    if scale_suffix:
        scale_key = scale_suffix.lower().strip()
        if scale_key in SCALE_MULTIPLIERS:
            multiplier = SCALE_MULTIPLIERS[scale_key]
    
    # If no suffix scale, check the rest of the text for scale words
    if multiplier == 1.0:
        text_lower = text.lower()
        for scale_key, mult in SCALE_MULTIPLIERS.items():
            # Check for word boundaries to avoid false matches
            if re.search(rf'\b{re.escape(scale_key)}\b', text_lower):
                multiplier = mult
                break
    
    # If still no scale detected, apply external scale parameter
    if multiplier == 1.0 and scale:
        scale_key = scale.lower().strip()
        multiplier = SCALE_MULTIPLIERS.get(scale_key, 1.0)
    
    result = number * multiplier
    return -result if is_negative else result


def verify_numeric_fact(
    claim_value: Union[str, float, int],
    source_value: Union[str, float, int],
    tolerance: float = 0.01,
    claim_scale: Optional[str] = None,
    source_scale: Optional[str] = None,
) -> Literal["exact_match", "approximate_match", "mismatch"]:
    """
    Verify a claimed numeric value against a source value.
    
    Args:
        claim_value: The value from the claim/report
        source_value: The value from the original source document
        tolerance: Allowed percentage difference (default 1%)
        claim_scale: Scale for claim value (e.g., "millions")
        source_scale: Scale for source value (e.g., "millions")
    
    Returns:
        - "exact_match": Values are identical after normalization
        - "approximate_match": Values differ by <= tolerance
        - "mismatch": Values differ by > tolerance
    
    Examples:
        >>> verify_numeric_fact("$10.5B", "10,500 million")
        'exact_match'
        >>> verify_numeric_fact("14514000000", "14,514", source_scale="millions")
        'exact_match'
        >>> verify_numeric_fact(10000000000, 10100000000)
        'approximate_match'
        >>> verify_numeric_fact("$10B", "$11B")
        'mismatch'
    """
    claim_normalized = normalize_financial_value(claim_value, scale=claim_scale)
    source_normalized = normalize_financial_value(source_value, scale=source_scale)
    
    # Handle zero comparison
    if source_normalized == 0:
        difference = 0.0 if claim_normalized == 0 else float('inf')
    else:
        difference = abs(claim_normalized - source_normalized) / abs(source_normalized)
    
    if difference == 0:
        return "exact_match"
    elif difference <= tolerance:
        return "approximate_match"
    else:
        return "mismatch"

