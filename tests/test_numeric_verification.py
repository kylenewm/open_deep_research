"""
Tests for numeric verification functions.
"""

import pytest

from open_deep_research.numeric_verification import (
    normalize_financial_value,
    verify_numeric_fact,
)


class TestNormalizeFinancialValuePassthrough:
    """Tests for int/float passthrough."""
    
    def test_int_passthrough(self):
        assert normalize_financial_value(10500000000) == 10500000000.0
    
    def test_float_passthrough(self):
        assert normalize_financial_value(10500000000.0) == 10500000000.0
    
    def test_int_with_scale(self):
        assert normalize_financial_value(100, scale="millions") == 100000000.0
    
    def test_float_with_scale(self):
        assert normalize_financial_value(14.514, scale="billions") == 14514000000.0


class TestNormalizeFinancialValueStrings:
    """Tests for string number parsing."""
    
    def test_plain_number_string(self):
        assert normalize_financial_value("10500000000") == 10500000000.0
    
    def test_with_commas(self):
        assert normalize_financial_value("10,500,000,000") == 10500000000.0
    
    def test_with_decimal(self):
        assert normalize_financial_value("10500000000.5") == 10500000000.5


class TestNormalizeFinancialValueBillions:
    """Tests for billion scale variations."""
    
    def test_uppercase_b(self):
        assert normalize_financial_value("$10.5B") == 10500000000.0
    
    def test_lowercase_b(self):
        assert normalize_financial_value("$10.5b") == 10500000000.0
    
    def test_bn_suffix(self):
        assert normalize_financial_value("$10.5bn") == 10500000000.0
    
    def test_billion_word(self):
        assert normalize_financial_value("$10.5 billion") == 10500000000.0
    
    def test_billion_word_after_number(self):
        assert normalize_financial_value("10.5 billion dollars") == 10500000000.0
    
    def test_billion_uppercase_word(self):
        assert normalize_financial_value("$10.5 Billion") == 10500000000.0


class TestNormalizeFinancialValueMillions:
    """Tests for million scale variations."""
    
    def test_uppercase_m(self):
        assert normalize_financial_value("$10,500M") == 10500000000.0
    
    def test_mm_suffix(self):
        assert normalize_financial_value("$10,500MM") == 10500000000.0
    
    def test_million_word(self):
        assert normalize_financial_value("$10,500 million") == 10500000000.0
    
    def test_million_in_sentence(self):
        assert normalize_financial_value("Revenue of $10,500 MM") == 10500000000.0
    
    def test_precise_millions(self):
        assert normalize_financial_value("$10,542MM") == 10542000000.0


class TestNormalizeFinancialValueNegatives:
    """Tests for negative number handling."""
    
    def test_parentheses_negative(self):
        assert normalize_financial_value("(500)") == -500.0
    
    def test_parentheses_with_currency_and_scale(self):
        assert normalize_financial_value("($500M)") == -500000000.0
    
    def test_minus_sign_with_scale(self):
        assert normalize_financial_value("-$500M") == -500000000.0
    
    def test_parentheses_negative_with_scale(self):
        assert normalize_financial_value("(500)", scale="millions") == -500000000.0


class TestNormalizeFinancialValueAccountingDashes:
    """Tests for accounting dash handling (CRITICAL)."""
    
    def test_hyphen_dash(self):
        assert normalize_financial_value("-") == 0.0
    
    def test_em_dash(self):
        assert normalize_financial_value("—") == 0.0
    
    def test_en_dash(self):
        assert normalize_financial_value("–") == 0.0
    
    def test_dollar_hyphen(self):
        assert normalize_financial_value("$ -") == 0.0
    
    def test_dollar_em_dash(self):
        assert normalize_financial_value("$ —") == 0.0
    
    def test_dollar_em_dash_in_parens(self):
        assert normalize_financial_value("$(—)") == 0.0


class TestNormalizeFinancialValueScaleParameter:
    """Tests for scale parameter (CRITICAL for tables)."""
    
    def test_millions_scale(self):
        assert normalize_financial_value("14,514", scale="millions") == 14514000000.0
    
    def test_thousands_scale(self):
        assert normalize_financial_value("14,514", scale="thousands") == 14514000.0
    
    def test_billions_scale(self):
        assert normalize_financial_value("14,514", scale="billions") == 14514000000000.0
    
    def test_no_scale(self):
        assert normalize_financial_value("14,514", scale=None) == 14514.0
    
    def test_thousands_large_number(self):
        assert normalize_financial_value("14,514,000", scale="thousands") == 14514000000.0


class TestNormalizeFinancialValueEdgeCases:
    """Tests for edge cases."""
    
    def test_zero_with_currency(self):
        assert normalize_financial_value("$0") == 0.0
    
    def test_plain_zero(self):
        assert normalize_financial_value("0") == 0.0
    
    def test_trillion(self):
        assert normalize_financial_value("$1.2T") == 1200000000000.0
    
    def test_decimal_precision(self):
        assert normalize_financial_value("$10.542B") == 10542000000.0
    
    def test_euro_currency(self):
        assert normalize_financial_value("€500M") == 500000000.0
    
    def test_empty_string(self):
        assert normalize_financial_value("") == 0.0
    
    def test_whitespace_only(self):
        assert normalize_financial_value("   ") == 0.0


class TestVerifyNumericFactExactMatch:
    """Tests for exact match verification."""
    
    def test_same_integers(self):
        assert verify_numeric_fact(10500000000, 10500000000) == "exact_match"
    
    def test_different_formats_same_value(self):
        assert verify_numeric_fact("$10.5B", "10,500 million") == "exact_match"
    
    def test_mixed_string_and_int(self):
        assert verify_numeric_fact("$10.5B", 10500000000) == "exact_match"
    
    def test_zero_comparison_dashes(self):
        assert verify_numeric_fact("-", "0") == "exact_match"
    
    def test_zero_comparison_em_dash_and_dollar(self):
        assert verify_numeric_fact("—", "$0") == "exact_match"


class TestVerifyNumericFactApproximateMatch:
    """Tests for approximate match verification."""
    
    def test_half_percent_difference(self):
        # 10,550,000,000 is ~0.48% more than 10,500,000,000
        assert verify_numeric_fact(10500000000, 10550000000) == "approximate_match"
    
    def test_exactly_one_percent(self):
        # 10,100,000,000 is exactly 1% more than 10,000,000,000
        assert verify_numeric_fact(10000000000, 10100000000) == "approximate_match"


class TestVerifyNumericFactMismatch:
    """Tests for mismatch verification."""
    
    def test_one_point_five_percent_difference(self):
        # 10,150,000,000 is 1.5% more than 10,000,000,000
        assert verify_numeric_fact(10000000000, 10150000000) == "mismatch"
    
    def test_large_difference(self):
        assert verify_numeric_fact("$10B", "$11B") == "mismatch"


class TestVerifyNumericFactScaleAware:
    """Tests for scale-aware verification (CRITICAL)."""
    
    def test_table_cell_vs_extracted(self):
        assert verify_numeric_fact(
            "14514000000", "14,514", source_scale="millions"
        ) == "exact_match"
    
    def test_both_have_same_scale(self):
        assert verify_numeric_fact(
            "14,514", "14,514", claim_scale="millions", source_scale="millions"
        ) == "exact_match"
    
    def test_scale_mismatch_detection(self):
        # 14,514 billions vs 14,514 millions is a 1000x difference
        assert verify_numeric_fact(
            "14,514", "14,514", claim_scale="billions", source_scale="millions"
        ) == "mismatch"


class TestVerifyNumericFactCustomTolerance:
    """Tests for custom tolerance values."""
    
    def test_strict_tolerance(self):
        # 0.5% difference should fail with 0.1% tolerance
        assert verify_numeric_fact(10000, 10050, tolerance=0.001) == "mismatch"
    
    def test_lenient_tolerance(self):
        # 5% difference should pass with 10% tolerance
        assert verify_numeric_fact(10000, 10500, tolerance=0.10) == "approximate_match"

