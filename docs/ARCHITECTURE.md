# Architecture Specification

## Overview

This system extracts verified financial facts from SEC filings and produces structured output. The LLM's role is "narrator over a verified fact table" — it cannot generate facts, only narrate facts that have been extracted and verified.

## Pipeline Flow

```
USER QUERY
    │
    ▼
┌─────────────────────────────────────┐
│ 1. ENTITY RESOLUTION                │
│ Map "NVIDIA" → {ticker: NVDA,       │
│ cik: 0001045810, fiscal_ye: Jan}    │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ 2. TIERED RETRIEVAL                 │
│ Filings: SEC EDGAR direct           │
│ News: Google PSE with date filters  │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ 3. DOCUMENT INGESTION               │
│ HTML from EDGAR                     │
│ Chunk by section (Item 7, etc.)     │
│ Preserve table structure            │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ 4. FACT EXTRACTION                  │
│ LLM extracts structured facts       │
│ Each fact has location pointer      │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ 5. FACT VERIFICATION (GATE)         │
│ Numeric: normalization + compare    │
│ Text: whitespace-normalized match   │
│ Failed facts do NOT enter store     │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ 6. VERIFIED FACT STORE              │
│ Only verified facts exist here      │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ 7. REPORT GENERATION                │
│ Narrates fact store only            │
│ Facts section + Thesis section      │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ 8. STRUCTURED OUTPUT                │
│ JSON primary, text secondary        │
└─────────────────────────────────────┘
```

## Data Structures

### Fact Model

```python
class Location(BaseModel):
    cik: str
    doc_date: str
    doc_type: str  # 10-K, 10-Q, 8-K
    section_id: str  # Item7, Item1A, etc.
    paragraph_index: int
    sentence_string: str

class FactContext(BaseModel):
    yoy_change: str | None = None
    vs_guidance: str | None = None

class Fact(BaseModel):
    fact_id: str
    entity: str  # ticker
    metric: str
    value: float | None
    unit: str
    period: str
    period_end_date: str
    location: Location
    source_format: str  # html_table, html_text
    doc_hash: str
    snapshot_id: str
    verification_status: str  # exact_match, approximate_match, mismatch, unverified
    negative_evidence: str | None = None
    context: FactContext | None = None
```

### Output Schema

```python
class ResearchOutput(BaseModel):
    query: str
    as_of_date: str | None
    facts: list[Fact]
    analysis: Analysis | None
    conflicts: list[Conflict]
    not_found: list[NotFoundMetric]

class Analysis(BaseModel):
    summary: str
    classification: Literal["thesis"]
    supporting_facts: list[str]  # fact_ids

class Conflict(BaseModel):
    metric: str
    values: list[ConflictingValue]

class ConflictingValue(BaseModel):
    value: float
    source: str

class NotFoundMetric(BaseModel):
    metric: str
    status: str  # "Not found in retrieved Tier 1/2 sources"
```

## Source Tiering

### Domain Tier
- **Tier 1:** sec.gov, bloomberg.com, reuters.com, wsj.com (news not opinion), company IR pages
- **Tier 2:** Industry publications, analyst reports, academic papers, federalreserve.gov
- **Tier 3:** General news, blogs, SEO content

### Document-Type Tier
- **Highest:** SEC filings (10-K, 10-Q, 8-K), XBRL, earnings releases, transcripts, press releases
- **High:** Central bank releases, regulatory body reports
- **Medium:** Straight news from Tier 1 domains
- **Lower:** Opinion, analysis, newsletters

**Rule:** Fact-mode claims only accept evidence from top two tiers of both dimensions.

## Numeric Verification

Uses unit normalization, not string matching or embeddings.

Must handle as equivalent:
- "$10.5B"
- "$10,500M"
- "$10,500 million"
- "10.5 billion dollars"
- "$10,542MM"
- "10542000000"

**Special case:** Accounting dashes ("-", "—", "–") represent zero.

**Tolerance:** 1% for rounding errors. >1% flagged for review.

## Text Verification

Uses whitespace-normalized string containment, not exact string match.

```python
def normalize_for_comparison(text: str) -> str:
    text = text.lower()
    text = text.replace('\xa0', ' ')  # &nbsp;
    text = text.replace('\n', ' ')
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Correct: normalized_sentence in normalized_source
# Wrong: raw_sentence in raw_source
```

## Location Storage

**Text Facts:**
- `paragraph_index`: Which paragraph in the section
- `sentence_string`: Exact quote from source

**Table Facts:**
- `table_index`: Which table in the section
- `row_index`, `column_index`: Cell coordinates
- `row_label`, `column_label`: Human-readable labels

Do not use character offsets (brittle).

## Table Scale Extraction

Financial tables define units in headers, not cells:
```
                (in millions, except per share data)
Revenue              14,514
```

The cell shows `14,514` but actual value is `$14,514,000,000`.

**Critical:** Extract scale from table header ("millions", "thousands") and apply during verification.

## Verification Bifurcation

**Text Facts (`source_format == "html_text"`):**
1. Normalize whitespace in both sentence and source
2. Check if normalized sentence exists in normalized source
3. Extract number from sentence, compare to fact.value

**Table Facts (`source_format == "html_table"`):**
1. Check if (row_index, column_index) exists in table
2. Extract cell value at coordinates
3. Apply table.scale to cell value
4. Compare to fact.value (which should already be in base units)

## Unknown Handling

1. **No evidence found:** Return null, render "Not found in retrieved Tier 1/2 sources"
2. **Explicit non-disclosure:** Extract the statement as negative_evidence
3. **Conflicting sources:** Surface both values in conflicts array, do not pick one
4. **Unknown fiscal year:** Explicitly fail with error, do not silently default to calendar year

## Out of Scope for Demo

- XBRL parsing (high implementation risk)
- Consensus estimates (requires Bloomberg/Refinitiv/FactSet)
- PDF parsing (use HTML from EDGAR)
