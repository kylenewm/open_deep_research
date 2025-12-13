# State Migration Guide

## The Problem

Your existing LangGraph pipeline passes **strings** between nodes:

```python
# OLD State Schema
class ResearchState(TypedDict):
    task: str
    messages: list[BaseMessage]
    reports: list[str]  # Strings - can't hold structured facts
```

The new components expect **objects** (`FactStore`, `DocumentSnapshot`, `Fact`).

If you try to inject new components without updating the State, old nodes will crash.

---

## The Solution: State Schema Migration

Before Step 6 (Integration), update your LangGraph State definition.

### New State Schema

```python
# In your existing deep_researcher.py (or state.py if separate)

from typing import TypedDict
from langchain_core.messages import BaseMessage

# Import new components
from src.store import FactStore
from src.models import DocumentSnapshot, EntityInfo

class ResearchState(TypedDict):
    # ========================================
    # LEGACY FIELDS (Keep for backward compatibility)
    # ========================================
    task: str
    messages: list[BaseMessage]
    
    # ========================================
    # NEW AUDITOR FIELDS
    # ========================================
    
    # The single source of truth - only verified facts
    fact_store: FactStore
    
    # Raw document snapshots (immutable evidence)
    snapshots: list[DocumentSnapshot]
    
    # Resolved entity info (ticker, CIK, fiscal year)
    entity: EntityInfo | None
    
    # Query metadata
    query: str
    as_of_date: str | None  # For time-machine mode
```

---

## The Strangle Pattern

Replace old nodes one at a time while keeping the pipeline running.

### Node 1: Search/Retrieval

**Old behavior:**
```python
def search_node(state: ResearchState) -> ResearchState:
    results = tavily.search(state["task"])
    state["messages"].append(HumanMessage(content=str(results)))
    return state
```

**New behavior:**
```python
from src.entities import resolve_entity
from src.ingestion import get_filing_html, create_document_snapshot

def search_node(state: ResearchState) -> ResearchState:
    # 1. Resolve entity
    entity = resolve_entity(state["task"])
    if entity is None:
        # Fallback to old behavior for unknown entities
        results = tavily.search(state["task"])
        state["messages"].append(HumanMessage(content=str(results)))
        return state
    
    # 2. Get SEC filing directly (Tier 1 source)
    html = get_filing_html(entity.cik, "10-Q")
    snapshot = create_document_snapshot(html, f"https://sec.gov/...", entity.cik, "10-Q")
    
    # 3. Update new state fields
    state["entity"] = entity
    state["snapshots"].append(snapshot)
    
    return state
```

### Node 2: Extraction

**Old behavior:**
```python
def extract_node(state: ResearchState) -> ResearchState:
    # LLM summarizes - can hallucinate
    summary = llm.invoke(f"Summarize: {state['messages']}")
    state["reports"].append(summary.content)
    return state
```

**New behavior:**
```python
from src.parsing import parse_filing_html
from src.extraction import extract_facts_from_section
from src.pipeline import process_extracted_facts

def extract_node(state: ResearchState) -> ResearchState:
    if not state["snapshots"]:
        return state  # Nothing to extract
    
    snapshot = state["snapshots"][-1]
    entity = state["entity"]
    
    # 1. Parse HTML into sections
    parsed = parse_filing_html(snapshot.raw_html)
    
    # 2. Extract facts from relevant sections (Item 7 = MD&A)
    for section in parsed.sections:
        if section.section_id in ("Item7", "Item8"):
            raw_facts = extract_facts_from_section(
                section, 
                entity.ticker, 
                snapshot
            )
            
            # 3. Verification GATE (not post-hoc)
            verified, rejected = process_extracted_facts(
                raw_facts, 
                section.raw_html
            )
            
            # 4. Only verified facts enter the store
            for fact in verified:
                state["fact_store"].add_fact(fact)
    
    return state
```

### Node 3: Report Generation

**Old behavior:**
```python
def report_node(state: ResearchState) -> ResearchState:
    # LLM writes freely - can hallucinate citations
    report = llm.invoke(f"Write report based on: {state['reports']}")
    return {"final_report": report.content}
```

**New behavior:**
```python
from src.report import generate_full_report

def report_node(state: ResearchState) -> ResearchState:
    # Report can ONLY narrate facts in the store
    # No free-form generation allowed
    report = generate_full_report(
        state["fact_store"], 
        state["query"]
    )
    return {"final_report": report}
```

---

## Migration Sequence

### Phase 1: Build Foundation (Steps 1-5)
- Build in `src/` folder
- Test in isolation
- **Do not touch existing code yet**

### Phase 2: State Migration (Before Step 6)
1. Update State schema (add new fields)
2. Initialize new fields in graph entry point:
   ```python
   initial_state = {
       # Legacy
       "task": user_query,
       "messages": [],
       
       # New
       "fact_store": FactStore(),
       "snapshots": [],
       "entity": None,
       "query": user_query,
       "as_of_date": None,
   }
   ```

### Phase 3: Node Replacement (Steps 6-10)
1. Replace search node → Uses `src/ingestion`
2. Replace extraction node → Uses `src/extraction` + `src/pipeline`
3. Replace report node → Uses `src/report`

### Phase 4: Cleanup
1. Remove old `verification.py` (replaced by `src/pipeline.py`)
2. Remove or repurpose `council.py`
3. Remove legacy state fields once stable

---

## Testing During Migration

After each node replacement, verify:

```bash
# Run existing tests (should still pass)
pytest tests/

# Run new component tests
pytest tests/test_verification.py -v
pytest tests/test_extraction.py -v

# Run integration test
python -c "
from deep_researcher import run_research
result = run_research('What was NVIDIA datacenter revenue in Q3 2024?')
print(result)
"
```

---

## Rollback Plan

If integration breaks, you can rollback by:

1. Reverting State schema changes
2. Keeping old node implementations commented out
3. New `src/` folder doesn't affect old code until you import it

---

## Files to Modify

| File | Change |
|------|--------|
| `deep_researcher.py` | Update State schema, replace nodes |
| `utils.py` | Add SEC EDGAR alongside Tavily |
| `verification.py` | Delete after Step 7 (replaced by `src/pipeline.py`) |
| `council.py` | Remove or repurpose |

---

## Critical Rule

**Do not modify `deep_researcher.py` until `tests/test_extraction.py` passes.**

Keep the old demo working until the new engine is ready to swap in.
