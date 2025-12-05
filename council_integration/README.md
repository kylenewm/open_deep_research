# LLM Council Integration for Open Deep Research

## Quick Start

### 1. Add the council module

Copy `council.py` to your repo:
```
src/open_deep_research/council.py
```

### 2. Update configuration.py

Add these fields to your `Configuration` class:

```python
from dataclasses import field
from typing import List

@dataclass(kw_only=True)
class Configuration:
    # ... existing fields ...
    
    # Council settings
    use_council: bool = True
    council_models: List[str] = field(default_factory=lambda: [
        "anthropic:claude-sonnet-4-20250514",
        "openai:gpt-4.1",
    ])
    council_min_consensus: float = 0.7
    council_max_revisions: int = 3
```

### 3. Update graph.py

Replace the `human_feedback` function:

```python
from open_deep_research.council import council_feedback

async def human_feedback(state, config):
    """Automated feedback using LLM council."""
    from langgraph.types import Command
    from langgraph.constants import Send
    from open_deep_research.configuration import Configuration
    
    configurable = Configuration.from_runnable_config(config)
    use_council = getattr(configurable, 'use_council', True)
    
    topic = state["topic"]
    sections = state["sections"]
    
    if use_council:
        from open_deep_research.council import council_feedback
        return await council_feedback(state, config)
    else:
        # Auto-approve fallback
        return Command(goto=[
            Send("build_section_with_web_research", {
                "topic": topic, "section": s, "search_iterations": 0
            })
            for s in sections if s.research
        ])
```

### 4. Set up API keys

Make sure you have API keys for the council models in your `.env`:

```bash
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

### 5. Run it

```bash
uvx --refresh --from "langgraph-cli[inmem]" --with-editable . --python 3.11 langgraph dev --allow-blocking
```

---

## How It Works

```
                     ┌─────────────────────────┐
                     │  generate_report_plan   │
                     └───────────┬─────────────┘
                                 │
                                 ▼
                     ┌─────────────────────────┐
                     │    council_feedback     │
                     │                         │
                     │  ┌───────┐  ┌───────┐  │
                     │  │Claude │  │GPT-4.1│  │  ← Parallel
                     │  └───┬───┘  └───┬───┘  │
                     │      └─────┬────┘      │
                     │            ▼           │
                     │     [Vote Counting]    │
                     └───────────┬────────────┘
                                 │
             ┌───────────────────┼───────────────────┐
             │                   │                   │
          APPROVE             REVISE              REJECT
          (≥70%)           (otherwise)         (unanimous)
             │                   │                   │
             ▼                   ▼                   ▼
     build_section...    generate_report_plan    END
                         (with feedback)
```

---

## Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `use_council` | `True` | Enable/disable council |
| `council_models` | `[claude-sonnet, gpt-4.1]` | Models to vote |
| `council_min_consensus` | `0.7` | Required agreement (70%) |
| `council_max_revisions` | `3` | Max revision attempts |

---

## Cost Estimate

Per research plan review:
- 2 models × ~800 tokens = ~1,600 tokens
- Cost: ~$0.01-0.02

With 3 revision rounds max: **~$0.05 per task**

---

## Files

```
src/open_deep_research/
├── council.py           ← ADD THIS (new file)
├── configuration.py     ← MODIFY (add council fields)
├── graph.py            ← MODIFY (replace human_feedback)
├── state.py            ← OPTIONAL (add tracking fields)
└── ... (other files unchanged)
```

---

# Future: Debate Protocol

Once the council is working, you can upgrade to a full debate protocol where models:

1. **Propose** - Each model creates their own solution
2. **Critique** - Each model reviews others' proposals
3. **Defend** - Models respond to critiques, may revise
4. **Judge** - A judge model picks winner or synthesizes best parts

### When to Upgrade to Debate

| Scenario | Use Council | Use Debate |
|----------|-------------|------------|
| Simple validation | ✓ | |
| Speed-critical | ✓ | |
| Budget-constrained | ✓ | |
| Complex research plans | | ✓ |
| High-stakes decisions | | ✓ |
| Need detailed reasoning | | ✓ |

### Debate Implementation (Future)

The debate protocol adds these rounds:

```python
# Round 1: Independent proposals
proposals = await gather([get_proposal(m) for m in models])

# Round 2: Cross-critique
critiques = await gather([
    critique(critic, target) 
    for critic in models 
    for target in models 
    if critic != target
])

# Round 3: Defend and revise
rebuttals = await gather([defend(m, critiques_for_m) for m in models])

# Round 4: Judge decides
decision = await judge(proposals, critiques, rebuttals)
```

### Debate Configuration (Future)

```python
@dataclass
class DebateConfig:
    debater_models: List[str]  # Models that argue
    judge_model: str           # Model that decides
    num_rounds: int = 1        # Critique/rebuttal rounds
    judge_can_synthesize: bool = True  # Can combine best parts
```

### Files for Debate (Future)

When ready to add debate, you'll need:
- `debate.py` - The debate protocol implementation
- Update `configuration.py` with debate settings
- Update `graph.py` to use `debate_feedback` instead of `council_feedback`

---

## Troubleshooting

### "Model not found" error
Make sure you have API keys set for all council models.

### Council always approves
Lower `council_min_consensus` to be stricter (e.g., 0.8).

### Council always rejects  
Raise `council_min_consensus` to be more lenient (e.g., 0.6).

### Too slow
Remove one model from `council_models` to speed up.

### Too expensive
Use cheaper models:
```python
council_models: List[str] = field(default_factory=lambda: [
    "anthropic:claude-haiku-3-5-20241022",
    "openai:gpt-4.1-mini",
])
```

---

## Testing

Quick test without running full system:

```python
# test_council.py
import asyncio
from open_deep_research.council import council_vote_on_plan, CouncilConfig

class MockSection:
    def __init__(self, name, desc, research):
        self.name = name
        self.description = desc
        self.research = research

async def test():
    sections = [
        MockSection("Intro", "Overview", False),
        MockSection("Analysis", "Deep dive", True),
        MockSection("Conclusion", "Summary", False),
    ]
    
    verdict = await council_vote_on_plan(
        topic="Test topic",
        sections=sections,
        config=CouncilConfig()
    )
    
    print(f"Decision: {verdict.decision}")
    print(f"Consensus: {verdict.consensus_score:.0%}")

asyncio.run(test())
```

---

## Summary

1. **Copy `council.py`** to `src/open_deep_research/`
2. **Add 4 fields** to `Configuration` class
3. **Replace `human_feedback`** function in `graph.py`
4. **Run and test**

That's it! Human-in-the-loop is now replaced with automated LLM council verification.
