# LLM Council Integration

Multi-model verification system for research briefs that replaces human-in-the-loop approval with automated consensus-based validation.

## Overview

The LLM Council validates research briefs before execution by querying multiple models in parallel and requiring consensus for approval. This catches vague, incomplete, or unfeasible research plans early - saving costs on wasted research steps.

### Flow

```
clarify_with_user -> write_research_brief -> validate_brief -> research_supervisor -> final_report
                            ^                     |
                            |_____ (revise) ______|
```

### How It Works

1. **Brief Generated**: User request is transformed into a structured research brief
2. **Council Votes**: GPT-4.1 and GPT-5 evaluate the brief in parallel
3. **Consensus Check**: Weighted votes are tallied based on confidence scores
4. **Routing**:
   - **Approve** (≥70% consensus): Proceed to research
   - **Revise**: Loop back to brief generation with synthesized feedback
   - **Reject** (unanimous) or max revisions: Force proceed with warning

## Configuration

Add these to your `.env` or configure via LangGraph Studio:

| Setting | Default | Description |
|---------|---------|-------------|
| `USE_COUNCIL` | `true` | Enable/disable council validation |
| `COUNCIL_MODELS` | `openai:gpt-4.1,openai:gpt-5` | Comma-separated list of council models |
| `COUNCIL_MIN_CONSENSUS` | `0.7` | Required agreement score (0.0-1.0) |
| `COUNCIL_MAX_REVISIONS` | `3` | Max revision attempts before force-proceed |

### Example Configuration

```python
# In configuration.py or via environment variables
use_council = True
council_models = ["openai:gpt-4.1", "openai:gpt-5"]
council_min_consensus = 0.7  # 70% agreement required
council_max_revisions = 3    # Max 3 revision loops
```

## Observability

### Console Output

The council logs decisions to console:

```
============================================================
COUNCIL DECISION: APPROVE
Consensus Score: 85%
------------------------------------------------------------
  ✓ openai:gpt-4.1: approve (90%)
  ✓ openai:gpt-5: approve (80%)
============================================================
```

### LangSmith Tracing

Council calls are tagged for LangSmith filtering:
- `langsmith:council_vote` - Individual model votes
- `langsmith:council_synthesis` - Feedback synthesis

Filter in LangSmith with: `tags:langsmith:council_vote`

## Cost Estimates

Per research brief validation:
- 2 models × ~1,000 tokens = ~2,000 tokens
- Cost: ~$0.02-0.04

With 3 revision rounds max: **~$0.10-0.15 per task**

This is high-leverage spending - catching a bad brief saves $2-5 of wasted research costs.

## Disabling the Council

To skip council validation entirely:

```bash
# Environment variable
USE_COUNCIL=false

# Or in configuration
use_council = False
```

The system will proceed directly from `write_research_brief` to `research_supervisor`.

## Troubleshooting

### Council always approves
Lower the consensus threshold to be stricter:
```python
council_min_consensus = 0.8  # Require 80% agreement
```

### Council always rejects/revises
Raise the threshold to be more lenient:
```python
council_min_consensus = 0.5  # Only need 50% agreement
```

### Too slow
Remove one model to speed up (still get value from single-model validation):
```python
council_models = ["openai:gpt-4.1"]
```

### API errors
Ensure you have valid API keys for all council models:
```bash
OPENAI_API_KEY=sk-...
```

## Files Modified

| File | Changes |
|------|---------|
| `src/open_deep_research/council.py` | New file - council voting logic |
| `src/open_deep_research/configuration.py` | Added 4 council config fields |
| `src/open_deep_research/state.py` | Added council tracking fields |
| `src/open_deep_research/deep_researcher.py` | Added validate_brief node |

---

## Next Steps (Future Enhancements)

### 1. Council 2: Validate Research Findings

Add a second checkpoint after research, before final report:

```
research_supervisor -> validate_findings -> final_report
```

This catches:
- Hallucinations in sub-agent findings
- Missing information that should trigger more research
- Contradictions between sub-agents

### 2. Debate Protocol

Upgrade from simple voting to full multi-model debate:

1. **Propose**: Each model creates their own solution
2. **Critique**: Each model reviews others' proposals
3. **Defend**: Models respond to critiques, may revise
4. **Judge**: A judge model picks winner or synthesizes best parts

### 3. Model Diversity

Add a third model from a different provider for true diversity:
- Google Gemini 2.0 Flash (fast, cheap)
- DeepSeek (strong reasoning)

### 4. Adaptive Consensus

Dynamically adjust threshold based on:
- Task complexity (higher stakes = stricter threshold)
- Historical success rates
- Cost constraints

### 5. Cost Dashboard

Track cumulative council costs in LangSmith:
- Per-run council cost
- Total council spend
- Cost-per-approval metrics

