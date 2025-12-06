"""
Offline Verification Benchmark - Test claim verification without running full pipeline.

This script allows rapid iteration on verification quality by testing against
known reports and sources. Run this to evaluate verification improvements.

Usage:
    cd open_deep_research
    python tests/test_verification_benchmark.py

Requirements:
    - OPENAI_API_KEY environment variable set
    - Virtual environment activated
"""

import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from langchain_core.runnables import RunnableConfig


# ============================================================================
# TEST CASES - Add your own test cases here
# ============================================================================

TEST_CASES = [
    {
        "name": "AI Market Statistics",
        "description": "Tests verification of market statistics claims",
        "report": """
# AI Market Analysis 2024

The global AI market reached $150 billion in 2024, representing significant growth.
OpenAI's ChatGPT has over 100 million weekly active users as of early 2024.
Meta released Llama 3.1 405B in July 2024, making it the largest open-source model.
The AI chip market is projected to reach $300 billion by 2030.
""",
        "sources": [
            {
                "url": "https://example.com/ai-market-size",
                "title": "AI Market Size Report 2024",
                "content": "The global artificial intelligence market size was valued at $150.2 billion in 2024. This represents a compound annual growth rate of 36.6% from the previous year. Key drivers include generative AI adoption and enterprise automation.",
                "query": "ai market size 2024",
                "timestamp": "2024-12-06T12:00:00"
            },
            {
                "url": "https://example.com/chatgpt-users",
                "title": "ChatGPT User Statistics 2024",
                "content": "OpenAI's ChatGPT platform reached 100 million weekly active users in early 2024, making it one of the fastest-growing consumer applications in history. The platform processes over 10 billion queries per month.",
                "query": "chatgpt users statistics",
                "timestamp": "2024-12-06T12:00:00"
            },
            {
                "url": "https://example.com/llama-release",
                "title": "Meta Releases Llama 3.1",
                "content": "Meta announced the release of Llama 3.1 in July 2024. The 405B parameter version is the largest openly available language model, outperforming many closed-source alternatives on standard benchmarks.",
                "query": "meta llama 3.1 release",
                "timestamp": "2024-12-06T12:00:00"
            }
        ],
        "expected_results": {
            "min_supported": 2,  # At least 2 claims should be supported
            "max_unsupported": 1,  # At most 1 claim should be unsupported
            "notes": "The $300B projection claim has no matching source, should be flagged"
        }
    },
    {
        "name": "Hallucinated Statistics",
        "description": "Tests detection of claims without source support",
        "report": """
# Company Revenue Report

Apple's revenue reached $500 billion in Q4 2024.
Tesla delivered 5 million vehicles in 2024.
Microsoft Azure holds 45% of cloud market share.
""",
        "sources": [
            {
                "url": "https://example.com/apple-revenue",
                "title": "Apple Q4 2024 Earnings",
                "content": "Apple reported quarterly revenue of $89 billion in Q4 2024, down slightly from the previous year. iPhone sales remained strong while Mac and iPad showed modest growth.",
                "query": "apple revenue q4 2024",
                "timestamp": "2024-12-06T12:00:00"
            },
            {
                "url": "https://example.com/cloud-market",
                "title": "Cloud Market Share 2024",
                "content": "AWS continues to lead the cloud market with 32% share, followed by Microsoft Azure at 23% and Google Cloud at 10%. The total cloud infrastructure market reached $250 billion.",
                "query": "cloud market share",
                "timestamp": "2024-12-06T12:00:00"
            }
        ],
        "expected_results": {
            "min_supported": 0,
            "max_unsupported": 3,  # All claims should fail verification
            "notes": "Apple revenue is wrong ($500B vs $89B), Tesla has no source, Azure share is wrong (45% vs 23%)"
        }
    },
    {
        "name": "Paraphrased Claims",
        "description": "Tests embedding matching for semantically similar but differently worded claims",
        "report": """
# Technology Trends

The artificial intelligence sector was valued at roughly one hundred fifty billion dollars last year.
OpenAI's conversational AI assistant has approximately 100M users every week.
""",
        "sources": [
            {
                "url": "https://example.com/ai-market",
                "title": "AI Market Size",
                "content": "The global AI market reached $150 billion in 2024.",
                "query": "ai market",
                "timestamp": "2024-12-06T12:00:00"
            },
            {
                "url": "https://example.com/chatgpt",
                "title": "ChatGPT Stats",
                "content": "ChatGPT has over 100 million weekly active users.",
                "query": "chatgpt users",
                "timestamp": "2024-12-06T12:00:00"
            }
        ],
        "expected_results": {
            "min_supported": 2,  # Both should match despite paraphrasing
            "max_unsupported": 0,
            "notes": "Tests embedding-based matching for paraphrased claims"
        }
    }
]


# ============================================================================
# BENCHMARK RUNNER
# ============================================================================

async def run_verification(report: str, sources: List[Dict], config: RunnableConfig) -> Dict[str, Any]:
    """Run verification on a single test case."""
    from open_deep_research.verification import verify_report
    return await verify_report(report, sources, config)


async def run_benchmark(test_cases: List[Dict] = None, verbose: bool = True) -> Dict[str, Any]:
    """Run the full benchmark suite."""
    
    if test_cases is None:
        test_cases = TEST_CASES
    
    # Configure
    config = RunnableConfig(configurable={
        "research_model": "openai:gpt-4o-mini",  # Use cheaper model for testing
        "max_claims_to_verify": 10,
        "verification_confidence_threshold": 0.8
    })
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "total_cases": len(test_cases),
        "passed": 0,
        "failed": 0,
        "cases": []
    }
    
    print("\n" + "=" * 70)
    print("VERIFICATION BENCHMARK")
    print("=" * 70)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n[{i}/{len(test_cases)}] Running: {test_case['name']}")
        print(f"    Description: {test_case['description']}")
        
        try:
            # Run verification
            result = await run_verification(
                test_case["report"],
                test_case["sources"],
                config
            )
            
            summary = result["summary"]
            expected = test_case["expected_results"]
            
            # Check expectations
            passed = True
            failures = []
            
            if summary["supported"] < expected["min_supported"]:
                passed = False
                failures.append(f"Supported: {summary['supported']} < expected min {expected['min_supported']}")
            
            if summary["unsupported"] > expected["max_unsupported"]:
                passed = False
                failures.append(f"Unsupported: {summary['unsupported']} > expected max {expected['max_unsupported']}")
            
            # Record result
            case_result = {
                "name": test_case["name"],
                "passed": passed,
                "summary": {
                    "total_claims": summary["total_claims"],
                    "supported": summary["supported"],
                    "partially_supported": summary["partially_supported"],
                    "unsupported": summary["unsupported"],
                    "uncertain": summary["uncertain"],
                    "overall_confidence": summary["overall_confidence"]
                },
                "failures": failures,
                "claims": result["claims"]
            }
            results["cases"].append(case_result)
            
            if passed:
                results["passed"] += 1
                print(f"    ✅ PASSED")
            else:
                results["failed"] += 1
                print(f"    ❌ FAILED")
                for f in failures:
                    print(f"       - {f}")
            
            if verbose:
                print(f"    Results: {summary['supported']}/{summary['total_claims']} supported, "
                      f"confidence: {summary['overall_confidence']:.0%}")
                if summary.get("warnings"):
                    print(f"    Warnings: {len(summary['warnings'])}")
                    for w in summary["warnings"][:3]:
                        print(f"       ⚠ {w[:70]}...")
                        
        except Exception as e:
            results["failed"] += 1
            results["cases"].append({
                "name": test_case["name"],
                "passed": False,
                "error": str(e)
            })
            print(f"    ❌ ERROR: {e}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"Total: {results['total_cases']} | Passed: {results['passed']} | Failed: {results['failed']}")
    print(f"Pass Rate: {results['passed'] / results['total_cases'] * 100:.0f}%")
    
    return results


async def save_results(results: Dict[str, Any], output_dir: str = "benchmark_results"):
    """Save benchmark results to JSON file."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    filename = f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    filepath = output_path / filename
    
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {filepath}")
    return filepath


# ============================================================================
# MAIN
# ============================================================================

async def main():
    """Run benchmark and save results."""
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY environment variable not set")
        print("Run: export OPENAI_API_KEY=your-key-here")
        sys.exit(1)
    
    # Run benchmark
    results = await run_benchmark(verbose=True)
    
    # Save results
    await save_results(results)
    
    # Exit with appropriate code
    sys.exit(0 if results["failed"] == 0 else 1)


if __name__ == "__main__":
    asyncio.run(main())

