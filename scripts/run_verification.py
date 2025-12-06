#!/usr/bin/env python3
"""
Run Verification Layer Standalone

This script allows running the verification layer on saved reports and sources,
without needing to run the full research pipeline. Useful for:
- Testing verification improvements quickly
- Re-verifying old runs
- Static evaluation benchmarks

Usage:
    # From a LangGraph thread (gets report from LangSmith, sources from store)
    python scripts/run_verification.py --thread-id d58fd825-... --project my-project
    
    # Or set project via environment variable
    export LANGSMITH_PROJECT=my-project
    python scripts/run_verification.py --thread-id d58fd825-...
    
    # From thread with manual sources (if store was cleared)
    python scripts/run_verification.py --thread-id <id> --project my-project --sources sources.json

    # From saved JSON files
    python scripts/run_verification.py --report report.json --sources sources.json
    
    # From a saved run directory
    python scripts/run_verification.py --run-dir runs/2024-12-06_123456/

Requirements:
    - OPENAI_API_KEY environment variable set
    - LANGSMITH_API_KEY environment variable set (for --thread-id)
    - Virtual environment activated
"""

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Load .env file
from dotenv import load_dotenv
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
    print(f"Loaded environment from {env_path}")


async def load_from_thread(thread_id: str, sources_file: str = None, project_name: str = None) -> tuple:
    """Load report from LangSmith and sources from store or file.
    
    Args:
        thread_id: LangGraph thread ID
        sources_file: Optional path to sources JSON (fallback if store empty)
        project_name: LangSmith project name (defaults to LANGSMITH_PROJECT env var)
    
    Returns:
        Tuple of (report, sources)
    """
    from langsmith import Client
    
    # Get LangSmith API key
    api_key = os.getenv("LANGSMITH_API_KEY")
    if not api_key:
        raise ValueError("LANGSMITH_API_KEY environment variable not set")
    
    # Get project name from arg or environment
    project = project_name or os.getenv("LANGSMITH_PROJECT") or os.getenv("LANGCHAIN_PROJECT")
    if not project:
        raise ValueError(
            "Project name required. Set LANGSMITH_PROJECT env var or use --project flag"
        )
    
    client = Client(api_key=api_key)
    
    print(f"\n{'='*70}")
    print(f"LOADING FROM THREAD: {thread_id}")
    print(f"PROJECT: {project}")
    print(f"{'='*70}")
    
    # Step 1: Get final_report from LangSmith
    print("Retrieving final_report from LangSmith...")
    report = None
    
    try:
        # Try to find runs - first attempt with trace_id (thread_id often equals trace_id)
        runs = []
        try:
            runs = list(client.list_runs(
                project_name=project,
                trace_id=thread_id,
                is_root=True,
                limit=10
            ))
            if runs:
                print(f"  Found {len(runs)} runs via trace_id")
        except Exception:
            pass  # trace_id might not match, fall through
        
        if not runs:
            # Search project and filter by metadata in Python
            print("  Searching project for runs with matching thread_id...")
            all_runs = list(client.list_runs(project_name=project, is_root=True, limit=100))
            runs = [
                r for r in all_runs 
                if r.extra and r.extra.get("metadata", {}).get("thread_id") == thread_id
            ]
            if runs:
                print(f"  Found {len(runs)} runs with matching thread_id in metadata")
        
        if not runs:
            # Last resort: just get runs with final_report output
            print("  No thread match, searching for any runs with final_report...")
            runs = list(client.list_runs(project_name=project, is_root=True, limit=50))
            runs = [r for r in runs if r.outputs and r.outputs.get("final_report")]
        
        # Find run with final_report
        for run in runs:
            if run.outputs and run.outputs.get("final_report"):
                report = run.outputs["final_report"]
                print(f"  ✓ Found report ({len(report)} chars) from run: {run.id}")
                break
        
        if not report:
            raise ValueError(f"No final_report found for thread {thread_id}")
            
    except Exception as e:
        print(f"  ✗ LangSmith error: {e}")
        raise ValueError(f"Could not retrieve report from LangSmith: {e}")
    
    # Step 2: Try to get sources from LangGraph store
    print("\nRetrieving sources from LangGraph store...")
    sources = []
    
    try:
        # Try to connect to the running LangGraph server's store
        import aiohttp
        
        async with aiohttp.ClientSession() as session:
            # Try to get sources via API (if server is running with store endpoint)
            store_url = f"http://127.0.0.1:2024/store/verification/verification_sources_{thread_id}"
            async with session.get(store_url) as response:
                if response.status == 200:
                    data = await response.json()
                    sources = data.get("value", []) if isinstance(data, dict) else data
                    print(f"  ✓ Found {len(sources)} sources in store")
    except Exception as e:
        print(f"  ✗ Store not accessible: {e}")
    
    # Step 3: Fall back to sources file if provided
    if not sources and sources_file:
        print(f"\nLoading sources from file: {sources_file}")
        with open(sources_file, 'r') as f:
            sources = json.load(f)
        print(f"  ✓ Loaded {len(sources)} sources from file")
    
    # Step 4: Error if no sources found
    if not sources:
        print("\n" + "="*70)
        print("⚠️  NO SOURCES FOUND")
        print("="*70)
        print("Sources are required for verification.")
        print("\nOptions:")
        print("  1. If LangGraph server is still running, sources should be in store")
        print("  2. Provide sources manually: --sources sources.json")
        print("  3. Re-run research with use_claim_verification=True")
        print("="*70 + "\n")
        raise ValueError("No sources available for verification")
    
    return report, sources


async def load_from_files(report_file: str, sources_file: str) -> tuple:
    """Load report and sources from JSON files."""
    with open(report_file, 'r') as f:
        report_data = json.load(f)
    
    with open(sources_file, 'r') as f:
        sources = json.load(f)
    
    # Handle different report formats
    if isinstance(report_data, dict):
        report = report_data.get("final_report") or report_data.get("report") or report_data.get("content", "")
    else:
        report = str(report_data)
    
    return report, sources


async def load_from_run_dir(run_dir: str) -> tuple:
    """Load report and sources from a saved run directory."""
    run_path = Path(run_dir)
    
    report_file = run_path / "report.json"
    sources_file = run_path / "sources.json"
    
    if not report_file.exists():
        raise FileNotFoundError(f"Report file not found: {report_file}")
    if not sources_file.exists():
        raise FileNotFoundError(f"Sources file not found: {sources_file}")
    
    return await load_from_files(str(report_file), str(sources_file))


async def save_run(output_dir: str, report: str, sources: list, result: dict = None, thread_id: str = None):
    """Save a run's data for later replay."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save report
    with open(output_path / "report.json", 'w') as f:
        json.dump({
            "final_report": report, 
            "saved_at": datetime.now().isoformat(),
            "thread_id": thread_id
        }, f, indent=2)
    
    # Save sources
    with open(output_path / "sources.json", 'w') as f:
        json.dump(sources, f, indent=2, default=str)
    
    # Save verification result if provided
    if result:
        with open(output_path / "verification_result.json", 'w') as f:
            json.dump(result, f, indent=2, default=str)
    
    print(f"Run saved to: {output_path}")
    return output_path


async def run_verification(report: str, sources: list, model: str = "openai:gpt-4o-mini") -> dict:
    """Run verification on a report and sources."""
    from langchain_core.runnables import RunnableConfig
    from open_deep_research.verification import verify_report
    
    config = RunnableConfig(configurable={
        "research_model": model,
        "max_claims_to_verify": 25,
        "verification_confidence_threshold": 0.8
    })
    
    print(f"\n{'='*70}")
    print("RUNNING VERIFICATION")
    print(f"{'='*70}")
    print(f"Report length: {len(report)} chars")
    print(f"Sources: {len(sources)}")
    print(f"Model: {model}")
    print(f"{'='*70}\n")
    
    result = await verify_report(report, sources, config)
    
    return result


def print_result(result: dict):
    """Print verification result in a readable format."""
    summary = result["summary"]
    
    print(f"\n{'='*70}")
    print("VERIFICATION RESULTS")
    print(f"{'='*70}")
    print(f"Total Claims: {summary['total_claims']}")
    print(f"Supported: {summary['supported']} ({summary['supported']/max(summary['total_claims'],1)*100:.0f}%)")
    print(f"Partially Supported: {summary['partially_supported']}")
    print(f"Unsupported: {summary['unsupported']}")
    print(f"Uncertain: {summary['uncertain']}")
    print(f"Overall Confidence: {summary['overall_confidence']:.0%}")
    
    if summary.get("warnings"):
        print(f"\n⚠️ WARNINGS ({len(summary['warnings'])}):")
        for w in summary["warnings"][:10]:
            print(f"  - {w}")
    
    if summary.get("data_issues"):
        print(f"\n📋 DATA ISSUES ({len(summary['data_issues'])}):")
        for issue in summary["data_issues"][:5]:
            print(f"  - {issue}")
    
    print(f"\n{'='*70}")
    print("PER-CLAIM BREAKDOWN")
    print(f"{'='*70}")
    
    for claim in result["claims"]:
        status_icon = {
            "SUPPORTED": "✅",
            "PARTIALLY_SUPPORTED": "⚠️",
            "UNSUPPORTED": "❌",
            "UNCERTAIN": "❓"
        }.get(claim["status"], "?")
        
        print(f"\n{status_icon} [{claim['claim_id']}] {claim['claim_text'][:70]}...")
        print(f"   Status: {claim['status']} ({claim['confidence']:.0%})")
        print(f"   Source: {claim.get('source_title', 'None')[:50] if claim.get('source_title') else 'No source matched'}")
        if claim.get("evidence_snippet"):
            print(f"   Evidence: {claim['evidence_snippet'][:80]}...")


async def main():
    parser = argparse.ArgumentParser(description='Run verification layer standalone')
    
    # Input options
    parser.add_argument('--thread-id', '-t', help='LangGraph thread ID (retrieves from LangSmith)')
    parser.add_argument('--project', '-p', help='LangSmith project name (or set LANGSMITH_PROJECT env var)')
    parser.add_argument('--report', '-r', help='Path to report JSON file')
    parser.add_argument('--sources', '-s', help='Path to sources JSON file')
    parser.add_argument('--run-dir', '-d', help='Path to saved run directory')
    
    # Model options
    parser.add_argument('--model', '-m', default='openai:gpt-4o-mini', 
                       help='Model to use for verification (default: openai:gpt-4o-mini)')
    
    # Output options
    parser.add_argument('--output', '-o', help='Directory to save results')
    parser.add_argument('--save', action='store_true', help='Save run data for later replay')
    
    # Utility options
    parser.add_argument('--list-runs', action='store_true', help='List saved runs')
    
    args = parser.parse_args()
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY environment variable not set")
        sys.exit(1)
    
    # Handle list-runs
    if args.list_runs:
        runs_dir = Path(__file__).parent.parent / "runs"
        if runs_dir.exists():
            runs = sorted(runs_dir.iterdir(), reverse=True)
            print(f"\nSaved runs in {runs_dir}:")
            for run in runs[:20]:
                if run.is_dir():
                    report_exists = (run / "report.json").exists()
                    sources_exists = (run / "sources.json").exists()
                    print(f"  {run.name} [report: {'✓' if report_exists else '✗'}, sources: {'✓' if sources_exists else '✗'}]")
        else:
            print("No saved runs found.")
        sys.exit(0)
    
    # Load data based on input method
    thread_id = None
    
    if args.thread_id:
        thread_id = args.thread_id
        report, sources = await load_from_thread(args.thread_id, args.sources, args.project)
    elif args.run_dir:
        report, sources = await load_from_run_dir(args.run_dir)
    elif args.report and args.sources:
        report, sources = await load_from_files(args.report, args.sources)
    else:
        parser.print_help()
        print("\nERROR: Must provide one of:")
        print("  --thread-id <id>                    (retrieves from LangSmith)")
        print("  --run-dir <path>                    (loads from saved run)")
        print("  --report <file> --sources <file>    (loads from JSON files)")
        sys.exit(1)
    
    # Run verification
    result = await run_verification(report, sources, args.model)
    
    # Print results
    print_result(result)
    
    # Save if requested (always save for thread-id runs)
    if args.save or args.output or args.thread_id:
        output_dir = args.output or f"runs/{datetime.now().strftime('%Y-%m-%d_%H%M%S')}_{thread_id[:8] if thread_id else 'manual'}"
        await save_run(output_dir, report, sources, result, thread_id)
    
    # Return exit code based on results
    unsupported = result["summary"]["unsupported"] + result["summary"]["uncertain"]
    if unsupported > result["summary"]["total_claims"] / 2:
        sys.exit(1)  # More than half unsupported
    sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())
