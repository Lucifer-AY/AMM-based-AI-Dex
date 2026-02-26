"""Complete project evaluation for the Crypto Trading Agent.

This script evaluates the WHOLE agent pipeline by running real prompts through
`run_trading_agent_sync(...)` and scoring the generated output quality.

Evaluation basis (what is measured):
1) Runtime reliability
   - Query execution success/failure
   - Latency per query
2) Output completeness
   - Recommendation presence (BUY/SELL/HOLD language)
   - Entry strategy mention
   - Exit strategy mention
   - Risk management mention (risk/stop-loss/position size)
   - Numerical grounding (prices/percentages/time horizon)
   - Disclaimer presence
3) Output quality sanity checks
   - Avoids obvious failure text and empty answers
   - Minimum output length threshold
4) Overall score
   - Weighted score per test case, then aggregated project score

It generates:
- JSON report with case-level and project-level metrics
- Markdown report with human-readable summary

Usage:
    python completeevaluation.py
    python completeevaluation.py --queries-file eval_queries.json
    python completeevaluation.py --output-json reports/complete_evaluation_report.json
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from config import settings
from trading_agent import run_trading_agent_sync


DEFAULT_QUERIES = [
    "Should I invest $2000 in Bitcoin for 7 days with medium risk tolerance?",
    "Analyze Ethereum for short-term trading with low risk tolerance and $1500 capital.",
    "Give a trading recommendation for SOL with high risk tolerance for 14 days and $3000.",
    "Should I buy or hold ADA today for a 10-day horizon with medium risk?",
    "Analyze BTC and provide entry, stop-loss, and target levels for a cautious investor.",
]

RECOMMENDATION_PATTERN = re.compile(r"\b(strong\s+buy|buy|hold|sell|strong\s+sell)\b", re.IGNORECASE)
ENTRY_PATTERN = re.compile(r"\b(entry|buy\s+zone|dca|accumulate|entry\s+price)\b", re.IGNORECASE)
EXIT_PATTERN = re.compile(r"\b(exit|target|take\s*profit|tp\b|profit\s+booking)\b", re.IGNORECASE)
RISK_PATTERN = re.compile(r"\b(risk|stop[-\s]?loss|position\s*sizing|drawdown|capital\s*allocation)\b", re.IGNORECASE)
DISCLAIMER_PATTERN = re.compile(r"\b(not\s+financial\s+advice|educational\s+purposes|do\s+your\s+own\s+research|consult\s+with\s+financial)\b", re.IGNORECASE)
NUMBER_PATTERN = re.compile(r"(\$\s?\d[\d,]*(?:\.\d+)?|\d+(?:\.\d+)?\s?%|\b\d+\s?(?:day|days|week|weeks)\b)", re.IGNORECASE)
ERROR_PATTERN = re.compile(r"\b(error|exception|traceback|unable\s+to\s+generate|no\s+recommendation\s+generated)\b", re.IGNORECASE)


@dataclass
class EvalCaseResult:
    query: str
    success: bool
    latency_seconds: float
    output_length: int
    score: float
    checks: Dict[str, bool]
    notes: List[str]
    output_excerpt: str
    error: Optional[str] = None


def load_queries(path: Optional[str]) -> List[str]:
    if not path:
        return DEFAULT_QUERIES

    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Queries file not found: {path}")

    with open(file_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, list) or not all(isinstance(item, str) and item.strip() for item in payload):
        raise ValueError("Queries file must be a JSON array of non-empty strings.")

    return payload


def evaluate_output_text(output: str) -> Dict:
    text = (output or "").strip()

    checks = {
        "has_recommendation": bool(RECOMMENDATION_PATTERN.search(text)),
        "has_entry_strategy": bool(ENTRY_PATTERN.search(text)),
        "has_exit_strategy": bool(EXIT_PATTERN.search(text)),
        "has_risk_management": bool(RISK_PATTERN.search(text)),
        "has_numerical_grounding": bool(NUMBER_PATTERN.search(text)),
        "has_disclaimer": bool(DISCLAIMER_PATTERN.search(text)),
        "no_error_language": not bool(ERROR_PATTERN.search(text)),
        "sufficient_length": len(text) >= 400,
    }

    weights = {
        "has_recommendation": 20,
        "has_entry_strategy": 12,
        "has_exit_strategy": 12,
        "has_risk_management": 15,
        "has_numerical_grounding": 15,
        "has_disclaimer": 8,
        "no_error_language": 10,
        "sufficient_length": 8,
    }

    score = sum(weights[key] for key, passed in checks.items() if passed)

    notes: List[str] = []
    for check_name, passed in checks.items():
        if not passed:
            notes.append(f"Missing or weak: {check_name}")

    return {
        "checks": checks,
        "score": float(score),
        "notes": notes,
    }


def evaluate_case(query: str) -> EvalCaseResult:
    start = time.perf_counter()
    try:
        output = run_trading_agent_sync(query)
        latency = time.perf_counter() - start

        text_eval = evaluate_output_text(output)
        return EvalCaseResult(
            query=query,
            success=True,
            latency_seconds=round(latency, 3),
            output_length=len(output or ""),
            score=round(text_eval["score"], 2),
            checks=text_eval["checks"],
            notes=text_eval["notes"],
            output_excerpt=(output or "")[:600],
            error=None,
        )
    except Exception as exc:
        latency = time.perf_counter() - start
        return EvalCaseResult(
            query=query,
            success=False,
            latency_seconds=round(latency, 3),
            output_length=0,
            score=0.0,
            checks={
                "has_recommendation": False,
                "has_entry_strategy": False,
                "has_exit_strategy": False,
                "has_risk_management": False,
                "has_numerical_grounding": False,
                "has_disclaimer": False,
                "no_error_language": False,
                "sufficient_length": False,
            },
            notes=["Pipeline execution failed"],
            output_excerpt="",
            error=str(exc),
        )


def aggregate_results(results: List[EvalCaseResult]) -> Dict:
    total = len(results)
    successes = sum(1 for item in results if item.success)
    success_rate = (successes / total * 100.0) if total else 0.0

    scores = [item.score for item in results]
    latencies = [item.latency_seconds for item in results]

    avg_score = statistics.mean(scores) if scores else 0.0
    median_score = statistics.median(scores) if scores else 0.0
    avg_latency = statistics.mean(latencies) if latencies else 0.0

    check_pass_counts: Dict[str, int] = {}
    for item in results:
        for key, value in item.checks.items():
            check_pass_counts[key] = check_pass_counts.get(key, 0) + (1 if value else 0)

    check_pass_rate = {
        key: round((count / total * 100.0), 2) if total else 0.0
        for key, count in check_pass_counts.items()
    }

    if avg_score >= 85 and success_rate >= 90:
        verdict = "excellent"
    elif avg_score >= 70 and success_rate >= 80:
        verdict = "good"
    elif avg_score >= 55 and success_rate >= 60:
        verdict = "needs_improvement"
    else:
        verdict = "poor"

    return {
        "total_cases": total,
        "success_cases": successes,
        "success_rate_pct": round(success_rate, 2),
        "avg_score": round(avg_score, 2),
        "median_score": round(median_score, 2),
        "avg_latency_seconds": round(avg_latency, 3),
        "check_pass_rate_pct": check_pass_rate,
        "overall_verdict": verdict,
    }


def write_markdown_report(report: Dict, markdown_path: str) -> None:
    md_file = Path(markdown_path)
    md_file.parent.mkdir(parents=True, exist_ok=True)

    project = report["project_summary"]
    cases = report["case_results"]

    lines = [
        "# Complete Project Evaluation Report",
        "",
        f"Generated at: **{report['meta']['timestamp_utc']}**",
        "",
        "## Evaluation Basis",
        "",
        "The agent output is evaluated on these weighted criteria:",
        "",
        "- Recommendation clarity (BUY/SELL/HOLD language)",
        "- Entry strategy presence",
        "- Exit strategy presence",
        "- Risk management details",
        "- Numerical grounding (prices/%, horizon)",
        "- Disclaimer presence",
        "- Error-free response language",
        "- Minimum response depth (length threshold)",
        "",
        "## Project-Level Summary",
        "",
        f"- Total cases: **{project['total_cases']}**",
        f"- Success rate: **{project['success_rate_pct']}%**",
        f"- Average score: **{project['avg_score']} / 100**",
        f"- Median score: **{project['median_score']} / 100**",
        f"- Average latency: **{project['avg_latency_seconds']}s**",
        f"- Verdict: **{project['overall_verdict']}**",
        "",
        "## Per-Case Results",
        "",
    ]

    for index, case in enumerate(cases, start=1):
        lines.extend(
            [
                f"### Case {index}",
                f"- Query: `{case['query']}`",
                f"- Success: **{case['success']}**",
                f"- Latency: **{case['latency_seconds']}s**",
                f"- Score: **{case['score']} / 100**",
                f"- Output length: **{case['output_length']} chars**",
            ]
        )

        if case.get("error"):
            lines.append(f"- Error: `{case['error']}`")

        if case.get("notes"):
            lines.append("- Notes:")
            for note in case["notes"]:
                lines.append(f"  - {note}")

        lines.append("")

    lines.extend(
        [
            "## Check Pass Rates",
            "",
        ]
    )

    for check_name, pass_rate in project["check_pass_rate_pct"].items():
        lines.append(f"- `{check_name}`: **{pass_rate}%**")

    with open(md_file, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def run_complete_evaluation(
    queries: List[str],
    output_json: str,
    output_markdown: str,
) -> Dict:
    case_results: List[EvalCaseResult] = []

    for query in queries:
        result = evaluate_case(query)
        case_results.append(result)

    project_summary = aggregate_results(case_results)

    report = {
        "meta": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "model_provider": "groq",
            "model_name": settings.model_name,
            "queries_count": len(queries),
        },
        "evaluation_basis": {
            "criteria": {
                "has_recommendation": 20,
                "has_entry_strategy": 12,
                "has_exit_strategy": 12,
                "has_risk_management": 15,
                "has_numerical_grounding": 15,
                "has_disclaimer": 8,
                "no_error_language": 10,
                "sufficient_length": 8,
            },
            "scoring_scale": "0 to 100",
        },
        "project_summary": project_summary,
        "case_results": [
            {
                "query": item.query,
                "success": item.success,
                "latency_seconds": item.latency_seconds,
                "output_length": item.output_length,
                "score": item.score,
                "checks": item.checks,
                "notes": item.notes,
                "error": item.error,
                "output_excerpt": item.output_excerpt,
            }
            for item in case_results
        ],
    }

    output_file = Path(output_json)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    write_markdown_report(report, output_markdown)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run complete end-to-end project evaluation.")
    parser.add_argument("--queries-file", type=str, default=None, help="Optional JSON file with evaluation queries")
    parser.add_argument("--output-json", type=str, default="reports/complete_evaluation_report.json")
    parser.add_argument("--output-markdown", type=str, default="reports/complete_evaluation_report.md")

    args = parser.parse_args()

    if not settings.groq_api_key or settings.groq_api_key == "your_groq_api_key_here":
        raise ValueError("GROQ_API_KEY is not configured. Please set it in .env before evaluation.")

    queries = load_queries(args.queries_file)
    report = run_complete_evaluation(
        queries=queries,
        output_json=args.output_json,
        output_markdown=args.output_markdown,
    )

    summary = report["project_summary"]
    print("\n✅ Complete evaluation finished")
    print(f"Cases: {summary['total_cases']}")
    print(f"Success rate: {summary['success_rate_pct']}%")
    print(f"Average score: {summary['avg_score']} / 100")
    print(f"Verdict: {summary['overall_verdict']}")
    print(f"JSON report: {args.output_json}")
    print(f"Markdown report: {args.output_markdown}")


if __name__ == "__main__":
    main()
