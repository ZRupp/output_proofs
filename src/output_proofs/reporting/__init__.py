"""Reporting and result aggregation."""

from .report import BenchmarkReport, aggregate_results, generate_report

__all__ = [
    "BenchmarkReport",
    "aggregate_results",
    "generate_report",
]
