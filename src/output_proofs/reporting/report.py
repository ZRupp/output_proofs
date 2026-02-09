"""Benchmark reporting and result aggregation."""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from ..verification.lean_verifier import VerificationResult


@dataclass
class TransformMetrics:
    """Metrics for a specific transform type."""

    transform_name: str
    total_variants: int
    python_generation_success: int
    translation_success: int
    compilation_success: int
    tests_passed_total: int
    tests_failed_total: int
    full_success: int  # All tests pass

    @property
    def python_generation_rate(self) -> float:
        if self.total_variants == 0:
            return 0.0
        return self.python_generation_success / self.total_variants

    @property
    def translation_rate(self) -> float:
        if self.python_generation_success == 0:
            return 0.0
        return self.translation_success / self.python_generation_success

    @property
    def compilation_rate(self) -> float:
        if self.translation_success == 0:
            return 0.0
        return self.compilation_success / self.translation_success

    @property
    def pass_rate(self) -> float:
        if self.total_variants == 0:
            return 0.0
        return self.full_success / self.total_variants


@dataclass
class PipelineStageMetrics:
    """Metrics for each pipeline stage."""

    python_generation_rate: float
    translation_success_rate: float
    lean_compilation_rate: float
    verification_pass_rate: float


@dataclass
class RobustnessMetrics:
    """Robustness metrics comparing original vs variant performance."""

    original_pass_rate: float
    variant_pass_rate: float
    robustness_score: float  # Difference between rates


@dataclass
class BenchmarkReport:
    """Complete benchmark report."""

    # Metadata
    model: str
    formalizer: str
    timestamp: str
    datasets: Dict[str, str]

    # Summary counts
    total_tasks: int
    total_variants: int

    # Pipeline metrics
    pipeline_stages: PipelineStageMetrics

    # Robustness metrics
    robustness: RobustnessMetrics

    # Per-transform breakdown
    by_transform: Dict[str, float]

    # Detailed results
    results: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "model": self.model,
            "formalizer": self.formalizer,
            "timestamp": self.timestamp,
            "datasets": self.datasets,
            "total_tasks": self.total_tasks,
            "total_variants": self.total_variants,
            "pipeline_stages": {
                "python_generation_rate": self.pipeline_stages.python_generation_rate,
                "translation_success_rate": self.pipeline_stages.translation_success_rate,
                "lean_compilation_rate": self.pipeline_stages.lean_compilation_rate,
                "verification_pass_rate": self.pipeline_stages.verification_pass_rate,
            },
            "robustness": {
                "original_pass_rate": self.robustness.original_pass_rate,
                "variant_pass_rate": self.robustness.variant_pass_rate,
                "robustness_score": self.robustness.robustness_score,
            },
            "by_transform": self.by_transform,
        }

    def save(self, path: Path) -> None:
        """Save report to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def save_detailed(self, path: Path) -> None:
        """Save detailed report including all results."""
        data = self.to_dict()
        data["results"] = self.results
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)


def aggregate_results(
    results: List[VerificationResult],
    model_name: str = "unknown",
    formalizer_name: str = "unknown",
) -> BenchmarkReport:
    """Aggregate verification results into a benchmark report.

    Args:
        results: List of verification results
        model_name: Name of the code generation model
        formalizer_name: Name of the translation model

    Returns:
        BenchmarkReport with aggregated metrics
    """
    if not results:
        return BenchmarkReport(
            model=model_name,
            formalizer=formalizer_name,
            timestamp=datetime.now().isoformat(),
            datasets={
                "python_source": "Muennighoff/mbpp",
                "lean_specs": "sunblaze-ucb/verina",
            },
            total_tasks=0,
            total_variants=0,
            pipeline_stages=PipelineStageMetrics(0, 0, 0, 0),
            robustness=RobustnessMetrics(0, 0, 0),
            by_transform={},
        )

    # Count unique tasks
    task_ids = set(r.task_id for r in results)
    total_tasks = len(task_ids)
    total_variants = len(results)

    # Pipeline stage metrics
    python_success = sum(1 for r in results if r.python_generated)
    translation_success = sum(1 for r in results if r.translation_success)
    compilation_success = sum(1 for r in results if r.lean_compiles)
    verification_success = sum(1 for r in results if r.success)

    pipeline_stages = PipelineStageMetrics(
        python_generation_rate=python_success / total_variants if total_variants > 0 else 0,
        translation_success_rate=translation_success / python_success if python_success > 0 else 0,
        lean_compilation_rate=(
            compilation_success / translation_success if translation_success > 0 else 0
        ),
        verification_pass_rate=verification_success / total_variants if total_variants > 0 else 0,
    )

    # Robustness metrics (original vs variants)
    original_results = [
        r for r in results if r.variant_id.endswith("_v0") or r.variant_id == "original"
    ]
    variant_results = [
        r for r in results if not (r.variant_id.endswith("_v0") or r.variant_id == "original")
    ]

    original_pass = sum(1 for r in original_results if r.success)
    variant_pass = sum(1 for r in variant_results if r.success)

    original_rate = original_pass / len(original_results) if original_results else 0
    variant_rate = variant_pass / len(variant_results) if variant_results else 0

    robustness = RobustnessMetrics(
        original_pass_rate=original_rate,
        variant_pass_rate=variant_rate,
        robustness_score=original_rate - variant_rate,
    )

    # Per-transform breakdown
    by_transform = _compute_transform_metrics(results)

    # Convert results to dicts for storage
    results_dicts = [
        {
            "task_id": r.task_id,
            "variant_id": r.variant_id,
            "translation_success": r.translation_success,
            "lean_compiles": r.lean_compiles,
            "tests_passed": r.tests_passed,
            "tests_failed": r.tests_failed,
            "success": r.success,
            "error": r.error,
        }
        for r in results
    ]

    return BenchmarkReport(
        model=model_name,
        formalizer=formalizer_name,
        timestamp=datetime.now().isoformat(),
        datasets={
            "python_source": "Muennighoff/mbpp",
            "lean_specs": "sunblaze-ucb/verina",
        },
        total_tasks=total_tasks,
        total_variants=total_variants,
        pipeline_stages=pipeline_stages,
        robustness=robustness,
        by_transform=by_transform,
        results=results_dicts,
    )


def _compute_transform_metrics(results: List[VerificationResult]) -> Dict[str, float]:
    """Compute pass rates by transform type."""
    transform_results: Dict[str, List[VerificationResult]] = {}

    for r in results:
        # Extract transform name from variant_id (e.g., "123_v_noise_rename_v0")
        vid = r.variant_id
        if "_v" in vid:
            # Try to extract transform name
            parts = vid.split("_")
            # Find transform keywords
            for i, part in enumerate(parts):
                if part in ("noise", "rename", "reorder"):
                    transform_name = "_".join(parts[i - 1 : i + 1]) if i > 0 else part
                    if transform_name not in transform_results:
                        transform_results[transform_name] = []
                    transform_results[transform_name].append(r)
                    break
            else:
                # Default category
                if "default" not in transform_results:
                    transform_results["default"] = []
                transform_results["default"].append(r)
        else:
            if "original" not in transform_results:
                transform_results["original"] = []
            transform_results["original"].append(r)

    return {
        name: sum(1 for r in res if r.success) / len(res) if res else 0
        for name, res in transform_results.items()
    }


def generate_report(
    results: List[VerificationResult],
    output_path: Path,
    model_name: str = "Llama-3.2-3B",
    formalizer_name: str = "Goedel-Formalizer-V2-8B",
    detailed: bool = True,
) -> BenchmarkReport:
    """Generate and save a benchmark report.

    Args:
        results: List of verification results
        output_path: Path to save the report
        model_name: Name of code generation model
        formalizer_name: Name of translation model
        detailed: Whether to include detailed results

    Returns:
        Generated BenchmarkReport
    """
    report = aggregate_results(results, model_name, formalizer_name)

    if detailed:
        report.save_detailed(output_path)
    else:
        report.save(output_path)

    return report
