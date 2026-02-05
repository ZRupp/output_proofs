"""Lean verification using Verina's verification infrastructure."""

import sys
import asyncio
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from pathlib import Path
import logging

from ..config import PathConfig
from ..tasks.schema import CombinedTask, TaskVariant

logger = logging.getLogger(__name__)


@dataclass
class VerificationResult:
    """Result of Lean verification against Verina spec."""

    task_id: int
    variant_id: str

    # Python generation
    python_generated: str

    # Translation
    lean_translated: str
    translation_success: bool

    # Lean verification
    lean_compiles: bool
    tests_passed: int
    tests_failed: int
    tests_total: int
    proof_valid: Optional[bool]

    # Error info
    error: Optional[str] = None
    compiler_output: str = ""

    # Additional metrics
    verification_time_ms: float = 0.0

    @property
    def success(self) -> bool:
        """Overall success: translation + compilation + all tests pass."""
        return (
            self.translation_success
            and self.lean_compiles
            and self.tests_failed == 0
            and self.tests_total > 0
        )

    @property
    def partial_success(self) -> bool:
        """Partial success: at least compilation succeeded."""
        return self.translation_success and self.lean_compiles


def _setup_verina_import():
    """Setup Verina import path dynamically."""
    config = PathConfig()
    verina_path = str(config.verina_path)
    if verina_path not in sys.path:
        sys.path.insert(0, verina_path)
    # Test that verina can actually be imported
    import verina
    return True


class LeanVerifier:
    """Verify Lean code using Verina's verification infrastructure."""

    def __init__(self, config: PathConfig = None):
        self.config = config or PathConfig()
        self._verina_available = False
        self._setup_verina()

    def _setup_verina(self):
        """Setup Verina imports."""
        try:
            _setup_verina_import()
            self._verina_available = True
            logger.info("Verina loaded successfully")
        except Exception as e:
            logger.warning(f"Verina not available: {e}")
            self._verina_available = False

    def verify(
        self,
        lean_code: str,
        task: CombinedTask,
        variant_id: str = "original",
        python_code: str = "",
    ) -> VerificationResult:
        """Verify Lean code against Verina specification.

        Args:
            lean_code: Lean code to verify
            task: Combined task with Verina spec
            variant_id: Identifier for this variant
            python_code: Original Python code (for result tracking)

        Returns:
            VerificationResult with verification details
        """
        if not self._verina_available:
            return self._verify_fallback(
                lean_code, task, variant_id, python_code,
                "Verina not available"
            )

        try:
            # Run async verification
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(
                self._verify_async(lean_code, task, variant_id, python_code)
            )
        except RuntimeError:
            # No event loop, create one
            return asyncio.run(
                self._verify_async(lean_code, task, variant_id, python_code)
            )

    async def _verify_async(
        self,
        lean_code: str,
        task: CombinedTask,
        variant_id: str,
        python_code: str,
    ) -> VerificationResult:
        """Async verification using Verina APIs."""
        import time
        start_time = time.time()

        try:
            # Import Verina modules
            from verina.benchmark.metrics import metric_generated_code
            from verina.benchmark.report import EvaluationTaskArtifact
            from verina.dataset.template import LeanGenerationTaskTemplate

            # Build artifact from translated Lean code
            artifact = EvaluationTaskArtifact(code=lean_code)

            # Create template using Verina spec
            # Note: LeanGenerationTaskTemplate expects specific format
            template = self._create_template(task)

            # Get benchmark data for the task
            benchmark_data = self._get_benchmark_data(task)

            # Run verification
            score = await metric_generated_code(template, benchmark_data, artifact)

            # Parse results
            tests_passed = sum(
                1 for s in score.unit_tests.values()
                if hasattr(s, 'value') and s.value == "pass"
            )
            tests_failed = sum(
                1 for s in score.unit_tests.values()
                if hasattr(s, 'value') and s.value == "fail"
            )
            tests_total = len(score.unit_tests)

            elapsed_ms = (time.time() - start_time) * 1000

            return VerificationResult(
                task_id=task.task_id,
                variant_id=variant_id,
                python_generated=python_code,
                lean_translated=lean_code,
                translation_success=True,
                lean_compiles=score.can_compile,
                tests_passed=tests_passed,
                tests_failed=tests_failed,
                tests_total=tests_total,
                proof_valid=None,  # Proof checking is separate
                verification_time_ms=elapsed_ms,
            )

        except ImportError as e:
            logger.error(f"Verina import error: {e}")
            return self._verify_fallback(
                lean_code, task, variant_id, python_code,
                f"Verina import error: {e}"
            )
        except Exception as e:
            logger.error(f"Verification error: {e}")
            elapsed_ms = (time.time() - start_time) * 1000
            return VerificationResult(
                task_id=task.task_id,
                variant_id=variant_id,
                python_generated=python_code,
                lean_translated=lean_code,
                translation_success=True,
                lean_compiles=False,
                tests_passed=0,
                tests_failed=0,
                tests_total=0,
                proof_valid=None,
                error=str(e),
                verification_time_ms=elapsed_ms,
            )

    def _create_template(self, task: CombinedTask):
        """Create Lean generation template from task."""
        try:
            from verina.dataset.template import LeanGenerationTaskTemplate
            return LeanGenerationTaskTemplate(task.verina.signature)
        except Exception:
            # Return a minimal template if creation fails
            return None

    def _get_benchmark_data(self, task: CombinedTask):
        """Get benchmark data object from task's Verina spec."""
        try:
            from verina.dataset.schema import BenchmarkData

            # Create BenchmarkData from VerinaSpec
            return BenchmarkData(
                data_id=task.verina.data_id,
                signature=task.verina.signature,
                lean_code=task.verina.lean_code,
                precond=task.verina.precond,
                postcond=task.verina.postcond,
                spec_desc=task.verina.spec_desc,
                test_cases=task.verina.test_cases,
            )
        except Exception:
            # Return the verina spec directly as fallback
            return task.verina

    def _verify_fallback(
        self,
        lean_code: str,
        task: CombinedTask,
        variant_id: str,
        python_code: str,
        error_msg: str,
    ) -> VerificationResult:
        """Fallback verification when Verina is not available.

        Performs basic syntax checking only.
        """
        # Basic check: does the code look like valid Lean?
        has_def = 'def ' in lean_code or 'theorem ' in lean_code
        has_assignment = ':=' in lean_code

        return VerificationResult(
            task_id=task.task_id,
            variant_id=variant_id,
            python_generated=python_code,
            lean_translated=lean_code,
            translation_success=bool(lean_code),
            lean_compiles=has_def and has_assignment,  # Very basic check
            tests_passed=0,
            tests_failed=0,
            tests_total=0,
            proof_valid=None,
            error=error_msg,
        )

    def verify_batch(
        self,
        lean_codes: List[str],
        tasks: List[CombinedTask],
        variant_ids: List[str],
        python_codes: List[str],
    ) -> List[VerificationResult]:
        """Verify multiple Lean codes.

        Args:
            lean_codes: List of Lean codes
            tasks: Corresponding tasks
            variant_ids: Variant identifiers
            python_codes: Original Python codes

        Returns:
            List of verification results
        """
        results = []
        for lean, task, vid, py in zip(lean_codes, tasks, variant_ids, python_codes):
            result = self.verify(lean, task, vid, py)
            results.append(result)
        return results


async def verify_translated_lean(
    python_code: str,
    lean_code: str,
    task: CombinedTask,
    variant_id: str,
) -> VerificationResult:
    """Standalone async function to verify translated Lean code.

    Args:
        python_code: Original Python code
        lean_code: Translated Lean code
        task: Combined task with Verina spec
        variant_id: Variant identifier

    Returns:
        VerificationResult
    """
    verifier = LeanVerifier()
    return verifier.verify(lean_code, task, variant_id, python_code)
