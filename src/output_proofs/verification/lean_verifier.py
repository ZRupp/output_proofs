"""Lean verification using Verina's verification infrastructure."""

import asyncio
import json
import logging
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..config import PathConfig
from ..tasks.schema import CombinedTask

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
    return True


def _parse_test_value(val_str: str) -> Any:
    """Parse a stringified test value back to its Python type.

    The HF dataset serializes values via str(), so True → "True", 5 → "5", etc.
    """
    if not isinstance(val_str, str):
        return val_str
    low = val_str.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    try:
        return int(val_str)
    except ValueError:
        pass
    try:
        return float(val_str)
    except ValueError:
        pass
    return val_str


def _build_signature(sig_dict: Dict[str, Any]):
    """Convert a signature dict to Verina's Signature Pydantic model."""
    from verina.dataset.schema import Parameter, Signature

    params = [
        Parameter(param_name=p["param_name"], param_type=p["param_type"])
        for p in sig_dict.get("parameters", [])
    ]
    return Signature(
        name=sig_dict.get("name", ""),
        parameters=params,
        return_type=sig_dict.get("return_type", ""),
    )


def _build_test_cases(tests_raw: List[Dict[str, Any]]) -> list:
    """Convert HF test dicts to Verina TestCase models.

    HF format: {input: JSON_string, expected: [str, ...], unexpected: [str, ...]}
    Verina format: TestCase(input: Dict, expected: Any, unexpected: List[Any])
    """
    from verina.dataset.schema import TestCase

    cases = []
    for t in tests_raw:
        # input: JSON string → dict, or already a dict
        inp = t.get("input", {})
        if isinstance(inp, str):
            inp = json.loads(inp)

        # expected: list of strings → single value (take first)
        exp_raw = t.get("expected", [])
        if isinstance(exp_raw, list) and len(exp_raw) > 0:
            expected = _parse_test_value(exp_raw[0])
        else:
            expected = _parse_test_value(exp_raw) if isinstance(exp_raw, str) else exp_raw

        # unexpected: list of strings → list of values
        unexp_raw = t.get("unexpected", [])
        if isinstance(unexp_raw, list):
            unexpected = [_parse_test_value(v) for v in unexp_raw]
        else:
            unexpected = []

        cases.append(TestCase(input=inp, expected=expected, unexpected=unexpected))
    return cases


def _build_reject_inputs(reject_raw: List[Dict[str, Any]]) -> list:
    """Convert HF reject_inputs to Verina RejectInput models."""
    from verina.dataset.schema import RejectInput

    results = []
    for r in reject_raw:
        inp = r.get("input", {})
        if isinstance(inp, str):
            inp = json.loads(inp)
        results.append(RejectInput(input=inp))
    return results


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
                lean_code, task, variant_id, python_code, "Verina not available"
            )

        try:
            # Run async verification
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(
                self._verify_async(lean_code, task, variant_id, python_code)
            )
        except RuntimeError:
            # No event loop, create one
            return asyncio.run(self._verify_async(lean_code, task, variant_id, python_code))

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
                1 for s in score.unit_tests.values() if hasattr(s, "value") and s.value == "pass"
            )
            tests_failed = sum(
                1 for s in score.unit_tests.values() if hasattr(s, "value") and s.value == "fail"
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
                lean_code, task, variant_id, python_code, f"Verina import error: {e}"
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
        """Create Lean generation template from task.

        Converts our signature dict to Verina's Signature Pydantic model.
        """
        try:
            from verina.dataset.template import LeanGenerationTaskTemplate

            sig = _build_signature(task.verina.signature)
            return LeanGenerationTaskTemplate(sig)
        except Exception as e:
            logger.warning(f"Failed to create template: {e}")
            return None

    def _get_benchmark_data(self, task: CombinedTask):
        """Get benchmark data object from task's Verina spec.

        Converts our VerinaSpec fields to Verina's Pydantic models:
        - lean_code (annotated string) → BenchmarkLeanData via parse_benchmark_lean_data()
        - signature dict → Signature model
        - tests list → List[TestCase]
        - reject_inputs list → List[RejectInput]
        """
        try:
            from verina.dataset.parsing import parse_benchmark_lean_data
            from verina.dataset.schema import BenchmarkData, SpecDesc

            spec = task.verina

            lean_data = parse_benchmark_lean_data(spec.lean_code)
            signature = _build_signature(spec.signature)
            tests = _build_test_cases(spec.tests)
            reject_inputs = _build_reject_inputs(spec.reject_inputs)

            return BenchmarkData(
                data_id=spec.data_id,
                description=spec.description,
                signature=signature,
                lean_data=lean_data,
                spec_desc=SpecDesc(precond_desc="", postcond_desc=""),
                reject_inputs=reject_inputs,
                tests=tests,
                metadata=None,
            )
        except Exception as e:
            logger.error(f"Failed to build BenchmarkData: {e}")
            raise

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
        has_def = "def " in lean_code or "theorem " in lean_code
        has_assignment = ":=" in lean_code

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
