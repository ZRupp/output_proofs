"""Lean verification using Verina's verification infrastructure."""

import ast
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


def _parse_dict_string(s: str) -> Dict[str, Any]:
    """Parse a string that may be JSON or a Python dict literal.

    The HF dataset uses json.dumps() for inputs, but some entries may use
    Python repr format (single quotes). ast.literal_eval is safe — it only
    parses literal expressions (strings, numbers, tuples, lists, dicts,
    booleans, None), never arbitrary code.
    """
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError):
        pass
    # Safe fallback: ast.literal_eval only handles literal data structures
    try:
        result = ast.literal_eval(s)
        if isinstance(result, dict):
            return result
    except (ValueError, SyntaxError):
        pass
    return {}


def _parse_test_value(val_str: str) -> Any:
    """Parse a stringified test value back to its Python type.

    The HF dataset serializes values via str(), so True -> "True", 5 -> "5", etc.
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


def _columnar_to_rows(data: Any) -> List[Dict[str, Any]]:
    """Convert HuggingFace column-oriented Sequence data to row-oriented.

    HF Sequence of structs returns {"col1": [v1, v2], "col2": [v3, v4]}
    instead of [{"col1": v1, "col2": v3}, {"col1": v2, "col2": v4}].
    """
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        keys = list(data.keys())
        if not keys:
            return []
        n = len(data[keys[0]])
        return [{k: data[k][i] for k in keys} for i in range(n)]
    return []


def _build_signature(sig_dict: Dict[str, Any]):
    """Convert a signature dict to Verina's Signature Pydantic model."""
    from verina.dataset.schema import Parameter, Signature

    params_raw = _columnar_to_rows(sig_dict.get("parameters", []))
    params = [
        Parameter(param_name=p["param_name"], param_type=p["param_type"])
        for p in params_raw
    ]
    return Signature(
        name=sig_dict.get("name", ""),
        parameters=params,
        return_type=sig_dict.get("return_type", ""),
    )


def _build_test_cases(tests_raw) -> list:
    """Convert HF test data to Verina TestCase models.

    Handles both row-oriented (list of dicts) and column-oriented (dict of
    lists) formats that HuggingFace datasets may return for Sequence features.
    """
    from verina.dataset.schema import TestCase

    rows = _columnar_to_rows(tests_raw)
    cases = []
    for t in rows:
        # input: JSON or Python dict string → dict, or already a dict
        inp = t.get("input", {})
        if isinstance(inp, str):
            inp = _parse_dict_string(inp)

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


def _build_reject_inputs(reject_raw) -> list:
    """Convert HF reject_inputs to Verina RejectInput models."""
    from verina.dataset.schema import RejectInput

    rows = _columnar_to_rows(reject_raw)
    results = []
    for r in rows:
        inp = r.get("input", {})
        if isinstance(inp, str):
            inp = _parse_dict_string(inp)
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

        Bypasses Verina's Prefect @task decorators and calls the Lean
        compiler directly via subprocess, avoiding issues with running
        Prefect tasks outside a Prefect flow context.

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

        import time
        from uuid import uuid4

        start_time = time.time()

        try:
            from verina.benchmark.report import EvaluationTaskArtifact
            from verina.lean import check_lean_compile, create_lean_file

            template = self._create_template(task)
            benchmark_data = self._get_benchmark_data(task)
            artifact = EvaluationTaskArtifact(code=lean_code)

            # Render Lean content — same logic as Verina's metric_generated_code
            lean_content = (
                template.render_imports(benchmark_data.lean_data.task_imports, "task")
                + "\n"
            )
            lean_content += (
                template.render_imports(artifact.imports, "llm_solution") + "\n"
            )
            lean_content += (
                template.render_aux(benchmark_data.lean_data.task_aux, "task") + "\n"
            )
            lean_content += (
                template.render_aux(artifact.precond_aux, "precond") + "\n"
            )
            precond = artifact.precond if artifact.precond else "True -- no precondition"
            lean_content += template.render_precond(precond) + "\n"
            lean_content += template.render_aux(artifact.code_aux, "code") + "\n"
            lean_content += template.render_code(artifact.code) + "\n"

            # Compile directly (bypassing Prefect @task)
            lean_file = create_lean_file(str(uuid4()), lean_content)
            can_compile, compile_output = check_lean_compile(lean_file)

            logger.debug(
                f"Lean compile {'OK' if can_compile else 'FAIL'} for "
                f"task {task.task_id} variant {variant_id}"
            )

            # Run unit tests if compilation succeeds
            tests_passed = 0
            tests_failed = 0
            tests_total = len(benchmark_data.tests)

            if can_compile and tests_total > 0:
                tests_passed, tests_failed = self._run_unit_tests(
                    template, lean_content, benchmark_data.tests
                )
                tests_total = tests_passed + tests_failed

            elapsed_ms = (time.time() - start_time) * 1000

            return VerificationResult(
                task_id=task.task_id,
                variant_id=variant_id,
                python_generated=python_code,
                lean_translated=lean_code,
                translation_success=True,
                lean_compiles=can_compile,
                tests_passed=tests_passed,
                tests_failed=tests_failed,
                tests_total=tests_total,
                proof_valid=None,
                compiler_output=compile_output[:500] if not can_compile else "",
                verification_time_ms=elapsed_ms,
            )

        except ImportError as e:
            logger.error(f"Verina import error: {e}")
            return self._verify_fallback(
                lean_code, task, variant_id, python_code, f"Verina import error: {e}"
            )
        except Exception as e:
            logger.error(f"Verification error for task {task.task_id}: {e}")
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

    def _run_unit_tests(self, template, base_lean_content, test_cases) -> tuple:
        """Run Lean unit tests by compiling test file directly.

        Returns (tests_passed, tests_failed).
        """
        from uuid import uuid4

        from verina.lean import check_lean_compile, create_lean_file

        test_content = (
            template.render_test_imports() + "\n" + base_lean_content
        )
        for idx, test_case in enumerate(test_cases):
            test_content += "\n\n" + template.render_code_unit_test(
                test_case, test_idx=idx
            )

        test_file = create_lean_file(str(uuid4()), test_content)
        tests_compile, test_output = check_lean_compile(test_file)

        if tests_compile:
            return len(test_cases), 0

        # Parse individual test results from compiler output
        passed = 0
        failed = 0
        marker_start = f"<{template.CODE_TEST_MSG_MARKER}>"
        marker_end = f"</{template.CODE_TEST_MSG_MARKER}>"

        for idx in range(len(test_cases)):
            tag = f"{marker_start}{idx}{marker_end}"
            if tag in test_output:
                # Find the output after this test's marker
                after_tag = test_output.split(tag, 1)[1]
                # Check up to next marker or end
                next_marker = after_tag.find(marker_start)
                segment = after_tag[:next_marker] if next_marker != -1 else after_tag
                if template.DECIDABLE_ERR_MSG in segment:
                    failed += 1
                else:
                    passed += 1
            else:
                failed += 1

        return passed, failed

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
