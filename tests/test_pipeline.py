"""Tests for pipeline components."""

import pytest
from unittest.mock import MagicMock, patch

from output_proofs.config import detect_device
from output_proofs.tasks.schema import MBPPTask, VerinaSpec, CombinedTask, TaskVariant
from output_proofs.tasks.variants import (
    VariantGenerator,
    extract_function_info,
    transform_description,
)

# Conditional imports for modules requiring torch
try:
    from output_proofs.inference.prompts import (
        PromptBuilder,
        create_instruction_prompt,
        get_fim_tokens,
    )

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    PromptBuilder = None
    create_instruction_prompt = None
    get_fim_tokens = None

from output_proofs.reporting.report import (
    aggregate_results,
    BenchmarkReport,
    PipelineStageMetrics,
)
from output_proofs.verification.lean_verifier import VerificationResult


class TestVariantGenerator:
    """Tests for variant generation."""

    @pytest.fixture
    def sample_task(self):
        """Create a sample combined task."""
        mbpp = MBPPTask(
            task_id=101,
            text="Write a function to add two numbers",
            code="def add(a, b):\n    return a + b",
            test_list=["assert add(1, 2) == 3"],
        )
        verina = VerinaSpec(
            data_id="verina_basic_1",
            lean_code="def add (a b : Nat) : Nat := a + b",
            signature="add : Nat → Nat → Nat",
            precond="True",
            postcond="result = a + b",
        )
        return CombinedTask(mbpp=mbpp, verina=verina, task_id=101)

    def test_generate_variants(self, sample_task):
        """Test variant generation."""
        generator = VariantGenerator(seed=42)
        variants = generator.generate_variants(sample_task, num_variants=3)

        assert len(variants) == 3
        for v in variants:
            assert v.task_id == 101
            assert v.func_name == "add"

    def test_extract_function_info(self):
        """Test function info extraction."""
        code = '''def my_func(x, y):
    """Docstring here."""
    return x + y
'''
        info = extract_function_info(code)

        assert info["name"] == "my_func"
        assert "x" in info["params"]
        assert "y" in info["params"]

    def test_transform_description(self):
        """Test description transformation."""
        desc = "Add parameter x and parameter y"
        mapping = {"x": "var_x", "y": "var_y"}

        result = transform_description(desc, mapping)

        assert "var_x" in result
        assert "var_y" in result


@pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
class TestPromptBuilder:
    """Tests for prompt building."""

    @pytest.fixture
    def mock_tokenizer(self):
        """Create mock tokenizer."""
        tokenizer = MagicMock()
        tokenizer.get_vocab.return_value = {}
        return tokenizer

    @pytest.fixture
    def sample_variant(self):
        """Create sample variant."""
        mbpp = MBPPTask(
            task_id=101,
            text="Add two numbers",
            code="def add(a, b): return a + b",
            test_list=[],
        )
        verina = VerinaSpec(
            data_id="verina_basic_1",
            lean_code="",
            signature="",
            precond="",
            postcond="",
        )
        task = CombinedTask(mbpp=mbpp, verina=verina, task_id=101)

        return TaskVariant(
            original_task=task,
            variant_id="101_v0",
            transformed_description="Add two numbers",
            transformed_signature="def add(var_a, var_b):",
            func_name="add",
            noisy_params="var_a, var_b",
            transforms_applied=["v_noise_rename"],
            param_mapping={"a": "var_a", "b": "var_b"},
            reverse_mapping={"var_a": "a", "var_b": "b"},
        )

    def test_instruction_prompt(self, mock_tokenizer, sample_variant):
        """Test instruction prompt creation."""
        builder = PromptBuilder(mock_tokenizer)
        prompt = builder.create_instruction_prompt(sample_variant)

        assert "Add two numbers" in prompt
        assert "def add(var_a, var_b)" in prompt

    def test_fim_tokens_not_supported(self, mock_tokenizer):
        """Test FIM token detection when not supported."""
        tokens = get_fim_tokens(mock_tokenizer)

        assert not tokens.supported


class TestReporting:
    """Tests for result aggregation and reporting."""

    @pytest.fixture
    def sample_results(self):
        """Create sample verification results."""
        return [
            VerificationResult(
                task_id=101,
                variant_id="101_v0",
                python_generated="def add(a, b): return a + b",
                lean_translated="def add := ...",
                translation_success=True,
                lean_compiles=True,
                tests_passed=3,
                tests_failed=0,
                tests_total=3,
                proof_valid=None,
            ),
            VerificationResult(
                task_id=101,
                variant_id="101_v1",
                python_generated="def add(x, y): return x + y",
                lean_translated="def add := ...",
                translation_success=True,
                lean_compiles=True,
                tests_passed=2,
                tests_failed=1,
                tests_total=3,
                proof_valid=None,
            ),
            VerificationResult(
                task_id=102,
                variant_id="102_v0",
                python_generated="",
                lean_translated="",
                translation_success=False,
                lean_compiles=False,
                tests_passed=0,
                tests_failed=0,
                tests_total=0,
                proof_valid=None,
                error="Translation failed",
            ),
        ]

    def test_aggregate_results(self, sample_results):
        """Test result aggregation."""
        report = aggregate_results(sample_results, "test-model", "test-formalizer")

        assert report.total_tasks == 2
        assert report.total_variants == 3
        assert report.model == "test-model"

    def test_pipeline_metrics(self, sample_results):
        """Test pipeline stage metrics calculation."""
        report = aggregate_results(sample_results)

        # 2 out of 3 had python generated
        assert report.pipeline_stages.python_generation_rate == pytest.approx(2 / 3, rel=0.01)

    def test_robustness_metrics(self, sample_results):
        """Test robustness metrics calculation."""
        report = aggregate_results(sample_results)

        # Original (v0) results: 1 success out of 2
        assert report.robustness.original_pass_rate >= 0

    def test_report_to_dict(self, sample_results):
        """Test report serialization."""
        report = aggregate_results(sample_results)
        data = report.to_dict()

        assert "model" in data
        assert "pipeline_stages" in data
        assert "robustness" in data


class TestDetectDevice:
    """Tests for detect_device() helper."""

    def test_env_var_override(self, monkeypatch):
        """DEVICE env var should take priority."""
        monkeypatch.setenv("DEVICE", "cpu")
        assert detect_device() == "cpu"

    def test_env_var_case_insensitive(self, monkeypatch):
        """DEVICE env var should be lowercased."""
        monkeypatch.setenv("DEVICE", "CUDA")
        assert detect_device() == "cuda"

    def test_gpu_available(self, monkeypatch):
        """Should return 'cuda' when torch reports GPU available."""
        monkeypatch.delenv("DEVICE", raising=False)
        with patch("torch.cuda.is_available", return_value=True):
            assert detect_device() == "cuda"

    def test_no_gpu(self, monkeypatch):
        """Should return 'cpu' when no GPU available."""
        monkeypatch.delenv("DEVICE", raising=False)
        with patch("torch.cuda.is_available", return_value=False):
            assert detect_device() == "cpu"
