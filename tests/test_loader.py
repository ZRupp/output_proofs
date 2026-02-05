"""Tests for task loading."""

import pytest
from unittest.mock import patch, MagicMock

from output_proofs.tasks.schema import MBPPTask, VerinaSpec, CombinedTask

# Skip loader tests if datasets not available
try:
    from output_proofs.tasks.loader import (
        MBPP_TASK_IDS,
        _extract_function_name,
        build_task_mapping,
    )
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
    MBPP_TASK_IDS = []
    _extract_function_name = None
    build_task_mapping = None


@pytest.mark.skipif(not HAS_DATASETS, reason="datasets module not available")
class TestMBPPTaskIds:
    """Tests for MBPP task ID list."""

    def test_task_ids_count(self):
        """Should have 49 task IDs (excluding 644)."""
        assert len(MBPP_TASK_IDS) == 49

    def test_task_id_644_excluded(self):
        """Task ID 644 should be excluded."""
        assert 644 not in MBPP_TASK_IDS

    def test_task_ids_unique(self):
        """All task IDs should be unique."""
        assert len(MBPP_TASK_IDS) == len(set(MBPP_TASK_IDS))


@pytest.mark.skipif(not HAS_DATASETS, reason="datasets module not available")
class TestFunctionNameExtraction:
    """Tests for function name extraction."""

    def test_simple_function(self):
        """Extract name from simple function."""
        code = "def my_func(x):\n    return x"
        assert _extract_function_name(code) == "my_func"

    def test_function_with_docstring(self):
        """Extract name from function with docstring."""
        code = '''def compute(a, b):
    """Compute something."""
    return a + b
'''
        assert _extract_function_name(code) == "compute"

    def test_no_function(self):
        """Return None when no function found."""
        code = "x = 1\ny = 2"
        assert _extract_function_name(code) is None


class TestSchema:
    """Tests for data schemas."""

    def test_mbpp_task_from_dict(self):
        """Test MBPPTask creation from dict."""
        data = {
            "task_id": 101,
            "text": "Write a function",
            "code": "def foo(): pass",
            "test_list": ["assert foo() is None"],
        }
        task = MBPPTask.from_dict(data)

        assert task.task_id == 101
        assert task.text == "Write a function"
        assert task.code == "def foo(): pass"
        assert len(task.test_list) == 1

    def test_verina_spec_from_dict(self):
        """Test VerinaSpec creation from dict."""
        data = {
            "data_id": "verina_basic_1",
            "lean_code": "def foo := 0",
            "signature": "foo : Nat",
            "precond": "True",
            "postcond": "result = 0",
        }
        spec = VerinaSpec.from_dict(data)

        assert spec.data_id == "verina_basic_1"
        assert spec.signature == "foo : Nat"

    def test_combined_task(self):
        """Test CombinedTask properties."""
        mbpp = MBPPTask(
            task_id=101,
            text="Test task",
            code="def test(): pass",
            test_list=[],
        )
        verina = VerinaSpec(
            data_id="verina_basic_1",
            lean_code="def test := 0",
            signature="test : Nat",
            precond="True",
            postcond="True",
        )
        combined = CombinedTask(mbpp=mbpp, verina=verina, task_id=101)

        assert combined.description == "Test task"
        assert combined.reference_python == "def test(): pass"
        assert combined.reference_lean == "def test := 0"
