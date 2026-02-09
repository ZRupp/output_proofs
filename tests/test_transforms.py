"""Tests for LibCST transformations."""

from output_proofs.transforms.rename import (
    VariableRenamer,
    generate_noise_variants,
)
from output_proofs.transforms.reorder import (
    ParameterReorderer,
    generate_reorder_variants,
)
from output_proofs.transforms.test_adapter import (
    adapt_single_test,
    adapt_tests_for_variant,
)


class TestVariableRenamer:
    """Tests for variable renaming transform."""

    def test_basic_rename(self):
        """Test basic variable renaming."""
        code = """def foo(x, y):
    z = x + y
    return z
"""
        renamer = VariableRenamer(seed=42)
        result = renamer.transform(code)

        assert result.success
        assert result.transformed != code
        assert result.param_mapping

    def test_preserves_function_name(self):
        """Function name should not be renamed."""
        code = """def my_function(a, b):
    return a + b
"""
        renamer = VariableRenamer(seed=42)
        result = renamer.transform(code)

        assert result.success
        assert "def my_function" in result.transformed

    def test_protected_names_not_renamed(self):
        """Builtin names should not be renamed."""
        code = """def foo(x):
    result = len(x)
    return result
"""
        renamer = VariableRenamer(seed=42)
        result = renamer.transform(code)

        assert result.success
        assert "len(" in result.transformed

    def test_generate_variants(self):
        """Test generating multiple variants."""
        code = """def bar(a, b):
    c = a * b
    return c
"""
        variants = generate_noise_variants(code, num_variants=3, seed=42)

        assert len(variants) == 3
        # Each variant should be different
        transformed_codes = [v.transformed for v in variants]
        assert len(set(transformed_codes)) == 3


class TestParameterReorderer:
    """Tests for parameter reordering transform."""

    def test_basic_reorder(self):
        """Test basic parameter reordering."""
        code = """def foo(a, b, c):
    return a + b + c
"""
        reorderer = ParameterReorderer(permutation=[2, 0, 1])
        result = reorderer.transform(code)

        assert result.success
        assert result.transformed != code

    def test_single_param_unchanged(self):
        """Single parameter should not be reordered."""
        code = """def foo(x):
    return x * 2
"""
        reorderer = ParameterReorderer(seed=42)
        result = reorderer.transform(code)

        assert result.success
        # Should remain unchanged
        assert result.transformed == code

    def test_star_args_not_reordered(self):
        """Functions with *args should not be reordered."""
        code = """def foo(a, *args):
    return a + sum(args)
"""
        reorderer = ParameterReorderer(seed=42)
        can_apply = reorderer.can_apply(code)

        assert not can_apply

    def test_generate_reorder_variants(self):
        """Test generating multiple reorder variants."""
        code = """def bar(x, y, z):
    return x - y + z
"""
        variants = generate_reorder_variants(code, num_variants=3, seed=42)

        # Some variants may be the same due to random permutations
        assert len(variants) >= 1


class TestTestAdapter:
    """Tests for test harness adaptation."""

    def test_adapt_keyword_args(self):
        """Test adapting keyword arguments."""
        test = "assert foo(x=1, y=2) == 3"
        mapping = {"x": "var_x", "y": "var_y"}

        adapted = adapt_single_test(test, "foo", mapping)

        assert "var_x=" in adapted
        assert "var_y=" in adapted

    def test_adapt_positional_unchanged(self):
        """Positional arguments should remain unchanged."""
        test = "assert foo(1, 2) == 3"
        mapping = {"x": "var_x", "y": "var_y"}

        adapted = adapt_single_test(test, "foo", mapping)

        # Positional args unchanged
        assert "foo(1, 2)" in adapted

    def test_adapt_test_list(self):
        """Test adapting a list of tests."""
        tests = [
            "assert bar(a=1) == 1",
            "assert bar(a=2) == 4",
            "assert bar(a=3) == 9",
        ]
        mapping = {"a": "_arg0"}

        adapted = adapt_tests_for_variant(tests, "bar", mapping)

        assert len(adapted) == 3
        for test in adapted:
            assert "_arg0=" in test


class TestTransformIntegration:
    """Integration tests for combined transforms."""

    def test_rename_then_adapt_tests(self):
        """Test renaming variables then adapting tests."""
        code = """def square(n):
    result = n * n
    return result
"""
        tests = [
            "assert square(n=2) == 4",
            "assert square(n=3) == 9",
        ]

        # Apply rename
        renamer = VariableRenamer(seed=42)
        result = renamer.transform(code)

        # Adapt tests
        adapted_tests = adapt_tests_for_variant(
            tests,
            "square",
            result.param_mapping,
        )

        # Verify tests were adapted
        assert result.success
        for test in adapted_tests:
            # Original 'n' should be replaced if it was in mapping
            if "n" in result.param_mapping:
                new_name = result.param_mapping["n"]
                assert f"{new_name}=" in test
