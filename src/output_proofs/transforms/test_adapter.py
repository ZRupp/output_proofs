"""Adapt test harnesses for transformed code variants.

Handles the critical Warning C: Noisy params break keyword argument calls in test suites.
"""

import re
from typing import Dict, List, Optional

import libcst as cst


class TestCallTransformer(cst.CSTTransformer):
    """Transform test assertions to use new parameter names."""

    def __init__(self, func_name: str, param_mapping: Dict[str, str]):
        self.func_name = func_name
        self.param_mapping = param_mapping  # old_name -> new_name

    def leave_Call(self, original_node: cst.Call, updated_node: cst.Call) -> cst.Call:
        """Transform keyword arguments in function calls."""
        # Check if this is a call to our target function
        if isinstance(updated_node.func, cst.Name):
            if updated_node.func.value != self.func_name:
                return updated_node
        elif isinstance(updated_node.func, cst.Attribute):
            # Handle method calls like assert_equal(func(...), ...)
            return updated_node
        else:
            return updated_node

        # Transform keyword arguments
        new_args = []
        for arg in updated_node.args:
            if arg.keyword and arg.keyword.value in self.param_mapping:
                new_keyword = cst.Name(self.param_mapping[arg.keyword.value])
                new_arg = arg.with_changes(keyword=new_keyword)
                new_args.append(new_arg)
            else:
                new_args.append(arg)

        return updated_node.with_changes(args=new_args)


def extract_function_name(test_assertion: str) -> Optional[str]:
    """Extract the function name being tested from an assertion.

    Examples:
        "assert foo(1, 2) == 3" -> "foo"
        "assert bar(x=1) == 2" -> "bar"
    """
    # Pattern: assert func_name(...)
    match = re.search(r"assert\s+(\w+)\s*\(", test_assertion)
    if match:
        return match.group(1)

    # Pattern: func_name(...) ==
    match = re.search(r"(\w+)\s*\([^)]*\)\s*==", test_assertion)
    if match:
        return match.group(1)

    return None


def adapt_single_test(
    test: str,
    func_name: str,
    param_mapping: Dict[str, str],
) -> str:
    """Adapt a single test assertion for noisy parameter names.

    Args:
        test: Test assertion string (e.g., "assert foo(k=2) == 4")
        func_name: Name of the function being tested
        param_mapping: Mapping from original param names to noisy names

    Returns:
        Transformed test assertion
    """
    if not param_mapping:
        return test

    try:
        # Try to parse as a module (wrap in a function for context)
        wrapper = f"def _test_wrapper():\n    {test}"
        tree = cst.parse_module(wrapper)

        transformer = TestCallTransformer(func_name, param_mapping)
        modified = tree.visit(transformer)

        # Extract the transformed test
        modified_code = modified.code
        # Remove the wrapper
        lines = modified_code.strip().split("\n")
        if len(lines) >= 2:
            return lines[1].strip()

        return test

    except (cst.ParserSyntaxError, Exception):
        # Fallback: use regex replacement
        return _regex_adapt_test(test, param_mapping)


def _regex_adapt_test(test: str, param_mapping: Dict[str, str]) -> str:
    """Fallback regex-based test adaptation."""
    result = test
    for old_name, new_name in param_mapping.items():
        # Replace keyword arguments: old_name= -> new_name=
        result = re.sub(rf"\b{re.escape(old_name)}\s*=", f"{new_name}=", result)
    return result


def adapt_tests_for_variant(
    test_list: List[str],
    func_name: str,
    param_mapping: Dict[str, str],
) -> List[str]:
    """Transform test assertions to use noisy parameter names.

    Args:
        test_list: List of test assertion strings
        func_name: Name of the function being tested
        param_mapping: Mapping from original param names to noisy names

    Returns:
        List of transformed test assertions
    """
    if not param_mapping:
        return test_list.copy()

    adapted = []
    for test in test_list:
        adapted_test = adapt_single_test(test, func_name, param_mapping)
        adapted.append(adapted_test)

    return adapted


def adapt_test_code_block(
    test_code: str,
    func_name: str,
    param_mapping: Dict[str, str],
) -> str:
    """Adapt a block of test code (multiple statements) for noisy params.

    Args:
        test_code: Block of Python test code
        func_name: Name of the function being tested
        param_mapping: Mapping from original param names to noisy names

    Returns:
        Transformed test code block
    """
    if not param_mapping:
        return test_code

    try:
        tree = cst.parse_module(test_code)
        transformer = TestCallTransformer(func_name, param_mapping)
        modified = tree.visit(transformer)
        return modified.code
    except cst.ParserSyntaxError:
        # Fallback: adapt line by line
        lines = test_code.split("\n")
        adapted_lines = []
        for line in lines:
            adapted_lines.append(_regex_adapt_test(line, param_mapping))
        return "\n".join(adapted_lines)
