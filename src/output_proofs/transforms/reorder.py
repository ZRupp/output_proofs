"""Parameter reordering transformations using LibCST."""

import random
from typing import Dict, List, Tuple, Optional
import libcst as cst

from .base import BaseTransform, TransformResult


class FunctionSignatureCollector(cst.CSTVisitor):
    """Collect function signature information."""

    def __init__(self):
        self.func_name: Optional[str] = None
        self.params: List[str] = []
        self.has_star_args: bool = False
        self.has_kwargs: bool = False

    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool:
        if self.func_name is None:
            self.func_name = node.name.value
            # Check for *args and **kwargs
            params = node.params
            if params.star_arg and not isinstance(params.star_arg, cst.MaybeSentinel):
                self.has_star_args = True
            if params.star_kwarg:
                self.has_kwargs = True
        return True

    def visit_Param(self, node: cst.Param) -> bool:
        if node.name:
            self.params.append(node.name.value)
        return False


class ParameterReorderTransformer(cst.CSTTransformer):
    """Reorder function parameters according to a permutation."""

    def __init__(self, permutation: List[int]):
        self.permutation = permutation
        self.original_order: List[str] = []
        self.reordered_params: List[str] = []

    def leave_Parameters(
        self, original_node: cst.Parameters, updated_node: cst.Parameters
    ) -> cst.Parameters:
        """Reorder the parameters list."""
        params = list(updated_node.params)

        if len(params) < 2:
            return updated_node

        # Only reorder regular params (not *args, **kwargs)
        if len(self.permutation) != len(params):
            return updated_node

        # Store original order for mapping
        self.original_order = [p.name.value for p in params if p.name]

        # Apply permutation
        reordered = [params[i] for i in self.permutation]
        self.reordered_params = [p.name.value for p in reordered if p.name]

        # Handle trailing commas
        new_params = []
        for i, param in enumerate(reordered):
            if i < len(reordered) - 1:
                # Add comma after all but last
                new_params.append(param.with_changes(
                    comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" "))
                ))
            else:
                # Last param has no comma
                new_params.append(param.with_changes(comma=cst.MaybeSentinel.DEFAULT))

        return updated_node.with_changes(params=new_params)


class CallArgumentReorderTransformer(cst.CSTTransformer):
    """Reorder function call arguments to match new parameter order."""

    def __init__(self, func_name: str, param_mapping: Dict[str, int]):
        self.func_name = func_name
        self.param_mapping = param_mapping  # param_name -> new_position

    def leave_Call(
        self, original_node: cst.Call, updated_node: cst.Call
    ) -> cst.Call:
        """Reorder arguments in function calls."""
        # Check if this is a call to our target function
        if isinstance(updated_node.func, cst.Name):
            if updated_node.func.value != self.func_name:
                return updated_node
        else:
            return updated_node

        args = list(updated_node.args)
        if not args:
            return updated_node

        # Separate positional and keyword args
        positional = [a for a in args if a.keyword is None]
        keyword = [a for a in args if a.keyword is not None]

        # Reorder positional args
        if len(positional) > 1 and len(positional) <= len(self.param_mapping):
            # Create inverse mapping (new_pos -> old_pos)
            new_order = sorted(self.param_mapping.items(), key=lambda x: x[1])
            reordered_pos = []
            for _, new_pos in new_order:
                if new_pos < len(positional):
                    reordered_pos.append(positional[new_pos])

            # Combine reordered positional with keyword args
            args = reordered_pos + keyword

            # Fix commas
            new_args = []
            for i, arg in enumerate(args):
                if i < len(args) - 1:
                    new_args.append(arg.with_changes(
                        comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" "))
                    ))
                else:
                    new_args.append(arg.with_changes(comma=cst.MaybeSentinel.DEFAULT))

            return updated_node.with_changes(args=new_args)

        return updated_node


class ParameterReorderer(BaseTransform):
    """Reorder function parameters to test positional argument handling."""

    def __init__(self, permutation: List[int] = None, seed: int = None):
        self.permutation = permutation
        self.rng = random.Random(seed)

    @property
    def name(self) -> str:
        return "param_reorder"

    def _generate_permutation(self, num_params: int) -> List[int]:
        """Generate a random permutation of parameter indices."""
        perm = list(range(num_params))
        self.rng.shuffle(perm)
        return perm

    def _collect_signature(self, tree: cst.Module) -> Tuple[str, List[str], bool]:
        """Collect function signature info."""
        collector = FunctionSignatureCollector()
        # Use MetadataWrapper to enable visitor pattern
        wrapper = cst.MetadataWrapper(tree)
        wrapper.visit(collector)
        can_reorder = not collector.has_star_args and not collector.has_kwargs
        return collector.func_name, collector.params, can_reorder

    def can_apply(self, code: str) -> bool:
        """Check if parameter reordering is possible."""
        try:
            tree = cst.parse_module(code)
            _, params, can_reorder = self._collect_signature(tree)
            return can_reorder and len(params) >= 2
        except cst.ParserSyntaxError:
            return False

    def transform(self, code: str) -> TransformResult:
        """Apply parameter reordering transformation."""
        try:
            tree = cst.parse_module(code)
        except cst.ParserSyntaxError as e:
            return TransformResult(
                original=code,
                transformed=code,
                transform_name=self.name,
                success=False,
                error=f"Parse error: {e}",
            )

        func_name, params, can_reorder = self._collect_signature(tree)

        if not can_reorder or len(params) < 2:
            return TransformResult(
                original=code,
                transformed=code,
                transform_name=self.name,
                success=True,
            )

        # Generate or use provided permutation
        perm = self.permutation if self.permutation else self._generate_permutation(len(params))

        # Validate permutation
        if len(perm) != len(params):
            perm = self._generate_permutation(len(params))

        # Apply transformation
        transformer = ParameterReorderTransformer(perm)
        modified_tree = tree.visit(transformer)

        # Build mapping
        param_mapping = {}
        for old_idx, new_idx in enumerate(perm):
            if old_idx < len(params):
                param_mapping[params[old_idx]] = new_idx

        return TransformResult(
            original=code,
            transformed=modified_tree.code,
            transform_name=self.name,
            param_mapping=param_mapping,
        )


def generate_reorder_variants(
    code: str,
    num_variants: int = 3,
    seed: int = None,
) -> List[TransformResult]:
    """Generate multiple variants with different parameter orderings.

    Args:
        code: Python source code
        num_variants: Number of variants to generate
        seed: Random seed for reproducibility

    Returns:
        List of TransformResult with different orderings
    """
    variants = []
    base_seed = seed if seed is not None else random.randint(0, 10000)

    for i in range(num_variants):
        reorderer = ParameterReorderer(seed=base_seed + i)
        if reorderer.can_apply(code):
            result = reorderer.transform(code)
            if result.success and result.transformed != code:
                result.transform_name = f"{result.transform_name}_v{i}"
                variants.append(result)

    return variants
