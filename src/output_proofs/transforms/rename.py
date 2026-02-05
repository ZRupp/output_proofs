"""Variable renaming transformations using LibCST.

Implements V_noise obfuscation from the adversarial task definition.
"""

import random
from typing import Dict, List, Set, Optional
import libcst as cst
from libcst import matchers as m

from .base import BaseTransform, TransformResult


# Variable noise set from circuitproofs Phase 1
V_NOISE = [
    "__tmp0", "__tmp1", "__tmp2",
    "var_x", "var_y", "var_z",
    "_arg0", "_arg1", "_arg2",
    "_v0", "_v1", "_v2",
]

# Names to never rename (Python builtins, common functions)
PROTECTED_NAMES = {
    # Builtins
    "True", "False", "None",
    "print", "len", "range", "str", "int", "float", "bool", "list", "dict", "set", "tuple",
    "sum", "max", "min", "abs", "round", "sorted", "reversed", "enumerate", "zip", "map", "filter",
    "isinstance", "type", "hasattr", "getattr", "setattr",
    "open", "input", "format",
    # Common imports
    "math", "re", "os", "sys",
    # Control flow
    "return", "if", "else", "elif", "for", "while", "break", "continue",
    "try", "except", "finally", "raise", "assert",
    "and", "or", "not", "in", "is",
    # Class/function keywords
    "def", "class", "self", "cls", "lambda",
    "import", "from", "as",
    # Common patterns
    "result", "output", "ans", "answer",
}


class VariableCollector(cst.CSTVisitor):
    """Collect all variable names from function parameters and assignments."""

    def __init__(self):
        self.param_names: Set[str] = set()
        self.local_names: Set[str] = set()
        self.func_name: Optional[str] = None

    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool:
        """Capture function name."""
        if self.func_name is None:
            self.func_name = node.name.value
        return True

    def visit_Param(self, node: cst.Param) -> bool:
        """Collect parameter names."""
        if node.name:
            self.param_names.add(node.name.value)
        return False

    def visit_Assign(self, node: cst.Assign) -> bool:
        """Collect assignment target names."""
        for target in node.targets:
            if isinstance(target.target, cst.Name):
                self.local_names.add(target.target.value)
        return False

    def visit_AnnAssign(self, node: cst.AnnAssign) -> bool:
        """Collect annotated assignment names."""
        if isinstance(node.target, cst.Name):
            self.local_names.add(node.target.value)
        return False

    def visit_For(self, node: cst.For) -> bool:
        """Collect for loop variable names."""
        if isinstance(node.target, cst.Name):
            self.local_names.add(node.target.value)
        return True


class VariableRenamerTransformer(cst.CSTTransformer):
    """Rename variables according to a mapping."""

    def __init__(self, rename_map: Dict[str, str]):
        self.rename_map = rename_map

    def leave_Name(
        self, original_node: cst.Name, updated_node: cst.Name
    ) -> cst.Name:
        """Rename variable references."""
        if updated_node.value in self.rename_map:
            return updated_node.with_changes(
                value=self.rename_map[updated_node.value]
            )
        return updated_node

    def leave_Param(
        self, original_node: cst.Param, updated_node: cst.Param
    ) -> cst.Param:
        """Rename function parameters."""
        if updated_node.name and updated_node.name.value in self.rename_map:
            new_name = cst.Name(self.rename_map[updated_node.name.value])
            return updated_node.with_changes(name=new_name)
        return updated_node


class VariableRenamer(BaseTransform):
    """Rename variables to noise names (ablate semantic shortcuts)."""

    def __init__(
        self,
        noise_names: List[str] = None,
        rename_params: bool = True,
        rename_locals: bool = True,
        seed: int = None,
    ):
        self.noise_names = noise_names or V_NOISE.copy()
        self.rename_params = rename_params
        self.rename_locals = rename_locals
        self.rng = random.Random(seed)

    @property
    def name(self) -> str:
        return "v_noise_rename"

    def _collect_names(self, tree: cst.Module) -> tuple:
        """Collect variable names from the code."""
        collector = VariableCollector()
        # Use the wrapper to enable visitor pattern
        wrapper = cst.MetadataWrapper(tree)
        wrapper.visit(collector)
        return collector.param_names, collector.local_names, collector.func_name

    def _create_mapping(self, names: Set[str]) -> Dict[str, str]:
        """Create mapping from original names to noise names."""
        mapping = {}
        available = self.noise_names.copy()
        self.rng.shuffle(available)

        for name in sorted(names):  # Sort for determinism
            if name in PROTECTED_NAMES:
                continue
            if available:
                mapping[name] = available.pop(0)
            else:
                # If we run out of noise names, generate indexed versions
                idx = len(mapping)
                mapping[name] = f"_var{idx}"

        return mapping

    def transform(self, code: str) -> TransformResult:
        """Apply variable renaming transformation."""
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

        # Collect names
        param_names, local_names, func_name = self._collect_names(tree)

        # Build mapping
        names_to_rename = set()
        if self.rename_params:
            names_to_rename.update(param_names)
        if self.rename_locals:
            names_to_rename.update(local_names)

        mapping = self._create_mapping(names_to_rename)

        if not mapping:
            return TransformResult(
                original=code,
                transformed=code,
                transform_name=self.name,
                success=True,
            )

        # Apply transformation
        transformer = VariableRenamerTransformer(mapping)
        modified_tree = tree.visit(transformer)
        transformed_code = modified_tree.code

        return TransformResult(
            original=code,
            transformed=transformed_code,
            transform_name=self.name,
            param_mapping=mapping,
        )


def generate_noise_variants(
    code: str,
    num_variants: int = 3,
    seed: int = None,
) -> List[TransformResult]:
    """Generate multiple variants with V_noise variable names.

    Args:
        code: Python source code
        num_variants: Number of variants to generate
        seed: Random seed for reproducibility

    Returns:
        List of TransformResult with different renaming mappings
    """
    variants = []
    base_seed = seed if seed is not None else random.randint(0, 10000)

    for i in range(num_variants):
        renamer = VariableRenamer(seed=base_seed + i)
        result = renamer.transform(code)
        if result.success:
            result.transform_name = f"{result.transform_name}_v{i}"
            variants.append(result)

    return variants
