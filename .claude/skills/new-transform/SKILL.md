# New Transform

Create a new LibCST code transformation following the project's registry pattern.

## Instructions

When the user wants to create a new transform:

1. **Choose a name** — ask the user for the transform name (e.g. `docstring_strip`, `type_hint_remove`)

2. **Create the file** at `src/output_proofs/transforms/{name}.py` using this template:

```python
"""<Description> transformation using LibCST."""

from typing import Dict, Optional
import libcst as cst

from .base import BaseTransform, TransformResult


class <ClassName>(BaseTransform):
    """<Description>."""

    def __init__(self, seed: int = None):
        # Add any configuration here
        pass

    @property
    def name(self) -> str:
        return "<transform_name>"

    def can_apply(self, code: str) -> bool:
        """Check if this transform is applicable."""
        # Add validation logic
        return bool(code.strip())

    def transform(self, code: str) -> TransformResult:
        """Apply the transformation."""
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

        # Implement your LibCST transformer here
        # transformer = YourTransformer()
        # modified_tree = tree.visit(transformer)

        return TransformResult(
            original=code,
            transformed=modified_tree.code,
            transform_name=self.name,
        )
```

3. **Register the transform** — add an import in `src/output_proofs/transforms/__init__.py` and include it in the registry if one exists

4. **Add tests** in `tests/test_transforms.py` or a new test file

5. **Verify** — run `pytest tests/ -v -k transform`

## Key Patterns

- All transforms inherit from `BaseTransform` (in `transforms/base.py`)
- `TransformResult` carries `param_mapping` for variable renames and `reverse_mapping` for undo
- Use `CompositeTransform` to chain multiple transforms together
- See `transforms/rename.py` (`VariableRenamer`) for a complete example with LibCST visitors/transformers
- Protected names in `PROTECTED_NAMES` should never be renamed

## Reference Files

- `src/output_proofs/transforms/base.py` — `BaseTransform`, `TransformResult`, `CompositeTransform`
- `src/output_proofs/transforms/rename.py` — `VariableRenamer` (full example)
