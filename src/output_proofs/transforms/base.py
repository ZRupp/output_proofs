"""Base classes for code transformations."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class TransformResult:
    """Result of a code transformation."""

    original: str
    transformed: str
    transform_name: str
    param_mapping: Dict[str, str] = field(default_factory=dict)  # original -> transformed
    reverse_mapping: Dict[str, str] = field(default_factory=dict)  # transformed -> original
    success: bool = True
    error: Optional[str] = None

    def __post_init__(self):
        # Build reverse mapping if not provided
        if self.param_mapping and not self.reverse_mapping:
            self.reverse_mapping = {v: k for k, v in self.param_mapping.items()}


class BaseTransform(ABC):
    """Abstract base class for code transformations."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of this transformation."""
        pass

    @abstractmethod
    def transform(self, code: str) -> TransformResult:
        """Apply transformation to code.

        Args:
            code: Python source code to transform

        Returns:
            TransformResult with original and transformed code
        """
        pass

    def can_apply(self, code: str) -> bool:
        """Check if this transform can be applied to the given code.

        Override in subclasses for specific validation.
        """
        return bool(code.strip())


class CompositeTransform(BaseTransform):
    """Compose multiple transforms together."""

    def __init__(self, transforms: List[BaseTransform]):
        self.transforms = transforms

    @property
    def name(self) -> str:
        return "composite_" + "_".join(t.name for t in self.transforms)

    def transform(self, code: str) -> TransformResult:
        """Apply all transforms in sequence."""
        current = code
        all_mappings = {}
        transform_names = []

        for t in self.transforms:
            if t.can_apply(current):
                result = t.transform(current)
                if result.success:
                    current = result.transformed
                    all_mappings.update(result.param_mapping)
                    transform_names.append(t.name)

        return TransformResult(
            original=code,
            transformed=current,
            transform_name="_".join(transform_names),
            param_mapping=all_mappings,
        )
