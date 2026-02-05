"""Transform registry for managing available transformations."""

from typing import Dict, List, Type, Optional
from .base import BaseTransform, CompositeTransform
from .rename import VariableRenamer
from .reorder import ParameterReorderer


class TransformRegistry:
    """Registry for code transformation classes."""

    _transforms: Dict[str, Type[BaseTransform]] = {}

    @classmethod
    def register(cls, name: str, transform_cls: Type[BaseTransform]) -> None:
        """Register a transform class."""
        cls._transforms[name] = transform_cls

    @classmethod
    def get(cls, name: str) -> Optional[Type[BaseTransform]]:
        """Get a transform class by name."""
        return cls._transforms.get(name)

    @classmethod
    def list_transforms(cls) -> List[str]:
        """List all registered transform names."""
        return list(cls._transforms.keys())

    @classmethod
    def create(cls, name: str, **kwargs) -> Optional[BaseTransform]:
        """Create a transform instance by name."""
        transform_cls = cls.get(name)
        if transform_cls:
            return transform_cls(**kwargs)
        return None


# Register built-in transforms
TransformRegistry.register("v_noise_rename", VariableRenamer)
TransformRegistry.register("param_reorder", ParameterReorderer)


def get_default_transforms(seed: int = None) -> List[BaseTransform]:
    """Get the default set of transforms for adversarial variant generation.

    Args:
        seed: Random seed for reproducibility

    Returns:
        List of transform instances
    """
    return [
        VariableRenamer(seed=seed),
        ParameterReorderer(seed=seed),
    ]


def create_composite_transform(
    transform_names: List[str],
    seed: int = None,
) -> CompositeTransform:
    """Create a composite transform from a list of transform names.

    Args:
        transform_names: List of transform names to compose
        seed: Random seed for reproducibility

    Returns:
        CompositeTransform that applies all transforms in sequence
    """
    transforms = []
    for name in transform_names:
        transform = TransformRegistry.create(name, seed=seed)
        if transform:
            transforms.append(transform)

    return CompositeTransform(transforms)
