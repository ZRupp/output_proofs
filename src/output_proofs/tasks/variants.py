"""Generate adversarial task variants using LibCST transformations."""

import re
import random
from typing import List, Dict, Optional
from dataclasses import dataclass

from .schema import CombinedTask, TaskVariant
from ..transforms.base import BaseTransform, TransformResult
from ..transforms.rename import VariableRenamer, V_NOISE
from ..transforms.reorder import ParameterReorderer
from ..transforms.test_adapter import adapt_tests_for_variant


def extract_function_info(code: str) -> Dict[str, str]:
    """Extract function name and parameters from Python code.

    Returns:
        Dict with 'name', 'params', 'docstring'
    """
    # Match function definition
    match = re.search(
        r'def\s+(\w+)\s*\(([^)]*)\)\s*:(?:\s*(?:"""([^"]*)"""|\'\'\'([^\']*)\'\'\'))?',
        code,
        re.DOTALL
    )

    if match:
        return {
            "name": match.group(1),
            "params": match.group(2).strip(),
            "docstring": match.group(3) or match.group(4) or "",
        }

    return {"name": "", "params": "", "docstring": ""}


def transform_description(
    description: str,
    param_mapping: Dict[str, str],
) -> str:
    """Transform task description to use noisy parameter names.

    Args:
        description: Original task description
        param_mapping: Mapping from original param names to noisy names (str -> str only)

    Returns:
        Description with parameter names replaced
    """
    result = description
    for old_name, new_name in param_mapping.items():
        # Only process string -> string mappings (skip reorder int mappings)
        if isinstance(new_name, str):
            # Replace whole word occurrences
            result = re.sub(rf'\b{re.escape(old_name)}\b', new_name, result)
    return result


class VariantGenerator:
    """Generate adversarial variants of tasks."""

    def __init__(
        self,
        transforms: List[BaseTransform] = None,
        seed: int = None,
    ):
        self.rng = random.Random(seed)
        self.seed = seed

        # Default transforms
        if transforms is None:
            transforms = [
                VariableRenamer(seed=seed),
                ParameterReorderer(seed=seed),
            ]
        self.transforms = transforms

    def generate_variants(
        self,
        task: CombinedTask,
        num_variants: int = 5,
    ) -> List[TaskVariant]:
        """Generate adversarial variants for a task.

        Args:
            task: Combined MBPP + Verina task
            num_variants: Number of variants to generate

        Returns:
            List of TaskVariant objects
        """
        variants = []
        reference_code = task.mbpp.code
        func_info = extract_function_info(reference_code)

        for i in range(num_variants):
            # Use different seed for each variant
            variant_seed = (self.seed or 0) + i

            # Randomly select which transforms to apply
            applied_transforms = []
            combined_mapping = {}

            for transform in self.transforms:
                # Set seed for this specific variant
                if hasattr(transform, 'rng'):
                    transform.rng = random.Random(variant_seed + len(applied_transforms))

                if transform.can_apply(reference_code):
                    result = transform.transform(reference_code)
                    if result.success and result.transformed != reference_code:
                        reference_code = result.transformed
                        applied_transforms.append(transform.name)
                        combined_mapping.update(result.param_mapping)

            # If no transforms were applied, create a basic V_noise variant
            if not applied_transforms:
                renamer = VariableRenamer(seed=variant_seed, rename_params=True)
                result = renamer.transform(task.mbpp.code)
                if result.success:
                    reference_code = result.transformed
                    applied_transforms.append(result.transform_name)
                    combined_mapping = result.param_mapping

            # Extract new function info
            new_func_info = extract_function_info(reference_code)

            # Transform description
            transformed_desc = transform_description(
                task.mbpp.text,
                combined_mapping,
            )

            # Filter param_mapping to only include string -> string mappings
            # (exclude reorder mappings which are string -> int)
            str_mapping = {k: v for k, v in combined_mapping.items() if isinstance(v, str)}

            variant = TaskVariant(
                original_task=task,
                variant_id=f"{task.task_id}_v{i}",
                transformed_description=transformed_desc,
                transformed_signature=f"def {new_func_info['name']}({new_func_info['params']}):",
                func_name=new_func_info['name'],
                noisy_params=new_func_info['params'],
                transforms_applied=applied_transforms,
                param_mapping=str_mapping,
                reverse_mapping={v: k for k, v in str_mapping.items()},
            )
            variants.append(variant)

            # Reset reference code for next variant
            reference_code = task.mbpp.code

        return variants

    def get_adapted_tests(
        self,
        variant: TaskVariant,
    ) -> List[str]:
        """Get test assertions adapted for a variant's parameter names.

        Args:
            variant: The task variant

        Returns:
            List of adapted test assertion strings
        """
        original_tests = variant.original_task.mbpp.test_list
        return adapt_tests_for_variant(
            original_tests,
            variant.func_name,
            variant.param_mapping,
        )


def generate_all_variants(
    tasks: List[CombinedTask],
    num_variants_per_task: int = 5,
    seed: int = None,
) -> List[TaskVariant]:
    """Generate adversarial variants for all tasks.

    Args:
        tasks: List of combined tasks
        num_variants_per_task: Number of variants per task
        seed: Random seed for reproducibility

    Returns:
        List of all generated variants
    """
    generator = VariantGenerator(seed=seed)
    all_variants = []

    for task in tasks:
        variants = generator.generate_variants(task, num_variants_per_task)
        all_variants.extend(variants)

    return all_variants
