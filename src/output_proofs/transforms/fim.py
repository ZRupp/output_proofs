"""FIM (Fill-In-The-Middle) template generation for code generation.

Implements the FIM formatting from the adversarial task definition.
Uses V_noise variable names to ablate semantic shortcuts.
"""

from dataclasses import dataclass
from typing import Dict, List

from ..tasks.schema import TaskVariant


@dataclass
class FIMTemplate:
    """A Fill-In-The-Middle template for code generation."""

    prefix: str  # Code before the hole
    suffix: str  # Code after the hole (empty per Warning A)
    description: str  # Task description
    func_name: str  # Function name
    params: str  # Parameter string
    param_mapping: Dict[str, str]  # Original -> noisy param names


def create_fim_template(
    variant: TaskVariant,
    include_docstring: bool = True,
) -> FIMTemplate:
    """Create a FIM template from a task variant.

    Args:
        variant: Task variant with transformed description and params
        include_docstring: Whether to include docstring in prefix

    Returns:
        FIMTemplate ready for model input
    """
    func_name = variant.func_name
    params = variant.noisy_params
    description = variant.transformed_description

    # Build prefix
    if include_docstring:
        prefix = f'''def {func_name}({params}):
    """{description}"""
'''
    else:
        prefix = f"""def {func_name}({params}):
"""

    # Empty suffix per Warning A
    suffix = ""

    return FIMTemplate(
        prefix=prefix,
        suffix=suffix,
        description=description,
        func_name=func_name,
        params=params,
        param_mapping=variant.param_mapping,
    )


def format_for_model(
    template: FIMTemplate,
    fim_prefix_token: str = "<|fim_prefix|>",
    fim_suffix_token: str = "<|fim_suffix|>",
    fim_middle_token: str = "<|fim_middle|>",
) -> str:
    """Format a FIM template with model-specific tokens.

    Args:
        template: FIM template
        fim_prefix_token: Token marking start of prefix
        fim_suffix_token: Token marking start of suffix
        fim_middle_token: Token marking where to fill in

    Returns:
        Formatted string ready for tokenization
    """
    return (
        f"{fim_prefix_token}{template.prefix}{fim_suffix_token}{template.suffix}{fim_middle_token}"
    )


def create_instruction_from_template(
    template: FIMTemplate,
    bos_token: str = "<|begin_of_text|>",
    user_header: str = "<|start_header_id|>user<|end_header_id|>",
    assistant_header: str = "<|start_header_id|>assistant<|end_header_id|>",
    eot_token: str = "<|eot_id|>",
) -> str:
    """Create instruction-tuned prompt from FIM template.

    Fallback for models that don't support FIM format.

    Args:
        template: FIM template
        bos_token: Beginning of sequence token
        user_header: User message header
        assistant_header: Assistant message header
        eot_token: End of turn token

    Returns:
        Instruction-formatted prompt string
    """
    return f"""{bos_token}{user_header}

Write a Python function to solve the following task:

{template.description}

The function signature should be:
def {template.func_name}({template.params}):

Write only the function implementation, no explanations.

{eot_token}{assistant_header}

```python
def {template.func_name}({template.params}):
"""


def batch_create_templates(
    variants: List[TaskVariant],
) -> List[FIMTemplate]:
    """Create FIM templates for a batch of variants.

    Args:
        variants: List of task variants

    Returns:
        List of FIM templates
    """
    return [create_fim_template(v) for v in variants]
