"""Prompt templates for code generation.

Implements both instruction-tuned and FIM (Fill-In-The-Middle) formats.
Handles Warning B: Tokenizer Compatibility for Llama-3.2.
"""

from dataclasses import dataclass

from ..tasks.schema import TaskVariant

# Llama 3 instruction format tokens
LLAMA3_BOS = "<|begin_of_text|>"
LLAMA3_USER_START = "<|start_header_id|>user<|end_header_id|>"
LLAMA3_ASSISTANT_START = "<|start_header_id|>assistant<|end_header_id|>"
LLAMA3_EOT = "<|eot_id|>"

# FIM tokens (for code-specialized models)
FIM_PREFIX = "<|fim_prefix|>"
FIM_SUFFIX = "<|fim_suffix|>"
FIM_MIDDLE = "<|fim_middle|>"


@dataclass
class FIMTokens:
    """FIM special tokens for a model."""

    prefix: str
    suffix: str
    middle: str
    supported: bool = True


def get_fim_tokens(tokenizer) -> FIMTokens:
    """Get FIM tokens for a tokenizer.

    Different models use different FIM token conventions.
    Returns appropriate tokens or indicates FIM is not supported.
    """
    # Check for common FIM token patterns
    vocab = tokenizer.get_vocab() if hasattr(tokenizer, "get_vocab") else {}

    # CodeLlama / Llama 3 code format
    if "<|fim_prefix|>" in vocab:
        return FIMTokens(
            prefix="<|fim_prefix|>",
            suffix="<|fim_suffix|>",
            middle="<|fim_middle|>",
        )

    # StarCoder format
    if "<fim_prefix>" in vocab:
        return FIMTokens(
            prefix="<fim_prefix>",
            suffix="<fim_suffix>",
            middle="<fim_middle>",
        )

    # Codex format
    if "<|fim▁begin|>" in vocab:
        return FIMTokens(
            prefix="<|fim▁begin|>",
            suffix="<|fim▁hole|>",
            middle="<|fim▁end|>",
        )

    # FIM not supported
    return FIMTokens(
        prefix="",
        suffix="",
        middle="",
        supported=False,
    )


class PromptBuilder:
    """Build prompts for code generation models."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.fim_tokens = get_fim_tokens(tokenizer)

    def supports_fim(self) -> bool:
        """Check if the model supports FIM format."""
        return self.fim_tokens.supported

    def create_instruction_prompt(self, variant: TaskVariant) -> str:
        """Create instruction-tuned prompt for Python code generation.

        Uses Llama 3 chat format for instruction-tuned models.
        """
        description = variant.transformed_description
        func_name = variant.func_name
        params = variant.noisy_params

        # Build the prompt
        prompt = f"""{LLAMA3_BOS}{LLAMA3_USER_START}

Write a Python function to solve the following task:

{description}

The function signature should be:
def {func_name}({params}):

Write only the function implementation, no explanations.

{LLAMA3_EOT}{LLAMA3_ASSISTANT_START}

```python
def {func_name}({params}):
"""
        return prompt

    def create_fim_prompt(self, variant: TaskVariant) -> str:
        """Create FIM prompt for Python code generation.

        Warning A: Leave suffix EMPTY - let model generate its own return statement.
        """
        if not self.supports_fim():
            # Fall back to instruction format
            return self.create_instruction_prompt(variant)

        description = variant.transformed_description
        func_name = variant.func_name
        params = variant.noisy_params

        # Build prefix: function signature and docstring
        prefix = f'''def {func_name}({params}):
    """{description}"""
'''

        # Empty suffix - let model generate complete function (Warning A)
        suffix = ""

        return f"{self.fim_tokens.prefix}{prefix}{self.fim_tokens.suffix}{suffix}{self.fim_tokens.middle}"

    def create_completion_prompt(self, variant: TaskVariant) -> str:
        """Create a simple completion prompt (non-chat format).

        For base models that aren't instruction-tuned.
        """
        description = variant.transformed_description
        func_name = variant.func_name
        params = variant.noisy_params

        return f'''# {description}

def {func_name}({params}):
    """
    {description}
    """
'''


def create_instruction_prompt(variant: TaskVariant) -> str:
    """Standalone function to create instruction prompt.

    Uses standard Llama 3 format.
    """
    description = variant.transformed_description
    func_name = variant.func_name
    params = variant.noisy_params

    return f"""{LLAMA3_BOS}{LLAMA3_USER_START}

Write a Python function to solve the following task:

{description}

The function signature should be:
def {func_name}({params}):

Write only the function implementation, no explanations.

{LLAMA3_EOT}{LLAMA3_ASSISTANT_START}

```python
def {func_name}({params}):
"""


def create_fim_prompt(
    variant: TaskVariant,
    fim_tokens: FIMTokens,
) -> str:
    """Standalone function to create FIM prompt.

    Warning A: Empty suffix lets model generate complete function.
    """
    description = variant.transformed_description
    func_name = variant.func_name
    params = variant.noisy_params

    prefix = f'''def {func_name}({params}):
    """{description}"""
'''
    suffix = ""  # Empty - Warning A

    return f"{fim_tokens.prefix}{prefix}{fim_tokens.suffix}{suffix}{fim_tokens.middle}"
