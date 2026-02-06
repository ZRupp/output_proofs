"""Translate Python code to Lean 4 using Goedel-Formalizer-V2-8B."""

from dataclasses import dataclass, field
from typing import Optional, List
import logging
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..config import FormalizerConfig
from ..tasks.schema import CombinedTask, TaskVariant

logger = logging.getLogger(__name__)


@dataclass
class TranslationResult:
    """Result of Python to Lean translation."""

    python_code: str
    lean_code: str
    success: bool
    error: Optional[str] = None
    prompt_used: str = ""


class GoedelFormalizer:
    """Translate Python to Lean 4 using Goedel-Formalizer-V2-8B."""

    def __init__(self, config: FormalizerConfig = None):
        self.config = config or FormalizerConfig()
        self.model = None
        self.tokenizer = None
        self._loaded = False

    def load(self) -> None:
        """Load model and tokenizer."""
        if self._loaded:
            return

        logger.info(f"Loading Goedel-Formalizer: {self.config.model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Use float16 only if GPU is available, otherwise float32 for CPU
        use_gpu = torch.cuda.is_available() and self.config.device != "cpu"
        dtype = torch.float16 if use_gpu else torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=dtype,
            attn_implementation="eager",  # Avoid flash_attn compatibility issues
            device_map="auto" if use_gpu else None,
            trust_remote_code=True,
        )

        # Move to CPU explicitly if not using GPU
        if not use_gpu:
            self.model = self.model.to("cpu")

        self._loaded = True
        logger.info("Goedel-Formalizer loaded successfully")

    def translate(
        self,
        python_code: str,
        task: CombinedTask,
    ) -> TranslationResult:
        """Translate Python code to Lean 4.

        Args:
            python_code: Python code to translate
            task: Combined task with Verina spec for guidance

        Returns:
            TranslationResult with Lean code
        """
        if not self._loaded:
            self.load()

        try:
            # Build prompt
            prompt = self._build_prompt(python_code, task)

            # Tokenize
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
            ).to(self.model.device)

            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    do_sample=self.config.do_sample,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            # Decode
            response = self.tokenizer.decode(
                outputs[0],
                skip_special_tokens=True,
            )

            # Extract Lean code
            lean_code = self._extract_lean_code(response)

            if not lean_code:
                return TranslationResult(
                    python_code=python_code,
                    lean_code="",
                    success=False,
                    error="Failed to extract Lean code from response",
                    prompt_used=prompt,
                )

            return TranslationResult(
                python_code=python_code,
                lean_code=lean_code,
                success=True,
                prompt_used=prompt,
            )

        except Exception as e:
            logger.error(f"Translation failed: {e}")
            return TranslationResult(
                python_code=python_code,
                lean_code="",
                success=False,
                error=str(e),
            )

    def translate_with_variant(
        self,
        python_code: str,
        variant: TaskVariant,
    ) -> TranslationResult:
        """Translate Python code using variant's task context.

        Args:
            python_code: Generated Python code
            variant: Task variant with original task info

        Returns:
            TranslationResult with Lean code
        """
        return self.translate(python_code, variant.original_task)

    def _build_prompt(self, python_code: str, task: CombinedTask) -> str:
        """Build prompt for Goedel-Formalizer.

        Includes Verina signature for guidance on expected Lean structure.
        """
        # Get Lean signature from Verina spec if available
        lean_signature = task.verina.signature if task.verina.signature else ""

        prompt = f"""Translate the following Python code to Lean 4:

Python:
```python
{python_code}
```

"""
        if lean_signature:
            prompt += f"""The Lean function should match this signature:
{lean_signature}

"""

        prompt += """Lean 4:
```lean
"""
        return prompt

    def _extract_lean_code(self, response: str) -> str:
        """Extract Lean code from model response.

        Handles markdown code blocks and raw Lean code.
        """
        # Try markdown code block
        if "```lean" in response:
            match = re.search(r"```lean\n?(.*?)```", response, re.DOTALL)
            if match:
                return match.group(1).strip()

        # Try generic code block
        if "```" in response:
            # Find the last code block (likely the output)
            blocks = re.findall(r"```(?:\w*\n)?(.*?)```", response, re.DOTALL)
            if blocks:
                # Return the last block that looks like Lean
                for block in reversed(blocks):
                    if "def " in block or "theorem " in block or ":=" in block:
                        return block.strip()
                # If no Lean-looking block, return the last one
                return blocks[-1].strip()

        # Try to find Lean code directly
        if "def " in response and ":=" in response:
            # Find function definition
            match = re.search(r"(def\s+\w+.*?)(?=\n\n|\Z)", response, re.DOTALL)
            if match:
                return match.group(1).strip()

        return ""

    def translate_batch(
        self,
        python_codes: List[str],
        tasks: List[CombinedTask],
    ) -> List[TranslationResult]:
        """Translate multiple Python codes to Lean.

        Args:
            python_codes: List of Python codes
            tasks: Corresponding tasks

        Returns:
            List of translation results
        """
        if len(python_codes) != len(tasks):
            raise ValueError("Number of codes and tasks must match")

        results = []
        for code, task in zip(python_codes, tasks):
            result = self.translate(code, task)
            results.append(result)

        return results

    def unload(self) -> None:
        """Unload model to free memory."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        self._loaded = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
