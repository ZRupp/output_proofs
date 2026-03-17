"""Translate Python code to Lean 4 using Goedel-Formalizer-V2-8B."""

import logging
import os
import re
from dataclasses import dataclass
from typing import List, Optional

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


def _build_quantization_config(quant_config):
    """Build a BitsAndBytesConfig from our QuantizationConfig.

    Returns None if quantization is disabled.
    """
    if not quant_config.enabled:
        return None

    try:
        from transformers import BitsAndBytesConfig
    except ImportError:
        raise ImportError(
            "bitsandbytes is required for quantization. "
            "Install with: pip install output_proofs[gpu]"
        )

    if quant_config.bits == 8:
        return BitsAndBytesConfig(load_in_8bit=True)
    elif quant_config.bits == 4:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=quant_config.double_quant,
            bnb_4bit_quant_type=quant_config.quant_type,
            bnb_4bit_compute_dtype=torch.float16,
        )
    else:
        raise ValueError(f"Unsupported quantization bits: {quant_config.bits}. Use 4 or 8.")


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

        hf_token = os.environ.get("HF_TOKEN") or None

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
            token=hf_token,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Left-padding is required for correct batched autoregressive generation
        self.tokenizer.padding_side = "left"

        use_gpu = torch.cuda.is_available() and self.config.device != "cpu"

        # Build quantization config if enabled
        bnb_config = _build_quantization_config(self.config.quantization)

        if bnb_config is not None:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                attn_implementation="sdpa",
                token=hf_token,
            )
        else:
            dtype = torch.float16 if use_gpu else torch.float32
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                torch_dtype=dtype,
                attn_implementation="sdpa",
                device_map="auto" if use_gpu else None,
                trust_remote_code=True,
                token=hf_token,
            )
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
        batch_size: int = 4,
    ) -> List[TranslationResult]:
        """Translate multiple Python codes to Lean using real batched inference.

        Chunks inputs into batches, tokenizes together with left-padding,
        and runs a single model.generate() call per batch.
        Falls back to sequential processing if a batch fails.

        Args:
            python_codes: List of Python codes
            tasks: Corresponding tasks
            batch_size: Number of items per batch

        Returns:
            List of translation results
        """
        if len(python_codes) != len(tasks):
            raise ValueError("Number of codes and tasks must match")

        if not self._loaded:
            self.load()

        results: List[Optional[TranslationResult]] = [None] * len(python_codes)

        # Build all prompts upfront
        prompts = [self._build_prompt(code, task) for code, task in zip(python_codes, tasks)]

        # Process in chunks
        total_batches = (len(python_codes) + batch_size - 1) // batch_size
        for batch_num, chunk_start in enumerate(range(0, len(python_codes), batch_size), 1):
            chunk_end = min(chunk_start + batch_size, len(python_codes))
            chunk_prompts = prompts[chunk_start:chunk_end]
            chunk_codes = python_codes[chunk_start:chunk_end]
            chunk_indices = list(range(chunk_start, chunk_end))

            logger.info(
                f"  Translation batch [{batch_num}/{total_batches}] "
                f"({chunk_end}/{len(python_codes)} variants)"
            )

            try:
                chunk_results = self._translate_batch_chunk(chunk_prompts, chunk_codes)
                for idx, result in zip(chunk_indices, chunk_results):
                    results[idx] = result
            except Exception as e:
                logger.warning(f"Batch translation failed, falling back to sequential: {e}")
                for idx in chunk_indices:
                    results[idx] = self.translate(python_codes[idx], tasks[idx])

        return results

    def _translate_batch_chunk(
        self, prompts: List[str], python_codes: List[str]
    ) -> List[TranslationResult]:
        """Translate a single batch chunk of prompts."""
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(self.model.device)

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

        results = []
        for i, (prompt, python_code) in enumerate(zip(prompts, python_codes)):
            response = self.tokenizer.decode(outputs[i], skip_special_tokens=True)
            lean_code = self._extract_lean_code(response)

            if not lean_code:
                results.append(
                    TranslationResult(
                        python_code=python_code,
                        lean_code="",
                        success=False,
                        error="Failed to extract Lean code from response",
                        prompt_used=prompt,
                    )
                )
            else:
                results.append(
                    TranslationResult(
                        python_code=python_code,
                        lean_code=lean_code,
                        success=True,
                        prompt_used=prompt,
                    )
                )

        return results

    def unload(self) -> None:
        """Unload model to free GPU memory.

        Forces garbage collection before clearing the CUDA cache so that
        cyclic references created by device_map="auto" (Accelerate dispatch
        hooks) are broken and GPU tensors are actually deallocated.
        """
        import gc

        self.model = None
        self.tokenizer = None
        self._loaded = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
