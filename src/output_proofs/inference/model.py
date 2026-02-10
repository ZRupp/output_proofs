"""Llama-3.2-3B model wrapper for code generation."""

import logging
from dataclasses import dataclass
from typing import List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..config import ModelConfig
from ..tasks.schema import TaskVariant
from .prompts import PromptBuilder

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    """Result of code generation."""

    prompt: str
    generated_code: str
    full_response: str
    success: bool
    error: Optional[str] = None
    tokens_generated: int = 0


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
        raise ValueError(
            f"Unsupported quantization bits: {quant_config.bits}. Use 4 or 8."
        )


class CodeGenerator:
    """Wrapper for Llama-3.2-3B code generation."""

    def __init__(self, config: ModelConfig = None):
        self.config = config or ModelConfig()
        self.model = None
        self.tokenizer = None
        self.prompt_builder = None
        self._loaded = False

    def load(self) -> None:
        """Load model and tokenizer."""
        if self._loaded:
            return

        logger.info(f"Loading model: {self.config.model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Left-padding is required for correct batched autoregressive generation
        self.tokenizer.padding_side = "left"

        use_gpu = torch.cuda.is_available() and self.config.device != "cpu"

        bnb_config = _build_quantization_config(self.config.quantization)

        if bnb_config is not None:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                attn_implementation="sdpa",
            )
        else:
            dtype = torch.float16 if use_gpu else torch.float32
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                torch_dtype=dtype,
                device_map="auto" if use_gpu else None,
                trust_remote_code=True,
                attn_implementation="sdpa",
            )
            if not use_gpu:
                self.model = self.model.to("cpu")

        self.prompt_builder = PromptBuilder(self.tokenizer)
        self._loaded = True
        logger.info("Model loaded successfully")

    def generate(
        self,
        variant: TaskVariant,
        use_fim: bool = False,
    ) -> GenerationResult:
        """Generate Python code for a task variant.

        Args:
            variant: Task variant to solve
            use_fim: Whether to use FIM format (if supported)

        Returns:
            GenerationResult with generated code
        """
        if not self._loaded:
            self.load()

        try:
            if use_fim and self.prompt_builder.supports_fim():
                prompt = self.prompt_builder.create_fim_prompt(variant)
            else:
                prompt = self.prompt_builder.create_instruction_prompt(variant)

            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
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

            full_response = self.tokenizer.decode(
                outputs[0],
                skip_special_tokens=True,
            )

            # Extract generated code (remove prompt)
            generated = full_response[len(prompt) :].strip()

            # Clean up the generated code
            generated_code = self._extract_code(generated)

            return GenerationResult(
                prompt=prompt,
                generated_code=generated_code,
                full_response=full_response,
                success=True,
                tokens_generated=len(outputs[0]) - len(inputs["input_ids"][0]),
            )

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return GenerationResult(
                prompt=prompt if "prompt" in locals() else "",
                generated_code="",
                full_response="",
                success=False,
                error=str(e),
            )

    def generate_batch(
        self,
        variants: List[TaskVariant],
        use_fim: bool = False,
        batch_size: int = 8,
    ) -> List[GenerationResult]:
        """Generate code for multiple variants using real batched inference.

        Chunks variants into batches, tokenizes each batch together with
        left-padding, and runs a single model.generate() call per batch.
        Falls back to sequential processing if a batch fails.

        Args:
            variants: List of task variants
            use_fim: Whether to use FIM format
            batch_size: Number of variants per batch

        Returns:
            List of generation results
        """
        if not self._loaded:
            self.load()

        results: List[Optional[GenerationResult]] = [None] * len(variants)

        prompts = []
        for variant in variants:
            if use_fim and self.prompt_builder.supports_fim():
                prompts.append(self.prompt_builder.create_fim_prompt(variant))
            else:
                prompts.append(self.prompt_builder.create_instruction_prompt(variant))

        # Process in chunks
        total_batches = (len(variants) + batch_size - 1) // batch_size
        for batch_num, chunk_start in enumerate(range(0, len(variants), batch_size), 1):
            chunk_end = min(chunk_start + batch_size, len(variants))
            chunk_prompts = prompts[chunk_start:chunk_end]
            chunk_indices = list(range(chunk_start, chunk_end))

            logger.info(
                f"  Generation batch [{batch_num}/{total_batches}] "
                f"({chunk_end}/{len(variants)} variants)"
            )

            try:
                chunk_results = self._generate_batch_chunk(chunk_prompts)
                for idx, result in zip(chunk_indices, chunk_results):
                    results[idx] = result
            except Exception as e:
                logger.warning(
                    f"Batch generation failed, falling back to sequential: {e}"
                )
                for idx in chunk_indices:
                    results[idx] = self.generate(variants[idx], use_fim=use_fim)

        return results

    def _generate_batch_chunk(self, prompts: List[str]) -> List[GenerationResult]:
        """Generate results for a single batch chunk of prompts."""
        # Tokenize all prompts together with padding
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(self.model.device)

        input_lengths = (inputs["attention_mask"]).sum(dim=1).tolist()

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
        for i, (prompt, input_len) in enumerate(zip(prompts, input_lengths)):
            full_response = self.tokenizer.decode(
                outputs[i],
                skip_special_tokens=True,
            )
            generated = full_response[len(prompt) :].strip()
            generated_code = self._extract_code(generated)

            tokens_generated = len(outputs[i]) - inputs["input_ids"].shape[1]
            results.append(
                GenerationResult(
                    prompt=prompt,
                    generated_code=generated_code,
                    full_response=full_response,
                    success=True,
                    tokens_generated=max(0, tokens_generated),
                )
            )

        return results

    def _extract_code(self, response: str) -> str:
        """Extract Python code from model response.

        Handles markdown code blocks and raw code.
        """
        # Check for markdown code block
        if "```python" in response:
            start = response.find("```python") + 9
            end = response.find("```", start)
            if end > start:
                return response[start:end].strip()

        if "```" in response:
            start = response.find("```") + 3
            end = response.find("```", start)
            if end > start:
                return response[start:end].strip()

        # Check for function definition
        if "def " in response:
            # Find the function and return everything from there
            start = response.find("def ")
            return response[start:].strip()

        return response.strip()

    def unload(self) -> None:
        """Unload model to free GPU memory.

        Forces garbage collection before clearing the CUDA cache so that
        cyclic references created by device_map="auto" (Accelerate dispatch
        hooks) are broken and GPU tensors are actually deallocated.
        """
        import gc

        self.model = None
        self.tokenizer = None
        self.prompt_builder = None
        self._loaded = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
