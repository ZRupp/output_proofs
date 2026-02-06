"""Llama-3.2-3B model wrapper for code generation."""

from dataclasses import dataclass, field
from typing import List, Optional, Union
import logging

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

        # Set padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Use float16 only if GPU is available, otherwise float32 for CPU
        use_gpu = torch.cuda.is_available() and self.config.device != "cpu"
        dtype = torch.float16 if use_gpu else torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=dtype,
            device_map="auto" if use_gpu else None,
            trust_remote_code=True,
            attn_implementation="eager",  # Avoid flash_attn compatibility issues
        )

        # Move to CPU explicitly if not using GPU
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
            # Build prompt
            if use_fim and self.prompt_builder.supports_fim():
                prompt = self.prompt_builder.create_fim_prompt(variant)
            else:
                prompt = self.prompt_builder.create_instruction_prompt(variant)

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
    ) -> List[GenerationResult]:
        """Generate code for multiple variants.

        Args:
            variants: List of task variants
            use_fim: Whether to use FIM format

        Returns:
            List of generation results
        """
        results = []
        for variant in variants:
            result = self.generate(variant, use_fim=use_fim)
            results.append(result)
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
