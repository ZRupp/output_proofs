"""End-to-end benchmark pipeline orchestration."""

import asyncio
import logging
from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path

from .config import PipelineConfig, ModelConfig, FormalizerConfig, PathConfig
from .tasks.loader import load_combined_tasks, load_mbpp_only
from .tasks.schema import CombinedTask, TaskVariant
from .tasks.variants import VariantGenerator, generate_all_variants
from .inference.model import CodeGenerator, GenerationResult
from .inference.prompts import PromptBuilder
from .translation.goedel_formalizer import GoedelFormalizer, TranslationResult
from .verification.lean_verifier import LeanVerifier, VerificationResult
from .reporting.report import BenchmarkReport, aggregate_results, generate_report

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Result from a single pipeline run for one variant."""

    variant: TaskVariant
    generation_result: GenerationResult
    translation_result: Optional[TranslationResult]
    verification_result: Optional[VerificationResult]

    @property
    def success(self) -> bool:
        """Full pipeline success."""
        return (
            self.generation_result.success
            and self.translation_result is not None
            and self.translation_result.success
            and self.verification_result is not None
            and self.verification_result.success
        )


class BenchmarkPipeline:
    """End-to-end benchmark pipeline.

    Orchestrates:
    1. Loading MBPP tasks and Verina specs
    2. Generating adversarial variants (LibCST transforms)
    3. Running Llama-3.2-3B inference (Python code generation)
    4. Translating Python to Lean (Goedel-Formalizer)
    5. Verifying Lean code (Verina)
    6. Aggregating results
    """

    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig()
        self.variant_generator = VariantGenerator()
        self.code_generator: Optional[CodeGenerator] = None
        self.translator: Optional[GoedelFormalizer] = None
        self.verifier: Optional[LeanVerifier] = None
        self._initialized = False

    def _initialize(self):
        """Initialize pipeline components lazily."""
        if self._initialized:
            return

        logger.info("Initializing pipeline components...")

        # Initialize code generator
        self.code_generator = CodeGenerator(self.config.model)

        # Initialize translator
        self.translator = GoedelFormalizer(self.config.formalizer)

        # Initialize verifier
        self.verifier = LeanVerifier(self.config.paths)

        self._initialized = True
        logger.info("Pipeline initialized")

    def run(
        self,
        tasks: List[CombinedTask] = None,
        num_variants_per_task: int = None,
        use_fim: bool = False,
    ) -> BenchmarkReport:
        """Run the complete benchmark pipeline.

        Args:
            tasks: Tasks to benchmark (loads from datasets if None)
            num_variants_per_task: Number of variants per task
            use_fim: Whether to use FIM format for generation

        Returns:
            BenchmarkReport with aggregated results
        """
        self._initialize()

        # Load tasks if not provided
        if tasks is None:
            logger.info("Loading combined tasks...")
            tasks = load_combined_tasks()
            logger.info(f"Loaded {len(tasks)} tasks")

        if not tasks:
            logger.warning("No tasks loaded")
            return aggregate_results([], self.config.model.model_name, self.config.formalizer.model_name)

        # Generate variants
        num_variants = num_variants_per_task or self.config.num_variants_per_task
        logger.info(f"Generating {num_variants} variants per task...")
        all_variants = generate_all_variants(tasks, num_variants)
        logger.info(f"Generated {len(all_variants)} total variants")

        # Run pipeline on all variants
        all_results = []
        for i, variant in enumerate(all_variants):
            logger.info(f"Processing variant {i+1}/{len(all_variants)}: {variant.variant_id}")

            result = self._process_variant(variant, use_fim)
            if result.verification_result:
                all_results.append(result.verification_result)

        # Aggregate and return report
        report = aggregate_results(
            all_results,
            self.config.model.model_name,
            self.config.formalizer.model_name,
        )

        return report

    def _process_variant(
        self,
        variant: TaskVariant,
        use_fim: bool = False,
    ) -> PipelineResult:
        """Process a single variant through the pipeline.

        Args:
            variant: Task variant to process
            use_fim: Whether to use FIM format

        Returns:
            PipelineResult with all stage results
        """
        # Stage 1: Generate Python code
        logger.debug(f"Generating Python for {variant.variant_id}")
        gen_result = self.code_generator.generate(variant, use_fim=use_fim)

        if not gen_result.success:
            logger.warning(f"Python generation failed for {variant.variant_id}: {gen_result.error}")
            return PipelineResult(
                variant=variant,
                generation_result=gen_result,
                translation_result=None,
                verification_result=None,
            )

        # Stage 2: Translate to Lean
        logger.debug(f"Translating to Lean for {variant.variant_id}")
        trans_result = self.translator.translate_with_variant(
            gen_result.generated_code,
            variant,
        )

        if not trans_result.success:
            logger.warning(f"Translation failed for {variant.variant_id}: {trans_result.error}")
            # Create partial verification result
            verify_result = VerificationResult(
                task_id=variant.task_id,
                variant_id=variant.variant_id,
                python_generated=gen_result.generated_code,
                lean_translated="",
                translation_success=False,
                lean_compiles=False,
                tests_passed=0,
                tests_failed=0,
                tests_total=0,
                proof_valid=None,
                error=f"Translation failed: {trans_result.error}",
            )
            return PipelineResult(
                variant=variant,
                generation_result=gen_result,
                translation_result=trans_result,
                verification_result=verify_result,
            )

        # Stage 3: Verify Lean code
        logger.debug(f"Verifying Lean for {variant.variant_id}")
        verify_result = self.verifier.verify(
            trans_result.lean_code,
            variant.original_task,
            variant.variant_id,
            gen_result.generated_code,
        )

        return PipelineResult(
            variant=variant,
            generation_result=gen_result,
            translation_result=trans_result,
            verification_result=verify_result,
        )

    def run_single_task(
        self,
        task: CombinedTask,
        num_variants: int = 5,
        use_fim: bool = False,
    ) -> List[PipelineResult]:
        """Run pipeline on a single task with variants.

        Useful for testing and debugging.
        """
        self._initialize()

        variants = self.variant_generator.generate_variants(task, num_variants)
        results = []

        for variant in variants:
            result = self._process_variant(variant, use_fim)
            results.append(result)

        return results

    def cleanup(self):
        """Release model resources."""
        if self.code_generator:
            self.code_generator.unload()
        if self.translator:
            self.translator.unload()


async def run_pipeline_async(
    config: PipelineConfig = None,
    tasks: List[CombinedTask] = None,
    num_variants_per_task: int = 5,
) -> BenchmarkReport:
    """Async version of pipeline run.

    Useful for integration with async applications.
    """
    pipeline = BenchmarkPipeline(config)
    try:
        return pipeline.run(tasks, num_variants_per_task)
    finally:
        pipeline.cleanup()
