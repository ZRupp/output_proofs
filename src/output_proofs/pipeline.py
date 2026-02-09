"""End-to-end benchmark pipeline orchestration."""

import logging
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

from .cache import ResultCache
from .config import PipelineConfig
from .inference.model import CodeGenerator, GenerationResult
from .reporting.report import BenchmarkReport, aggregate_results
from .tasks.loader import load_combined_tasks
from .tasks.schema import CombinedTask, TaskVariant
from .tasks.variants import VariantGenerator, generate_all_variants
from .translation.goedel_formalizer import GoedelFormalizer, TranslationResult
from .verification.lean_verifier import LeanVerifier, VerificationResult

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
        self.cache: Optional[ResultCache] = None
        self._initialized = False

    def _initialize(self):
        """Initialize pipeline components lazily.

        Only sets up the verifier and cache — models are loaded on-demand
        in each phase to avoid holding both in VRAM simultaneously.
        """
        if self._initialized:
            return

        logger.info("Initializing pipeline components...")

        # Initialize verifier (no GPU model)
        self.verifier = LeanVerifier(self.config.paths)

        # Initialize cache
        default_cache_dir = self.config.paths.results_dir / "cache"
        self.cache = ResultCache(self.config.cache, default_cache_dir=default_cache_dir)

        self._initialized = True
        logger.info("Pipeline initialized")

    def run(
        self,
        tasks: List[CombinedTask] = None,
        num_variants_per_task: int = None,
        use_fim: bool = False,
    ) -> BenchmarkReport:
        """Run the complete benchmark pipeline using phased execution.

        Phase 1: Load generation model, batch-generate all variants, unload.
        Phase 2: Load translation model, batch-translate all variants, unload.
        Phase 3: Verify all translated Lean code (CPU-only).

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
            return aggregate_results(
                [], self.config.model.model_name, self.config.formalizer.model_name
            )

        # Generate variants
        num_variants = num_variants_per_task or self.config.num_variants_per_task
        logger.info(f"Generating {num_variants} variants per task...")
        all_variants = generate_all_variants(tasks, num_variants)
        logger.info(f"Generated {len(all_variants)} total variants")

        # Phase 1: Code generation
        gen_results = self._phase_generate(all_variants, use_fim)

        # Phase 2: Translation
        trans_results = self._phase_translate(all_variants, gen_results)

        # Phase 3: Verification
        all_verification_results = self._phase_verify(all_variants, gen_results, trans_results)

        # Aggregate and return report
        report = aggregate_results(
            all_verification_results,
            self.config.model.model_name,
            self.config.formalizer.model_name,
        )

        return report

    # ------------------------------------------------------------------
    # Phase 1: Code generation
    # ------------------------------------------------------------------

    def _phase_generate(
        self,
        variants: List[TaskVariant],
        use_fim: bool,
    ) -> Dict[str, GenerationResult]:
        """Generate Python code for all variants.

        Loads the generation model, batch-processes uncached variants,
        caches results, then unloads the model.
        """
        logger.info("=== Phase 1: Code Generation ===")
        results: Dict[str, GenerationResult] = {}
        uncached_variants: List[TaskVariant] = []
        uncached_indices: List[int] = []

        # Check cache
        for i, variant in enumerate(variants):
            cached = self.cache.get_generation(
                variant.task_id,
                variant.variant_id,
                variant.transforms_applied,
                self.config.model.model_name,
            )
            if cached is not None:
                results[variant.variant_id] = GenerationResult(**cached)
                logger.debug(f"Cache hit for generation: {variant.variant_id}")
            else:
                uncached_variants.append(variant)
                uncached_indices.append(i)

        if not uncached_variants:
            logger.info(f"All {len(variants)} generation results from cache")
            return results

        logger.info(
            f"Generating {len(uncached_variants)} variants "
            f"({len(variants) - len(uncached_variants)} cached)"
        )

        # Load model, generate, unload
        generator = CodeGenerator(self.config.model)
        generator.load()

        try:
            batch_results = generator.generate_batch(
                uncached_variants,
                use_fim=use_fim,
                batch_size=self.config.generation_batch_size,
            )

            for variant, result in zip(uncached_variants, batch_results):
                results[variant.variant_id] = result
                if self.cache.enabled:
                    self.cache.put_generation(
                        variant.task_id,
                        variant.variant_id,
                        variant.transforms_applied,
                        self.config.model.model_name,
                        asdict(result),
                    )
        finally:
            generator.unload()
            del generator

        logger.info(f"Phase 1 complete: {sum(1 for r in results.values() if r.success)} succeeded")
        return results

    # ------------------------------------------------------------------
    # Phase 2: Translation
    # ------------------------------------------------------------------

    def _phase_translate(
        self,
        variants: List[TaskVariant],
        gen_results: Dict[str, GenerationResult],
    ) -> Dict[str, TranslationResult]:
        """Translate generated Python code to Lean 4.

        Skips variants where generation failed. Loads the translation model,
        batch-processes uncached variants, caches results, then unloads.
        """
        logger.info("=== Phase 2: Translation ===")
        results: Dict[str, TranslationResult] = {}

        # Filter to variants with successful generation
        translatable = [
            v
            for v in variants
            if gen_results.get(v.variant_id, None) is not None and gen_results[v.variant_id].success
        ]

        uncached_variants: List[TaskVariant] = []

        # Check cache
        for variant in translatable:
            cached = self.cache.get_translation(
                variant.task_id,
                variant.variant_id,
                variant.transforms_applied,
                self.config.formalizer.model_name,
            )
            if cached is not None:
                results[variant.variant_id] = TranslationResult(**cached)
                logger.debug(f"Cache hit for translation: {variant.variant_id}")
            else:
                uncached_variants.append(variant)

        if not uncached_variants:
            logger.info(
                f"All {len(translatable)} translation results from cache "
                f"({len(variants) - len(translatable)} skipped due to generation failure)"
            )
            return results

        logger.info(
            f"Translating {len(uncached_variants)} variants "
            f"({len(translatable) - len(uncached_variants)} cached, "
            f"{len(variants) - len(translatable)} skipped)"
        )

        # Load model, translate, unload
        translator = GoedelFormalizer(self.config.formalizer)
        translator.load()

        try:
            python_codes = [gen_results[v.variant_id].generated_code for v in uncached_variants]
            tasks = [v.original_task for v in uncached_variants]

            batch_results = translator.translate_batch(
                python_codes,
                tasks,
                batch_size=self.config.translation_batch_size,
            )

            for variant, result in zip(uncached_variants, batch_results):
                results[variant.variant_id] = result
                if self.cache.enabled:
                    self.cache.put_translation(
                        variant.task_id,
                        variant.variant_id,
                        variant.transforms_applied,
                        self.config.formalizer.model_name,
                        asdict(result),
                    )
        finally:
            translator.unload()
            del translator

        logger.info(f"Phase 2 complete: {sum(1 for r in results.values() if r.success)} succeeded")
        return results

    # ------------------------------------------------------------------
    # Phase 3: Verification
    # ------------------------------------------------------------------

    def _phase_verify(
        self,
        variants: List[TaskVariant],
        gen_results: Dict[str, GenerationResult],
        trans_results: Dict[str, TranslationResult],
    ) -> List[VerificationResult]:
        """Verify Lean code for all variants.

        Builds VerificationResult for every variant, including failures
        at earlier stages.
        """
        logger.info("=== Phase 3: Verification ===")
        all_results = []

        for variant in variants:
            gen = gen_results.get(variant.variant_id)
            trans = trans_results.get(variant.variant_id)

            # Generation failed
            if gen is None or not gen.success:
                continue

            # Translation failed or skipped
            if trans is None or not trans.success:
                error_msg = trans.error if trans else "Translation skipped"
                all_results.append(
                    VerificationResult(
                        task_id=variant.task_id,
                        variant_id=variant.variant_id,
                        python_generated=gen.generated_code if gen else "",
                        lean_translated="",
                        translation_success=False,
                        lean_compiles=False,
                        tests_passed=0,
                        tests_failed=0,
                        tests_total=0,
                        proof_valid=None,
                        error=f"Translation failed: {error_msg}",
                    )
                )
                continue

            # Run verification
            logger.debug(f"Verifying Lean for {variant.variant_id}")
            verify_result = self.verifier.verify(
                trans.lean_code,
                variant.original_task,
                variant.variant_id,
                gen.generated_code,
            )
            all_results.append(verify_result)

        logger.info(
            f"Phase 3 complete: {sum(1 for r in all_results if r.success)}/{len(all_results)} "
            "passed verification"
        )
        return all_results

    # ------------------------------------------------------------------
    # Single-variant processing (for debugging)
    # ------------------------------------------------------------------

    def _process_variant(
        self,
        variant: TaskVariant,
        use_fim: bool = False,
    ) -> PipelineResult:
        """Process a single variant through the pipeline.

        Lazy-loads models as needed. Kept for debugging and single-task runs.

        Args:
            variant: Task variant to process
            use_fim: Whether to use FIM format

        Returns:
            PipelineResult with all stage results
        """
        # Ensure models are available for single-variant mode
        if self.code_generator is None:
            self.code_generator = CodeGenerator(self.config.model)
        if self.translator is None:
            self.translator = GoedelFormalizer(self.config.formalizer)

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
