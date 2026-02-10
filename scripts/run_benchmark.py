#!/usr/bin/env python3
"""CLI script to run the output_proofs benchmark."""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from output_proofs.config import (
    CacheConfig,
    FormalizerConfig,
    ModelConfig,
    PipelineConfig,
    QuantizationConfig,
)
from output_proofs.pipeline import BenchmarkPipeline


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
        ],
    )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run output_proofs benchmark pipeline")

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Code generation model name (default: from config.py)",
    )

    parser.add_argument(
        "--formalizer",
        type=str,
        default=None,
        help="Python to Lean translation model (default: from config.py)",
    )

    parser.add_argument(
        "--variants",
        type=int,
        default=5,
        help="Number of adversarial variants per task",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./results/benchmark_report.json"),
        help="Output path for benchmark report",
    )

    parser.add_argument(
        "--tasks",
        type=int,
        nargs="+",
        help="Specific MBPP task IDs to run (runs all if not specified)",
    )

    parser.add_argument(
        "--use-fim",
        action="store_true",
        help="Use FIM format for code generation (if model supports it)",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load tasks and variants without running inference",
    )

    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu", "auto"],
        default=None,
        help="Compute device: cuda, cpu, or auto (default: auto-detect)",
    )

    # Quantization
    parser.add_argument(
        "--quantize",
        type=int,
        choices=[4, 8],
        default=None,
        help="Enable int4 or int8 quantization (requires bitsandbytes)",
    )

    # Caching (on by default so Phase 1 results survive a Phase 2 crash)
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable disk-based result caching",
    )

    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear cache before running",
    )

    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Custom cache directory (default: results/cache)",
    )

    # Batch sizes
    parser.add_argument(
        "--generation-batch-size",
        type=int,
        default=8,
        help="Batch size for code generation (default: 8)",
    )

    parser.add_argument(
        "--translation-batch-size",
        type=int,
        default=4,
        help="Batch size for translation (default: 4)",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    # Build quantization config
    quant_config = QuantizationConfig(
        enabled=args.quantize is not None,
        bits=args.quantize or 8,
    )

    # Build model configs
    model_config = ModelConfig(quantization=quant_config)
    if args.model:
        model_config.model_name = args.model

    formalizer_config = FormalizerConfig(quantization=quant_config)
    if args.formalizer:
        formalizer_config.model_name = args.formalizer

    if args.device:
        model_config.device = args.device
        formalizer_config.device = args.device

    # Build cache config (enabled by default)
    cache_config = CacheConfig(
        enabled=not args.no_cache,
        cache_dir=args.cache_dir,
    )

    logger.info("Starting output_proofs benchmark")
    logger.info(f"Model: {model_config.model_name}")
    logger.info(f"Formalizer: {formalizer_config.model_name}")
    logger.info(f"Variants per task: {args.variants}")
    if quant_config.enabled:
        logger.info(f"Quantization: {quant_config.bits}-bit")
    if cache_config.enabled:
        logger.info(f"Caching: enabled (dir={args.cache_dir or 'results/cache'})")
    logger.info(
        f"Batch sizes: generation={args.generation_batch_size}, "
        f"translation={args.translation_batch_size}"
    )

    config = PipelineConfig(
        model=model_config,
        formalizer=formalizer_config,
        num_variants_per_task=args.variants,
        generation_batch_size=args.generation_batch_size,
        translation_batch_size=args.translation_batch_size,
        cache=cache_config,
    )

    # Create pipeline
    pipeline = BenchmarkPipeline(config)

    try:
        # Clear cache if requested
        if args.clear_cache:
            from output_proofs.cache import ResultCache

            cache = ResultCache(cache_config)
            cache.clear()
            logger.info("Cache cleared")

        if args.dry_run:
            # Dry run: just load and show task info
            from output_proofs.tasks.loader import MBPP_TASK_IDS, load_combined_tasks
            from output_proofs.tasks.variants import generate_all_variants

            logger.info("Dry run mode - loading tasks...")
            tasks = load_combined_tasks()

            if args.tasks:
                tasks = [t for t in tasks if t.task_id in args.tasks]

            logger.info(f"Loaded {len(tasks)} tasks")
            logger.info(f"Available MBPP task IDs: {MBPP_TASK_IDS}")

            variants = generate_all_variants(tasks, args.variants)
            logger.info(f"Would generate {len(variants)} variants")

            for v in variants[:5]:
                logger.info(f"  - {v.variant_id}: transforms={v.transforms_applied}")

            return 0

        # Run full benchmark
        logger.info("Running benchmark pipeline...")
        report = pipeline.run(
            num_variants_per_task=args.variants,
            use_fim=args.use_fim,
        )

        # Save report
        logger.info(f"Saving report to {args.output}")
        report.save_detailed(args.output)

        # Log summary
        logger.info("")
        logger.info("=" * 60)
        logger.info("BENCHMARK RESULTS")
        logger.info("=" * 60)
        logger.info(f"Model: {report.model}")
        logger.info(f"Formalizer: {report.formalizer}")
        logger.info(f"Total tasks: {report.total_tasks}")
        logger.info(f"Total variants: {report.total_variants}")
        logger.info("")
        logger.info("Pipeline Stage Success Rates:")
        logger.info(f"  Python generation: {report.pipeline_stages.python_generation_rate:.2%}")
        logger.info(f"  Translation:       {report.pipeline_stages.translation_success_rate:.2%}")
        logger.info(f"  Lean compilation:  {report.pipeline_stages.lean_compilation_rate:.2%}")
        logger.info(f"  Verification:      {report.pipeline_stages.verification_pass_rate:.2%}")
        logger.info("")
        logger.info("Robustness Metrics:")
        logger.info(f"  Original pass rate: {report.robustness.original_pass_rate:.2%}")
        logger.info(f"  Variant pass rate:  {report.robustness.variant_pass_rate:.2%}")
        logger.info(f"  Robustness score:   {report.robustness.robustness_score:.2%}")
        logger.info("")
        logger.info("By Transform:")
        for name, rate in report.by_transform.items():
            logger.info(f"  {name}: {rate:.2%}")
        logger.info("=" * 60)

        return 0

    except KeyboardInterrupt:
        logger.info("Benchmark interrupted by user")
        return 1

    except Exception as e:
        logger.exception(f"Benchmark failed: {e}")
        return 1

    finally:
        pipeline.cleanup()


if __name__ == "__main__":
    sys.exit(main())
