# Benchmark Analyzer

Analyze benchmark result JSON reports in `./results/`.

## Report Schema

Each report JSON contains:

```json
{
  "model": "model-name",
  "formalizer": "formalizer-name",
  "total_tasks": 10,
  "total_variants": 50,
  "pipeline_stages": {
    "python_generation_rate": 0.85,
    "translation_success_rate": 0.70,
    "lean_compilation_rate": 0.60,
    "verification_pass_rate": 0.45
  },
  "robustness": {
    "original_pass_rate": 0.80,
    "variant_pass_rate": 0.55,
    "robustness_score": 0.69
  },
  "by_transform": {
    "v_noise_rename": 0.52,
    "param_reorder": 0.61
  },
  "results": [...]
}
```

## Metrics to Compute

1. **Pipeline funnel** — drop-off rates between each stage
2. **Robustness delta** — difference between original and variant pass rates
3. **Transform impact ranking** — which transforms degrade performance most
4. **Per-task breakdown** — identify specific tasks where the model consistently fails
5. **Cross-run comparison** — if multiple reports exist, compare models/configurations

## Output Format

Present results as a structured summary with:
- Key findings (2-3 bullet points)
- Table of pipeline stage drop-offs
- Bar-chart-style ranking of transform impact (use text/ASCII)
- List of most-fragile tasks (lowest variant pass rates)
- Recommendations for improving robustness

## Instructions

1. Read all JSON files in `./results/`
2. Parse and validate against the schema above
3. Compute the metrics listed
4. Present the structured summary
5. If multiple reports exist, include cross-run comparison
