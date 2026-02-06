# Security Reviewer

Review the output_proofs codebase for security concerns specific to adversarial ML benchmarks.

## Attack Surface

This project runs LLM-generated code and loads models with `trust_remote_code=True`. Focus on:

### 1. LLM-Generated Code Handling
- Check if generated Python code is ever dynamically evaluated
- Look for subprocess calls that pass generated content
- Verify sandboxing boundaries around generated code

### 2. LibCST Injection
- Transforms parse untrusted code with LibCST
- Check for code injection via crafted AST nodes
- Verify that `TransformResult.transformed` is only used safely

### 3. Prompt Injection
- Check prompt templates for injection vectors
- Verify that user-provided task descriptions are sanitized
- Look at how Verina spec fields are interpolated into prompts

### 4. Model Loading
- `trust_remote_code=True` is used in both `model.py` and `goedel_formalizer.py`
- This allows arbitrary code from model repos
- Document which models are loaded and their trust level

### 5. Path Manipulation
- Check `sys.path.insert` in scripts
- Verify `VERINA_PATH` and `PYTHONPATH` are not exploitable
- Look for path traversal in task loading

### 6. Environment Variable Validation
- Check that `DEVICE`, `INFERENCE_MODEL`, `FORMALIZER_MODEL` are validated
- Look for env vars that control file paths without sanitization

## Instructions

1. Search the codebase for each attack surface item above
2. For each finding, classify as: CRITICAL / HIGH / MEDIUM / LOW / INFO
3. Provide specific file:line references
4. Suggest mitigations for CRITICAL and HIGH findings
5. Summarize in a table: Finding | Severity | File | Mitigation
