# Quick Start Guide

## Installation

```bash
# 1. Install Python dependencies
pip install torch transformers scipy numpy python-Levenshtein

# 2. Download models (if not already available)
# - LLM Model: Qwen3-4B, GPT-4, or similar
# - Embedding Model: all-MiniLM-L6-v2 or similar sentence transformer
```

## Minimal Example

```bash
python receipt_evaluator.py \
  --gt-dir ./your_test_data \
  --pred-dir ./your_predictions \
  --output-dir ./evaluation_results \
  --llm-model /path/to/Qwen3-4B \
  --embedding-model /path/to/all-MiniLM-L6-v2
```

## Expected Output

```
================================================================================
Receipt Information Extraction Evaluation
================================================================================
Ground Truth Directory: ./your_test_data
Prediction Directory: ./your_predictions
Output Directory: ./evaluation_results
LLM Model: /path/to/Qwen3-4B
Embedding Model: /path/to/all-MiniLM-L6-v2
================================================================================

Loading LLM from /path/to/Qwen3-4B on cuda...
LLM loaded successfully.
Loading Sentence Transformer from /path/to/all-MiniLM-L6-v2 on cuda...
Sentence Transformer loaded successfully.

Starting evaluation...
Evaluating full test set (sequential mode)...
Processing 100/2000 | Elapsed: 2m 30s | Remaining: ~47m 30s
...

================================================================================
✅ Evaluation completed successfully!
⏱️  Total time: 50m 15s
================================================================================

[Full Test] Overall: Acc=0.9450, Prec=0.9380, Rec=0.9520, F1=0.9449
```

## Check Results

Results are saved in the output directory:
- `evaluation_results_YYYYMMDD_HHMMSS.json` - Detailed metrics in JSON format
- `evaluation_report_YYYYMMDD_HHMMSS.md` - Human-readable Markdown report

## Next Steps

- See [README.md](README.md) for advanced usage
