# ReceiptBench

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

A comprehensive evaluation framework for assessing receipt/invoice information extraction systems with support for multi-field evaluation, semantic similarity judgment, and multi-task performance analysis.

## 📋 Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Evaluation Metrics](#evaluation-metrics)
- [Field Types](#field-types)
- [Advanced Usage](#advanced-usage)
- [License](#license)

## ✨ Features<a id="features"></a>

- **Multi-Field Evaluation**: Supports 15+ fields including exact match, amount comparison, and semantic similarity
- **LLM-Based Semantic Judgment**: Uses large language models (Qwen, GPT, etc.) for sophisticated text comparison
- **Hungarian Algorithm**: Optimal matching for list-based fields (e.g., itemized details)
- **Multi-Model Parallel Inference**: Load multiple LLM instances for faster evaluation
- **Multi-Task Support**: Evaluate performance across different subtasks (Perception, Normalization, Reasoning, Structure)
- **Comprehensive Metrics**: Accuracy, Precision, Recall, F1-Score for each field and overall
- **Efficient Caching**: Automatic result caching to avoid redundant LLM calls

## 🚀 Installation<a id="installation"></a>

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for LLM inference)
- 8GB+ GPU memory (24GB+ recommended for multi-model parallel inference)

### Install from Source

```bash
# Clone the repository
git clone https://github.com/Safeoffellow/ReceiptBench.git
cd receipt-eval

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

```bash
pip install torch transformers scipy numpy python-Levenshtein
```

## 🎯 Quick Start<a id="quick-start"></a>

### Basic Usage

```bash
python receipt_evaluator.py \
  --gt-dir ./data/test \
  --pred-dir ./data/predictions \
  --output-dir ./results \
  --llm-model ./models/Qwen3-4B \
  --embedding-model ./models/all-MiniLM-L6-v2
```

### With Parallel Processing

```bash
python receipt_evaluator.py \
  --gt-dir ./data/test \
  --pred-dir ./data/predictions \
  --output-dir ./results \
  --llm-model ./models/Qwen3-4B \
  --embedding-model ./models/all-MiniLM-L6-v2 \
  --num-workers 3 \
  --num-model-instances 3
```

## 📖 Usage<a id="usage"></a>

### Command Line Arguments

```
Required Arguments:
  --gt-dir PATH              Ground truth directory containing JSON files
  --pred-dir PATH            Prediction directory containing JSON files
  --output-dir PATH          Output directory for evaluation results
  --llm-model PATH           Path to LLM model (e.g., Qwen3-4B, GPT-4)
  --embedding-model PATH     Path to sentence transformer model

Optional Arguments:
  --subset-base-dir PATH     Base directory containing subtask test sets
  --num-workers INT          Number of parallel workers (default: 1)
  --num-model-instances INT  Number of LLM instances for parallel inference (default: 1)
```

## 📊 Evaluation Metrics<a id="evaluation-metrics"></a>

The framework computes the following metrics for each field and overall:

- **Accuracy**: (TP + TN) / (TP + TN + FP + FN)
- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **F1-Score**: 2 × (Precision × Recall) / (Precision + Recall)

### Confusion Matrix

- **TP (True Positive)**: Correctly predicted non-empty value
- **TN (True Negative)**: Correctly identified empty value
- **FP (False Positive)**: Predicted non-empty when should be empty
- **FN (False Negative)**: Predicted empty when should be non-empty

## 🏷️ Field Types<a id="field-types"></a>

The framework supports three evaluation strategies:

### 1. Exact Match
Fields: `type`, `invoice_number`, `tax_number`, `std_start_time`, `std_end_time`, `std_invoice_time`, `std_curr`

Comparison: String equality (case-sensitive)

### 2. Amount Comparison
Fields: `std_total`, `orig_total`

Comparison: Numeric equality with normalization (handles currency symbols, commas)

### 3. Semantic Similarity
Fields: `orig_start_time`, `orig_end_time`, `orig_invoice_time`, `place`, `departure`, `arrival`, `seller_name`, `seller_address`

Comparison: 
1. String equality check first
2. LLM-based semantic judgment if not equal

### 4. List Fields
Fields: `orig_curr` (using Hungarian algorithm), `detail` (using multi-metric matching)

Comparison: Optimal assignment with similarity scoring

## 🔧 Advanced Usage<a id="advanced-usage"></a>

### Multi-Task Evaluation

The framework automatically evaluates four subtasks if subtask directories are provided:

- **Task 1 - Perception**: OCR accuracy (invoice_number, tax_number, etc.)
- **Task 2 - Normalization**: Data standardization (std_start_time, std_total, etc.)
- **Task 3 - Reasoning**: Inference tasks (type, place, departure, arrival, etc.)
- **Task 4 - Structure**: Itemized detail extraction

```bash
python receipt_evaluator.py \
  --gt-dir ./data/test \
  --pred-dir ./data/predictions \
  --output-dir ./results \
  --llm-model ./models/Qwen3-4B \
  --embedding-model ./models/all-MiniLM-L6-v2 \
  --subset-base-dir ./data/test_subsets
```

### Performance Optimization

#### Single Model (Conservative)
```bash
# Recommended for: GPU with 8-16GB memory
python receipt_evaluator.py [args] --num-workers 1 --num-model-instances 1
# Expected: 40-60 minutes for 2000 files
```

#### Multi-Model Parallel (Recommended)
```bash
# Recommended for: GPU with 24GB+ memory (RTX 3090/4090, A100)
python receipt_evaluator.py [args] --num-workers 3 --num-model-instances 3
# Expected: 20-30 minutes for 2000 files
```

### Python API

```python
from receipt_evaluator import ReceiptEvaluator

evaluator = ReceiptEvaluator(
    gt_dir='./data/test',
    pred_dir='./data/predictions',
    output_dir='./results',
    llm_model_path='./models/Qwen3-4B',
    embedding_model_path='./models/all-MiniLM-L6-v2',
    num_workers=3,
    num_model_instances=3
)

evaluator.evaluate()
```

## 📄 Output Format

The evaluation generates two files:

### 1. JSON Report (`evaluation_results_YYYYMMDD_HHMMSS.json`)

```json
{
  "metadata": {
    "timestamp": "2024-01-01T00:00:00",
    "gt_dir": "./data/test",
    "pred_dir": "./data/predictions"
  },
  "full_test": {
    "overall_metrics": {
      "accuracy": 0.95,
      "precision": 0.94,
      "recall": 0.96,
      "f1": 0.95
    },
    "field_metrics": {
      "type": {
        "Accuracy": 0.98,
        "Precision": 0.97,
        "Recall": 0.99,
        "F1": 0.98
      },
      ...
    }
  },
  "subtasks": {...}
}
```

### 2. Markdown Report (`evaluation_report_YYYYMMDD_HHMMSS.md`)

Human-readable report with tables for overall metrics, field-level metrics, and subtask performance.

## 📝 License<a id="license"></a>

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or issues, please open an issue on GitHub.

## 🙏 Acknowledgments

- Built with [PyTorch](https://pytorch.org/) and [Transformers](https://huggingface.co/transformers/)
- LLM semantic judgment inspired by recent advances in large language models
- Hungarian algorithm implementation using [SciPy](https://scipy.org/)

---

**Note**: This framework is designed for research purposes. For production use, please consider additional error handling, validation, and optimization.
