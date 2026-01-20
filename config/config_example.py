# Configuration Example for Receipt IE Evaluation

# Paths
GT_DIR = "./data/test"
PRED_DIR = "./data/predictions"
OUTPUT_DIR = "./results"
LLM_MODEL_PATH = "./models/Qwen3-4B"
EMBEDDING_MODEL_PATH = "./models/all-MiniLM-L6-v2"

# Multi-task evaluation (optional)
SUBSET_BASE_DIR = "./data/test_subsets"

# Performance settings
NUM_WORKERS = 3              # Number of parallel workers
NUM_MODEL_INSTANCES = 3      # Number of LLM instances for parallel inference

# Field types (default, can be customized)
FIELD_TYPES = {
    "type": "exact",
    "std_start_time": "exact",
    "orig_start_time": "semantic",
    "std_end_time": "exact",
    "orig_end_time": "semantic",
    "std_invoice_time": "exact",
    "orig_invoice_time": "semantic",
    "place": "semantic",
    "departure": "semantic",
    "arrival": "semantic",
    "std_curr": "exact",
    "orig_curr": "list",
    "std_total": "amount",
    "orig_total": "amount",
    "detail": "list",
    "seller_name": "semantic",
    "seller_address": "semantic",
    "invoice_number": "exact",
    "tax_number": "exact"
}

# Evaluation thresholds
LEVENSHTEIN_THRESHOLD = 0.75

# Detail field matching weights
DETAIL_LEVENSHTEIN_WEIGHT = 0.30
DETAIL_TOKEN_SORT_WEIGHT = 0.20
DETAIL_LCS_WEIGHT = 0.10
DETAIL_SEMANTIC_WEIGHT = 0.40
