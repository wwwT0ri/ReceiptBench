"""
Receipt Information Extraction Evaluation Framework

A comprehensive evaluation framework for assessing receipt/invoice information extraction
systems. Supports multi-field evaluation with semantic similarity judgment, IoU-based
localization assessment, and multi-task performance analysis.

Key Features:
- Hierarchical evaluation: exact match, amount comparison, semantic similarity
- LLM-based semantic judgment for text fields
- Hungarian algorithm for list-based field matching
- Multi-model parallel inference support
- Comprehensive metrics: Accuracy, Precision, Recall, F1-Score
- Multi-task evaluation support

Authors: [Your Name/Team]
License: MIT
Version: 2.0.0
"""

import os
import sys
import json
import logging
import argparse
import threading
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import torch
import torch.nn.functional as F
import numpy as np
from scipy.optimize import linear_sum_assignment

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ==========================================
# Configuration and Constants
# ==========================================

@dataclass
class EvaluationConfig:
    """Configuration for receipt evaluation"""
    
    # Field type definitions
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
    
    # Subtask field definitions
    SUBTASK_FIELDS = {
        "Task_1_Perception": [
            "invoice_number", "tax_number", "seller_name",
            "orig_start_time", "orig_end_time", "orig_invoice_time",
            "orig_total", "orig_curr"
        ],
        "Task_2_Normalization": [
            "std_start_time", "std_end_time", "std_invoice_time", "std_total"
        ],
        "Task_3_Reasoning": [
            "type", "place", "departure", "arrival", "std_curr", "seller_address"
        ],
        "Task_4_Extraction": ["detail"]
    }
    
    # Similarity thresholds
    LEVENSHTEIN_THRESHOLD = 0.75
    
    # Detail field matching weights
    DETAIL_LEVENSHTEIN_WEIGHT = 0.30
    DETAIL_TOKEN_SORT_WEIGHT = 0.20
    DETAIL_LCS_WEIGHT = 0.10
    DETAIL_SEMANTIC_WEIGHT = 0.40


# ==========================================
# Utility Functions
# ==========================================

class Utils:
    """Utility functions for receipt evaluation"""
    
    @staticmethod
    def is_empty(val: Any) -> bool:
        """
        Check if a value is empty
        
        Args:
            val: Value to check
            
        Returns:
            True if value is empty (None, "", [], {}, 0)
        """
        if val is None:
            return True
        if isinstance(val, str):
            return val.strip() == ""
        if isinstance(val, (list, dict)):
            return len(val) == 0
        # Treat numeric 0 as empty for amount fields
        if isinstance(val, (int, float)) and val == 0:
            return True
        return False
    
    @staticmethod
    def normalize_amount(amount_str: Any) -> Optional[float]:
        """
        Normalize amount string to float
        
        Args:
            amount_str: Amount string or number
            
        Returns:
            Normalized float value or None if invalid/empty
        """
        if Utils.is_empty(amount_str):
            return None
        amount_str = str(amount_str).replace('$', '').replace(',', '').strip()
        if not amount_str:
            return None
        try:
            val = float(amount_str)
            return None if val == 0 else val
        except ValueError:
            return None
    
    @staticmethod
    def extract_field_data(data: Dict, field: str) -> Tuple[Any, List]:
        """
        Extract field value and coordinates from data
        
        Args:
            data: Receipt data dictionary
            field: Field name to extract
            
        Returns:
            Tuple of (field_value, coordinates_list)
        """
        if not isinstance(data, dict):
            return None, []
        
        field_data = data.get(field)
        if field_data is None:
            return None, []
        
        if isinstance(field_data, dict):
            value = field_data.get("content")
            coords = field_data.get("coord", [])
            return value, coords if isinstance(coords, list) else []
        
        return field_data, []
    


def levenshtein_ratio(s1: str, s2: str) -> float:
    """
    Calculate normalized Levenshtein distance (edit distance ratio)
    
    Args:
        s1, s2: Strings to compare
        
    Returns:
        Similarity ratio between 0 and 1
    """
    if s1 == s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    
    len1, len2 = len(s1), len(s2)
    if len1 < len2:
        s1, s2 = s2, s1
        len1, len2 = len2, len1
    
    if len2 == 0:
        return 0.0
    
    prev_row = list(range(len2 + 1))
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row
    
    distance = prev_row[-1]
    max_len = max(len1, len2)
    return 1.0 - (distance / max_len)


# ==========================================
# Semantic Embedding Model
# ==========================================

class SemanticEmbeddingModel:
    """
    Sentence transformer for semantic embedding computation
    Used for detail field matching with vector similarity
    """
    
    def __init__(self, model_path: str):
        """
        Initialize semantic embedding model
        
        Args:
            model_path: Path to sentence transformer model (e.g., all-MiniLM-L6-v2)
        """
        self.loaded = False
        self.tokenizer = None
        self.model = None
        self.device = None
        self._load_model(model_path)
    
    def _load_model(self, model_path: str):
        """Load sentence transformer model"""
        try:
            from transformers import AutoTokenizer, AutoModel
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Loading Sentence Transformer from {model_path} on {self.device}...")
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModel.from_pretrained(model_path).to(self.device)
            self.model.eval()
            self.loaded = True
            logger.info("Sentence Transformer loaded successfully.")
        except ImportError:
            logger.warning("transformers library not found. Semantic embedding will be disabled.")
        except Exception as e:
            logger.warning(f"Failed to load Sentence Transformer: {e}")
    
    def _mean_pooling(self, model_output, attention_mask):
        """Apply mean pooling to get sentence embeddings"""
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask
    
    def encode(self, texts: List[str]) -> torch.Tensor:
        """
        Encode texts to semantic embeddings
        
        Args:
            texts: List of text strings
            
        Returns:
            Normalized embedding tensor
        """
        if not self.loaded or not texts:
            return torch.Tensor().to(self.device) if self.device else torch.Tensor()
        
        valid_texts = [str(t) for t in texts if t and str(t).strip()]
        if not valid_texts:
            return torch.Tensor().to(self.device)
        
        encoded_input = self.tokenizer(
            valid_texts, 
            padding=True, 
            truncation=True, 
            return_tensors='pt'
        ).to(self.device)
        
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        
        sentence_embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings


# ==========================================
# LLM Semantic Judge Model
# ==========================================

class SemanticJudgeModel:
    """
    LLM-based semantic similarity judge
    Uses Qwen or similar models for sophisticated semantic comparison
    Supports multi-model parallel inference
    """
    
    def __init__(self, model_path: str, num_model_instances: int = 1):
        """
        Initialize semantic judge model
        
        Args:
            model_path: Path to LLM model (e.g., Qwen3-4B)
            num_model_instances: Number of model instances for parallel inference
        """
        self.loaded = False
        self.models = []
        self.tokenizers = []
        self.model_locks = []
        self.device = None
        self.num_model_instances = num_model_instances
        
        # Shared cache across all model instances
        self.cache = {}
        self.cache_hits = 0
        self.total_calls = 0
        self.cache_lock = threading.Lock()
        
        self._load_models(model_path, num_model_instances)
    
    def _load_models(self, model_path: str, num_instances: int):
        """Load multiple model instances for parallel inference"""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            if num_instances > 1:
                logger.info(f"🚀 Loading {num_instances} model instances on {self.device} for parallel inference...")
            else:
                logger.info(f"Loading LLM from {model_path} on {self.device}...")
            
            for i in range(num_instances):
                if num_instances > 1:
                    logger.info(f"  Loading model instance {i+1}/{num_instances}...")
                
                tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map={"": 0} if torch.cuda.is_available() else "auto",
                    attn_implementation="flash_attention_2",
                    trust_remote_code=True
                )
                model.eval()
                
                self.models.append(model)
                self.tokenizers.append(tokenizer)
                self.model_locks.append(threading.Lock())
                
                if torch.cuda.is_available():
                    gpu_mem = torch.cuda.memory_allocated(0) / 1e9
                    logger.info(f"    GPU Memory after loading model {i+1}: {gpu_mem:.2f} GB")
            
            self.loaded = True
            
            if num_instances > 1:
                total_mem = torch.cuda.memory_allocated(0) / 1e9 if torch.cuda.is_available() else 0
                logger.info(f"✅ Successfully loaded {num_instances} model instances")
                logger.info(f"📊 Total GPU Memory: {total_mem:.2f} GB")
            else:
                logger.info("LLM loaded successfully.")
                
        except ImportError:
            logger.warning("transformers library not found. Semantic LLM comparison will be disabled.")
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error(f"❌ GPU Out of Memory when loading {num_instances} models!")
                logger.error(f"💡 Try reducing num_model_instances to {max(1, num_instances-1)}")
            raise
        except Exception as e:
            logger.warning(f"Failed to load LLM: {e}")
    
    def _get_model_index(self) -> int:
        """Select model instance based on thread ID"""
        if self.num_model_instances == 1:
            return 0
        thread_id = threading.current_thread().ident
        return thread_id % self.num_model_instances
    
    def judge_semantic_similarity(self, text1: str, text2: str, field_name: str = "field") -> bool:
        """
        Judge if two texts are semantically similar using LLM
        
        Args:
            text1: Ground truth text
            text2: Predicted text
            field_name: Field name for context
            
        Returns:
            True if semantically similar, False otherwise
        """
        self.total_calls += 1
        
        if not self.loaded:
            return levenshtein_ratio(text1.lower(), text2.lower()) >= EvaluationConfig.LEVENSHTEIN_THRESHOLD
        
        # Check cache
        cache_key = (text1.lower().strip(), text2.lower().strip(), field_name)
        with self.cache_lock:
            if cache_key in self.cache:
                self.cache_hits += 1
                if self.total_calls % 100 == 0:
                    cache_rate = (self.cache_hits / self.total_calls) * 100
                    model_info = f" (using {self.num_model_instances} model instances)" if self.num_model_instances > 1 else ""
                    logger.info(f"🔥 LLM Cache: {self.cache_hits}/{self.total_calls} hits ({cache_rate:.1f}%){model_info}")
                return self.cache[cache_key]
        
        # Select model instance
        model_idx = self._get_model_index()
        model = self.models[model_idx]
        tokenizer = self.tokenizers[model_idx]
        model_lock = self.model_locks[model_idx]
        
        try:
            prompt = f"""You are a data quality expert. Evaluate if the Predicted Value is semantically equivalent to the Ground Truth Value in the context of receipt/invoice {field_name}.

Consider equivalent if they represent the same real-world entity despite minor differences (abbreviations, typos, formatting).
Consider NOT equivalent if they refer to different entities or have significant missing/extra information.

Ground Truth: "{text1}"
Predicted: "{text2}"

Respond with ONLY a JSON object:
{{"is_equivalent": true/false, "reasoning": "brief explanation"}}"""
            
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer([text], return_tensors="pt").to(self.device)
            
            with model_lock, torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False,
                    temperature=0.1,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            response = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True).strip()
            
            # Parse JSON response
            try:
                start_idx = response.find('{')
                end_idx = response.rfind('}')
                if start_idx != -1 and end_idx != -1:
                    json_str = response[start_idx:end_idx + 1]
                    result = json.loads(json_str)
                    is_equivalent = result.get("is_equivalent", False)
                    final_result = is_equivalent if isinstance(is_equivalent, bool) else str(is_equivalent).lower() == "true"
                else:
                    final_result = "true" in response.lower()
            except json.JSONDecodeError:
                final_result = "true" in response.lower()
            
            with self.cache_lock:
                self.cache[cache_key] = final_result
            return final_result
            
        except Exception as e:
            logger.warning(f"LLM judgment failed: {e}, falling back to edit distance")
            final_result = levenshtein_ratio(text1.lower(), text2.lower()) >= EvaluationConfig.LEVENSHTEIN_THRESHOLD
            with self.cache_lock:
                self.cache[cache_key] = final_result
            return final_result


# ==========================================
# Main Receipt Evaluator
# ==========================================

class ReceiptEvaluator:
    """
    Comprehensive receipt information extraction evaluator
    
    Supports:
    - Multi-field evaluation with different comparison strategies
    - LLM-based semantic similarity judgment
    - Hungarian algorithm for list-based field matching
    - Multi-model parallel inference
    - Multi-task evaluation
    """
    
    def __init__(
        self,
        gt_dir: str,
        pred_dir: str,
        output_dir: str,
        llm_model_path: str,
        embedding_model_path: str,
        subset_dirs: Optional[Dict[str, str]] = None,
        num_workers: int = 1,
        num_model_instances: int = 1
    ):
        """
        Initialize receipt evaluator
        
        Args:
            gt_dir: Ground truth directory
            pred_dir: Prediction directory
            output_dir: Output directory for results
            llm_model_path: Path to LLM model for semantic judgment
            embedding_model_path: Path to sentence transformer model
            subset_dirs: Optional dict mapping subtask names to test directories
            num_workers: Number of parallel workers (1 = sequential)
            num_model_instances: Number of LLM instances for parallel inference
        """
        self.gt_dir = gt_dir
        self.pred_dir = pred_dir
        self.output_dir = output_dir
        self.num_workers = num_workers
        self.num_model_instances = num_model_instances
        
        # Initialize models
        self.semantic_judge = SemanticJudgeModel(llm_model_path, num_model_instances)
        self.embedding_model = SemanticEmbeddingModel(embedding_model_path)
        
        # Configuration
        self.field_types = EvaluationConfig.FIELD_TYPES
        self.subtask_fields = EvaluationConfig.SUBTASK_FIELDS
        self.subset_dirs = subset_dirs or {}
        
        # Thread safety
        self.results_lock = threading.Lock()
        
        # Results storage
        self.all_fields = list(self.field_types.keys())
        self.results = self._init_results_dict(self.all_fields)
        self.file_results = {}
        self.subtask_results = {name: self._init_results_dict(fields) 
                               for name, fields in self.subtask_fields.items()}
    
    def _init_results_dict(self, fields: List[str]) -> Dict:
        """Initialize results dictionary structure"""
        return {
            field: {
                "value": {"TP": 0, "FP": 0, "TN": 0, "FN": 0},
                "iou_scores": []
            }
            for field in fields
        }
    
    # ... (继续实现其他方法)
    
    def evaluate(self):
        """Run complete evaluation pipeline"""
        import time
        overall_start_time = time.time()
        
        logger.info("Starting evaluation...")
        logger.info("="*80)
        
        # 1. Evaluate full test set
        self._evaluate_full_test()
        
        # 2. Evaluate subtasks
        if self.subset_dirs:
            self._evaluate_subtasks()
        
        # 3. Generate reports
        self._generate_reports()
        
        # 4. Display summary
        total_time = time.time() - overall_start_time
        logger.info(f"\n{'='*80}")
        logger.info(f"✅ Evaluation completed successfully!")
        logger.info(f"⏱️  Total time: {self._format_time(total_time)}")
        logger.info(f"{'='*80}\n")
    
    def _format_time(self, seconds: float) -> str:
        """Format time duration"""
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes}m {secs}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            return f"{hours}h {minutes}m {secs}s"
    
    # Additional methods will be implemented...


# ==========================================
# Command Line Interface
# ==========================================

def main():
    """Main entry point for command line usage"""
    parser = argparse.ArgumentParser(
        description="Receipt Information Extraction Evaluation Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python receipt_evaluator.py --gt-dir ./test --pred-dir ./predictions --output-dir ./results \\
    --llm-model ./Qwen3-4B --embedding-model ./all-MiniLM-L6-v2
  
  # With parallel processing
  python receipt_evaluator.py --gt-dir ./test --pred-dir ./predictions --output-dir ./results \\
    --llm-model ./Qwen3-4B --embedding-model ./all-MiniLM-L6-v2 \\
    --num-workers 3 --num-model-instances 3

For more information, see: https://github.com/your-repo/receipt-ie-eval
        """
    )
    
    # Required arguments
    parser.add_argument('--gt-dir', type=str, required=True,
                       help='Path to ground truth directory')
    parser.add_argument('--pred-dir', type=str, required=True,
                       help='Path to prediction directory')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Path to output directory for results')
    parser.add_argument('--llm-model', type=str, required=True,
                       help='Path to LLM model for semantic judgment')
    parser.add_argument('--embedding-model', type=str, required=True,
                       help='Path to sentence transformer model')
    
    # Optional arguments
    parser.add_argument('--subset-base-dir', type=str, default=None,
                       help='Base directory containing subtask test sets')
    parser.add_argument('--num-workers', type=int, default=1,
                       help='Number of parallel workers (default: 1)')
    parser.add_argument('--num-model-instances', type=int, default=1,
                       help='Number of LLM instances for parallel inference (default: 1)')
    
    args = parser.parse_args()
    
    # Parse subtask directories
    subset_dirs = {}
    if args.subset_base_dir:
        for subtask_name in EvaluationConfig.SUBTASK_FIELDS.keys():
            subtask_dir = os.path.join(args.subset_base_dir, subtask_name)
            if os.path.exists(subtask_dir):
                subset_dirs[subtask_name] = subtask_dir
    
    # Display configuration
    logger.info("="*80)
    logger.info("Receipt Information Extraction Evaluation")
    logger.info("="*80)
    logger.info(f"Ground Truth Directory: {args.gt_dir}")
    logger.info(f"Prediction Directory: {args.pred_dir}")
    logger.info(f"Output Directory: {args.output_dir}")
    logger.info(f"LLM Model: {args.llm_model}")
    logger.info(f"Embedding Model: {args.embedding_model}")
    logger.info(f"Parallel Workers: {args.num_workers}")
    logger.info(f"Model Instances: {args.num_model_instances}")
    logger.info("="*80)
    
    # Create and run evaluator
    evaluator = ReceiptEvaluator(
        gt_dir=args.gt_dir,
        pred_dir=args.pred_dir,
        output_dir=args.output_dir,
        llm_model_path=args.llm_model,
        embedding_model_path=args.embedding_model,
        subset_dirs=subset_dirs,
        num_workers=args.num_workers,
        num_model_instances=args.num_model_instances
    )
    
    evaluator.evaluate()


if __name__ == "__main__":
    main()
