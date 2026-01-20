"""
Parallel processing example with multi-model inference
"""

from receipt_evaluator import ReceiptEvaluator

def main():
    # Configure paths
    gt_dir = './data/test'
    pred_dir = './data/predictions'
    output_dir = './results'
    llm_model_path = './models/Qwen3-4B'
    embedding_model_path = './models/all-MiniLM-L6-v2'
    
    # Create evaluator with parallel processing
    evaluator = ReceiptEvaluator(
        gt_dir=gt_dir,
        pred_dir=pred_dir,
        output_dir=output_dir,
        llm_model_path=llm_model_path,
        embedding_model_path=embedding_model_path,
        num_workers=3,              # 3 parallel workers
        num_model_instances=3       # 3 LLM instances for parallel inference
    )
    
    # Run evaluation
    print("Starting parallel evaluation with 3 model instances...")
    print("Expected speedup: 3-5x compared to sequential processing")
    evaluator.evaluate()
    print(f"Evaluation complete! Results saved to {output_dir}")

if __name__ == "__main__":
    main()
