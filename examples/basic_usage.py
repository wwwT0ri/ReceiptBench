"""
Basic usage example for Receipt IE Evaluation Framework
"""

from receipt_evaluator import ReceiptEvaluator

def main():
    # Configure paths
    gt_dir = './data/test'
    pred_dir = './data/predictions'
    output_dir = './results'
    llm_model_path = './models/Qwen3-4B'
    embedding_model_path = './models/all-MiniLM-L6-v2'
    
    # Create evaluator
    evaluator = ReceiptEvaluator(
        gt_dir=gt_dir,
        pred_dir=pred_dir,
        output_dir=output_dir,
        llm_model_path=llm_model_path,
        embedding_model_path=embedding_model_path
    )
    
    # Run evaluation
    print("Starting evaluation...")
    evaluator.evaluate()
    print(f"Evaluation complete! Results saved to {output_dir}")

if __name__ == "__main__":
    main()
