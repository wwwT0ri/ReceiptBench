"""
Example with multi-task evaluation
"""

from receipt_evaluator import ReceiptEvaluator

def main():
    # Configure paths
    gt_dir = './data/test'
    pred_dir = './data/predictions'
    output_dir = './results'
    llm_model_path = './models/Qwen3-4B'
    embedding_model_path = './models/all-MiniLM-L6-v2'
    
    # Define subtask directories
    subset_dirs = {
        'Task_1_Perception': './data/test_subsets/Task_1_Perception',
        'Task_2_Normalization': './data/test_subsets/Task_2_Normalization',
        'Task_3_Reasoning': './data/test_subsets/Task_3_Reasoning',
        'Task_4_Structure': './data/test_subsets/Task_4_Structure'
    }
    
    # Create evaluator with multi-task support
    evaluator = ReceiptEvaluator(
        gt_dir=gt_dir,
        pred_dir=pred_dir,
        output_dir=output_dir,
        llm_model_path=llm_model_path,
        embedding_model_path=embedding_model_path,
        subset_dirs=subset_dirs,
        num_workers=3,
        num_model_instances=3
    )
    
    # Run evaluation
    print("Starting multi-task evaluation...")
    evaluator.evaluate()
    print(f"Evaluation complete! Results saved to {output_dir}")
    print("\nSubtask results:")
    print("- Task 1 (Perception): OCR accuracy")
    print("- Task 2 (Normalization): Data standardization")
    print("- Task 3 (Reasoning): Inference accuracy")
    print("- Task 4 (Structure): Detail extraction")

if __name__ == "__main__":
    main()
