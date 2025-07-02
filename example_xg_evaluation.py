"""
example_xg_evaluation.py
-----------------------
Example usage of the xG evaluation functionality.
"""

import os
from xg_evaluation import evaluate_xg_model_detailed

def run_xg_evaluation_example():
    """
    Example function showing how to run xG model evaluation.
    
    This assumes you have:
    1. A trained xG model checkpoint
    2. Match CSV data 
    3. A pretrained encoder
    """
    
    # Example paths - adjust these to your actual file locations
    model_path = "checkpoints/best_model.pth"  # Path to your trained xG model
    data_dir = "/path/to/your/match/csv/files"  # Directory with match CSV files
    encoder_path = "/path/to/your/encoder.pth"  # Path to pretrained encoder
    
    # Check if files exist before running
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        print("   Please train an xG model first using:")
        print("   python3 main.py --task=xg_prediction --model-type=regression ...")
        return
    
    if not os.path.exists(data_dir):
        print(f"❌ Data directory not found at {data_dir}")
        print("   Please update data_dir to point to your match CSV files")
        return
        
    if not os.path.exists(encoder_path):
        print(f"❌ Encoder not found at {encoder_path}")
        print("   Please update encoder_path to point to your pretrained encoder")
        return
    
    print("🚀 Starting xG model evaluation...")
    print(f"   Model: {model_path}")
    print(f"   Data: {data_dir}")
    print(f"   Encoder: {encoder_path}")
    print()
    
    # Run the detailed evaluation
    evaluate_xg_model_detailed(
        model_path=model_path,
        data_dir=data_dir,
        encoder_path=encoder_path,
        cache_dir="cache",
        sequence_length=200,
        num_samples=20  # 10 goals + 10 non-goals
    )


if __name__ == "__main__":
    run_xg_evaluation_example() 