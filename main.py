import argparse
from sequence_classifier.main_transformer import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a sequence transformer for football event classification")

    # Data and model paths
    parser.add_argument("--data-dir", type=str, default="../match_csv", help="Directory containing match CSV files")
    parser.add_argument("--encoder-path", type=str, required=True, help="Path to pretrained encoder model")
    parser.add_argument("--cache-dir", type=str, default="cache", help="Directory to cache embeddings")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Directory to save model checkpoints")
    parser.add_argument("--artifacts-dir", type=str, default="artifacts", help="Directory to save output artifacts")

    # Dataset parameters
    parser.add_argument("--sequence-length", type=int, default=10, help="Number of events in each sequence")
    parser.add_argument("--min-gap", type=int, default=1, help="Minimum number of events between sequences")
    parser.add_argument("--max-gap", type=int, default=None, help="Maximum number of events between sequences")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Proportion of matches for training")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Proportion of matches for validation")
    parser.add_argument("--max-samples-per-match", type=int, default=10000, help="Maximum samples per match")
    parser.add_argument("--max-samples-total", type=int, default=10000000, help="Maximum total samples")

    # Transformer model parameters
    parser.add_argument("--transformer-heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--transformer-layers", type=int, default=6, help="Number of transformer layers")
    parser.add_argument("--transformer-dim-feedforward", type=int, default=2048,
                        help="Dimension of feedforward network")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")

    # Training parameters
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--clip-grad-norm", type=float, default=1.0, help="Gradient clipping norm")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of data loader workers")

    # Utility flags
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--force-recompute", action="store_true", help="Force recomputation of embeddings")
    parser.add_argument("--skip-training", action="store_true", help="Skip training (evaluation only)")
    parser.add_argument("--skip-evaluation", action="store_true", help="Skip evaluation (training only)")
    parser.add_argument("--cpu", action="store_true", help="Force CPU usage even if GPU is available")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--quiet", action="store_true", help="Disable progress bars")

    args = parser.parse_args()
    main(args)