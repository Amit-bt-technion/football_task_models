"""
main_transformer.py
------------------
Main script for training the sequence transformer model.
"""

import logging
import os
import torch
import random
import numpy as np
from pathlib import Path

from sequence_classifier.dataset import create_data_loaders
from sequence_classifier.preprocessing import load_and_embed_matches, get_class_weights
from sequence_classifier.sequence_transformer import EventSequenceTransformer, ContinuousValueTransformer
from sequence_classifier.training import train_model, evaluate_model, plot_training_history
from sequence_classifier.regression_training import train_model as train_regression_model
from sequence_classifier.regression_training import evaluate_model as evaluate_regression_model
from sequence_classifier.regression_training import plot_training_history as plot_regression_history


def setup_logging(log_level=logging.INFO) -> None:
    """
    Set up basic logging configuration.
    """
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main(args):
    # Set up logging
    setup_logging(logging.INFO if not args.debug else logging.DEBUG)
    logger = logging.getLogger(__name__)
    
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Check for task and model_type compatibility
    # Define which tasks are regression tasks
    regression_tasks = ['xg_prediction', 'dominating_team_regression']
    classification_tasks = ['event_type_classification', 'random_classification', 'dominating_team_classification']
    
    is_regression_task = args.task in regression_tasks or 'regression' in args.task
    is_classification_task = args.task in classification_tasks or 'classification' in args.task
    
    if args.model_type == 'classification' and is_regression_task:
        logger.error(f"Incompatible configuration: Cannot use classification model with regression task '{args.task}'")
        return 1
    
    if args.model_type == 'regression' and is_classification_task:
        logger.error(f"Incompatible configuration: Cannot use regression model with classification task '{args.task}'")
        return 1
    
    # Ensure output directories exist
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.artifacts_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    logger.info(f"Using device: {device}")
    
    # Log task configuration
    if args.task == 'xg_prediction':
        logger.info(f"Using regression model for xG prediction task (predicting goal probability)")
    
    # Determine mask_list based on task
    mask_list = None
    if args.task == "xg_prediction":
        # Masked fields for shot events: end_location[0,1,2], statsbomb_xg, deflected, outcome.id
        # Based on tokenized_event.py shot event structure starting at index 71:
        # 71: shot.type.id, 72: end_location[0], 73: end_location[1], 74: end_location[2], 
        # 75: aerial_won, 76: follows_dribble, 77: first_time, 78: open_goal,
        # 79: statsbomb_xg, 80: deflected, 81: technique.id, 82: body_part.id, 83: outcome.id
        mask_list = [72, 73, 74, 79, 80, 83]  # end_location[0,1,2], statsbomb_xg, deflected, outcome.id
        logger.info(f"Using mask_list for xG prediction: {mask_list}")
    
    # Step 1: Load and embed match data
    logger.info("Loading and embedding match data...")
    events_dict, embeddings_dict = load_and_embed_matches(
        csv_root_dir=args.data_dir,
        encoder_model_path=args.encoder_path,
        cache_dir=args.cache_dir,
        device=device,
        batch_size=args.batch_size,
        force_recompute=args.force_recompute,
        verbose=not args.quiet,
        mask_list=mask_list,
        task=args.task
    )
    
    # Step 2: Create data loaders
    logger.info("Creating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(
        task=args.task,
        events_dict=events_dict,
        embeddings_dict=embeddings_dict,
        sequence_length=args.sequence_length,
        min_gap=args.min_gap,
        max_gap=args.max_gap,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        batch_size=args.batch_size,
        max_samples_per_match=args.max_samples_per_match,
        max_samples_total=args.max_samples_total,
        num_workers=args.num_workers,
        verbose=not args.quiet
    )
    
    logger.info(f"Created {len(train_loader)} training batches, "
               f"{len(val_loader)} validation batches, "
               f"{len(test_loader)} test batches")
    
    # Step 3: Create model
    logger.info("Creating model...")
    if args.model_type == 'classification':
        # Classification model
        model = EventSequenceTransformer(
            embedding_dim=32,  # Matches the latent dim from the encoder
            nhead=args.transformer_heads,
            num_encoder_layers=args.transformer_layers,
            dim_feedforward=args.transformer_dim_feedforward,
            dropout=args.dropout
        )
    else:
        # Regression model
        model = ContinuousValueTransformer(
            embedding_dim=32,  # Matches the latent dim from the encoder
            nhead=args.transformer_heads,
            num_encoder_layers=args.transformer_layers,
            dim_feedforward=args.transformer_dim_feedforward,
            dropout=args.dropout
        )
    
    # Log model architecture
    logger.info(f"Model architecture:\n{model}")
    
    # Step 4: Train model
    if not args.skip_training:
        logger.info("Training model...")
        if args.model_type == 'classification':
            # Classification training
            model, history = train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                num_epochs=args.epochs,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                patience=args.patience,
                clip_grad_norm=args.clip_grad_norm,
                checkpoint_dir=args.checkpoint_dir,
                device=device
            )
        else:
            # Regression training
            model, history = train_regression_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                num_epochs=args.epochs,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                patience=args.patience,
                clip_grad_norm=args.clip_grad_norm,
                checkpoint_dir=args.checkpoint_dir,
                device=device
            )
        
        # Plot training history
        plot_path = Path(args.artifacts_dir) / "training_history.png"
        if args.model_type == 'classification':
            plot_training_history(history, save_path=plot_path)
        else:
            plot_regression_history(history, save_path=plot_path)
        logger.info(f"Training history plot saved to {plot_path}")
        
        # Save final model
        final_model_path = Path(args.checkpoint_dir) / "final_model.pth"
        torch.save(model.state_dict(), final_model_path)
        logger.info(f"Final model saved to {final_model_path}")
    
    # Step 5: Evaluate model
    if not args.skip_evaluation:
        # Load best model if not training
        if args.skip_training:
            best_model_path = Path(args.checkpoint_dir) / "best_model.pth"
            if best_model_path.exists():
                logger.info(f"Loading best model from {best_model_path}")
                checkpoint = torch.load(best_model_path, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                logger.warning(f"No best model found at {best_model_path}. Using the current model.")
        
        logger.info("Evaluating model...")
        if args.model_type == 'classification':
            metrics = evaluate_model(model, test_loader, device=device)
        else:
            metrics = evaluate_regression_model(model, test_loader, device=device)
        
        # Save metrics
        metrics_path = Path(args.artifacts_dir) / "test_metrics.txt"
        with open(metrics_path, "w") as f:
            for metric, value in metrics.items():
                f.write(f"{metric}: {value}\n")
        logger.info(f"Test metrics saved to {metrics_path}")
    
    logger.info("Pipeline complete!")

