"""
regression_training.py
--------------------
Functions for training and evaluating the transformer model for regression tasks.
"""

import logging
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    clip_grad_norm: Optional[float] = 1.0
) -> Tuple[float, float]:
    """
    Train model for one epoch.
    
    Args:
        model: The model to train
        dataloader: DataLoader for training data
        criterion: Loss function (MSE)
        optimizer: Optimizer
        device: Device to train on
        clip_grad_norm: Max norm for gradient clipping (None for no clipping)
        
    Returns:
        Tuple of (average loss, average MSE)
    """
    model.train()
    total_loss = 0.0
    total_mse = 0.0
    total = 0
    
    for batch_idx, (sequences, targets) in enumerate(tqdm(dataloader, desc="Training", leave=False)):
        sequences, targets = sequences.to(device), targets.to(device)
        
        # Ensure targets have the right shape for regression
        if targets.dim() == 1:
            targets = targets.unsqueeze(1)  # Convert [batch_size] to [batch_size, 1]
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(sequences)
        loss = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        mse = nn.MSELoss()(outputs, targets).item()
        total_mse += mse * targets.size(0)
        total += targets.size(0)
    
    avg_loss = total_loss / len(dataloader)
    avg_mse = total_mse / total
    
    return avg_loss, avg_mse


def validate(
    model: nn.Module, 
    dataloader: DataLoader, 
    criterion: nn.Module, 
    device: torch.device
) -> Tuple[float, float]:
    """
    Validate the model.
    
    Args:
        model: The model to validate
        dataloader: DataLoader for validation data
        criterion: Loss function (MSE)
        device: Device to validate on
        
    Returns:
        Tuple of (average loss, average MSE)
    """
    model.eval()
    total_loss = 0.0
    total_mse = 0.0
    total = 0
    
    with torch.no_grad():
        for sequences, targets in tqdm(dataloader, desc="Validating", leave=False):
            sequences, targets = sequences.to(device), targets.to(device)
            
            # Ensure targets have the right shape for regression
            if targets.dim() == 1:
                targets = targets.unsqueeze(1)  # Convert [batch_size] to [batch_size, 1]
            
            # Forward pass
            outputs = model(sequences)
            loss = criterion(outputs, targets)
            
            # Track metrics
            total_loss += loss.item()
            mse = nn.MSELoss()(outputs, targets).item()
            total_mse += mse * targets.size(0)
            total += targets.size(0)
    
    avg_loss = total_loss / len(dataloader)
    avg_mse = total_mse / total
    
    return avg_loss, avg_mse


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 10,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-5,
    patience: int = 3,
    clip_grad_norm: Optional[float] = 1.0,
    checkpoint_dir: str = "checkpoints",
    device: Optional[torch.device] = None
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """
    Train the model with early stopping.
    
    Args:
        model: The model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        num_epochs: Maximum number of epochs to train for
        learning_rate: Learning rate
        weight_decay: Weight decay for L2 regularization
        patience: Number of epochs to wait for improvement before early stopping
        clip_grad_norm: Max norm for gradient clipping
        checkpoint_dir: Directory to save checkpoints
        device: Device to train on (defaults to GPU if available)
        
    Returns:
        Tuple of (trained model, dictionary of training history)
    """
    logger = logging.getLogger(__name__)
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logger.info(f"Training on {device}")
    model = model.to(device)
    
    # Setup criterion and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    
    # Create checkpoint directory
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    
    # Track metrics
    history = {
        'train_loss': [],
        'train_mse': [],
        'val_loss': [],
        'val_mse': []
    }
    
    best_val_loss = float('inf')
    best_epoch = 0
    epochs_no_improve = 0
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch+1}/{num_epochs}")
        
        # Train
        train_loss, train_mse = train_epoch(
            model, train_loader, criterion, optimizer, device, clip_grad_norm
        )
        
        # Validate
        val_loss, val_mse = validate(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Log metrics
        logger.info(f"Train Loss: {train_loss:.4f} | Train MSE: {train_mse:.4f}")
        logger.info(f"Val Loss: {val_loss:.4f} | Val MSE: {val_mse:.4f}")
        
        # Save metrics
        history['train_loss'].append(train_loss)
        history['train_mse'].append(train_mse)
        history['val_loss'].append(val_loss)
        history['val_mse'].append(val_mse)
        
        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_no_improve = 0
            
            # Save best model
            best_model_path = checkpoint_dir / "best_model.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_mse': val_mse
            }, best_model_path)
            
            logger.info(f"New best model saved to {best_model_path}")
        else:
            epochs_no_improve += 1
            
        # Save checkpoint every epoch
        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pth"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'val_mse': val_mse
        }, checkpoint_path)
        
        # Early stopping
        if epochs_no_improve >= patience:
            logger.info(f"Early stopping triggered after epoch {epoch+1}")
            break
    
    # Training time
    total_time = time.time() - start_time
    logger.info(f"Training completed in {total_time:.2f} seconds")
    logger.info(f"Best validation loss: {best_val_loss:.4f} at epoch {best_epoch+1}")
    
    # Load best model
    best_model_path = checkpoint_dir / "best_model.pth"
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, history


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: Optional[torch.device] = None
) -> Dict[str, float]:
    """
    Evaluate the model on test data.
    
    Args:
        model: The model to evaluate
        test_loader: DataLoader for test data
        device: Device to evaluate on
        
    Returns:
        Dictionary of metrics
    """
    logger = logging.getLogger(__name__)
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model.to(device)
    model.eval()
    
    total_mse = 0.0
    total_mae = 0.0
    total = 0
    all_targets = []
    all_predictions = []
    
    with torch.no_grad():
        for sequences, targets in tqdm(test_loader, desc="Testing"):
            sequences, targets = sequences.to(device), targets.to(device)
            
            # Ensure targets have the right shape for regression
            if targets.dim() == 1:
                targets = targets.unsqueeze(1)  # Convert [batch_size] to [batch_size, 1]
            
            # Forward pass
            outputs = model(sequences)
            
            # Track metrics
            mse = nn.MSELoss()(outputs, targets).item()
            mae = nn.L1Loss()(outputs, targets).item()
            
            total_mse += mse * targets.size(0)
            total_mae += mae * targets.size(0)
            total += targets.size(0)
            
            # Save predictions and targets for later analysis
            # Flatten outputs and targets for correlation calculation
            all_targets.extend(targets.view(-1).cpu().numpy())
            all_predictions.extend(outputs.view(-1).cpu().numpy())
    
    # Calculate average metrics
    avg_mse = total_mse / total
    avg_mae = total_mae / total
    rmse = np.sqrt(avg_mse)
    
    # Convert to numpy arrays for correlation calculation
    all_targets = np.array(all_targets)
    all_predictions = np.array(all_predictions)
    correlation = np.corrcoef(all_targets, all_predictions)[0, 1]
    
    metrics = {
        'mse': avg_mse,
        'rmse': rmse,
        'mae': avg_mae,
        'correlation': correlation
    }
    
    logger.info(f"Test MSE: {avg_mse:.4f}")
    logger.info(f"Test RMSE: {rmse:.4f}")
    logger.info(f"Test MAE: {avg_mae:.4f}")
    logger.info(f"Test Correlation: {correlation:.4f}")
    
    return metrics


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None
) -> None:
    """
    Plot training history.
    
    Args:
        history: Dictionary of training metrics
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(12, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot MSE
    plt.subplot(1, 2, 2)
    plt.plot(history['train_mse'], label='Train')
    plt.plot(history['val_mse'], label='Validation')
    plt.title('Mean Squared Error')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        
    plt.show() 