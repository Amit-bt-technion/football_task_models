"""
visualization.py
--------------
Visualization tools for the sequence transformer project.
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
import torch
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional, Union, Any


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Optional[str] = None
) -> None:
    """
    Plot a confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        save_path: Optional path to save the plot
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    
    # Plot confusion matrix
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d',
        cmap='Blues',
        xticklabels=['First Before Second', 'Second Before First'],
        yticklabels=['First Before Second', 'Second Before First']
    )
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        
    plt.show()


def get_model_predictions(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get model predictions and probabilities.
    
    Args:
        model: The model
        dataloader: DataLoader for evaluation data
        device: Device to run on
        
    Returns:
        Tuple of (true labels, predicted labels, predicted probabilities)
    """
    model.eval()
    
    all_targets = []
    all_predictions = []
    all_probabilities = []
    
    with torch.no_grad():
        for sequences, targets in dataloader:
            sequences, targets = sequences.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(sequences)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            # Save results
            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_probabilities.extend(probabilities[:, 1].cpu().numpy())  # Probability of class 1
    
    return np.array(all_targets), np.array(all_predictions), np.array(all_probabilities)


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    save_path: Optional[str] = None
) -> float:
    """
    Plot ROC curve and return AUC.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities for class 1
        save_path: Optional path to save the plot
        
    Returns:
        Area under the ROC curve
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        
    plt.show()
    
    return roc_auc


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    save_path: Optional[str] = None
) -> float:
    """
    Plot precision-recall curve and return average precision.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities for class 1
        save_path: Optional path to save the plot
        
    Returns:
        Average precision
    """
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    
    # Calculate average precision
    avg_precision = np.sum(np.diff(recall[::-1]) * precision[::-1][:-1])
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AP = {avg_precision:.3f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        
    plt.show()
    
    return avg_precision


def visualize_attention(
    model: torch.nn.Module,
    sequence: torch.Tensor,
    device: torch.device,
    save_path: Optional[str] = None
) -> None:
    """
    Visualize attention weights for a single sequence.
    
    Args:
        model: The transformer model
        sequence: Input sequence tensor of shape [1, seq_len, embedding_dim]
        device: Device to run on
        save_path: Optional path to save the plot
    """
    model.eval()
    sequence = sequence.unsqueeze(0).to(device)  # Add batch dimension
    
    # Get attention weights (assuming the model exposes attention)
    with torch.no_grad():
        # Forward pass - this assumes model has been modified to return attention weights
        # You may need to modify the model to expose attention weights
        outputs, attention_weights = model(sequence, return_attention=True)
    
    # Average attention weights across heads
    # This assumes attention_weights is a tensor of shape [num_layers, num_heads, seq_len, seq_len]
    avg_attention = attention_weights.mean(dim=1)  # Average across heads
    
    # Plot attention weights for each layer
    num_layers = avg_attention.shape[0]
    seq_len = avg_attention.shape[-1]
    
    # Create a grid of subplots
    fig, axes = plt.subplots(
        nrows=1, 
        ncols=num_layers, 
        figsize=(num_layers * 4, 4),
        sharey=True
    )
    
    if num_layers == 1:
        axes = [axes]
    
    # Plot each layer's attention
    for i, ax in enumerate(axes):
        im = ax.imshow(avg_attention[i].cpu(), cmap='viridis')
        ax.set_title(f'Layer {i+1}')
        ax.set_xlabel('Key position')
        if i == 0:
            ax.set_ylabel('Query position')
    
    plt.colorbar(im, ax=axes)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        
    plt.show()


def visualize_embeddings(
    embeddings: np.ndarray,
    labels: np.ndarray,
    perplexity: int = 30,
    n_neighbors: int = 15,
    save_path: Optional[str] = None
) -> None:
    """
    Visualize embeddings using t-SNE.
    
    Args:
        embeddings: Embedding vectors
        labels: Labels for coloring
        perplexity: t-SNE perplexity parameter
        n_neighbors: Number of neighbors for t-SNE
        save_path: Optional path to save the plot
    """
    from sklearn.manifold import TSNE
    
    # Reduce dimensionality with t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=1000)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        embeddings_2d[:, 0], 
        embeddings_2d[:, 1], 
        c=labels, 
        cmap='viridis', 
        alpha=0.8, 
        s=10
    )
    
    plt.colorbar(scatter, label='Label')
    plt.title('t-SNE Visualization of Event Embeddings')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        
    plt.show()
