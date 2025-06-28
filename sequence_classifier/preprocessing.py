"""
preprocessing.py
---------------
Functionality for preprocessing and embedding event data.
"""

import logging
import os
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Tuple, Optional, Union, List

from utils.data_loading import load_match_events
from utils.event_autoencoder import EventAutoencoder

# TODO: The problem is that events_df and embeddings are not consistent in order

def load_and_embed_matches(
    csv_root_dir: str,
    encoder_model_path: str,
    cache_dir: Optional[str] = "cache",
    device: Optional[torch.device] = None,
    batch_size: int = 128,
    force_recompute: bool = False,
    verbose: bool = True,
    mask_list: Optional[List[int]] = None,
    task: Optional[str] = None
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """
    Load all match events, embed them using the pretrained autoencoder,
    and cache the embeddings for faster loading.
    
    Args:
        csv_root_dir: Directory containing match event CSV files
        encoder_model_path: Path to the pretrained encoder model
        cache_dir: Directory to store/load cached embeddings
        device: Device to run embedding on
        batch_size: Batch size for embedding
        force_recompute: Force recomputation of embeddings even if cached
        verbose: Whether to show progress bars
        mask_list: List of column indices to mask (set to 0) before embedding
        task: Task name for task-specific caching (e.g., 'xg_prediction')
        
    Returns:
        Tuple of (Dictionary of match events, Dictionary of match embeddings)
    """
    logger = logging.getLogger(__name__)
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create cache directory if it doesn't exist
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    # Path to cached events_df pickle file
    events_cache_path = Path(cache_dir) / "events_df.pkl"

    # Load all match events
    if events_cache_path and events_cache_path.exists() and not force_recompute:
        logger.info(f"Loading cached match events from {events_cache_path}")
        events_df = pd.read_pickle(events_cache_path)
    else:
        logger.info(f"Loading match events from {csv_root_dir}")
        events_df = load_match_events(csv_root_dir=csv_root_dir)
        events_df = events_df.drop(columns=["Unnamed: 0"])
        logger.info(f"Caching events_df to: {events_cache_path}")
        events_df.to_pickle(events_cache_path)

    # Get unique match IDs
    match_ids = events_df["match_id"].unique()
    logger.info(f"Found {len(match_ids)} matches")
    
    # Log masking and task information
    if task:
        logger.info(f"Using task-specific caching for task: '{task}'")
    if mask_list is not None:
        logger.info(f"Will apply masking to {len(mask_list)} columns before embedding: {mask_list}")

    # Create embeddings and events dictionaries
    embeddings_by_match = {}
    events_by_match = {}
    
    # Load the encoder model
    input_dim = events_df.shape[1] - 1 # removing match_id column later
    model = EventAutoencoder(input_dim=input_dim, latent_dim=32)
    model.load_state_dict(torch.load(encoder_model_path, map_location=device, weights_only=False))
    model = model.to(device)
    model.eval()
    
    # Process each match
    for match_id in tqdm(match_ids, desc="Processing matches", disable=not verbose):
        embeddings_cache_path = None
        events_cache_path = None
        if cache_dir:
            # Use task-specific cache files for embeddings if task is provided
            task_suffix = f"_{task}" if task else ""
            embeddings_cache_path = Path(cache_dir) / f"{match_id}_embeddings{task_suffix}.npy"
            events_cache_path = Path(cache_dir) / f"{match_id}_events.npy"  # Events are always the same

        # Check if embeddings are cached (task-specific caching ensures consistency)
        use_cached_embeddings = (embeddings_cache_path and embeddings_cache_path.exists() and not force_recompute)
        use_cached_events = (events_cache_path and events_cache_path.exists() and not force_recompute)
        
        if use_cached_embeddings and use_cached_events:
            # Load both from cache
            embeddings_by_match[match_id] = np.load(embeddings_cache_path)
            events_by_match[match_id] = np.load(events_cache_path)
            cache_info = f"task='{task}'" if task else "default"
            logger.debug(f"Loaded cached embeddings and events for match {match_id} ({cache_info})")
        else:
            # Load events from cache if available, otherwise compute from events_df
            if use_cached_events:
                events_by_match[match_id] = np.load(events_cache_path)
                match_data = events_by_match[match_id]
                if not use_cached_embeddings:
                    cache_info = f"task='{task}'" if task else "default"
                    logger.debug(f"Loaded cached events for match {match_id}, will compute embeddings for {cache_info}")
            else:
                # Filter events for this match and process
                match_df = events_df[events_df["match_id"] == match_id]
                # Drop the match_id column
                match_data = match_df.drop(columns=["match_id"]).values
                # Add events to events_by_match (full unmasked data for label extraction)
                events_by_match[match_id] = match_data.copy()

            # Apply masking if mask_list is provided (only for embedding, not for label extraction)
            match_data_for_embedding = match_data.copy()
            if mask_list is not None:
                logger.debug(f"Applying masking to {len(mask_list)} columns for match {match_id}")
                for mask_idx in mask_list:
                    if mask_idx < match_data_for_embedding.shape[1]:
                        match_data_for_embedding[:, mask_idx] = 0.0
                    else:
                        logger.warning(f"Mask index {mask_idx} is out of bounds for {match_data_for_embedding.shape[1]} columns")

            # Convert to tensor (using masked data for embedding)
            match_tensor = torch.tensor(match_data_for_embedding, dtype=torch.float32)

            # Embed in batches
            match_embeddings = []
            with torch.no_grad():
                for i in range(0, len(match_tensor), batch_size):
                    batch = match_tensor[i:i+batch_size].to(device)
                    batch_embeddings = model.encode(batch).cpu().numpy()
                    match_embeddings.append(batch_embeddings)
            
            # Combine batches
            embeddings_by_match[match_id] = np.vstack(match_embeddings)
            
            # Cache embeddings and events if not already cached
            if cache_dir:
                if not use_cached_embeddings:
                    np.save(embeddings_cache_path, embeddings_by_match[match_id])
                    cache_info = f"task='{task}'" if task else "default"
                    logger.debug(f"Cached embeddings for match {match_id} ({cache_info})")
                if not use_cached_events:
                    np.save(events_cache_path, events_by_match[match_id])
                    logger.debug(f"Cached events for match {match_id}")
    
    return events_by_match, embeddings_by_match


def get_class_weights(dataset_or_loader: Union[torch.utils.data.Dataset, torch.utils.data.DataLoader]) -> torch.Tensor:
    """
    Calculate class weights based on class distribution.
    
    Args:
        dataset_or_loader: Dataset or DataLoader
        
    Returns:
        Tensor of class weights
    """
    labels = []
    
    if isinstance(dataset_or_loader, torch.utils.data.DataLoader):
        for _, batch_labels in dataset_or_loader:
            labels.extend(batch_labels.numpy())
    else:
        for _, label in dataset_or_loader:
            labels.append(label.item())
    
    labels = np.array(labels)
    class_counts = np.bincount(labels)
    class_weights = 1.0 / class_counts
    
    # Normalize weights
    class_weights = class_weights / np.sum(class_weights) * len(class_weights)
    
    return torch.tensor(class_weights, dtype=torch.float32)
