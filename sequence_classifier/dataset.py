"""
dataset.py
---------
Custom dataset for loading and preparing pairs of event sequences.
"""

import logging
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Dict, Optional
import random
from tqdm import tqdm


class EventSequenceDataset(Dataset):
    """
    Dataset for creating pairs of event sequences from embedded football events.
    Each item is a pair of sequences, with a label indicating their chronological order.
    """
    
    def __init__(
        self, 
        events_df: pd.DataFrame,
        sequence_length: int = 10,
        min_gap: int = 1,
        max_gap: Optional[int] = None,
        match_ids: Optional[List[str]] = None,
        shuffle: bool = True,
        max_samples_per_match: Optional[int] = None,
        max_total_samples: Optional[int] = None,
        precomputed_embeddings: Optional[Dict[str, np.ndarray]] = None,
        verbose: bool = True
    ):
        """
        Initialize the EventSequenceDataset.
        
        Args:
            events_df: DataFrame containing event embeddings with match_id column
            sequence_length: Number of events in each sequence
            min_gap: Minimum number of events between sequences
            max_gap: Maximum number of events between sequences (if None, no limit)
            match_ids: Optional list of match IDs to filter by
            shuffle: Whether to shuffle the samples
            max_samples_per_match: Maximum number of samples to generate per match
            max_total_samples: Maximum total samples across all matches
            precomputed_embeddings: Optional dictionary of precomputed embeddings keyed by match_id
            verbose: Whether to show progress bars
        """
        self.logger = logging.getLogger(__name__)
        self.sequence_length = sequence_length
        self.min_gap = min_gap
        self.max_gap = max_gap
        
        # Filter by match_ids if provided
        if match_ids is not None:
            events_df = events_df[events_df['match_id'].isin(match_ids)]
        
        # Get unique match IDs
        self.match_ids = events_df['match_id'].unique()
        self.logger.info(f"Processing {len(self.match_ids)} unique matches")
        
        # Group events by match_id
        self.match_events = {}
        self.embeddings = {}
        self.num_columns = events_df.shape[1] - 1  # Subtract match_id column
        
        # Organize data by match
        for match_id in tqdm(self.match_ids, desc="Organizing match data", disable=not verbose):
            match_df = events_df[events_df['match_id'] == match_id]
            
            # If precomputed embeddings are provided, use them
            if precomputed_embeddings is not None and match_id in precomputed_embeddings:
                self.embeddings[match_id] = precomputed_embeddings[match_id]
            else:
                # Otherwise, convert DataFrame to numpy array (excluding match_id column)
                match_df_no_id = match_df.drop(columns=['match_id'])
                self.embeddings[match_id] = match_df_no_id.values
                
            self.match_events[match_id] = len(self.embeddings[match_id])
        
        # Generate sequence pairs
        self.samples = self._generate_sequence_pairs(
            max_samples_per_match=max_samples_per_match,
            max_total_samples=max_total_samples,
            verbose=verbose
        )
        
        if shuffle:
            random.shuffle(self.samples)
            
        self.logger.info(f"Created dataset with {len(self.samples)} sequence pairs")
    
    def _generate_sequence_pairs(
        self, 
        max_samples_per_match: Optional[int] = None,
        max_total_samples: Optional[int] = None,
        verbose: bool = True
    ) -> List[Tuple]:
        """
        Generate pairs of sequences from the match events.
        
        Returns:
            List of tuples: (match_id, seq1_start, seq2_start, label)
        """
        samples = []
        total_samples = 0
        
        for match_id in tqdm(self.match_ids, desc="Generating sequence pairs", disable=not verbose):
            match_samples = []
            num_events = self.match_events[match_id]
            
            # Skip matches that don't have enough events
            if num_events < 2 * self.sequence_length + self.min_gap:
                self.logger.debug(f"Skipping match {match_id}: not enough events")
                continue
            
            # Compute the maximum valid starting position for the first sequence
            max_start = num_events - (2 * self.sequence_length + self.min_gap)
            
            # Generate random starting positions
            possible_starts = list(range(max_start + 1))
            random.shuffle(possible_starts)
            
            # Limit samples per match if specified
            if max_samples_per_match is not None:
                possible_starts = possible_starts[:max_samples_per_match]
            
            for seq1_start in possible_starts:
                seq1_end = seq1_start + self.sequence_length
                
                # Determine the range of valid starting positions for the second sequence
                seq2_start_min = seq1_end + self.min_gap
                seq2_start_max = num_events - self.sequence_length
                
                if self.max_gap is not None:
                    seq2_start_max = min(seq2_start_max, seq1_end + self.max_gap)
                
                if seq2_start_min <= seq2_start_max:
                    # Randomly choose to put the sequences in chronological or reverse order
                    if random.random() < 0.5:
                        # Chronological order (first before second)
                        seq2_start = random.randint(seq2_start_min, seq2_start_max)
                        label = 0
                    else:
                        # Reverse order (second before first)
                        temp = seq1_start
                        seq1_start = random.randint(seq2_start_min, seq2_start_max)
                        seq2_start = temp
                        label = 1
                    
                    match_samples.append((match_id, seq1_start, seq2_start, label))
            
            samples.extend(match_samples)
            total_samples += len(match_samples)
            
            if max_total_samples is not None and total_samples >= max_total_samples:
                self.logger.info(f"Reached maximum total samples ({max_total_samples})")
                samples = samples[:max_total_samples]
                break
                
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple containing:
                - Tensor of shape [2*sequence_length, embedding_dim] with concatenated sequences
                - Target label (0 or 1)
        """
        match_id, seq1_start, seq2_start, label = self.samples[idx]
        
        # Get embeddings for both sequences
        seq1 = self.embeddings[match_id][seq1_start:seq1_start + self.sequence_length]
        seq2 = self.embeddings[match_id][seq2_start:seq2_start + self.sequence_length]
        
        # Concatenate the sequences
        concatenated = np.concatenate([seq1, seq2], axis=0)
        
        # Convert to tensors
        X = torch.tensor(concatenated, dtype=torch.float32)
        y = torch.tensor(label, dtype=torch.long)
        
        return X, y


def create_data_loaders(
    events_df: pd.DataFrame,
    sequence_length: int = 10,
    min_gap: int = 1,
    max_gap: Optional[int] = None,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    batch_size: int = 32,
    precomputed_embeddings: Optional[Dict[str, np.ndarray]] = None,
    max_samples_per_match: Optional[int] = 1000,
    max_samples_total: Optional[int] = 100000,
    num_workers: int = 4,
    verbose: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders.
    
    Args:
        events_df: DataFrame containing event embeddings with match_id column
        sequence_length: Number of events in each sequence
        min_gap: Minimum number of events between sequences
        max_gap: Maximum number of events between sequences
        train_ratio: Proportion of matches to use for training
        val_ratio: Proportion of matches to use for validation
        test_ratio: Proportion of matches to use for testing
        batch_size: Batch size for data loaders
        precomputed_embeddings: Optional dictionary of precomputed embeddings
        max_samples_per_match: Maximum samples to generate per match
        max_samples_total: Maximum total samples
        num_workers: Number of workers for data loading
        verbose: Whether to show progress bars
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    logger = logging.getLogger(__name__)
    
    # Get unique match IDs and shuffle them
    match_ids = events_df['match_id'].unique().tolist()
    random.shuffle(match_ids)
    
    # Split match IDs into train, validation, and test sets
    num_matches = len(match_ids)
    train_size = int(train_ratio * num_matches)
    val_size = int(val_ratio * num_matches)
    
    train_match_ids = match_ids[:train_size]
    val_match_ids = match_ids[train_size:train_size + val_size]
    test_match_ids = match_ids[train_size + val_size:]
    
    logger.info(f"Split {num_matches} matches into {len(train_match_ids)} train, "
                f"{len(val_match_ids)} validation, and {len(test_match_ids)} test")
    
    # Create datasets
    train_dataset = EventSequenceDataset(
        events_df=events_df,
        sequence_length=sequence_length,
        min_gap=min_gap,
        max_gap=max_gap,
        match_ids=train_match_ids,
        shuffle=True,
        max_samples_per_match=max_samples_per_match,
        max_total_samples=max_samples_total,
        precomputed_embeddings=precomputed_embeddings,
        verbose=verbose
    )
    
    val_dataset = EventSequenceDataset(
        events_df=events_df,
        sequence_length=sequence_length,
        min_gap=min_gap,
        max_gap=max_gap,
        match_ids=val_match_ids,
        shuffle=True,
        max_samples_per_match=max_samples_per_match // 2,  # Use fewer samples for validation
        max_total_samples=max_samples_total // 10,  # Use fewer samples for validation
        precomputed_embeddings=precomputed_embeddings,
        verbose=verbose
    )
    
    test_dataset = EventSequenceDataset(
        events_df=events_df,
        sequence_length=sequence_length,
        min_gap=min_gap,
        max_gap=max_gap,
        match_ids=test_match_ids,
        shuffle=False,  # Don't shuffle test set
        max_samples_per_match=max_samples_per_match // 2,  # Use fewer samples for testing
        max_total_samples=max_samples_total // 10,  # Use fewer samples for testing
        precomputed_embeddings=precomputed_embeddings,
        verbose=verbose
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader
