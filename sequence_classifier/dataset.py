"""
dataset.py
---------
Custom dataset for loading and preparing sequences of football events with flexible sampling strategies.
"""

import logging
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Dict, Optional, Callable
import random
from tqdm import tqdm

# Task registry to store sampling and labeling functions
TASK_REGISTRY = {}


def register_task(task_name):
    """Decorator to register task functions in the global registry."""
    def decorator(func):
        TASK_REGISTRY[task_name] = func
        return func
    return decorator


class EventSequenceDataset(Dataset):
    """
    Dataset for creating samples of event sequences from embedded football events.
    Supports different sampling strategies based on the task.
    """
    
    def __init__(
        self, 
        events_df: pd.DataFrame,
        precomputed_embeddings: Dict[str, np.ndarray],
        sequence_length: int = 10,
        min_gap: int = 1,
        max_gap: Optional[int] = None,
        match_ids: Optional[List[str]] = None,
        shuffle: bool = True,
        max_samples_per_match: Optional[int] = None,
        max_total_samples: Optional[int] = None,
        task: str = "chronological_order",  # Default task
        task_params: Optional[Dict] = None,
        verbose: bool = True
    ):
        """
        Initialize the EventSequenceDataset.
        
        Args:
            events_df: DataFrame containing event embeddings with match_id column
            precomputed_embeddings: dictionary of precomputed embeddings keyed by match_id
            sequence_length: Number of events in each sequence
            min_gap: Minimum number of events between sequences
            max_gap: Maximum number of events between sequences (if None, no limit)
            match_ids: Optional list of match IDs to filter by
            shuffle: Whether to shuffle the samples
            max_samples_per_match: Maximum number of samples to generate per match
            max_total_samples: Maximum total samples across all matches
            task: Name of the task to determine sampling strategy
            task_params: Additional parameters specific to the task
            verbose: Whether to show progress bars
        """
        self.logger = logging.getLogger(__name__)
        self.sequence_length = sequence_length
        self.min_gap = min_gap
        self.max_gap = max_gap
        self.task = task
        self.task_params = task_params or {}

        # Check if task exists in registry
        if task not in TASK_REGISTRY:
            raise ValueError(f"Task '{task}' not found in registry. Available tasks: {list(TASK_REGISTRY.keys())}")

        # Filter by match_ids if provided
        if match_ids is not None:
            events_df = events_df[events_df['match_id'].isin(match_ids)]
        
        # Get unique match IDs
        self.match_ids = events_df['match_id'].unique()
        self.logger.info(f"Processing {len(self.match_ids)} unique matches for task '{task}'")
        
        # Group events by match_id
        self.match_events = {}
        self.embeddings = {}
        self.num_columns = events_df.shape[1] - 1  # Subtract match_id column
        
        # Organize data by match
        for match_id in tqdm(self.match_ids, desc="Organizing match data", disable=not verbose):
            match_df = events_df[events_df['match_id'] == match_id]
            
            # If no precomputed embeddings are provided, raise ValueError
            if match_id not in precomputed_embeddings:
                raise ValueError(f"No embeddings for match {match_id}.")

            # Set events and embeddings properties
            self.embeddings[match_id] = precomputed_embeddings[match_id]
            self.match_events[match_id] = len(self.embeddings[match_id])
        
        # Generate samples using the task-specific function
        self.samples = self._generate_samples(
            max_samples_per_match=max_samples_per_match,
            max_total_samples=max_total_samples,
            verbose=verbose
        )
        
        if shuffle:
            random.shuffle(self.samples)
            
        self.logger.info(f"Created dataset with {len(self.samples)} samples for task '{task}'")
    
    def _generate_samples(
        self, 
        max_samples_per_match: Optional[int] = None,
        max_total_samples: Optional[int] = None,
        verbose: bool = True
    ) -> List[Tuple]:
        """
        Generate samples using the task-specific sampling function.
        
        Returns:
            List of samples as defined by the task-specific function
        """
        # Get the task-specific sampling function
        sampling_func = TASK_REGISTRY[self.task]

        samples = []
        total_samples = 0
        
        for match_id in tqdm(self.match_ids, desc=f"Generating samples for {self.task}", disable=not verbose):
            num_events = self.match_events[match_id]
            
            # Skip matches that don't have enough events for the minimum sequence length
            if num_events < self.sequence_length:
                self.logger.debug(f"Skipping match {match_id}: not enough events")
                continue
            
            # Call the task-specific sampling function
            match_samples = sampling_func(
                match_id=match_id,
                num_events=num_events,
                sequence_length=self.sequence_length,
                min_gap=self.min_gap,
                max_gap=self.max_gap,
                max_samples=max_samples_per_match,
                **self.task_params
            )

            samples.extend(match_samples)
            total_samples += len(match_samples)

            if max_total_samples is not None and total_samples >= max_total_samples:
                self.logger.info(f"Reached maximum total samples ({max_total_samples})")
                samples = samples[:max_total_samples]
                break

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple:
        """
        Get a single sample.

        Args:
            idx: Sample index

        Returns:
            A tuple containing input tensor(s) and target tensor(s) as defined by the task
        """
        sample = self.samples[idx]

        # Unpack sample information (format depends on task)
        match_id = sample[0]  # First element is always match_id

        # For chronological_order task (maintain backward compatibility)
        if self.task == "chronological_order":
            seq1_start, seq2_start, label = sample[1:]

            # Get embeddings for both sequences
            seq1 = self.embeddings[match_id][seq1_start:seq1_start + self.sequence_length]
            seq2 = self.embeddings[match_id][seq2_start:seq2_start + self.sequence_length]

            # Concatenate the sequences
            concatenated = np.concatenate([seq1, seq2], axis=0)

            # Convert to tensors
            X = torch.tensor(concatenated, dtype=torch.float32)
            y = torch.tensor(label, dtype=torch.long)

            return X, y

        # For next_event_prediction task
        elif self.task == "next_event_prediction":
            seq_start, next_event_idx = sample[1:]

            # Get embeddings for sequence and next event
            seq = self.embeddings[match_id][seq_start:seq_start + self.sequence_length]
            next_event = self.embeddings[match_id][next_event_idx]

            # Convert to tensors
            X = torch.tensor(seq, dtype=torch.float32)
            y = torch.tensor(next_event, dtype=torch.float32)

            return X, y

        # For event_type_classification task
        elif self.task == "event_type_classification":
            seq_start, event_types = sample[1], sample[2]

            # Get embeddings for sequence
            seq = self.embeddings[match_id][seq_start:seq_start + self.sequence_length]

            # Convert to tensors
            X = torch.tensor(seq, dtype=torch.float32)
            y = torch.tensor(event_types, dtype=torch.long)

            return X, y

        # For custom tasks, parse the sample structure as needed
        else:
            # The specific implementation depends on how the task's sampling function formats its output
            # This is a flexible extension point for adding more tasks
            # Process remaining elements based on task-specific format
            task_data = sample[1:]

            # Use task-specific processing logic
            X_data, y_data = self._process_task_sample(match_id, task_data)

            # Convert to tensors
            X = torch.tensor(X_data, dtype=torch.float32)
            y = torch.tensor(y_data, dtype=torch.float32 if isinstance(y_data[0], float) else torch.long)

            return X, y

    def _process_task_sample(self, match_id, task_data):
        """
        Process task-specific sample data.
        This method can be overridden by subclasses for custom tasks.

        Args:
            match_id: Match ID
            task_data: Task-specific data from the sample

        Returns:
            Tuple of (X_data, y_data) for the model
        """
        raise NotImplementedError(f"Processing for task '{self.task}' not implemented")


# Register task functions

@register_task("chronological_order")
def sample_chronological_order(
    match_id: str,
    num_events: int,
    sequence_length: int,
    min_gap: int,
    max_gap: Optional[int] = None,
    max_samples: Optional[int] = None,
    **kwargs
) -> List[Tuple]:
    """
    Generate sequence pairs for chronological order classification.

    Args:
        match_id: Match ID
        num_events: Number of events in the match
        sequence_length: Number of events in each sequence
        min_gap: Minimum number of events between sequences
        max_gap: Maximum number of events between sequences
        max_samples: Maximum number of samples to generate

    Returns:
        List of tuples: (match_id, seq1_start, seq2_start, label)
    """
    samples = []

    # Skip matches that don't have enough events
    if num_events < 2 * sequence_length + min_gap:
        return samples

    # Compute the maximum valid starting position for the first sequence
    max_start = num_events - (2 * sequence_length + min_gap)

    # Generate random starting positions
    possible_starts = list(range(max_start + 1))
    random.shuffle(possible_starts)

    # Limit samples per match if specified
    if max_samples is not None:
        possible_starts = possible_starts[:max_samples]

    for seq1_start in possible_starts:
        seq1_end = seq1_start + sequence_length

        # Determine the range of valid starting positions for the second sequence
        seq2_start_min = seq1_end + min_gap
        seq2_start_max = num_events - sequence_length

        if max_gap is not None:
            seq2_start_max = min(seq2_start_max, seq1_end + max_gap)

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
            
            samples.append((match_id, seq1_start, seq2_start, label))

    return samples


@register_task("next_event_prediction")
def sample_next_event_prediction(
    match_id: str,
    num_events: int,
    sequence_length: int,
    min_gap: int = 1,
    max_gap: Optional[int] = None,
    max_samples: Optional[int] = None,
    prediction_distance: int = 1,
    **kwargs
) -> List[Tuple]:
    """
    Generate samples for next event prediction.

    Args:
        match_id: Match ID
        num_events: Number of events in the match
        sequence_length: Number of events in each sequence
        min_gap: Not used in this task but kept for API consistency
        max_gap: Not used in this task but kept for API consistency
        max_samples: Maximum number of samples to generate
        prediction_distance: How many events ahead to predict

    Returns:
        List of tuples: (match_id, seq_start, next_event_idx)
    """
    samples = []

    # Skip matches that don't have enough events
    if num_events < sequence_length + prediction_distance:
        return samples

    # Maximum starting position to ensure we have enough events for the sequence and the target
    max_start = num_events - (sequence_length + prediction_distance)

    # Generate random starting positions
    possible_starts = list(range(max_start + 1))
    random.shuffle(possible_starts)

    # Limit samples per match if specified
    if max_samples is not None:
        possible_starts = possible_starts[:max_samples]

    for seq_start in possible_starts:
        next_event_idx = seq_start + sequence_length + prediction_distance - 1
        samples.append((match_id, seq_start, next_event_idx))

    return samples


@register_task("event_type_classification")
def sample_event_type_classification(
    match_id: str,
    num_events: int,
    sequence_length: int,
    min_gap: int = 1,
    max_gap: Optional[int] = None,
    max_samples: Optional[int] = None,
    events_df: Optional[pd.DataFrame] = None,
    **kwargs
) -> List[Tuple]:
    """
    Generate samples for event type classification.

    Args:
        match_id: Match ID
        num_events: Number of events in the match
        sequence_length: Number of events in each sequence
        min_gap: Not used in this task but kept for API consistency
        max_gap: Not used in this task but kept for API consistency
        max_samples: Maximum number of samples to generate
        events_df: DataFrame containing event data with types

    Returns:
        List of tuples: (match_id, seq_start, event_types)
    """
    samples = []

    # Skip matches that don't have enough events
    if num_events < sequence_length:
        return samples

    # If events_df is not provided, we can't determine event types
    if events_df is None:
        raise ValueError("events_df must be provided for event_type_classification task")

    # Filter events for this match
    match_events = events_df[events_df['match_id'] == match_id]

    # Maximum starting position
    max_start = num_events - sequence_length

    # Generate random starting positions
    possible_starts = list(range(max_start + 1))
    random.shuffle(possible_starts)

    # Limit samples per match if specified
    if max_samples is not None:
        possible_starts = possible_starts[:max_samples]

    for seq_start in possible_starts:
        # Get event types for the sequence
        sequence_indices = range(seq_start, seq_start + sequence_length)
        event_types = match_events.iloc[sequence_indices]['event_type'].values

        samples.append((match_id, seq_start, event_types))

    return samples


# You can add more task functions here


def create_data_loaders(
    task: str,
    events_df: pd.DataFrame,
    sequence_length: int = 10,
    min_gap: int = 1,
    max_gap: Optional[int] = None,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    batch_size: int = 32,
    precomputed_embeddings: Optional[Dict[str, np.ndarray]] = None,
    max_samples_per_match: Optional[int] = 1000,
    max_samples_total: Optional[int] = 100000,
    num_workers: int = 4,
    task_params: Optional[Dict] = None,
    verbose: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders.
    
    Args:
        task: Name of the task to determine sampling strategy
        events_df: DataFrame containing event embeddings with match_id column
        sequence_length: Number of events in each sequence
        min_gap: Minimum number of events between sequences
        max_gap: Maximum number of events between sequences
        train_ratio: Proportion of matches to use for training
        val_ratio: Proportion of matches to use for validation
        batch_size: Batch size for data loaders
        precomputed_embeddings: Optional dictionary of precomputed embeddings
        max_samples_per_match: Maximum samples to generate per match
        max_samples_total: Maximum total samples
        num_workers: Number of workers for data loading
        task_params: Additional parameters specific to the task
        verbose: Whether to show progress bars
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    logger = logging.getLogger(__name__)
    
    # Merge task parameters with events_df for tasks that need it
    if task_params is None:
        task_params = {}

    # For tasks that need the events_df
    if task == "event_type_classification":
        task_params['events_df'] = events_df

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
        task=task,
        task_params=task_params,
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
        task=task,
        task_params=task_params,
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
        task=task,
        task_params=task_params,
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
