"""
data_loading.py
---------------
Module for loading and preparing the match event data.
"""

import logging
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from typing import Optional


def load_match_events(csv_root_dir: str = "csv") -> pd.DataFrame:
    """
    Load all match event CSV files from the specified directory into a single DataFrame.

    Args:
        csv_root_dir (str): Path to the root directory containing the CSV files.

    Returns:
        pd.DataFrame: Combined DataFrame containing all match events.

    Raises:
        ValueError: If no CSV files are found or none are valid.
    """
    logger = logging.getLogger(__name__)
    root_path = Path(csv_root_dir)

    csv_files = list(root_path.glob("*.csv"))
    if not csv_files:
        raise ValueError(f"No CSV files found in {csv_root_dir}")

    logger.info(f"Found {len(csv_files)} CSV files to process")

    dfs = []
    for csv_file in tqdm(csv_files, desc="Loading CSV files"):
        try:
            temp_df = pd.read_csv(csv_file)
            # Add match_id from filename
            match_id = csv_file.stem.replace(".json", "")
            temp_df["match_id"] = match_id
            dfs.append(temp_df)
        except Exception as e:
            logger.error(f"Error processing {csv_file}: {str(e)}")
            continue

    if not dfs:
        raise ValueError("No valid CSV files were processed.")

    logger.info("Combining DataFrames...")
    combined_df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Final DataFrame shape: {combined_df.shape}")

    return combined_df


def cleanup_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop unused columns and perform minimal cleanup.

    Args:
        df (pd.DataFrame): DataFrame to clean.

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    columns_to_drop = ["Unnamed: 0", "match_id"]
    for col in columns_to_drop:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)

    return df
