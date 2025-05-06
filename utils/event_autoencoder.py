"""
event_autoencoder.py
--------
Contains the EventAutoencoder model class definition.
"""

import torch
import torch.nn as nn
from typing import Tuple


class EventAutoencoder(nn.Module):
    """
    A multi-layer autoencoder for event vectors.
    """

    def __init__(self, input_dim: int = 128, latent_dim: int = 32) -> None:
        """
        Initializes the EventAutoencoder.

        Args:
            input_dim (int): Dimension of the input features.
            latent_dim (int): Dimension of the latent space.
        """
        super(EventAutoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, latent_dim)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (Reconstructed input, Latent code)
        """
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pass input only through the encoder to get embeddings.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Latent embeddings.
        """
        return self.encoder(x)
