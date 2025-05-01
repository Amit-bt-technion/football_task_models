"""
sequence_transformer.py
----------------------
A transformer model for sequence classification of football event embeddings.
"""

import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class PositionalEncoding(nn.Module):
    """
    Injects positional information into the input embeddings.
    """
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        # Create positional encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        # Register buffer (persistent state)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Args:
            x: Input embeddings [batch_size, seq_len, embedding_dim]
        Returns:
            Embeddings with positional encoding added
        """
        return x + self.pe[:, :x.size(1)]


class EventSequenceTransformer(nn.Module):
    """
    Transformer model for classifying the chronological order of two event sequences.
    """
    def __init__(self, embedding_dim=32, nhead=8, num_encoder_layers=6, dim_feedforward=2048, dropout=0.1):
        super(EventSequenceTransformer, self).__init__()
        
        self.embedding_dim = embedding_dim
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(embedding_dim)
        
        # Transformer encoder
        encoder_layers = TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_encoder_layers)
        
        # [CLS] token embedding - will be prepended to each sequence
        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))
        
        # Final classifier for the [CLS] token representation
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, 2)  # Binary classification: 0 = first before second, 1 = second before first
        )
        
    def forward(self, x, src_mask=None):
        """
        Args:
            x: Tensor of shape [batch_size, 2*seq_len, embedding_dim]
                Contains concatenated sequences to be classified
            src_mask: Optional mask for the transformer
        
        Returns:
            Classification logits (0 = first sequence is before second, 1 = second is before first)
        """
        batch_size = x.size(0)

        # Add CLS token to the beginning of the sequence
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Apply transformer encoder
        x = self.transformer_encoder(x, src_mask)

        # Use the [CLS] token representation for classification
        cls_representation = x[:, 0]

        # Classify
        logits = self.classifier(cls_representation)
        return logits
