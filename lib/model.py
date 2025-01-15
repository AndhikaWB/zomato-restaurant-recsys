import polars as pl
import torch

from torch import nn
from torch.utils.data import Dataset


class MixedDataset(Dataset):
    """Dataset with 2 distinct types (text and numeric).

    The input must be a Polars dataframe. The numeric (other) columns
    will be detected automatically based on text and target columns.
    """

    def __init__(self, df: pl.DataFrame, text_cols: pl.Expr, target_cols: pl.Expr):
        # Only convert to tensor when needed
        self.df = df
        # Text columns for embedding purpose
        self.text_cols = text_cols
        # Target columns (can be single or one-hot encoded)
        self.target_cols = target_cols

    def __getitem__(self, idx) -> dict:
        text_cols = self.df.select(self.text_cols)
        target_cols = self.df.select(self.target_cols)

        # Non-text columns that won't go to embedding
        other_cols = self.df.select(pl.exclude(text_cols.columns + target_cols.columns))

        # Automatically use GPU tensor if set by user
        device = str(torch.get_default_device())

        return {
            # Features
            'text': text_cols.to_torch(dtype=pl.Int32)[idx].to(device),
            'other': other_cols.to_torch(dtype=pl.Float32)[idx].to(device),
            # Target
            'target': target_cols.to_torch(dtype=pl.Float32)[idx].to(device),
        }

    def __len__(self):
        return len(self.df)


class TextEmbedding(nn.Module):
    """Classification/regression model for building text embedding.

    Args:
        embedding_vocab (int): Embedding vocabulary size.
        embedding_dim (int): Embedding dimension to represent each
            sentence.
        text_features (int): Dimension of text input features.
        other_features (int): Dimension of other input features.
        target_classes (int): Number of target classes.
    """

    def __init__(
        self,
        embedding_vocab: int,
        embedding_dim: int,
        text_features: int,
        other_features: int,
        target_classes: int,
    ):
        super().__init__()

        self.text_input = nn.Sequential(
            # If we use 0 to pad the sentences, then padding_idx = 0
            nn.EmbeddingBag(embedding_vocab, embedding_dim, padding_idx=0),
            nn.Linear(embedding_dim, 128),
        )

        self.other_input = nn.Linear(other_features, 128)

        self.combined_input = nn.Sequential(
            nn.ReLU(), nn.BatchNorm1d(256), nn.Linear(256, target_classes)
        )

    def forward(self, text_inputs: torch.Tensor, other_inputs: torch.Tensor):
        out1 = self.text_input(text_inputs)

        out2 = self.other_input(other_inputs)

        # Combine the previous layer output
        out3 = torch.cat([out1, out2], dim=1)
        out3 = self.combined_input(out3)

        return out3

    def reset_parameters(self, module=None):
        if not module:
            module = self

        for layer in module.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
            # Recursive reset on all children
            self.reset_parameters(layer)
