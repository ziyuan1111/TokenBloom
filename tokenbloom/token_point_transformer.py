"""ReSplat-style KNN Point Transformer for batched ATsplat tokens.

The attention and residual blocks are ported from ReSplat's
``PlainPointTransformer``.  ReSplat's optional CUDA ``pointops`` package is not
available in the TokenGS environment, so only its KNN/grouping primitives are
replaced by equivalent batched PyTorch operations.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def compute_batched_knn(
    token_xyz: torch.Tensor,
    token_valid_mask: torch.Tensor,
    knn_samples: int,
    query_chunk_size: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return XYZ KNN indices and a validity mask without XYZ gradients."""
    if token_xyz.ndim != 3 or token_xyz.shape[-1] != 3:
        raise ValueError("token_xyz must have shape [B,N,3]")
    if token_valid_mask.shape != token_xyz.shape[:2]:
        raise ValueError("token_valid_mask must have shape [B,N]")
    if knn_samples <= 0:
        raise ValueError("knn_samples must be positive")
    if query_chunk_size <= 0:
        raise ValueError("query_chunk_size must be positive")
    batch, num_tokens = token_xyz.shape[:2]
    if num_tokens == 0:
        raise ValueError("at least one token is required")
    actual_k = min(int(knn_samples), num_tokens)
    index_chunks: list[torch.Tensor] = []
    valid_chunks: list[torch.Tensor] = []
    with torch.no_grad():
        xyz = token_xyz.detach().float()
        key_valid = token_valid_mask.detach().bool()
        for start in range(0, num_tokens, query_chunk_size):
            end = min(num_tokens, start + query_chunk_size)
            distances = torch.cdist(xyz[:, start:end], xyz)
            distances = distances.masked_fill(
                ~key_valid[:, None, :], torch.inf
            )
            indices = distances.topk(
                actual_k, dim=-1, largest=False, sorted=True
            ).indices
            gathered_valid = torch.gather(
                key_valid[:, None, :].expand(batch, end - start, num_tokens),
                dim=2,
                index=indices,
            )
            query_valid = key_valid[:, start:end, None]
            index_chunks.append(indices)
            valid_chunks.append(gathered_valid & query_valid)
    return torch.cat(index_chunks, dim=1), torch.cat(valid_chunks, dim=1)


class TokenKNNAttention(nn.Module):
    """Single-head dot-product attention over precomputed XYZ neighbors."""

    def __init__(
        self,
        channels: int,
        knn_samples: int = 16,
        proj_channels: int | None = None,
    ) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        if knn_samples <= 0:
            raise ValueError("knn_samples must be positive")
        if proj_channels is not None and proj_channels <= 0:
            raise ValueError("proj_channels must be positive")
        self.knn_samples = int(knn_samples)
        self.proj_channels = channels if proj_channels is None else int(proj_channels)
        self.qkv = nn.Linear(channels, self.proj_channels * 3, bias=False)
        self.proj = nn.Linear(self.proj_channels, channels)

    def forward(
        self,
        features: torch.Tensor,
        knn_indices: torch.Tensor,
        neighbor_valid_mask: torch.Tensor,
        token_valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        if features.ndim != 3:
            raise ValueError("features must have shape [B,N,C]")
        batch, num_tokens = features.shape[:2]
        if knn_indices.shape[:2] != (batch, num_tokens):
            raise ValueError("knn_indices must have shape [B,N,K]")
        if neighbor_valid_mask.shape != knn_indices.shape:
            raise ValueError("neighbor_valid_mask must match knn_indices")
        if token_valid_mask.shape != (batch, num_tokens):
            raise ValueError("token_valid_mask must have shape [B,N]")

        query, key, value = self.qkv(features).chunk(3, dim=-1)
        batch_indices = torch.arange(batch, device=features.device)[:, None, None]
        neighbor_keys = key[batch_indices, knn_indices]
        neighbor_values = value[batch_indices, knn_indices]
        scale = self.proj_channels**-0.5
        scores = (query.unsqueeze(2) * neighbor_keys).sum(dim=-1) * scale
        valid = neighbor_valid_mask & token_valid_mask[:, :, None]
        finite_floor = -torch.finfo(scores.dtype).max
        attention = torch.softmax(scores.masked_fill(~valid, finite_floor), dim=-1)
        attention = attention * valid.to(attention.dtype)
        attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(
            torch.finfo(attention.dtype).eps
        )
        output = (attention.unsqueeze(-1) * neighbor_values).sum(dim=2)
        output = self.proj(output)
        return output * token_valid_mask.unsqueeze(-1).to(output.dtype)


class TokenPointTransformerMLP(nn.Module):
    """ReSplat's four-times expanded GELU feed-forward network."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(channels, channels * 4)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(channels * 4, channels)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(features)))


class TokenPointTransformerBlock(nn.Module):
    """Pre-norm KNN attention followed by a pre-norm residual MLP."""

    def __init__(
        self,
        channels: int,
        knn_samples: int,
        attn_proj_channels: int | None,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.attn = TokenKNNAttention(
            channels,
            knn_samples=knn_samples,
            proj_channels=attn_proj_channels,
        )
        self.norm2 = nn.LayerNorm(channels)
        self.mlp = TokenPointTransformerMLP(channels)

    def forward(
        self,
        features: torch.Tensor,
        knn_indices: torch.Tensor,
        neighbor_valid_mask: torch.Tensor,
        token_valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        mask = token_valid_mask.unsqueeze(-1).to(features.dtype)
        features = features + self.attn(
            self.norm1(features),
            knn_indices,
            neighbor_valid_mask,
            token_valid_mask,
        )
        features = features * mask
        features = features + self.mlp(self.norm2(features))
        return features * mask


class TokenPlainPointTransformer(nn.Module):
    """Four-block ReSplat Point Transformer with XYZ used only for KNN."""

    def __init__(
        self,
        channels: int,
        knn_samples: int = 8,
        num_blocks: int = 4,
        attn_proj_channels: int | None = 64,
        knn_query_chunk_size: int = 1024,
    ) -> None:
        super().__init__()
        if num_blocks <= 0:
            raise ValueError("num_blocks must be positive")
        self.knn_samples = int(knn_samples)
        self.knn_query_chunk_size = int(knn_query_chunk_size)
        self.blocks = nn.ModuleList(
            [
                TokenPointTransformerBlock(
                    channels,
                    knn_samples=self.knn_samples,
                    attn_proj_channels=attn_proj_channels,
                )
                for _ in range(num_blocks)
            ]
        )

    def compute_knn_indices(
        self,
        token_xyz: torch.Tensor,
        token_valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return compute_batched_knn(
            token_xyz,
            token_valid_mask,
            knn_samples=self.knn_samples,
            query_chunk_size=self.knn_query_chunk_size,
        )

    def forward(
        self,
        token_xyz: torch.Tensor,
        token_features: torch.Tensor,
        token_valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if token_features.ndim != 3:
            raise ValueError("token_features must have shape [B,N,C]")
        if token_xyz.shape[:2] != token_features.shape[:2]:
            raise ValueError("token_xyz and token_features dimensions differ")
        if token_valid_mask is None:
            token_valid_mask = torch.ones(
                token_features.shape[:2],
                dtype=torch.bool,
                device=token_features.device,
            )
        knn_indices, neighbor_valid_mask = self.compute_knn_indices(
            token_xyz,
            token_valid_mask,
        )
        features = token_features
        for block in self.blocks:
            features = block(
                features,
                knn_indices,
                neighbor_valid_mask,
                token_valid_mask,
            )
        return features

