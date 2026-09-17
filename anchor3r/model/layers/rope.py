# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0,
# included in LICENSE, section RoPE-ViT.txt.


# Implementation of 2D Rotary Position Embeddings (RoPE).

# This module provides a clean implementation of 2D Rotary Position Embeddings,
# which extends the original RoPE concept to handle 2D spatial positions.

# Inspired by:
#         https://github.com/meta-llama/codellama/blob/main/llama/model.py
#         https://github.com/naver-ai/rope-vit


import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class PositionGetter:
    """Generates and caches 2D spatial positions for patches in a grid.

    This class efficiently manages the generation of spatial coordinates for patches
    in a 2D grid, caching results to avoid redundant computations.

    Attributes:
        position_cache: Dictionary storing precomputed position tensors for different
            grid dimensions.
    """

    def __init__(self):
        """Initializes the position generator with an empty cache."""
        self.position_cache: Dict[Tuple[int, int], torch.Tensor] = {}

    def __call__(self, batch_size: int, height: int, width: int, device: torch.device) -> torch.Tensor:
        """Generates spatial positions for a batch of patches.

        Args:
            batch_size: Number of samples in the batch.
            height: Height of the grid in patches.
            width: Width of the grid in patches.
            device: Target device for the position tensor.

        Returns:
            Tensor of shape (batch_size, height*width, 2) containing y,x coordinates
            for each position in the grid, repeated for each batch item.
        """
        if (height, width) not in self.position_cache:
            y_coords = torch.arange(height, device=device)
            x_coords = torch.arange(width, device=device)
            positions = torch.cartesian_prod(y_coords, x_coords)
            self.position_cache[height, width] = positions

        cached_positions = self.position_cache[height, width]
        return cached_positions.view(1, height * width, 2).expand(batch_size, -1, -1).clone()


class RotaryPositionEmbedding2D(nn.Module):
    """2D Rotary Position Embedding implementation.

    This module applies rotary position embeddings to input tokens based on their
    2D spatial positions. It handles the position-dependent rotation of features
    separately for vertical and horizontal dimensions.

    Args:
        frequency: Base frequency for the position embeddings. Default: 100.0
        scaling_factor: Scaling factor for frequency computation. Default: 1.0

    Attributes:
        base_frequency: Base frequency for computing position embeddings.
        scaling_factor: Factor to scale the computed frequencies.
        frequency_cache: Cache for storing precomputed frequency components.
    """

    def __init__(self, frequency: float = 100.0, scaling_factor: float = 1.0):
        """Initializes the 2D RoPE module."""
        super().__init__()
        self.base_frequency = frequency
        self.scaling_factor = scaling_factor
        self.frequency_cache: Dict[Tuple, Tuple[torch.Tensor, torch.Tensor]] = {}

    def _compute_frequency_components(
        self, dim: int, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes frequency components for rotary embeddings.

        Args:
            dim: Feature dimension (must be even).
            seq_len: Maximum sequence length.
            device: Target device for computations.
            dtype: Data type for the computed tensors.

        Returns:
            Tuple of (cosine, sine) tensors for frequency components.
        """
        cache_key = (dim, seq_len, device, dtype)
        if cache_key not in self.frequency_cache:
            # Compute frequency bands
            exponents = torch.arange(0, dim, 2, device=device).float() / dim
            inv_freq = 1.0 / (self.base_frequency**exponents)

            # Generate position-dependent frequencies
            positions = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
            angles = torch.einsum("i,j->ij", positions, inv_freq)

            # Compute and cache frequency components
            angles = angles.to(dtype)
            angles = torch.cat((angles, angles), dim=-1)
            cos_components = angles.cos().to(dtype)
            sin_components = angles.sin().to(dtype)
            self.frequency_cache[cache_key] = (cos_components, sin_components)

        return self.frequency_cache[cache_key]

    @staticmethod
    def _rotate_features(x: torch.Tensor) -> torch.Tensor:
        """Performs feature rotation by splitting and recombining feature dimensions.

        Args:
            x: Input tensor to rotate.

        Returns:
            Rotated feature tensor.
        """
        feature_dim = x.shape[-1]
        x1, x2 = x[..., : feature_dim // 2], x[..., feature_dim // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def _apply_1d_rope(
        self, tokens: torch.Tensor, positions: torch.Tensor, cos_comp: torch.Tensor, sin_comp: torch.Tensor
    ) -> torch.Tensor:
        """Applies 1D rotary position embeddings along one dimension.

        Args:
            tokens: Input token features.
            positions: Position indices.
            cos_comp: Cosine components for rotation.
            sin_comp: Sine components for rotation.

        Returns:
            Tokens with applied rotary position embeddings.
        """
        # Embed positions with frequency components
        cos = F.embedding(positions, cos_comp)[:, None, :, :]
        sin = F.embedding(positions, sin_comp)[:, None, :, :]

        # Apply rotation
        return (tokens * cos) + (self._rotate_features(tokens) * sin)

    def forward(self, tokens: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Applies 2D rotary position embeddings to input tokens.

        Args:
            tokens: Input tensor of shape (batch_size, n_heads, n_tokens, dim).
                   The feature dimension (dim) must be divisible by 4.
            positions: Position tensor of shape (batch_size, n_tokens, 2) containing
                      the y and x coordinates for each token.

        Returns:
            Tensor of same shape as input with applied 2D rotary position embeddings.

        Raises:
            AssertionError: If input dimensions are invalid or positions are malformed.
        """
        # Validate inputs
        assert tokens.size(-1) % 2 == 0, "Feature dimension must be even"
        assert positions.ndim == 3 and positions.shape[-1] == 2, "Positions must have shape (batch_size, n_tokens, 2)"

        # Compute feature dimension for each spatial direction
        feature_dim = tokens.size(-1) // 2

        # Get frequency components
        max_position = int(positions.max()) + 1
        cos_comp, sin_comp = self._compute_frequency_components(feature_dim, max_position, tokens.device, tokens.dtype)

        # Split features for vertical and horizontal processing
        vertical_features, horizontal_features = tokens.chunk(2, dim=-1)

        # Apply RoPE separately for each dimension
        vertical_features = self._apply_1d_rope(vertical_features, positions[..., 0], cos_comp, sin_comp)
        horizontal_features = self._apply_1d_rope(horizontal_features, positions[..., 1], cos_comp, sin_comp)

        # Combine processed features
        return torch.cat((vertical_features, horizontal_features), dim=-1)


class RotaryPositionEmbedding3D(nn.Module):
    """3D RoPE over (t, y, x) with fixed periodic time harmonics."""

    _TIME_PERIOD: int = 128
    _TIME_DIM: int = 16
    DEFAULT_TIME_HARMONICS: Tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8)

    @property
    def time_period(self) -> int:
        return self._TIME_PERIOD

    @property
    def time_dim(self) -> int:
        return self._TIME_DIM

    def __init__(
        self,
        frequency: float = 100.0,
        scaling_factor: float = 1.0,
    ):
        super().__init__()
        self.base_frequency = frequency
        self.scaling_factor = scaling_factor
        self.frequency_cache: Dict[Tuple, Tuple[torch.Tensor, torch.Tensor]] = {}
        if self.time_dim // 2 != len(self.DEFAULT_TIME_HARMONICS):
            raise ValueError(
                f"_TIME_DIM={self._TIME_DIM} requires {self.time_dim // 2} time harmonics, "
                f"but {len(self.DEFAULT_TIME_HARMONICS)} are defined."
            )

    def _compute_frequency_components(
        self, dim: int, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cache_key = (dim, seq_len, device, dtype)
        if cache_key not in self.frequency_cache:
            exponents = torch.arange(0, dim, 2, device=device).float() / dim
            inv_freq = 1.0 / (self.base_frequency**exponents)
            positions = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
            angles = torch.einsum("i,j->ij", positions, inv_freq)
            angles = angles.to(dtype)
            angles = torch.cat((angles, angles), dim=-1)
            cos_components = angles.cos().to(dtype)
            sin_components = angles.sin().to(dtype)
            self.frequency_cache[cache_key] = (cos_components, sin_components)

        return self.frequency_cache[cache_key]

    @staticmethod
    def _rotate_features(x: torch.Tensor) -> torch.Tensor:
        feature_dim = x.shape[-1]
        x1, x2 = x[..., : feature_dim // 2], x[..., feature_dim // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def _apply_1d_rope(
        self, tokens: torch.Tensor, positions: torch.Tensor, cos_comp: torch.Tensor, sin_comp: torch.Tensor
    ) -> torch.Tensor:
        cos = F.embedding(positions, cos_comp)[:, None, :, :]
        sin = F.embedding(positions, sin_comp)[:, None, :, :]
        return (tokens * cos) + (self._rotate_features(tokens) * sin)

    def _apply_time_rope(self, tokens: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        d = tokens.size(-1)
        requested_pairs = d // 2
        max_unique_harmonic = (self.time_period - 1) // 2
        if max_unique_harmonic <= 0:
            return tokens
        harmonic_values = [
            harmonic for harmonic in self.DEFAULT_TIME_HARMONICS[:requested_pairs]
            if harmonic <= max_unique_harmonic
        ]
        if len(harmonic_values) != requested_pairs:
            raise ValueError(
                f"time_period={self.time_period} cannot support the first {requested_pairs} "
                f"time harmonics {self.DEFAULT_TIME_HARMONICS[:requested_pairs]}."
            )
        num_pairs = len(harmonic_values)
        rot_d = 2 * num_pairs
        if rot_d == 0:
            return tokens

        device = tokens.device
        harmonics = torch.tensor(harmonic_values, device=device, dtype=torch.float32)
        omega = (2.0 * torch.pi / float(self.time_period)) * harmonics

        time_pos = torch.remainder(positions, self.time_period).to(device=device, dtype=torch.float32)
        angles = time_pos[..., None] * omega.view(1, 1, -1)
        angles = torch.cat((angles, angles), dim=-1)
        cos = angles.cos().to(tokens.dtype)[:, None, :, :]
        sin = angles.sin().to(tokens.dtype)[:, None, :, :]

        rotated = (tokens[..., :rot_d] * cos) + (self._rotate_features(tokens[..., :rot_d]) * sin)
        if rot_d < d:
            return torch.cat([rotated, tokens[..., rot_d:]], dim=-1)
        return rotated

    def forward(self, tokens: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        assert positions.ndim == 3 and positions.shape[-1] == 3, "Positions must have shape (batch_size, n_tokens, 3)"

        head_dim = tokens.size(-1)
        if self.time_dim >= head_dim:
            raise ValueError(f"time_dim ({self.time_dim}) must be smaller than head_dim ({head_dim})")
        spatial_dim = head_dim - self.time_dim
        if spatial_dim % 4 != 0:
            raise ValueError(
                f"head_dim - time_dim must be divisible by 4 so y/x dims are even, "
                f"got head_dim={head_dim}, time_dim={self.time_dim}"
            )

        y_dim = spatial_dim // 2
        x_dim = spatial_dim - y_dim
        spatial_max_position = int(positions[..., 1:].max().item()) + 1

        def rotate_spatial_part(part: torch.Tensor, pos_idx: int) -> torch.Tensor:
            d = part.size(-1)
            rot_d = (d // 2) * 2
            if rot_d == 0:
                return part
            cos_c, sin_c = self._compute_frequency_components(rot_d, spatial_max_position, part.device, part.dtype)
            rotated = self._apply_1d_rope(part[..., :rot_d], positions[..., pos_idx], cos_c, sin_c)
            if rot_d < d:
                return torch.cat([rotated, part[..., rot_d:]], dim=-1)
            return rotated

        t_f = self._apply_time_rope(tokens[..., :self.time_dim], positions[..., 0])
        y_f = rotate_spatial_part(tokens[..., self.time_dim:self.time_dim + y_dim], 1)
        x_f = rotate_spatial_part(tokens[..., self.time_dim + y_dim:self.time_dim + y_dim + x_dim], 2)
        return torch.cat([t_f, y_f, x_f], dim=-1)
