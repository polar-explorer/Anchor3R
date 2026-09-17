from __future__ import annotations

import torch
import torch.nn as nn

from anchor3r.model.layers import Mlp
from anchor3r.model.layers.block import Block


class CameraHead(nn.Module):
    def __init__(
        self,
        dim_in: int = 2048,
        trunk_depth: int = 4,
        pose_encoding_type: str = "absT_quaR",
        num_heads: int = 16,
        mlp_ratio: int = 4,
        init_values: float = 0.01,
    ):
        super().__init__()
        if pose_encoding_type != "absT_quaR":
            raise ValueError("The public Anchor3R inference config only supports absT_quaR camera output.")

        self.target_dim = 7
        self.trunk = nn.ModuleList(
            [
                Block(
                    dim=dim_in,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    init_values=init_values,
                )
                for _ in range(trunk_depth)
            ]
        )
        self.token_norm = nn.LayerNorm(dim_in)
        self.trunk_norm = nn.LayerNorm(dim_in)
        self.empty_pose_tokens = nn.Parameter(torch.zeros(1, 1, self.target_dim))
        self.embed_pose = nn.Linear(self.target_dim, dim_in)
        self.poseLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim_in, 3 * dim_in, bias=True))
        self.adaln_norm = nn.LayerNorm(dim_in, elementwise_affine=False, eps=1e-6)
        self.pose_branch = Mlp(
            in_features=dim_in,
            hidden_features=dim_in // 2,
            out_features=self.target_dim,
        )

    def trunk_fn(self, pose_tokens: torch.Tensor, num_iterations: int, attn_mask: torch.Tensor | None = None) -> list:
        batch, seq_len, _ = pose_tokens.shape
        pred_pose_enc = None
        pred_pose_enc_list = []

        for _ in range(num_iterations):
            if pred_pose_enc is None:
                module_input = self.embed_pose(self.empty_pose_tokens.expand(batch, seq_len, -1))
            else:
                pred_pose_enc = pred_pose_enc.detach()
                module_input = self.embed_pose(pred_pose_enc)

            shift_msa, scale_msa, gate_msa = self.poseLN_modulation(module_input).chunk(3, dim=-1)
            pose_tokens_modulated = gate_msa * _modulate(self.adaln_norm(pose_tokens), shift_msa, scale_msa)
            pose_tokens_modulated = pose_tokens_modulated + pose_tokens

            for block in self.trunk:
                pose_tokens_modulated = block(pose_tokens_modulated, attn_mask=attn_mask)

            pred_pose_delta = self.pose_branch(self.trunk_norm(pose_tokens_modulated))
            pred_pose_enc = pred_pose_delta if pred_pose_enc is None else pred_pose_enc + pred_pose_delta
            pred_pose_enc_list.append(pred_pose_enc)

        return pred_pose_enc_list


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale) + shift


class BatchCameraHead(CameraHead):
    def forward(
        self,
        win_pose_tokens: torch.Tensor,
        chunk_idx: int,
        num_iterations: int = 4,
        first_chunk_full: bool = True,
    ) -> list:
        _, num_rows, window_size, _ = win_pose_tokens.shape
        if chunk_idx == 0 and num_rows != window_size:
            raise ValueError(f"First chunk rows ({num_rows}) must match window size ({window_size}).")
        if not first_chunk_full:
            raise ValueError("The public Anchor3R inference path requires first_chunk_full=True.")

        win_pose_tokens = win_pose_tokens.flatten(0, 1)
        win_pose_tokens = self.token_norm(win_pose_tokens)
        return self.trunk_fn(win_pose_tokens, num_iterations)
