# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is subject to the VGGT license included at
# LICENSE, section VGGT.txt.

import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Dict

from anchor3r.model.layers.block import Block
from anchor3r.model.layers.rope import (
    PositionGetter,
    RotaryPositionEmbedding2D,
    RotaryPositionEmbedding3D,
)
from anchor3r.model.layers.vision_transformer import vit_large

from dataclasses import dataclass


_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]

@dataclass
class KVCache:
    """
    Explicit KV cache delta for one attention layer (one sample).
    """
    kv_ori: Optional[torch.Tensor] = None  # [2, B, H, T_old, D]
    kv_new: Optional[torch.Tensor] = None  # [2, B, H, T_new, D]

# Anchor3R stream aggregator: frame/global cache streaming attention.

def slice_expand_and_flatten(
    token_tensor: torch.Tensor, # (2, X, C)
    B: int,
    F: int,
    Win: int,
    is_first_chunk: bool,
    device: torch.device,
) -> torch.Tensor:
    assert token_tensor.ndim == 3 and token_tensor.shape[0] == 2
    tok_ref = token_tensor[0]  # [X, C]
    tok_src = token_tensor[1]  # [X, C]

    if is_first_chunk:
        grid_idx = torch.arange(F, device=device)
        diagonal_mask = grid_idx[:, None] == grid_idx[None, :]
        assert F == Win, f"First chunk requires F({F}) == Win({Win})"
        final_mask = diagonal_mask.view(F, F, 1, 1)
    else:
        final_mask = torch.zeros(F, Win, 1, 1, device=device, dtype=torch.bool)
        final_mask[:, -1] = True

    out = torch.where(final_mask, tok_ref, tok_src) # (F, Win, X, C)

    return out.unsqueeze(0).expand(B, -1, -1, -1, -1).flatten(0, 2) # (B * F * Win, X, C)

class StreamAggregator(nn.Module):

    def __init__(
        self,
        patch_size=14,
        embed_dim=1024,
        num_pose_tokens=32,
        pose_attn_depth: int = 4,
        pose_window_attn_layer_idx=(2, 5, 8, 11, 14, 17, 20, 23),
        img_size=518,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        num_register_tokens=4,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        qk_norm=True,
        rope_freq=100,
        init_values=0.01,
        intermediate_layer_idx=(4, 11, 17, 23),
        gate_attn: bool = False,
    ):
        super().__init__()

        if float(rope_freq) <= 0:
            raise ValueError(
                "StreamAggregator requires rope_freq > 0 because "
                "frame/global streaming attention needs RoPE positions."
            )
        self.patch_embed = vit_large(
            img_size=img_size,
            patch_size=patch_size,
            num_register_tokens=num_register_tokens,
            interpolate_antialias=True,
            interpolate_offset=0.0,
            init_values=1.0,
        )
        self.patch_embed.mask_token.requires_grad_(False)

        self.rope = RotaryPositionEmbedding2D(frequency=rope_freq)
        self.position_getter = PositionGetter()

        block_kwargs = dict(
            dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            ffn_bias=ffn_bias,
            init_values=init_values,
            qk_norm=qk_norm,
            rope=self.rope,
            gate_attn=gate_attn,
        )
        self.frame_blocks = nn.ModuleList(
            [Block(**block_kwargs) for _ in range(depth)]
        )
        self.global_blocks = nn.ModuleList(
            [Block(**block_kwargs) for _ in range(depth)]
        )

        self.depth = int(depth)
        self.patch_size = patch_size
        self.intermediate_layer_idx = [int(layer_idx) for layer_idx in intermediate_layer_idx]
        if not self.intermediate_layer_idx:
            raise ValueError("intermediate_layer_idx must contain at least one layer for depth decoding.")
        invalid_layers = [
            layer_idx for layer_idx in self.intermediate_layer_idx
            if layer_idx < 0 or layer_idx >= self.depth
        ]
        if invalid_layers:
            raise ValueError(
                f"intermediate_layer_idx contains invalid layer(s) {invalid_layers}; "
                f"valid range is [0, {self.depth - 1}]."
            )
        self.output_layer_idx = self.intermediate_layer_idx[-1]
        self.collected_layer_set = frozenset(self.intermediate_layer_idx)

        for name, value in (("_resnet_mean", _RESNET_MEAN), ("_resnet_std", _RESNET_STD)):
            self.register_buffer(
                name,
                torch.FloatTensor(value).view(1, 1, 3, 1, 1),
                persistent=False,
            )

        self.global_rope = RotaryPositionEmbedding3D(frequency=rope_freq)
        for global_block in self.global_blocks:
            global_block.attn.rope = self.global_rope

        self.pose_token = nn.Parameter(torch.randn(2, num_pose_tokens, embed_dim))
        nn.init.normal_(self.pose_token, std=1e-6)

        self.pose_attn_depth = int(pose_attn_depth)
        if self.pose_attn_depth <= 0:
            raise ValueError("pose_attn_depth must be positive.")
        if self.depth % self.pose_attn_depth != 0:
            raise ValueError(
                f"Aggregator depth ({self.depth}) must be divisible by "
                f"pose_attn_depth ({self.pose_attn_depth})."
            )
        self.pose_attn_repeat = self.depth // self.pose_attn_depth
        self.pose_self_attn_repeat = self.pose_attn_repeat
        self.pose_window_attn_repeat = self.pose_attn_repeat
        self.pose_self_attn_blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=None,
                    gate_attn=False,
                )
                for _ in range(self.pose_attn_depth)
            ]
        )
        self.pose_window_attn_blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.global_rope,
                    gate_attn=gate_attn,
                )
                for _ in range(self.pose_attn_depth)
            ]
        )
        self.pose_window_attn_layer_idx = [int(layer_idx) for layer_idx in pose_window_attn_layer_idx]
        invalid_pose_window_layers = [
            layer_idx for layer_idx in self.pose_window_attn_layer_idx
            if layer_idx < 0 or layer_idx >= self.depth
        ]
        if invalid_pose_window_layers:
            raise ValueError(
                f"pose_window_attn_layer_idx contains invalid layer(s) {invalid_pose_window_layers}; "
                f"valid range is [0, {self.depth - 1}]."
            )
        self.pose_window_attn_layer_set = frozenset(self.pose_window_attn_layer_idx)
        active_pose_window_block_idx = {
            layer_idx // self.pose_window_attn_repeat
            for layer_idx in self.pose_window_attn_layer_set
        }
        self.pose_window_attn_active_block_idx = frozenset(active_pose_window_block_idx)
        for block_idx, pose_window_block in enumerate(self.pose_window_attn_blocks):
            if block_idx not in self.pose_window_attn_active_block_idx:
                pose_window_block.requires_grad_(False)
        self.num_pose_tokens = num_pose_tokens
        self.fov_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.register_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        nn.init.normal_(self.fov_token, std=1e-6)
        nn.init.normal_(self.register_token, std=1e-6)
        self.rope3d_start = 0

    def _build_split_rope_positions(
        self,
        B: int,
        F: int,
        P: int,
        X: int,
        Win: int,
        H: int,
        W: int,
        special_patch_tokens: int,
        rope3d_start: int,
        is_first_chunk: bool,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        patch_h, patch_w = H // self.patch_size, W // self.patch_size
        patch_xy = self.position_getter(B * F, patch_h, patch_w, device=device)
        patch_xy = (patch_xy + 1).view(B, F, -1, 2)
        zeros = patch_xy.new_zeros

        patch_xy = torch.cat([zeros((B, F, special_patch_tokens, 2)), patch_xy], dim=2)

        frame_patch_pos = patch_xy.reshape(B * F, P, 2).contiguous()

        current_t = rope3d_start + torch.arange(F, device=device, dtype=patch_xy.dtype)
        patch_t = current_t.view(1, F, 1, 1).expand(B, F, P, 1)
        global_patch_pos = torch.cat([patch_t, patch_xy], dim=-1)

        window_offsets = torch.arange(Win, device=device, dtype=patch_xy.dtype)
        if is_first_chunk:
            pose_window_t = window_offsets.view(1, Win).expand(F, Win)
        else:
            pose_window_t = current_t.view(F, 1) - (Win - 1) + window_offsets.view(1, Win)
            if int(pose_window_t.min().item()) < 0:
                raise ValueError(
                    "rope3d_start is inconsistent with non-first chunk KV caches: "
                    f"rope3d_start={rope3d_start}, Win={Win}."
                )

        # In global layers each pose-token group represents one source image in
        # the streaming window. Encode that source frame id with periodic time so
        # dense image windows cannot collapse different source-pose meanings.
        global_pose_pos = zeros((B, F, Win, X, 3))
        global_pose_pos[..., 0] = pose_window_t.reshape(1, F, Win, 1)
        global_pose_pos = global_pose_pos.reshape(B, F, Win * X, 3)
        all_pos = torch.cat([global_patch_pos, global_pose_pos], dim=2)
        all_pos = all_pos.reshape(B * F, P + Win * X, 3).contiguous()
        return frame_patch_pos, all_pos

    def forward(
        self,
        images: torch.Tensor,
        frame_kv_caches: List[KVCache],
        global_kv_caches: List[KVCache],
        pose_window_kv_caches: List[KVCache],
        window_size: int = 10,
        chunk_idx: int = 0,
    ) -> Tuple[Dict, torch.Tensor, int]:
        Win = int(window_size)
        chunk_idx = int(chunk_idx)
        if Win > self.global_rope.time_period:
            raise ValueError(
                "3D RoPE can only disambiguate pose source frames "
                f"within one time period; got window_size={Win}, "
                f"time_period={self.global_rope.time_period}."
            )
        if chunk_idx == 0:
            self.rope3d_start = 0
        rope3d_start = int(self.rope3d_start)

        B, S, C_in, H, W = images.shape
        assert B == 1, "Batch size must be one."
        F = S

        assert len(frame_kv_caches) == len(self.frame_blocks), "frame_kv_caches length mismatch"
        assert len(global_kv_caches) == len(self.global_blocks), "global_kv_caches length mismatch"
        assert len(pose_window_kv_caches) == len(self.global_blocks), "pose_window_kv_caches length mismatch"

        if C_in != 3:
            raise ValueError(f"Expected 3 input channels, got {C_in}")

        # Normalize images and reshape for patch embed
        images = (images - self._resnet_mean) / self._resnet_std
        images = images.reshape(B * S, C_in, H, W)

        patch_tokens = self.patch_embed(images)["x_norm_patchtokens"]

        _, _, C = patch_tokens.shape

        patch_tokens = patch_tokens.reshape(B * F, -1, C)
        fov_tokens = self.fov_token.expand(B * F, -1, -1)
        register_tokens = self.register_token.expand(B * F, -1, -1)
        patch_tokens = torch.cat([fov_tokens, register_tokens, patch_tokens], dim=1)

        _, P, C = patch_tokens.shape
        special_patch_tokens = 2

        X = self.num_pose_tokens
        is_first_chunk = chunk_idx == 0
        self._check_kv_cache(frame_kv_caches, global_kv_caches, pose_window_kv_caches, B, F, P, X, Win, is_first_chunk)

        pose_tokens = slice_expand_and_flatten(
            self.pose_token, B, F, Win, is_first_chunk=is_first_chunk, device=images.device
        ) # (B * F * Win, X, C)
        assert pose_tokens.shape == (B * F * Win, X, C)

        patch_pos, all_pos = self._build_split_rope_positions(
            B, F, P, X, Win, H, W, special_patch_tokens, rope3d_start,
            is_first_chunk, images.device,
        )

        frame_patch_dict: Dict[int, torch.Tensor] = {}
        global_patch_dict: Dict[int, torch.Tensor] = {}
        frame_pose_dict: Dict[int, torch.Tensor] = {}
        global_pose_dict: Dict[int, torch.Tensor] = {}

        for block_idx in range(self.depth):
            patch_tokens, pose_tokens, _, frame_patch_inter, frame_pose_inter = self._process_stream_frame_attention(
                patch_tokens, pose_tokens, B, F, P, C, X, Win, block_idx,
                patch_pos=patch_pos,
                kv_cache_list=frame_kv_caches,
            )
            patch_tokens, pose_tokens, _, global_patch_inter, global_pose_inter = self._process_stream_global_attention(
                patch_tokens, pose_tokens, B, F, P, C, X, Win, block_idx,
                all_pos=all_pos,
                kv_cache_list=global_kv_caches,
                pose_window_kv_cache_list=pose_window_kv_caches,
            )
            if frame_patch_inter is not None:
                frame_patch_dict[block_idx] = frame_patch_inter
            if global_patch_inter is not None:
                global_patch_dict[block_idx] = global_patch_inter
            if frame_pose_inter is not None:
                frame_pose_dict[block_idx] = frame_pose_inter
            if global_pose_inter is not None:
                global_pose_dict[block_idx] = global_pose_inter

        output_dict = {
            layer_idx: torch.cat([frame_patch_dict[layer_idx], global_patch_dict[layer_idx]], dim=-1)
            for layer_idx in self.intermediate_layer_idx
        }
        output_dict[-1] = output_dict[self.output_layer_idx]
        output_pose_tokens = torch.cat(
            [frame_pose_dict[self.output_layer_idx], global_pose_dict[self.output_layer_idx]],
            dim=-1,
        )

        # output_pose_tokens: (B, F, Win, 2C)  -> batch-first
        self.rope3d_start = rope3d_start + F
        return output_dict, output_pose_tokens, special_patch_tokens

    def _check_kv_cache(
        self,
        frame_kv_caches: List[KVCache],
        global_kv_caches: List[KVCache],
        pose_window_kv_caches: List[KVCache],
        B: int,
        F: int,
        P: int,
        X: int,
        Win: int,
        is_first_chunk: bool,
    ):
        if is_first_chunk:
            assert F == Win, f"First chunk requires F({F}) == Win({Win})"
            assert all(cache.kv_ori is None for cache in frame_kv_caches)
            assert all(cache.kv_new is None for cache in frame_kv_caches)
            assert all(cache.kv_ori is None for cache in global_kv_caches)
            assert all(cache.kv_new is None for cache in global_kv_caches)
            assert all(cache.kv_ori is None for cache in pose_window_kv_caches)
            assert all(cache.kv_new is None for cache in pose_window_kv_caches)
            return

        for cache in frame_kv_caches:
            assert cache.kv_ori is not None and cache.kv_new is None
            _, B_c, Win_1_c, _, P_c, _ = cache.kv_ori.shape
            assert B_c == B and Win_1_c == Win - 1 and P_c == P

        for cache in global_kv_caches:
            assert cache.kv_ori is not None and cache.kv_new is None
            _, B_c, _, Win_1_c, P_c, _ = cache.kv_ori.shape
            assert B_c == B and Win_1_c == Win - 1 and P_c == P

        G = Win * X
        for idx, cache in enumerate(pose_window_kv_caches):
            if idx not in self.pose_window_attn_layer_set:
                assert cache.kv_ori is None and cache.kv_new is None
                continue
            assert cache.kv_ori is not None and cache.kv_new is None
            _, B_c, _, Win_1_c, G_c, _ = cache.kv_ori.shape
            assert B_c == B and Win_1_c == Win - 1 and G_c == G

    def _process_stream_frame_attention(
        self, patch_tokens, pose_tokens, B, F, P, C, X, Win,
        frame_idx, patch_pos: torch.Tensor, kv_cache_list: List[KVCache],
    ):
        """
        patch_tokens: (B*F,P,C)
        pose_tokens : (B,F,Win,X,C)
        """
        assert patch_tokens.shape == (B * F, P, C)
        assert pose_tokens.shape == (B * F * Win, X, C)
        assert patch_pos.shape[:2] == (B * F, P)

        patch_intermediate = None
        pose_intermediate = None

        local_kv_cache = KVCache()
        patch_tokens = self.frame_blocks[frame_idx](patch_tokens, patch_pos, kv_cache=local_kv_cache)

        kv_remain_shape = local_kv_cache.kv_new.shape[-3:] # (H, P, C//H)
        kv_cache = kv_cache_list[frame_idx]
        local_kv_cache.kv_new = local_kv_cache.kv_new.reshape(2, B, F, *kv_remain_shape) # (2, B, F, H, P, C//H)

        old_kv_ori = kv_cache.kv_ori
        is_first_cache = old_kv_ori is None
        if is_first_cache:
            patch_history = local_kv_cache.kv_new
        else:
            patch_history = torch.cat([old_kv_ori, local_kv_cache.kv_new], dim=2) # (2, B, Win - 1 + F, H, P, C//H)
            kv_cache.kv_ori = None
            del old_kv_ori

        if kv_cache.kv_new is None:
            next_kv = patch_history[:, :, 1-Win:] # (2, B, Win-1, H, P, C//H)
            kv_cache.kv_new = next_kv.clone().detach()
            del next_kv

        if is_first_cache:
            src_patch_kv = patch_history.unsqueeze(2).expand(2, B, F, Win, *kv_remain_shape) # (2, B, F, Win, H, P, C//H)
        else:
            s0, s1, s2, s3, s4, s5 = patch_history.stride()
            src_patch_kv = patch_history.as_strided(size=(2, B, F, Win, *kv_remain_shape), stride=(s0, s1, s2, s2, s3, s4, s5)) # (2,B,F,Win,H,P,C//H)
        local_kv_cache.kv_ori = src_patch_kv.flatten(1, 3) # (2, B*F*Win, H, P, C//H)
        local_kv_cache.kv_new = torch.tensor([], device=local_kv_cache.kv_ori.device)

        pose_tokens = self.frame_blocks[frame_idx](
            pose_tokens,
            kv_cache=local_kv_cache,
        )
        pose_tokens = self._process_pose_self_attention(
            pose_tokens, B, F, Win, X, C, frame_idx
        )
        local_kv_cache.kv_ori = None
        local_kv_cache.kv_new = None
        del patch_history, src_patch_kv

        if frame_idx in self.collected_layer_set:
            patch_intermediate = patch_tokens.reshape(B, F, P, C)  # (B,F,P,C)
            pose_full = pose_tokens.reshape(B, F, Win, X, C)
            pose_intermediate = pose_full[..., 0, :]  # (B,F,Win,C)

        frame_idx += 1

        return patch_tokens, pose_tokens, frame_idx, patch_intermediate, pose_intermediate

    def _process_pose_self_attention(
        self,
        pose_tokens: torch.Tensor,
        B: int,
        F: int,
        Win: int,
        X: int,
        C: int,
        frame_idx: int,
    ) -> torch.Tensor:
        pose_tokens = pose_tokens.reshape(B * F, Win * X, C)
        pose_block = self.pose_self_attn_blocks[frame_idx // self.pose_self_attn_repeat]
        pose_tokens = pose_block(pose_tokens)
        return pose_tokens.reshape(B * F * Win, X, C)

    def _process_pose_window_attention(
        self, pose_tokens: torch.Tensor, B: int, F: int, Win: int, X: int, C: int,
        global_idx: int, kv_cache_list: List[KVCache], pose_pos: torch.Tensor,
    ) -> torch.Tensor:
        assert pose_tokens.shape == (B * F * Win, X, C)
        pose_block = self.pose_window_attn_blocks[global_idx // self.pose_window_attn_repeat]
        kv_cache = kv_cache_list[global_idx]
        G = Win * X

        pose_seq = pose_tokens.reshape(B * F, G, C)
        assert pose_pos.shape[:2] == (B * F, G)
        q_cache, k_cur, v_cur = pose_block.attn.qkv_forward(pose_block.norm1(pose_seq), pos=pose_pos)
        H, D = k_cur.shape[1], k_cur.shape[-1]
        kv_cur = torch.stack([k_cur, v_cur], dim=0).reshape(2, B, F, H, G, D).transpose(2, 3).contiguous()
        first_cache = kv_cache.kv_ori is None
        pose_history = kv_cur if first_cache else torch.cat([kv_cache.kv_ori, kv_cur], dim=3)
        if not first_cache:
            kv_cache.kv_ori = None

        if kv_cache.kv_new is None:
            next_kv = pose_history[:, :, :, 1-Win:]
            kv_cache.kv_new = next_kv.clone().detach()

        pose_window_kv = (
            pose_history.unsqueeze(2).expand(-1, -1, F, -1, Win, -1, -1)
            if first_cache else
            pose_history.unfold(dimension=3, size=Win, step=1).permute(
                    0, 1, 3, 2, 6, 4, 5
            )
        ).reshape(2, B * F, H, Win * G, D)
        pose_seq = pose_block(pose_seq, pos=pose_pos, kv_cache=KVCache(kv_ori=pose_window_kv), q_cache=q_cache)

        return pose_seq.reshape(B * F * Win, X, C)

    def _process_stream_global_attention(
        self, patch_tokens, pose_tokens, B, F, P, C, X, Win,
        global_idx,
        all_pos: torch.Tensor,
        kv_cache_list: List[KVCache],
        pose_window_kv_cache_list: List[KVCache],
    ):
        """
        patch_tokens: (B*F,P,C)
        pose_tokens : (B*F*Win,X,C)
        NOTE: all_pos already contains the per-token RoPE positions prepared in forward.
        """
        assert patch_tokens.shape == (B * F, P, C)
        assert pose_tokens.shape == (B * F * Win, X, C)

        # reshape patch tokens for global blocks
        A = P + Win * X
        pos_dim = all_pos.shape[-1]
        assert all_pos.shape == (B * F, A, pos_dim)
        pose_pos = all_pos[:, P:].contiguous()

        patch_intermediate = None
        pose_intermediate = None

        global_block = self.global_blocks[global_idx]
        all_tokens = torch.cat([patch_tokens, pose_tokens.reshape(B * F, Win * X, C)], dim=1)
        kv_cache = kv_cache_list[global_idx]
        q_cache, k, v = global_block.attn.qkv_forward(global_block.norm1(all_tokens), all_pos)
        _, num_heads, _, head_dim = k.shape
        kv = torch.stack([k, v], dim=0) # (2, B * F, H, A, C//H)
        patch_kv = kv.reshape(2, B, F, num_heads, A, head_dim).transpose(2, 3)[..., :P, :]
        pose_kv = kv[..., -Win * X:, :]
        patch_history = None

        if kv_cache.kv_ori is None:
            if kv_cache.kv_new is None:
                next_kv = patch_kv[..., 1:, :, :] # (2, B, H, Win - 1, P, C//H)
                kv_cache.kv_new = next_kv.clone().detach()
                del next_kv
            patch_window_kv = patch_kv.unsqueeze(2).expand(
                -1, -1, F, -1, Win, -1, -1
            ).reshape(2, B * F, num_heads, Win * P, head_dim)
        else:
            old_kv_ori = kv_cache.kv_ori
            patch_history = torch.cat([old_kv_ori, patch_kv], dim=3) # (2, B, H, Win - 1 + F, P, C//H)
            kv_cache.kv_ori = None
            del old_kv_ori
            if kv_cache.kv_new is None:
                next_kv = patch_history[..., 1-Win:, :, :] # (2, B, H, Win - 1, P, C//H)
                kv_cache.kv_new = next_kv.clone().detach()
                del next_kv
            patch_window_kv = patch_history.unfold(dimension=3, size=Win, step=1)
            patch_window_kv = patch_window_kv.permute(0, 1, 3, 2, 6, 4, 5).reshape(
                2, B * F, num_heads, Win * P, head_dim
            )

        swa_kv_cache = KVCache(kv_ori=torch.cat([patch_window_kv, pose_kv], dim=3))
        all_tokens = global_block(all_tokens, all_pos, kv_cache=swa_kv_cache, q_cache=q_cache)
        swa_kv_cache.kv_ori = None
        swa_kv_cache.kv_new = None
        del patch_window_kv, patch_kv, pose_kv, kv, q_cache, k, v, swa_kv_cache
        if patch_history is not None:
            del patch_history

        patch_tokens = all_tokens[:, :P].contiguous()
        pose_tokens = all_tokens[:, P:].reshape(B * F * Win, X, C).contiguous()
        if global_idx in self.pose_window_attn_layer_set:
            pose_tokens = self._process_pose_window_attention(
                pose_tokens,
                B,
                F,
                Win,
                X,
                C,
                global_idx,
                pose_window_kv_cache_list,
                pose_pos=pose_pos,
            )

        if global_idx in self.collected_layer_set:
            patch_intermediate = patch_tokens.reshape(B, F, P, C)
            pose_full = pose_tokens.reshape(B, F, Win, X, C)
            pose_intermediate = pose_full[..., 0, :]  # (B,F,Win,C)

        global_idx += 1

        return patch_tokens, pose_tokens, global_idx, patch_intermediate, pose_intermediate
