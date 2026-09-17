# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0,
# included in LICENSE, section DINO.txt.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/layers/patch_embed.py

from __future__ import annotations

from typing import Callable, Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        qk_norm: bool = False,
        rope=None,
        gate_attn: bool = False,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        if not isinstance(gate_attn, bool):
            raise TypeError("gate_attn must be a bool.")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.gate_attn = gate_attn
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.rope = rope
        if self.gate_attn:
            self.gate_proj = nn.Linear(dim, num_heads)
            nn.init.zeros_(self.gate_proj.weight)
            nn.init.constant_(self.gate_proj.bias, 2.0)

    def qkv_forward(self, x: Tensor, pos=None):
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        if self.rope is not None and pos is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)
        if k.dtype != v.dtype:
            k = k.to(v.dtype)
        return q, k, v

    def forward(self, x: Tensor, pos=None, attn_mask=None, kv_cache=None, q_cache=None) -> Tensor:
        b, n, c = x.shape
        gate = None
        if self.gate_attn:
            gate = torch.sigmoid(self.gate_proj(x.mean(dim=1)))
            gate = gate.view(b, self.num_heads, 1, 1)
        if q_cache is None:
            q, k, v = self.qkv_forward(x, pos=pos)
            if kv_cache is not None:
                if kv_cache.kv_ori is not None:
                    k_cache, v_cache = kv_cache.kv_ori.unbind(0)
                    k = torch.cat([k_cache, k], dim=2)
                    v = torch.cat([v_cache, v], dim=2)
                if kv_cache.kv_new is None:
                    if k.dtype != v.dtype:
                        k = k.to(v.dtype)
                    kv_cache.kv_new = torch.stack([k, v], dim=0)
        else:
            q = q_cache
            if kv_cache is None or kv_cache.kv_ori is None:
                raise ValueError("q_cache requires kv_cache.kv_ori.")
            k, v = kv_cache.kv_ori.unbind(0)

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        if gate is not None:
            out = out * gate.to(dtype=out.dtype)
        out = out.transpose(1, 2).reshape(b, q.shape[2], c)
        return self.proj(out)


class LayerScale(nn.Module):
    def __init__(
        self,
        dim: int,
        init_values: Union[float, Tensor] = 1e-5,
        inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        return self.fc2(x)


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        init_values=None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        qk_norm: bool = False,
        rope=None,
        gate_attn: bool = False,
    ) -> None:
        super().__init__()

        self.norm1 = norm_layer(dim)

        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            qk_norm=qk_norm,
            rope=rope,
            gate_attn=gate_attn,
        )

        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            bias=ffn_bias,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()

    def forward(self, x: Tensor, pos=None, attn_mask=None, kv_cache=None, q_cache=None) -> Tensor:
        delta_x = self.attn(
            self.norm1(x),
            pos=pos,
            attn_mask=attn_mask,
            kv_cache=kv_cache,
            q_cache=q_cache,
        )
        x = x + self.ls1(delta_x)
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x
