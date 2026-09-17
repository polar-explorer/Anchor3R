from __future__ import annotations

from contextlib import nullcontext

import numpy as np
import torch
from torch import nn

from anchor3r.model.heads.camera import BatchCameraHead
from anchor3r.model.heads.dpt_head import DPTHead
from anchor3r.model.stream_aggregator import StreamAggregator
from anchor3r.utils.core import InferenceModule, dotdict, pad_image


class MultiViewStreamSampler(InferenceModule):
    def __init__(
        self,
        stream_aggregator_cfg: dotdict | None = None,
        cam_decoder_cfg: dotdict | None = None,
        dpt_decoder_cfg: dotdict | None = None,
        fov_head_cfg: dotdict | None = None,
    ):
        super().__init__()

        stream_aggregator_cfg = dotdict(
            stream_aggregator_cfg or {"img_size": 518, "patch_size": 14, "embed_dim": 1024}
        )
        cam_decoder_cfg = dotdict(cam_decoder_cfg or {"dim_in": 2048})
        dpt_decoder_cfg = dotdict(
            dpt_decoder_cfg
            or {
                "dim_in": 2048,
                "output_dim": 2,
                "activation": "exp",
                "conf_activation": "expp1",
            }
        )
        fov_head_cfg = dotdict(fov_head_cfg or {"hidden_dims": [256, 128], "init_bias": 0.0})

        pose_encoding_type = cam_decoder_cfg.get("pose_encoding_type", "absT_quaR")
        if pose_encoding_type != "absT_quaR":
            raise ValueError(
                "MultiViewStreamSampler expects cam_decoder_cfg.pose_encoding_type='absT_quaR' "
                f"because FoV is decoded by the separate fov head, got {pose_encoding_type!r}."
            )
        cam_decoder_cfg["pose_encoding_type"] = pose_encoding_type

        if "embed_dim" in stream_aggregator_cfg:
            dim_in = stream_aggregator_cfg["embed_dim"] * 2
            cam_decoder_cfg["dim_in"] = dim_in
            dpt_decoder_cfg["dim_in"] = dim_in
        if "intermediate_layer_idx" in stream_aggregator_cfg:
            dpt_decoder_cfg["intermediate_layer_idx"] = stream_aggregator_cfg["intermediate_layer_idx"]

        self.stream_aggregator: StreamAggregator = StreamAggregator(
            **stream_aggregator_cfg,
        )
        self.cam_decoder: nn.Module = BatchCameraHead(
            **cam_decoder_cfg,
        )
        self.dpt_decoder: DPTHead = DPTHead(**dpt_decoder_cfg)
        self.fov_head = self._build_fov_head(cam_decoder_cfg["dim_in"], fov_head_cfg)

        self.stream_aggregator_cfg = stream_aggregator_cfg
        self.cam_decoder_cfg = cam_decoder_cfg
        self.dpt_decoder_cfg = dpt_decoder_cfg
        self.fov_head_cfg = fov_head_cfg
        self.patch_size = (self.stream_aggregator.patch_size, self.stream_aggregator.patch_size)

    def _build_fov_head(self, dim_in: int, fov_head_cfg: dotdict) -> nn.Sequential:
        hidden_dims = list(fov_head_cfg.get("hidden_dims", [256, 128]))
        init_bias = fov_head_cfg.get("init_bias", 0.0)
        layers = []
        last_dim = dim_in
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.GELU())
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, 2))
        fov_head = nn.Sequential(*layers)
        for module in fov_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        if isinstance(init_bias, (list, tuple)):
            fov_head[-1].bias.data.copy_(fov_head[-1].bias.new_tensor(init_bias))
        else:
            nn.init.constant_(fov_head[-1].bias, float(init_bias))
        return fov_head

    def _predict_fov(self, rgb_feats: dict[int, torch.Tensor], patch_start_idx: int):
        fov_token_idx = patch_start_idx - 2
        if fov_token_idx < 0:
            raise ValueError("fov token is enabled but patch_start_idx does not include it.")
        fov_token_features = rgb_feats[-1][:, :, fov_token_idx, :].float()
        predicted_fov = torch.relu(self.fov_head(fov_token_features)) + 0.01
        return predicted_fov

    def forward(self, batch: dotdict):
        output = dotdict()
        first_chunk_full = bool(batch.get("first_chunk_full", True))
        if not first_chunk_full:
            raise ValueError("MultiViewStreamSampler requires first_chunk_full=True.")

        B, N = batch.rgb.shape[:2]
        Ho, Wo = batch.meta.H[0].item(), batch.meta.W[0].item()
        Hn = int(np.ceil(Ho / self.patch_size[0])) * self.patch_size[0]
        Wn = int(np.ceil(Wo / self.patch_size[1])) * self.patch_size[1]
        assert Ho == Hn and Wo == Wn, f"Ho: {Ho}, Hn: {Hn}, Wo: {Wo}, Wn: {Wn}"

        rgbs = batch.rgb.reshape(B, N, Ho, Wo, -1).permute(0, 1, 4, 2, 3).contiguous()
        rgbs = pad_image(rgbs, size=[Hn, Wn])

        rgb_feats, win_pose_tokens, idx_patch = self.stream_aggregator(
            rgbs,
            global_kv_caches=batch.global_kv_caches,
            frame_kv_caches=batch.frame_kv_caches,
            pose_window_kv_caches=batch.pose_window_kv_caches,
            window_size=batch.window_size,
            chunk_idx=batch.chunk_idx,
        )

        decoder_autocast = (
            torch.amp.autocast(device_type="cuda", enabled=False)
            if rgbs.device.type == "cuda"
            else nullcontext()
        )
        with decoder_autocast:
            chunk_cam_maps = self.cam_decoder(
                win_pose_tokens,
                chunk_idx=batch.chunk_idx,
                first_chunk_full=first_chunk_full,
            )
            predicted_fov = self._predict_fov(rgb_feats, idx_patch)
            dpt_map, dpt_cnf = self.dpt_decoder(
                rgb_feats,
                images=rgbs,
                patch_start_idx=idx_patch,
            )
            dpt_map = dpt_map[..., :Ho, :Wo, :]
            dpt_cnf = dpt_cnf[..., :Ho, :Wo]

            output.chunk_cam_maps = chunk_cam_maps
            output.chunk_cam_map = chunk_cam_maps[-1]
            output.dpt_map = dpt_map.reshape(B, N, Ho * Wo, dpt_map.shape[-1])
            output.dpt_cnf = dpt_cnf.reshape(B, N, Ho * Wo, 1)
            output.fov = predicted_fov

        batch.output.update(output)
        return batch
