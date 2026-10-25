"""Inference-only adapter for trusted smallstep_v1 LSTM checkpoints.

The reference board-marker action head is exposed through the current MoveKey
ONNX ABI. No weights are fitted and no reference-project imports are required.
"""

from __future__ import annotations

from dataclasses import replace

import torch
from torch import nn
from torch.nn import functional as F

from .config import NetworkConfig
from .network import SinusoidalPositionEncoding, SuzeroNetwork
from .protocol import ACTION_ENCODING, VariantSpec


class _LegacyPositionEncoding(SinusoidalPositionEncoding):
    def __init__(self, d_model: int, table: torch.Tensor) -> None:
        super().__init__(d_model)
        self.register_buffer("pe", torch.empty_like(table))

    def _axis(self, coords: torch.Tensor) -> torch.Tensor:
        offset = self.pe.shape[0] // 2
        lookup = self.pe[(coords + offset).clamp(0, self.pe.shape[0] - 1)]
        return torch.where(
            ((coords >= -offset) & (coords < offset)).unsqueeze(-1),
            lookup, super()._axis(coords),
        )


class LegacyLstmNetwork(SuzeroNetwork):
    architecture = "lstm"

    def __init__(self, config: NetworkConfig, spec: VariantSpec, table: torch.Tensor) -> None:
        super().__init__(config, spec)
        del self.action_mlp
        self.pos_encoder = _LegacyPositionEncoding(config.d_model, table)
        self.board_projection = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2), nn.GELU(),
            nn.Linear(config.d_model // 2, spec.board_squares),
        )
        self.action_lstm = nn.LSTM(2 * spec.board_squares, 64, batch_first=True)
        self.action_head = nn.Linear(64, 1)
        self.parameter_count = sum(p.numel() for p in self.parameters())

    def action_markers(self, actions, l_coords, t_coords):
        side = self.spec.board_side
        source_board = (
            (l_coords[:, None, :] == actions[..., 3, None] + self.spec.line_shift)
            & (t_coords[:, None, :] == actions[..., 2, None])
        )
        target_board = (
            (l_coords[:, None, :] == actions[..., 7, None] + self.spec.line_shift)
            & (t_coords[:, None, :] == actions[..., 6, None])
        )
        source = source_board.unsqueeze(-1) & F.one_hot(
            actions[..., 1] * side + actions[..., 0], self.spec.board_squares
        ).bool().unsqueeze(2)
        target = target_board.unsqueeze(-1) & F.one_hot(
            actions[..., 5] * side + actions[..., 4], self.spec.board_squares
        ).bool().unsqueeze(2)
        # Historical dense encoding writes source=-1, then destination=+1.
        return torch.where(target, 1.0, -source.to(torch.float32))

    def forward(self, board_planes, l_coords, t_coords, used_board_counts,
                urgency, side_to_move, actions):
        cls, boards = self.encode_observation(
            board_planes, l_coords, t_coords, used_board_counts, urgency, side_to_move
        )
        markers = self.action_markers(actions, l_coords, t_coords)
        projected = self.board_projection(boards).unsqueeze(1) + torch.zeros_like(markers)
        combined = torch.cat((projected, markers), dim=-1)
        batch, candidates, count = combined.shape[:3]
        sequence = combined.reshape(batch * candidates, count, 2 * self.spec.board_squares)
        # Selecting the last real output is equivalent to packed-sequence h_n
        # for this unidirectional LSTM, and exports with dynamic B/N/q.
        output, _ = self.action_lstm(sequence)
        output = output.reshape(batch, candidates, count, 64)
        indices = (used_board_counts - 1)[:, None, None, None]
        indices = indices.expand(batch, candidates, 1, 64)
        hidden = output.gather(2, indices).squeeze(2)
        return self.value_head(cls), self.submit_head(cls), self.action_head(hidden).squeeze(-1)


def load_legacy_checkpoint(state: dict, spec: VariantSpec) -> LegacyLstmNetwork:
    if state.get("action_encoding") != ACTION_ENCODING:
        raise ValueError("legacy checkpoint must declare action_encoding=smallstep_v1")
    saved = state.get("config", {})
    if saved.get("variant_name") != spec.name:
        raise ValueError("legacy checkpoint variant does not match the match rules")
    raw = saved.get("network", {})
    if raw.get("architecture", "lstm") != "lstm":
        raise ValueError("only the legacy LSTM checkpoint architecture is supported")
    for key, expected in (("board_side", spec.board_side),
                          ("board_squares", spec.board_squares),
                          ("piece_channels", spec.piece_channels)):
        if raw.get(key) != expected:
            raise ValueError(f"legacy checkpoint {key} does not match the match rules")
    cfg = NetworkConfig(**{key: raw[key] for key in (
        "d_model", "n_heads", "n_layers", "d_ff", "dropout"
    ) if key in raw}, max_params=2**31 - 1)
    weights = state["model_state_dict"]
    table = weights["pos_encoder.pe"]
    if table.ndim != 2 or table.shape[1] != cfg.d_model // 2 or table.shape[0] % 2:
        raise ValueError("invalid legacy positional encoding table")
    network = LegacyLstmNetwork(
        cfg, replace(spec, line_shift=raw.get("l_coord_shift", 5)), table
    )
    converted = {}
    for key, value in weights.items():
        if key.startswith("transformer.layers."):
            key = key.replace("transformer.layers.", "blocks.", 1)
            key = key.replace("self_attn.in_proj_weight", "qkv.weight")
            key = key.replace("self_attn.in_proj_bias", "qkv.bias")
            key = key.replace("self_attn.out_proj.", "out_proj.")
        converted[key] = value
    network.load_state_dict(converted, strict=True)
    network.eval()
    return network
