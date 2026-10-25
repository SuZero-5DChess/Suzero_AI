"""``cls_action_mlp`` network for the smallstep_movekey_v1 protocol.

The layout follows the reference architecture (board tokenizer, sinusoidal
(L, T) encoding, CLS token carrying side-to-move, transformer trunk, value and
submit heads, and an MLP that scores ordered MoveKeys).  The default width is
still ``d_model=128``; one transformer layer and a 64-wide action MLP keep the
whole model below the 200k-parameter budget.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import NetworkConfig
from .protocol import VariantSpec, resolve_variant

_FROM_X, _FROM_Y, _FROM_T, _FROM_L = 0, 1, 2, 3
_TO_X, _TO_Y, _TO_T, _TO_L = 4, 5, 6, 7


class SinusoidalPositionEncoding(nn.Module):
    """Static sin/cos encoding of a timeline/turn pair into ``d_model`` dims.

    Each axis receives ``d_model/2`` dimensions.  Within an axis the columns
    alternate ``sin`` and ``cos`` so the ordering matches the reference model.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        if d_model % 4 != 0:
            raise ValueError("d_model must be divisible by 4 for (L, T) encoding")
        self.d_model = d_model
        half = d_model // 2
        self.register_buffer(
            "inv_freq",
            torch.exp(
                torch.arange(0, half, 2, dtype=torch.float32)
                * (-math.log(10_000.0) / half)
            ),
            persistent=False,
        )

    def _axis(self, coords: torch.Tensor) -> torch.Tensor:
        values = coords.to(torch.float32)
        angles = values.unsqueeze(-1) * self.inv_freq.to(values.device)
        return torch.stack((torch.sin(angles), torch.cos(angles)), dim=-1).flatten(-2)

    def forward(self, l_coords: torch.Tensor, t_coords: torch.Tensor) -> torch.Tensor:
        return torch.cat((self._axis(l_coords), self._axis(t_coords)), dim=-1)


class BoardTokenizer(nn.Module):
    """Flatten one board into ``d_model - 3`` features (3 urgency slots)."""

    def __init__(self, config: NetworkConfig, spec: VariantSpec) -> None:
        super().__init__()
        input_dim = spec.piece_channels * spec.board_squares
        output_dim = config.d_model - 3
        self.proj = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )

    def forward(self, board_planes: torch.Tensor) -> torch.Tensor:
        # Works for [B, N, C, S] and [B*N, C, S]: only the last two dims are
        # flattened.  Avoiding reshape with Python shape ints keeps the exported
        # graph dynamic in B and N.
        return self.proj(board_planes.flatten(-2))


class TransformerBlock(nn.Module):
    """Post-norm encoder layer, matching ``nn.TransformerEncoderLayer``.

    Manual multi-head attention keeps the padding mask explicit and exports
    cleanly to opset-18 ONNX.
    """

    def __init__(self, config: NetworkConfig) -> None:
        super().__init__()
        d_model = config.d_model
        if d_model % config.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = config.n_heads
        self.head_dim = d_model // config.n_heads
        self.scale = self.head_dim**-0.5
        self.dropout = config.dropout

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.linear1 = nn.Linear(d_model, config.d_ff)
        self.linear2 = nn.Linear(config.d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout_attn = nn.Dropout(config.dropout)
        self.dropout_ff = nn.Dropout(config.dropout)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        qkv = self.qkv(x).unflatten(-1, (3, self.n_heads, self.head_dim))
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]

        scores = torch.matmul(query, key.transpose(-1, -2)) * self.scale
        mask = key_padding_mask[:, None, None, :]
        scores = scores.masked_fill(mask, -1.0e9)
        probs = torch.softmax(scores, dim=-1)
        probs = F.dropout(probs, p=self.dropout, training=self.training)
        attended = torch.matmul(probs, value)
        attended = attended.transpose(1, 2).flatten(2, 3)
        x = self.norm1(x + self.dropout_attn(self.out_proj(attended)))
        feedforward = self.linear2(self.dropout_ff(self.activation(self.linear1(x))))
        return self.norm2(x + self.dropout_ff(feedforward))


class SuzeroNetwork(nn.Module):
    """AlphaZero policy/value network for ``smallstep_movekey_v1``."""

    architecture = "cls_action_mlp"
    action_input_encoding = "movekey_int64_v1"

    def __init__(
        self,
        network_config: NetworkConfig | None = None,
        variant: str | VariantSpec = "very_small",
    ) -> None:
        super().__init__()
        self.config = network_config or NetworkConfig()
        self.spec = resolve_variant(variant)
        cfg = self.config

        self.board_tokenizer = BoardTokenizer(cfg, self.spec)
        self.pos_encoder = SinusoidalPositionEncoding(cfg.d_model)
        # Urgency ramps: a * r + (1 - a), with the reference coefficients.
        self.register_buffer("urgency_a", torch.tensor(-0.4))
        self.register_buffer("urgency_b", torch.tensor(-0.05))
        self.register_buffer("urgency_c", torch.tensor(-0.2))

        self.blocks = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg.n_layers)]
        )
        self.value_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model // 2),
            nn.GELU(),
            nn.Linear(cfg.d_model // 2, 1),
            nn.Tanh(),
        )
        self.submit_head = nn.Linear(cfg.d_model, 1)

        self.action_feature_dim = 2 * cfg.d_model + 2 * self.spec.board_squares
        self.action_mlp = nn.Sequential(
            nn.Linear(cfg.d_model + self.action_feature_dim, cfg.action_mlp_hidden),
            nn.GELU(),
            nn.Linear(cfg.action_mlp_hidden, 1),
        )

        self.parameter_count = sum(p.numel() for p in self.parameters())
        if self.parameter_count > cfg.max_params:
            raise ValueError(
                f"network has {self.parameter_count:,} parameters, exceeding "
                f"the {cfg.max_params:,} budget; reduce n_layers, d_ff or "
                "action_mlp_hidden (d_model is fixed by policy)"
            )

    # ------------------------------------------------------------- features
    def action_features(self, actions: torch.Tensor) -> torch.Tensor:
        # Shape comparisons would be folded into trace constants; skip them
        # while tracing.  ``dtype`` is metadata, so it never warns.
        if not torch.jit.is_tracing():
            if actions.ndim != 3 or actions.shape[-1] != 8:
                raise ValueError(
                    f"actions must be [B, q, 8], got {tuple(actions.shape)}"
                )
        if actions.dtype != torch.int64:
            raise ValueError("actions must be int64")
        side = self.spec.board_side
        src_position = self.pos_encoder(
            actions[..., _FROM_L] + self.spec.line_shift, actions[..., _FROM_T]
        )
        dst_position = self.pos_encoder(
            actions[..., _TO_L] + self.spec.line_shift, actions[..., _TO_T]
        )
        src_index = actions[..., _FROM_Y] * side + actions[..., _FROM_X]
        dst_index = actions[..., _TO_Y] * side + actions[..., _TO_X]
        src_square = F.one_hot(
            src_index, num_classes=self.spec.board_squares
        ).to(src_position.dtype)
        dst_square = F.one_hot(
            dst_index, num_classes=self.spec.board_squares
        ).to(dst_position.dtype)
        return torch.cat(
            (src_position, dst_position, src_square, dst_square), dim=-1
        )

    def score_actions(
        self, cls_out: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        compact = self.action_features(actions)
        # Broadcast through a tensor whose B/q dims follow the input actions;
        # a Python-int ``expand`` would freeze q in the ONNX graph.
        broadcast = torch.zeros_like(compact[..., :1])
        cls_features = cls_out.unsqueeze(1) + broadcast
        return self.action_mlp(torch.cat((cls_features, compact), dim=-1)).squeeze(-1)

    def _check_inputs(
        self,
        board_planes: torch.Tensor,
        l_coords: torch.Tensor,
        t_coords: torch.Tensor,
        used_board_counts: torch.Tensor,
        urgency: torch.Tensor,
        side_to_move: torch.Tensor,
        actions: torch.Tensor,
    ) -> None:
        if board_planes.ndim != 4:
            raise ValueError("board_planes must be [B, N, C, S]")
        batch, boards, channels, squares = board_planes.shape
        if channels != self.spec.piece_channels or squares != self.spec.board_squares:
            raise ValueError(
                f"board_planes geometry must be [B, N, {self.spec.piece_channels}, "
                f"{self.spec.board_squares}], got {tuple(board_planes.shape)}"
            )
        if boards < 1:
            raise ValueError("at least one board is required")
        if l_coords.shape != (batch, boards) or t_coords.shape != (batch, boards):
            raise ValueError("coordinate tensors must be [B, N]")
        if used_board_counts.shape != (batch,):
            raise ValueError("used_board_counts must be [B]")
        if urgency.shape != (batch,) or side_to_move.shape != (batch,):
            raise ValueError("urgency and side_to_move must be [B]")
        if actions.ndim != 3 or actions.shape[0] != batch or actions.shape[1] < 1:
            raise ValueError("actions must be [B, q, 8] with q >= 1")

    # --------------------------------------------------------------- forward
    def forward(
        self,
        board_planes: torch.Tensor,
        l_coords: torch.Tensor,
        t_coords: torch.Tensor,
        used_board_counts: torch.Tensor,
        urgency: torch.Tensor,
        side_to_move: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # The shape checks compare traced sizes, which ONNX tracing folds into
        # constants and reports as TracerWarnings; they carry no data flow, so
        # run them only outside tracing.
        if not torch.jit.is_tracing():
            self._check_inputs(
                board_planes,
                l_coords,
                t_coords,
                used_board_counts,
                urgency,
                side_to_move,
                actions,
            )

        cls_out, _ = self.encode_observation(
            board_planes, l_coords, t_coords, used_board_counts, urgency, side_to_move
        )
        return self.value_head(cls_out), self.submit_head(cls_out), self.score_actions(cls_out, actions)

    def encode_observation(
        self, board_planes: torch.Tensor, l_coords: torch.Tensor,
        t_coords: torch.Tensor, used_board_counts: torch.Tensor,
        urgency: torch.Tensor, side_to_move: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.board_tokenizer(board_planes)

        r = urgency.to(torch.float32).unsqueeze(-1)
        urgency_features = torch.stack(
            (
                self.urgency_a * r + (1.0 - self.urgency_a),
                self.urgency_b * r + (1.0 - self.urgency_b),
                self.urgency_c * r + (1.0 - self.urgency_c),
            ),
            dim=-1,
        )
        # Broadcast over the dynamic N dimension of ``tokens``.
        urgency_features = urgency_features + torch.zeros_like(tokens[..., :3])
        tokens = torch.cat((tokens, urgency_features), dim=-1)
        tokens = tokens + self.pos_encoder(l_coords, t_coords)

        cls = (
            tokens[:, :1, :] * 0.0
            + side_to_move.to(torch.float32).view(-1, 1, 1)
        )
        tokens = torch.cat((cls, tokens), dim=1)

        # cumsum produces 0..N-1 with a dynamic N; torch.arange would freeze
        # the traced board count as a constant.
        positions = torch.cumsum(torch.ones_like(l_coords), dim=1) - 1
        valid = positions < used_board_counts.unsqueeze(1)
        cls_mask = torch.zeros_like(used_board_counts.unsqueeze(1), dtype=torch.bool)
        key_padding_mask = torch.cat((cls_mask, ~valid), dim=1)

        for block in self.blocks:
            tokens = block(tokens, key_padding_mask)
        return tokens[:, 0, :], tokens[:, 1:, :]


def build_network(
    network_config: NetworkConfig | None = None,
    variant: str | VariantSpec = "very_small",
) -> SuzeroNetwork:
    return SuzeroNetwork(network_config, variant)
