# alphazero/network.py
"""
Transformer network for semimove-level AlphaZero.

State encoder outputs:
  - value: scalar in [-1, 1] from current player's perspective
  - submit_logit: scalar prior logit for submit action
  - raw_logits: [N, 16] per-board source-square logits (the policy)

Inputs:
  - board_planes: [B, N, 14, 16]
  - l_coords, t_coords: [B, N] int
  - urgency: [B] long — natural number, boards remaining until forced end
    Embedded as 3 linear features (a*r + 1-a) cat'd into each board token
    after BoardTokenizer (output 125 → cat 3 → 128). At r=1 all three equal 1;
    at larger r they decay toward 0 at different rates (a=-0.4, -0.05, -0.2).
"""

import math
import torch
import torch.nn as nn

from .config import NetworkConfig


class SinusoidalPositionEncoding(nn.Module):
    """
    Static sinusoidal positional encoding for (L, T) coordinates.

    Pre-computes a full table covering both positive and negative coordinate
    ranges at init, then slices by index in forward — no learned parameters,
    no per-step sin/cos computation. Added directly to token embeddings.

    The table covers [-max_pos, max_pos) so that negative coordinates (common
    in 5D chess for past timelines / turns) map to the correct encoding.
    """

    def __init__(self, d_model: int, max_l: int = 10, max_t: int = 50):
        super().__init__()
        self.d_model = d_model
        half = d_model // 2  # each of L and T gets half the dims; cat = d_model

        div_term = torch.exp(
            torch.arange(0, half, 2, dtype=torch.float32) * (-math.log(10000.0) / half)
        )  # [half//2]

        # Pre-compute table covering [-max_pos, max_pos) so negative indices
        # are looked up correctly (offset = max_pos in forward).
        max_pos = max(max_l, max_t)
        pos = torch.arange(-max_pos, max_pos, dtype=torch.float32).unsqueeze(-1)  # [2*max_pos, 1]
        pe = torch.zeros(2 * max_pos, half)
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        self.register_buffer("pe", pe)
        self.pe_offset = max_pos

    def forward(self, l_coords: torch.Tensor, t_coords: torch.Tensor) -> torch.Tensor:
        """Slice pre-computed table by index for both L and T axes.

        Args:
            l_coords: [N] int — timeline indices (may be negative)
            t_coords: [N] int — turn indices (may be negative)
        Returns:
            [N, d_model] — positional encoding directly added to tokens
        """
        pe_l = self.pe[l_coords + self.pe_offset]  # [N, half]
        pe_t = self.pe[t_coords + self.pe_offset]  # [N, half]
        return torch.cat([pe_l, pe_t], dim=-1)  # [N, d_model]


class BoardTokenizer(nn.Module):
    """Convert one board's tensor representation into a d_model - 3 embedding."""

    def __init__(self, cfg: NetworkConfig):
        super().__init__()
        input_dim = cfg.piece_channels * cfg.board_squares  # 224
        out_dim = cfg.d_model - 3  # 125, leaving room for 3 urgency features
        self.proj = nn.Sequential(
            nn.Linear(input_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )

    def forward(self, board_planes: torch.Tensor) -> torch.Tensor:
        flat = board_planes.reshape(board_planes.size(0), -1)
        return self.proj(flat)


class AlphaZeroNetwork(nn.Module):
    """Transformer-based network for semimove-level 5D chess."""

    def __init__(self, cfg: NetworkConfig | None = None):
        super().__init__()
        if cfg is None:
            cfg = NetworkConfig()
        self.cfg = cfg

        self.board_tokenizer = BoardTokenizer(cfg)
        self.pos_encoder = SinusoidalPositionEncoding(cfg.d_model, cfg.max_timelines, cfg.max_turns)

        # Urgency features: a*r + (1-a) with negative slopes (decay as r grows)
        # r = boards remaining until forced end; at r=1 all three equal 1,
        # at larger r they decay toward 0 at different rates.
        self.register_buffer("urgency_a", torch.tensor(-0.4))
        self.register_buffer("urgency_b", torch.tensor(-0.05))
        self.register_buffer("urgency_c", torch.tensor(-0.2))

        # Side-to-move is encoded by setting the [CLS] token:
#   side_to_move=0 (white): CLS = all zeros
#   side_to_move=1 (black): CLS = all ones
# (No learned CLS token parameter — the value is a hyperplane in
#  d_model space that every transformer layer can attend to.)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_ff,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=cfg.n_layers,
            enable_nested_tensor=False,
        )

        # Value and submit heads from [CLS]
        self.value_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model // 2),
            nn.GELU(),
            nn.Linear(cfg.d_model // 2, 1),
            nn.Tanh(),
        )
        self.submit_head = nn.Linear(cfg.d_model, 1)

        # Auxiliary per-board source-square logits (kept for diagnostics)
        self.policy_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model // 2),
            nn.GELU(),
            nn.Linear(cfg.d_model // 2, cfg.board_squares),
        )

        # (No destination-aware action scorer — policy is read from raw_logits)

    def forward(
        self,
        board_planes: torch.Tensor,
        l_coords: torch.Tensor,
        t_coords: torch.Tensor,
        urgency: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        return_latent: bool = False,
        side_to_move: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        """
        Args:
            board_planes: [B, N, 12, 16]
            l_coords: [B, N] int
            t_coords: [B, N] int
            urgency: [B] long — natural number, boards remaining until forced end
            padding_mask: [B, N] bool (True = padding)
            side_to_move: [B] long — 0=white, 1=black. Sets [CLS] token to
                          all-zeros (white) or all-ones (black).
        Returns:
            value: [B, 1]
            submit_logit: [B, 1]
            raw_logits: [B, N, 16]
            if return_latent: also (board_out [B, N, d_model], cls_out [B, d_model])
        """
        B, N = board_planes.shape[:2]
        device = board_planes.device

        bp_flat = board_planes.reshape(B * N, self.cfg.piece_channels, self.cfg.board_squares)
        # BoardTokenizer produces [B*N, 253]
        tokens = self.board_tokenizer(bp_flat).reshape(B, N, self.cfg.d_model - 3)

        # Urgency: [B] natural number r → [B, N, 3] linear features a*r + (1-a)
        # At r=1 all three equal 1; at larger r they diverge by slope.
        r = urgency.float()  # [B]
        urg_feats = torch.stack([
            self.urgency_a * r + (1.0 - self.urgency_a),
            self.urgency_b * r + (1.0 - self.urgency_b),
            self.urgency_c * r + (1.0 - self.urgency_c),
        ], dim=-1)  # [B, 3]
        urg_feats = urg_feats.unsqueeze(1).expand(-1, N, -1)  # [B, N, 3]
        tokens = torch.cat([tokens, urg_feats], dim=-1)  # [B, N, 128]

        l_flat = l_coords.reshape(B * N)
        t_flat = t_coords.reshape(B * N)
        pos_enc = self.pos_encoder(l_flat, t_flat).reshape(B, N, self.cfg.d_model)
        tokens = tokens + pos_enc

        # [CLS] token encodes side-to-move: all zeros for white, all ones for black.
        if side_to_move is None:
            side_to_move = torch.zeros(B, dtype=torch.long, device=device)
        cls_value = side_to_move.float().view(B, 1, 1)  # [B, 1, 1]
        cls = cls_value.expand(B, 1, self.cfg.d_model)  # [B, 1, d_model]
        tokens = torch.cat([cls, tokens], dim=1)

        if padding_mask is not None:
            cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=device)
            padding_mask = torch.cat([cls_mask, padding_mask], dim=1)

        out = self.transformer(tokens, src_key_padding_mask=padding_mask)

        cls_out = out[:, 0, :]
        board_out = out[:, 1:, :]

        value = self.value_head(cls_out)
        submit_logit = self.submit_head(cls_out)
        raw_logits = self.policy_head(board_out)

        if return_latent:
            return value, submit_logit, raw_logits, board_out, cls_out
        return value, submit_logit, raw_logits

    # ── inference helpers (score_legal_actions, predict, etc.) removed ──
    # These were thin wrappers around forward() + policy_head() + indexing
    # that belonged at the call site, not on the network class.  Each caller
    # now inlines the 2-5 lines it actually needs.
