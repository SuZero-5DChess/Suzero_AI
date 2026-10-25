"""Masked policy cross-entropy plus value MSE."""

from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn.functional as F

from .network import SuzeroNetwork


class AlphaZeroObjective:
    """Loss for the smallstep_movekey_v1 action layout.

    Non-submit candidates occupy columns ``[0, qmax)`` of ``other_logits``.
    The submit logit is appended as column ``qmax``.  ``policy_mask`` marks
    real slots, so padded candidates never contribute to the loss.
    """

    def __init__(self, value_weight: float = 1.0, policy_weight: float = 1.0) -> None:
        if value_weight < 0 or policy_weight < 0:
            raise ValueError("loss weights must be non-negative")
        self.value_weight = float(value_weight)
        self.policy_weight = float(policy_weight)

    def __call__(
        self, model: SuzeroNetwork, data: Mapping[str, Any]
    ) -> tuple[torch.Tensor, Mapping[str, torch.Tensor]]:
        device = next(model.parameters()).device
        moved = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in data.items()
        }
        value, submit_logit, other_logits = model(
            moved["board_planes"],
            moved["l_coords"],
            moved["t_coords"],
            moved["used_board_counts"],
            moved["urgency"],
            moved["side_to_move"],
            moved["actions"],
        )
        target_value = moved["value_target"].to(value.dtype)
        value_loss = F.mse_loss(value, target_value, reduction="mean")

        mask = moved["policy_mask"]
        target = moved["policy"].to(other_logits.dtype) * mask.to(
            other_logits.dtype
        )
        # Re-normalise after masking so numerical noise or a partially
        # truncated target still yields a distribution.
        totals = target.sum(dim=1, keepdim=True)
        target = target / totals.clamp_min(1e-12)

        logits = torch.cat((other_logits, submit_logit), dim=1)
        assert logits.shape == target.shape, (
            f"policy logits {tuple(logits.shape)} do not match target "
            f"{tuple(target.shape)}"
        )
        masked_logits = logits.masked_fill(~mask, -1.0e30)
        log_probs = F.log_softmax(masked_logits, dim=1)
        per_sample = -(target * log_probs).sum(dim=1)

        valid = mask.any(dim=1) & (totals.squeeze(1) > 0)
        if bool(valid.any()):
            policy_loss = per_sample[valid].mean()
        else:
            policy_loss = other_logits.sum() * 0.0

        loss = (
            self.value_weight * value_loss + self.policy_weight * policy_loss
        )
        batch = value.shape[0]
        metrics = {
            "loss_sum": loss.detach() * batch,
            "value_loss_sum": value_loss.detach() * batch,
            "policy_loss_sum": policy_loss.detach() * batch,
        }
        return loss, metrics
