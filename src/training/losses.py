"""Safety-Aware Asymmetric Huber Loss and related loss functions.

This module provides the core loss functions used in the EL defect severity
regression pipeline. The flagship class is `SafetyAwareAsymmetricHuberLoss`
(SAHL), which combines Huber's smooth L1/L2 transition with asymmetric
weighting to penalise missed high-severity defects more heavily.

Example:
    >>> criterion = SafetyAwareAsymmetricHuberLoss(
    ...     threshold=0.70, critical_weight=2.5, beta=0.5
    ... )
    >>> loss = criterion(predictions, targets)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SafetyAwareAsymmetricHuberLoss(nn.Module):
    """Safety-Aware Asymmetric Huber Loss (SAHL) for defect severity regression.

    Combines asymmetric sample weighting with the Huber loss function.
    The critical penalty is applied only when both conditions hold:
    (1) the ground-truth severity is at or above ``threshold``, **and**
    (2) the model under-predicts (``prediction < threshold``).  This
    two-condition gating directs the optimiser to penalise missed
    high-severity defects without distorting the calibration of correctly
    identified critical samples.

    The Huber component transitions from quadratic (L2) to linear (L1) at
    a prediction error of ``beta``, providing outlier robustness while
    maintaining smooth gradients near zero.

    **Backward compatibility**: When ``beta=0.0`` (default), the loss
    degenerates to a purely asymmetric L1 loss with the same two-condition
    gating, producing identical behaviour to the original
    ``WeightedL1Loss`` used in earlier experiments when the prediction
    condition is inactive.

    Mathematically, for a single prediction-target pair:

    .. math::

        \\delta &= |\\hat{y} - y| \\\\
        w &=
        \\begin{cases}
            w_{\\text{critical}} &
                \\text{if } y \\geq \\tau \\text{ and } \\hat{y} < \\tau \\\\
            w_{\\text{normal}}   &
                \\text{otherwise}
        \\end{cases} \\\\[4pt]
        H(\\delta) &=
        \\begin{cases}
            \\frac{1}{2}\\delta^2 / \\beta &
                \\text{if } \\beta > 0 \\text{ and } \\delta \\leq \\beta \\\\
            \\delta - \\frac{1}{2}\\beta &
                \\text{if } \\beta > 0 \\text{ and } \\delta > \\beta \\\\
            \\delta &
                \\text{if } \\beta = 0
        \\end{cases} \\\\[4pt]
        \\mathcal{L} &= \\frac{1}{N} \\sum w \\cdot H(\\delta)

    Attributes:
        threshold: Target threshold τ for the two-condition gate.
        critical_weight: Loss multiplier for safety-critical under-predictions.
        normal_weight: Loss multiplier for all other samples.
        beta: Huber transition parameter (0.0 → pure L1).

    Args:
        threshold: Target severity threshold τ defining the boundary for both
            ground-truth severity and model under-prediction (default: ``0.70``).
        critical_weight: Loss multiplier for safety-critical samples
            (default: ``2.5``).
        normal_weight: Loss multiplier for non-critical samples
            (default: ``1.0``).
        beta: Huber transition point. When ``beta=0.0`` the loss reduces
            to weighted L1. When ``beta > 0``, residuals ≤ beta use a
            quadratic penalty and residuals > beta use a linear penalty
            (default: ``0.0``).
    """

    def __init__(
        self,
        threshold: float = 0.70,
        critical_weight: float = 2.5,
        normal_weight: float = 1.0,
        beta: float = 0.0,
    ) -> None:
        super().__init__()
        if threshold < 0.0 or threshold > 1.0:
            raise ValueError(
                f"threshold must be in [0, 1], got {threshold}"
            )
        if critical_weight < 0.0:
            raise ValueError(
                f"critical_weight must be non-negative, got {critical_weight}"
            )
        if normal_weight < 0.0:
            raise ValueError(
                f"normal_weight must be non-negative, got {normal_weight}"
            )
        if beta < 0.0:
            raise ValueError(f"beta must be non-negative, got {beta}")

        self.threshold = threshold
        self.critical_weight = critical_weight
        self.normal_weight = normal_weight
        self.beta = beta

    def forward(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute the forward pass of the SAHL loss.

        Args:
            predictions: Model output tensor of shape ``(N, 1)`` or ``(N,)``.
            targets: Ground-truth tensor of matching shape.

        Returns:
            Scalar loss value averaged over the batch.
        """
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        abs_error = torch.abs(predictions - targets)

        # --- Huber component ---
        if self.beta > 0.0:
            # Smooth transition: quadratic for small errors, linear for large
            huber = torch.where(
                abs_error <= self.beta,
                0.5 * (abs_error ** 2) / self.beta,
                abs_error - 0.5 * self.beta,
            )
        else:
            # Degenerate case: pure L1 (backward-compatible)
            huber = abs_error

        # --- Safety-aware asymmetric weighting (two-condition gate) ---
        # Per manuscript Eq. (6): critical weight applies only when
        #   y >= τ  AND  ŷ < τ  (high-severity sample + model under-predicts)
        critical_mask = (targets >= self.threshold) & (predictions < self.threshold)
        weights = torch.where(
            critical_mask,
            torch.full_like(targets, self.critical_weight),
            torch.full_like(targets, self.normal_weight),
        )

        return torch.mean(huber * weights)

    def extra_repr(self) -> str:
        return (
            f"threshold={self.threshold}, critical_weight={self.critical_weight}, "
            f"normal_weight={self.normal_weight}, beta={self.beta}"
        )


class WeightedMSELoss(nn.Module):
    """Asymmetric weighted Mean Squared Error loss.

    Applies a higher penalty to high-severity samples, defined as those
    whose target value is at or above ``threshold``. This loss is provided
    for ablation comparison against SAHL.

    .. math::

        \\mathcal{L} = \\frac{1}{N} \\sum w_i \\cdot (\\hat{y}_i - y_i)^2

    where :math:`w_i = w_{\\text{critical}}` if :math:`y_i \\geq \\tau`,
    else :math:`w_i = w_{\\text{normal}}`.

    Args:
        threshold: Target value above which the critical weight applies.
        critical_weight: MSE multiplier for high-severity samples.
        normal_weight: MSE multiplier for normal samples (default: ``1.0``).
    """

    def __init__(
        self,
        threshold: float,
        critical_weight: float,
        normal_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.threshold = threshold
        self.critical_weight = critical_weight
        self.normal_weight = normal_weight

    def forward(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute the weighted MSE.

        Args:
            predictions: Model output tensor.
            targets: Ground-truth tensor of matching shape.

        Returns:
            Scalar loss value averaged over the batch.
        """
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        squared_error = (predictions - targets) ** 2
        weights = torch.where(
            targets >= self.threshold,
            torch.full_like(targets, self.critical_weight),
            torch.full_like(targets, self.normal_weight),
        )
        return torch.mean(squared_error * weights)

    def extra_repr(self) -> str:
        return (
            f"threshold={self.threshold}, "
            f"critical_weight={self.critical_weight}, "
            f"normal_weight={self.normal_weight}"
        )


def build_loss(
    loss_type: str,
    huber_beta: float = 0.5,
    loss_weight_threshold: float = 0.70,
    loss_weight_multiplier: float = 2.5,
) -> nn.Module:
    """Loss-function factory mapping a string identifier to a callable.

    Args:
        loss_type: One of ``"smoothl1"``, ``"mse"``, ``"weighted_l1"``,
            ``"weighted_mse"``, or ``"sahl"``.
        huber_beta: Beta parameter for ``smoothl1`` and the ``sahl``
            loss type (only used when ``sahl`` has beta > 0).
        loss_weight_threshold: Target threshold for asymmetric losses.
        loss_weight_multiplier: Critical-sample weight for asymmetric
            losses.

    Returns:
        An ``nn.Module`` loss criterion.

    Raises:
        ValueError: If ``loss_type`` is unrecognised.
    """
    if loss_type == "smoothl1":
        return nn.SmoothL1Loss(beta=huber_beta)
    if loss_type == "mse":
        return nn.MSELoss()
    if loss_type in ("weighted_l1", "sahl"):
        # SAHL with beta=0 behaves identically to the original WeightedL1Loss.
        return SafetyAwareAsymmetricHuberLoss(
            threshold=loss_weight_threshold,
            critical_weight=loss_weight_multiplier,
            normal_weight=1.0,
            beta=huber_beta if loss_type == "sahl" else 0.0,
        )
    if loss_type == "weighted_mse":
        return WeightedMSELoss(
            threshold=loss_weight_threshold,
            critical_weight=loss_weight_multiplier,
            normal_weight=1.0,
        )
    raise ValueError(f"Unsupported loss_type: {loss_type}")
