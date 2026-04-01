"""
src/embedding/losses/triplet.py
==================================
Chapitre 9 — Triplet Loss.

Perte sur des triplets (ancre, positif, négatif) en espace cosinus :
  L(a, p, n) = max(0, d(a, p) - d(a, n) + margin)

Garantit que d(ancre, positif) < d(ancre, négatif) - margin.
Travaille en distance cosinus sur des embeddings normalisés L2 :
  d_cos(u, v) = 1 - cos(u, v)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as functional


class TripletLoss(nn.Module):
    """
    Triplet Loss en espace cosinus.

    Args:
        margin:    Marge minimale entre d(a,p) et d(a,n). Défaut 0.5.
        reduction: "mean" (défaut) ou "sum".

    Exemple :
        loss_fn = TripletLoss(margin=0.5)
        anchor   = F.normalize(torch.randn(8, 768), dim=1)
        positive = F.normalize(torch.randn(8, 768), dim=1)
        negative = F.normalize(torch.randn(8, 768), dim=1)
        loss = loss_fn(anchor, positive, negative)
    """

    def __init__(self, margin: float = 0.5, reduction: str = "mean") -> None:
        super().__init__()
        self.margin = margin
        self.reduction = reduction

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            anchor:   (B, d) — embeddings normalisés L2
            positive: (B, d) — embeddings normalisés L2
            negative: (B, d) — embeddings normalisés L2

        Returns:
            Perte scalaire (ou somme si reduction="sum").
        """
        # Distances cosinus : d = 1 - cos(u, v)
        # Pour des embeddings normalisés L2 : cos(u,v) = u · v
        d_pos = 1.0 - (anchor * positive).sum(dim=-1)  # (B,)
        d_neg = 1.0 - (anchor * negative).sum(dim=-1)  # (B,)

        # Triplet loss : max(0, d_pos - d_neg + margin)
        losses = functional.relu(d_pos - d_neg + self.margin)  # (B,)

        if self.reduction == "mean":
            return losses.mean()
        if self.reduction == "sum":
            return losses.sum()
        return losses
