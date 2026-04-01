"""
src/embedding/losses/contrastive.py
=====================================
Chapitre 9 — Contrastive Loss.

Perte contrastive sur des paires (emb1, emb2, label) :
  - label=0 : paire similaire -> minimiser la distance
  - label=1 : paire dissemblable -> maximiser la distance jusqu'au margin

L(emb1, emb2, y) = (1-y) · 0.5·d² + y · 0.5·max(0, margin - d)²

Inconvénient principal : nécessite des labels binaires explicites et
traite chaque paire indépendamment (pas d'apprentissage contrastif
à grande échelle). Voir MNRLoss pour l'alternative moderne.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as functional


class ContrastiveLoss(nn.Module):
    """
    Contrastive Loss (Hadsell et al. 2006).

    Args:
        margin: Distance minimale souhaitée entre paires dissemblables.

    Exemple :
        loss_fn = ContrastiveLoss(margin=0.5)
        emb1 = F.normalize(torch.randn(8, 768), dim=1)
        emb2 = F.normalize(torch.randn(8, 768), dim=1)
        labels = torch.randint(0, 2, (8,))
        loss = loss_fn(emb1, emb2, labels)
    """

    def __init__(self, margin: float = 0.5) -> None:
        super().__init__()
        self.margin = margin

    def forward(
        self,
        emb1: torch.Tensor,
        emb2: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            emb1:   (B, d) — embeddings normalisés L2
            emb2:   (B, d) — embeddings normalisés L2
            labels: (B,) — 0=similaire, 1=dissemblable

        Returns:
            Perte scalaire.
        """
        distances = functional.pairwise_distance(emb1, emb2, p=2)  # (B,)

        # Terme pour les paires similaires (label=0)
        loss_similar = 0.5 * distances.pow(2)

        # Terme pour les paires dissemblables (label=1)
        loss_dissim = 0.5 * functional.relu(self.margin - distances).pow(2)

        loss = (1 - labels.float()) * loss_similar + labels.float() * loss_dissim

        return loss.mean()
