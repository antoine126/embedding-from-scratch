"""
src/embedding/losses/matryoshka.py
=====================================
Chapitre 9 — Matryoshka Representation Learning (MRL).

MRL entraîne le modèle à produire des embeddings utiles à plusieurs
dimensions simultanément. Les premières d dimensions d'un vecteur de
dimension D sont elles-mêmes un bon embedding de dimension d.

Propriété clé : un embedding MRL de dimension 1024 peut être tronqué
à 128 dimensions et rester performant — sans ré-entraînement.

L_MRL = Σ_{d ∈ dims} w_d · L(emb[:d], targets)

Référence : Kusupati et al. 2022 — "Matryoshka Representation Learning"
"""
from __future__ import annotations

import torch
import torch.nn as nn

from embedding.losses.mnr import MNRLoss


class MatryoshkaLoss(nn.Module):
    """
    Matryoshka Representation Learning Loss.

    Entraîne le modèle avec une perte MNR à plusieurs dimensions.
    Les dimensions sont évaluées de la plus petite à la plus grande.

    Args:
        dimensions:  Liste croissante de dimensions à entraîner.
                     Exemple : [64, 128, 256, 512, 768]
        temperature: Température pour la MNR Loss interne.
        weights:     Poids optionnels par dimension (par défaut égaux).

    Exemple :
        loss_fn = MatryoshkaLoss(
            dimensions=[64, 128, 256, 512, 768],
            temperature=0.05,
        )
        queries   = model(q_ids, q_mask)   # (B, 768)
        positives = model(p_ids, p_mask)   # (B, 768)
        loss = loss_fn(queries, positives)
    """

    def __init__(
        self,
        dimensions: list[int],
        temperature: float = 0.05,
        weights: list[float] | None = None,
    ) -> None:
        super().__init__()
        if not dimensions or any(d <= 0 for d in dimensions):
            raise ValueError("dimensions doit être une liste de dimensions positives.")

        self.dimensions = sorted(dimensions)
        self.mnr = MNRLoss(temperature=temperature)

        if weights is not None:
            if len(weights) != len(dimensions):
                raise ValueError(
                    f"len(weights)={len(weights)} != len(dimensions)={len(dimensions)}"
                )
            total = sum(weights)
            self.weights = [w / total for w in weights]
        else:
            # Poids égaux par défaut
            self.weights = [1.0 / len(dimensions)] * len(dimensions)

    def forward(
        self,
        queries: torch.Tensor,
        positives: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            queries:   (B, D) — embeddings full-dimension (non normalisés L2)
            positives: (B, D) — embeddings full-dimension (non normalisés L2)

        Returns:
            Perte scalaire combinant toutes les dimensions.

        Note: Les embeddings sont normalisés L2 par dimension dans cette fonction.
        """
        import torch.nn.functional as functionnal

        total_loss = torch.tensor(0.0, device=queries.device)

        for dim, weight in zip(self.dimensions, self.weights):
            # Tronquer et re-normaliser à la dimension d
            q_trunc = functionnal.normalize(queries[:, :dim], p=2, dim=-1)
            p_trunc = functionnal.normalize(positives[:, :dim], p=2, dim=-1)

            loss_d = self.mnr(q_trunc, p_trunc)
            total_loss = total_loss + weight * loss_d

        return total_loss
