"""
src/embedding/losses/mnr.py
==============================
Chapitre 9 — Multiple Negative Ranking Loss (MNR).

La MNR Loss est la fonction de coût recommandée pour entraîner
des modèles d'embedding à partir de paires (query, positive).

Principe : pour chaque query q_i, son positif p_i doit être
plus similaire que tous les autres positifs du batch {p_j, j≠i},
qui servent de négatifs in-batch.

L = -1/B Σ_i log [ exp(sim(q_i, p_i)/τ) / Σ_j exp(sim(q_i, p_j)/τ) ]

C'est une InfoNCE Loss où la cible est la diagonale de la matrice
de similarité. Un batch de taille B génère B-1 négatifs par exemple
sans nécessiter de labels négatifs explicites.

Référence : Henderson et al. 2017 (Efficient Natural Language Response
            Suggestion for Smart Reply)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as functional


class MNRLoss(nn.Module):
    """
    Multiple Negative Ranking Loss (InfoNCE avec négatifs in-batch).

    Args:
        temperature: Température τ pour le softmax. Plus τ est petit,
                     plus la distribution est concentrée.
                     Valeurs typiques : 0.01-0.1.

    Exemple :
        loss_fn  = MNRLoss(temperature=0.05)
        queries  = F.normalize(torch.randn(32, 768), dim=1)
        positives = F.normalize(torch.randn(32, 768), dim=1)
        loss = loss_fn(queries, positives)
    """

    def __init__(self, temperature: float = 0.05) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        queries: torch.Tensor,
        positives: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            queries:   (B, d) — embeddings de requêtes, normalisés L2
            positives: (B, d) — embeddings de passages positifs, normalisés L2

        Returns:
            Perte scalaire (cross-entropie sur la diagonale).
        """
        # Matrice de similarité : sim[i,j] = cos(q_i, p_j) / τ
        sim_matrix = (queries @ positives.T) / self.temperature  # (B, B)

        # Cible : la diagonale (q_i doit être similaire à p_i uniquement)
        targets = torch.arange(len(queries), device=queries.device)

        # Cross-entropie : équivalent à softmax + log sur la diagonale
        return functional.cross_entropy(sim_matrix, targets)


class InfoNCELoss(nn.Module):
    """
    InfoNCE Loss bidirectionnelle (symétrique).

    Combine la MNR Loss dans les deux sens :
      - Chaque query doit retrouver son positif
      - Chaque positif doit retrouver sa query

    Utilisée dans CLIP (Radford 2021) et de nombreux modèles bimodaux.
    La perte symétrique stabilise l'entraînement et accélère la convergence.
    """

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        queries: torch.Tensor,
        positives: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            queries:   (B, d) — embeddings normalisés L2
            positives: (B, d) — embeddings normalisés L2

        Returns:
            Perte scalaire (moyenne des deux directions).
        """
        sim = (queries @ positives.T) / self.temperature  # (B, B)
        targets = torch.arange(len(queries), device=queries.device)

        # Perte query -> positive
        loss_q2p = functional.cross_entropy(sim, targets)

        # Perte positive -> query (matrice transposée)
        loss_p2q = functional.cross_entropy(sim.T, targets)

        return (loss_q2p + loss_p2q) / 2.0
