"""
src/embedding/training/metrics.py
====================================
Chapitre 11 — Métriques d'évaluation pour les modèles d'embedding.

Métriques :
  - cosine_similarity_matrix : matrice de similarités cosinus
  - mean_reciprocal_rank     : MRR@K pour la récupération
  - recall_at_k              : Recall@K
  - average_precision        : AP pour une requête
  - mean_average_precision   : MAP sur un corpus

Ces métriques évaluent la qualité de récupération d'information
sans nécessiter de labels fins — juste une notion de "pertinence".
"""
from __future__ import annotations

import torch
import torch.nn.functional as functional


def cosine_similarity_matrix(
    embeddings_a: torch.Tensor,
    embeddings_b: torch.Tensor,
) -> torch.Tensor:
    """
    Calcule la matrice de similarités cosinus entre deux ensembles d'embeddings.

    Args:
        embeddings_a: (N, d) — normalisés L2 ou non
        embeddings_b: (M, d) — normalisés L2 ou non

    Returns:
        (N, M) — matrice de similarités cosinus dans [-1, 1]
    """
    a = functional.normalize(embeddings_a, p=2, dim=-1)
    b = functional.normalize(embeddings_b, p=2, dim=-1)
    return a @ b.T


def recall_at_k(
    queries: torch.Tensor,
    corpus: torch.Tensor,
    relevant_indices: list[list[int]],
    k: int = 10,
) -> float:
    """
    Calcule le Recall@K moyen sur un ensemble de requêtes.

    Recall@K = (nombre de documents pertinents dans les K premiers) / (total pertinents)

    Args:
        queries:          (N, d) — embeddings des requêtes
        corpus:           (M, d) — embeddings du corpus
        relevant_indices: Liste de N listes d'indices pertinents par requête
        k:                Nombre de résultats à considérer

    Returns:
        Recall@K moyen sur toutes les requêtes.
    """
    sim_matrix = cosine_similarity_matrix(queries, corpus)  # (N, M)
    top_k_indices = sim_matrix.topk(k, dim=1).indices  # (N, K)

    recalls = []
    for i, relevant in enumerate(relevant_indices):
        if not relevant:
            continue
        retrieved = set(top_k_indices[i].tolist())
        relevant_set = set(relevant)
        recall = len(retrieved & relevant_set) / len(relevant_set)
        recalls.append(recall)

    return float(sum(recalls) / len(recalls)) if recalls else 0.0


def mean_reciprocal_rank(
    queries: torch.Tensor,
    corpus: torch.Tensor,
    relevant_indices: list[list[int]],
    k: int = 100,
) -> float:
    """
    Calcule le MRR@K (Mean Reciprocal Rank) moyen.

    MRR@K = 1/N Σ_i (1 / rang_du_premier_document_pertinent)
    Si aucun document pertinent dans les K premiers : contribution = 0.

    Args:
        queries:          (N, d) — embeddings des requêtes
        corpus:           (M, d) — embeddings du corpus
        relevant_indices: Liste de N listes d'indices pertinents
        k:                Profondeur de recherche maximale

    Returns:
        MRR@K moyen.
    """
    sim_matrix = cosine_similarity_matrix(queries, corpus)  # (N, M)
    top_k_indices = sim_matrix.topk(k, dim=1).indices  # (N, K)

    reciprocal_ranks = []
    for i, relevant in enumerate(relevant_indices):
        relevant_set = set(relevant)
        rr = 0.0
        for rank, idx in enumerate(top_k_indices[i].tolist(), start=1):
            if idx in relevant_set:
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)

    return float(sum(reciprocal_ranks) / len(reciprocal_ranks))


def average_precision(
    ranked_indices: list[int],
    relevant_set: set[int],
) -> float:
    """
    Calcule l'Average Precision (AP) pour une requête.

    AP = (1 / |R|) Σ_{k=1}^{N} P@k · rel(k)
    où P@k est la précision au rang k et rel(k)=1 si le doc k est pertinent.

    Args:
        ranked_indices: Liste d'indices de documents ordonnés par score décroissant
        relevant_set:   Ensemble des indices de documents pertinents

    Returns:
        AP dans [0, 1].
    """
    if not relevant_set:
        return 0.0

    num_relevant = 0
    sum_precision = 0.0

    for k, idx in enumerate(ranked_indices, start=1):
        if idx in relevant_set:
            num_relevant += 1
            sum_precision += num_relevant / k

    return sum_precision / len(relevant_set)


def mean_average_precision(
    queries: torch.Tensor,
    corpus: torch.Tensor,
    relevant_indices: list[list[int]],
    k: int = 1000,
) -> float:
    """
    Calcule la MAP@K (Mean Average Precision) sur un ensemble de requêtes.

    Args:
        queries:          (N, d) — embeddings des requêtes
        corpus:           (M, d) — embeddings du corpus
        relevant_indices: Liste de N listes d'indices pertinents
        k:                Profondeur de recherche maximale

    Returns:
        MAP@K dans [0, 1].
    """
    sim_matrix = cosine_similarity_matrix(queries, corpus)
    top_k_indices = sim_matrix.topk(k, dim=1).indices

    aps = []
    for i, relevant in enumerate(relevant_indices):
        if not relevant:
            continue
        ap = average_precision(
            ranked_indices=top_k_indices[i].tolist(),
            relevant_set=set(relevant),
        )
        aps.append(ap)

    return float(sum(aps) / len(aps)) if aps else 0.0


def log_gradient_norms(model: torch.nn.Module, step: int) -> dict[str, float]:
    """
    Logue la norme des gradients par couche.

    Utile pour diagnostiquer les explosions ou disparitions de gradients.
    Une norme totale > 10 indique souvent un problème de learning rate.

    Args:
        model: Modèle PyTorch (après .backward()).
        step:  Étape courante (pour le seuil de logging).

    Returns:
        Dict {"total_grad_norm": float, "layer_name": float, ...}
    """
    from loguru import logger

    grad_norms: dict[str, float] = {}
    total_norm = 0.0

    for name, param in model.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2).item()
            grad_norms[name] = param_norm
            total_norm += param_norm**2

    total_norm = total_norm**0.5

    if step % 500 == 0:
        logger.debug(f"Norme totale des gradients : {total_norm:.4f}")
        for name, norm in sorted(
            grad_norms.items(), key=lambda x: x[1], reverse=True
        )[:5]:
            logger.debug(f"  {name}: {norm:.4f}")

    return {"total_grad_norm": total_norm, **grad_norms}
