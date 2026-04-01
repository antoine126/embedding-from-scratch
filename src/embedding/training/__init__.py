"""Boucle d'entraînement et métriques (Chapitres 2, 10-13)."""
from embedding.training.metrics import mean_average_precision, mean_reciprocal_rank, recall_at_k
from embedding.training.optimizer import get_cosine_schedule_with_warmup, get_optimizer
from embedding.training.trainer import EmbeddingTrainer

__all__ = [
    "EmbeddingTrainer",
    "get_optimizer",
    "get_cosine_schedule_with_warmup",
    "recall_at_k",
    "mean_reciprocal_rank",
    "mean_average_precision",
]
