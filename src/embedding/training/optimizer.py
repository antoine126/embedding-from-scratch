"""
src/embedding/training/optimizer.py
======================================
Chapitre 10 — Optimiseurs et schedulers de learning rate.

Fonctions :
  - get_optimizer : construit AdamW avec séparation weight decay / no-decay
  - get_scheduler : scheduler cosine avec warmup linéaire

Bonne pratique : ne pas appliquer le weight decay aux biais et
aux paramètres de LayerNorm — cela dégrade les performances.
"""
from __future__ import annotations

import math

import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from embedding.model.config import TrainingConfig


def get_optimizer(
    model: nn.Module,
    config: TrainingConfig,
) -> AdamW:
    """
    Construit AdamW avec une séparation propre des groupes de paramètres.

    Paramètres avec weight decay :
      - Matrices de projection (poids des couches linéaires)

    Paramètres sans weight decay :
      - Biais
      - Paramètres de LayerNorm (gain et biais)
      - Embeddings

    Cette séparation suit les recommandations de BERT et GPT-2.

    Args:
        model:  Modèle PyTorch.
        config: TrainingConfig avec learning_rate et weight_decay.

    Returns:
        Optimiseur AdamW configuré.
    """
    # Identifier les paramètres qui ne doivent pas avoir de weight decay
    no_decay = {"bias", "LayerNorm.weight", "LayerNorm.bias"}

    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(nd in name for nd in no_decay):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": config.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    return AdamW(
        param_groups,
        lr=config.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
    )


def get_cosine_schedule_with_warmup(
    optimizer: AdamW,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.0,
) -> LambdaLR:
    """
    Scheduler cosine avec warmup linéaire.

    Phase 1 (warmup) : lr augmente linéairement de 0 -> lr_max
    Phase 2 (cosine) : lr décroît de lr_max -> lr_max * min_lr_ratio

    C'est le scheduler standard pour les modèles d'embedding (2022+).

    Args:
        optimizer:            Optimiseur AdamW.
        num_warmup_steps:     Nombre d'étapes de warmup linéaire.
        num_training_steps:   Nombre total d'étapes d'entraînement.
        min_lr_ratio:         Ratio lr minimal (défaut 0.0 = décroît jusqu'à 0).

    Returns:
        LambdaLR scheduler.
    """

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            # Phase warmup : interpolation linéaire
            return float(current_step) / float(max(1, num_warmup_steps))

        # Phase cosine
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        # Interpoler entre min_lr_ratio et 1.0
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return LambdaLR(optimizer, lr_lambda)
