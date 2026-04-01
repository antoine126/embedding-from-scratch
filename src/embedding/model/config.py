"""
src/embedding/model/config.py
==============================
Chapitre 1 — Configuration typée du modèle et de l'entraînement.

Toute configuration passe par ces dataclasses — jamais de constantes
hardcodées dans le code. Les fichiers TOML du dossier configs/ sont
chargés via ExperimentConfig.from_toml().
"""
from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass
class ModelConfig:
    """Configuration de l'architecture du modèle d'embedding."""

    d_model: int = 768
    n_heads: int = 12
    n_layers: int = 6
    d_ff: int = 3072
    max_seq_len: int = 512
    vocab_size: int = 32_000
    dropout: float = 0.1
    pooling: Literal["mean", "cls", "weighted_mean"] = "mean"
    pos_encoding: Literal["sinusoidal", "learned", "rope"] = "learned"
    activation: Literal["relu", "gelu", "swiglu", "geglu"] = "gelu"

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) doit être divisible "
                f"par n_heads ({self.n_heads})."
            )

    @property
    def d_head(self) -> int:
        """Dimension par tête d'attention."""
        return self.d_model // self.n_heads


@dataclass
class TrainingConfig:
    """Configuration de la boucle d'entraînement."""

    batch_size: int = 256
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_steps: int = 1_000
    max_steps: int = 50_000
    num_epochs: int = 0  # 0 = mode étapes (max_steps), > 0 = mode époques
    gradient_clip: float = 1.0
    mixed_precision: Literal["bf16", "fp16", "fp32"] = "bf16"
    grad_accum_steps: int = 1
    device: str = "auto"  # "auto" | "cpu" | "cuda" | "cuda:0" | "cuda:1" | "mps"


@dataclass
class LossConfig:
    """Configuration de la fonction de coût contrastive."""

    type: Literal["contrastive", "triplet", "mnr", "matryoshka"] = "mnr"
    temperature: float = 0.05
    margin: float = 0.5


@dataclass
class DataConfig:
    """Configuration des données d'entraînement."""

    train_path: str = "data/pairs_train.jsonl"
    val_path: str = "data/pairs_val.jsonl"
    num_workers: int = 4


@dataclass
class LoggingConfig:
    """Configuration du logging et du suivi d'expériences."""

    project: str = "embedding-from-scratch"
    run_name: str = "base-mnr-v1"
    log_every: int = 100
    eval_every: int = 1_000


@dataclass
class ExperimentConfig:
    """Configuration complète d'une expérience d'entraînement."""

    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    data: DataConfig = field(default_factory=DataConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @classmethod
    def from_toml(cls, path: str | Path) -> ExperimentConfig:
        """Charge une configuration depuis un fichier TOML."""
        with open(path, "rb") as f:
            raw = tomllib.load(f)
        return cls(
            model=ModelConfig(**raw.get("model", {})),
            training=TrainingConfig(**raw.get("training", {})),
            loss=LossConfig(**raw.get("loss", {})),
            data=DataConfig(**raw.get("data", {})),
            logging=LoggingConfig(**raw.get("logging", {})),
        )
