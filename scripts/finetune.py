"""
scripts/finetune.py
====================
Chapitre 13 — Fine-tuning contrastif d'un modèle d'embedding.

Ce script lance le fine-tuning d'un modèle (pré-entraîné ou from scratch)
avec la MNR Loss sur des paires (query, positive).

Usage :
    uv run python scripts/finetune.py --config configs/embedding_base.toml \\
        --tokenizer checkpoints/pretrain/tokenizer/ \\
        --checkpoint checkpoints/pretrain/pretrained_model.pt \\
        --output checkpoints/finetune/

Données attendues (config.data.train_path) :
    Format JSONL : {"query": "...", "positive": "..."}
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from loguru import logger
from torch.utils.data import DataLoader

from embedding.data.collators import PairCollator
from embedding.data.dataset import PairDataset
from embedding.data.tokenizer import load_tokenizer
from embedding.losses import MatryoshkaLoss, MNRLoss
from embedding.model.config import ExperimentConfig
from embedding.model.encoder import EmbeddingModel
from embedding.training.optimizer import get_cosine_schedule_with_warmup, get_optimizer
from embedding.training.trainer import EmbeddingTrainer
from embedding.utils.device import DeviceManager


def build_criterion(config: ExperimentConfig) -> torch.nn.Module:
    """Instancie la fonction de coût selon la configuration."""
    loss_type = config.loss.type
    temperature = config.loss.temperature

    if loss_type == "mnr":
        return MNRLoss(temperature=temperature)

    if loss_type == "matryoshka":
        return MatryoshkaLoss(
            dimensions=[64, 128, 256, 512, config.model.d_model],
            temperature=temperature,
        )

    if loss_type == "triplet":
        from embedding.losses import TripletLoss

        return TripletLoss(margin=config.loss.margin)

    if loss_type == "contrastive":
        from embedding.losses import ContrastiveLoss

        return ContrastiveLoss(margin=config.loss.margin)

    raise ValueError(f"Type de perte inconnu : {loss_type!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tuning contrastif")
    parser.add_argument("--config", required=True, help="Chemin vers le fichier TOML")
    parser.add_argument("--tokenizer", required=True, help="Répertoire du tokenizer")
    parser.add_argument(
        "--checkpoint", default=None, help="Checkpoint de pré-entraînement"
    )
    parser.add_argument(
        "--output", default="checkpoints/finetune", help="Répertoire de sortie"
    )
    args = parser.parse_args()

    # --- Configuration ---
    config = ExperimentConfig.from_toml(args.config)
    logger.info(f"Devices disponibles : {DeviceManager.available_devices()}")
    dm = DeviceManager.setup(
        seed=42,
        mixed_precision=config.training.mixed_precision,
        device=config.training.device,
    )
    logger.info(f"Device : {dm}")

    # --- Tokenizer ---
    tokenizer = load_tokenizer(Path(args.tokenizer) / "tokenizer.json")
    logger.info(f"Tokenizer chargé : {tokenizer.get_vocab_size()} tokens")

    # --- Datasets ---
    train_dataset = PairDataset(config.data.train_path)
    val_dataset = PairDataset(config.data.val_path)
    logger.info(
        f"Données : {len(train_dataset)} paires train, {len(val_dataset)} paires val"
    )

    collator = PairCollator(tokenizer=tokenizer, max_length=config.model.max_seq_len)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=config.data.num_workers,
        pin_memory=(dm.device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=config.data.num_workers,
    )

    # --- Modèle ---
    model = EmbeddingModel(config.model).to(dm.device)

    if args.checkpoint:
        state_dict = torch.load(args.checkpoint, map_location=dm.device)
        model.load_state_dict(state_dict, strict=False)
        logger.info(f"Checkpoint chargé : {args.checkpoint}")

    # --- Optimiseur, scheduler, perte ---
    optimizer = get_optimizer(model, config.training)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.training.warmup_steps,
        num_training_steps=config.training.max_steps,
    )
    criterion = build_criterion(config)
    logger.info(f"Fonction de coût : {criterion.__class__.__name__}")

    # --- Trainer ---
    trainer = EmbeddingTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        config=config.training,
        dm=dm,
        grad_accum_steps=config.training.grad_accum_steps,
        checkpoint_dir=Path(args.output),
    )

    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        eval_every=config.logging.eval_every,
    )


if __name__ == "__main__":
    main()
