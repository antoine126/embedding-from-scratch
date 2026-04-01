"""
scripts/pretrain.py
====================
Chapitre 12 — Pré-entraînement MLM (Masked Language Modeling).

Ce script lance le pré-entraînement d'un modèle d'embedding
via la tâche MLM sur un corpus de texte brut.

Usage :
    uv run python scripts/pretrain.py --config configs/embedding_base.toml \\
        --corpus data/corpus.txt --output checkpoints/pretrain/

Le pré-entraînement MLM apprend des représentations générales du langage
avant le fine-tuning contrastif (scripts/finetune.py).
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from loguru import logger
from torch.utils.data import DataLoader

from embedding.data.collators import MLMDataCollator
from embedding.data.dataset import MLMDataset
from embedding.data.tokenizer import train_bpe_tokenizer
from embedding.model.config import ExperimentConfig
from embedding.model.encoder import EmbeddingModel
from embedding.training.optimizer import get_cosine_schedule_with_warmup, get_optimizer
from embedding.utils.device import DeviceManager


def build_mlm_head(d_model: int, vocab_size: int) -> torch.nn.Module:
    """Tête de prédiction MLM : projection vers le vocabulaire."""
    return torch.nn.Sequential(
        torch.nn.Linear(d_model, d_model),
        torch.nn.GELU(),
        torch.nn.LayerNorm(d_model),
        torch.nn.Linear(d_model, vocab_size),
    )


def train_mlm(
    config: ExperimentConfig,
    corpus_path: str,
    tokenizer_path: str | None,
    output_dir: Path,
) -> None:
    """Lance le pré-entraînement MLM."""
    logger.info(f"Devices disponibles : {DeviceManager.available_devices()}")
    dm = DeviceManager.setup(
        seed=42,
        mixed_precision=config.training.mixed_precision,
        device=config.training.device,
    )
    logger.info(f"Device : {dm}")

    # --- Tokenizer ---
    if tokenizer_path and Path(tokenizer_path).exists():
        from embedding.data.tokenizer import load_tokenizer

        tokenizer = load_tokenizer(tokenizer_path)
        logger.info(f"Tokenizer chargé depuis {tokenizer_path}")
    else:
        logger.info("Entraînement du tokenizer BPE...")
        tokenizer = train_bpe_tokenizer(
            corpus_files=[corpus_path],
            vocab_size=config.model.vocab_size,
            save_path=output_dir / "tokenizer",
        )

    # --- Dataset et DataLoader ---
    dataset = MLMDataset(
        path=corpus_path,
        tokenizer=tokenizer,
        max_seq_len=config.model.max_seq_len,
    )
    collator = MLMDataCollator(
        mask_token_id=tokenizer.token_to_id("[MASK]"),
        vocab_size=config.model.vocab_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        collate_fn=collator,
        num_workers=config.data.num_workers,
    )

    # --- Modèle + tête MLM ---
    model = EmbeddingModel(config.model).to(dm.device)
    mlm_head = build_mlm_head(config.model.d_model, config.model.vocab_size)
    mlm_head = mlm_head.to(dm.device)

    # --- Optimiseur et scheduler ---
    optimizer = get_optimizer(model, config.training)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.training.warmup_steps,
        num_training_steps=config.training.max_steps,
    )

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)

    # --- Boucle d'entraînement MLM ---
    model.train()
    mlm_head.train()
    optimizer.zero_grad()
    step = 0

    num_epochs = config.training.num_epochs

    def _mlm_step(batch: dict) -> float:
        """Une étape MLM avec accumulation de gradients. Retourne la loss brute."""
        batch = {k: v.to(dm.device) for k, v in batch.items()}

        with dm.autocast_context:
            # On passe par les blocs transformer sans la couche de pooling
            x = model.token_embedding(batch["input_ids"])
            x = model.position_encoding(x)
            x = model.embedding_dropout(x)
            for block in model.blocks:
                x = block(x, batch["attention_mask"])
            x = model.post_norm(x)  # (B, L, d)

            logits = mlm_head(x)  # (B, L, vocab_size)
            loss = loss_fn(
                logits.view(-1, config.model.vocab_size),
                batch["labels"].view(-1),
            )
            loss = loss / config.training.grad_accum_steps

        loss.backward()

        if (step + 1) % config.training.grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(mlm_head.parameters()),
                max_norm=config.training.gradient_clip,
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        return loss.item() * config.training.grad_accum_steps

    if num_epochs > 0:
        # ------------------------------------------------------------------
        # Mode époques : num_epochs passes complètes sur le corpus
        # ------------------------------------------------------------------
        for epoch in range(1, num_epochs + 1):
            logger.info(f"--- Époque {epoch}/{num_epochs} ---")
            epoch_loss_sum = 0.0
            epoch_steps = 0
            for batch in loader:
                raw_loss = _mlm_step(batch)
                epoch_loss_sum += raw_loss
                epoch_steps += 1
                if step % config.logging.log_every == 0:
                    logger.info(
                        f"Époque {epoch}/{num_epochs} | "
                        f"Étape {step:>6d} | loss={raw_loss:.4f} | "
                        f"lr={scheduler.get_last_lr()[0]:.2e}"
                    )
                step += 1

            # Bilan de fin d'époque
            train_loss_avg = epoch_loss_sum / max(epoch_steps, 1)
            perplexity = math.exp(min(train_loss_avg, 20.0))  # cap pour éviter overflow
            sep = "─" * 50
            logger.info(f"┌{sep}")
            logger.info(f"│  Bilan — époque {epoch}/{num_epochs}")
            logger.info(f"│  {'train_loss':<12}: {train_loss_avg:.4f}")
            logger.info(f"│  {'perplexité':<12}: {perplexity:.2f}")
            logger.info(f"│  {'lr':<12}: {scheduler.get_last_lr()[0]:.2e}")
            logger.info(f"└{sep}")
    else:
        # ------------------------------------------------------------------
        # Mode étapes : s'arrête après config.training.max_steps mises à jour
        # ------------------------------------------------------------------
        for batch in loader:
            if step >= config.training.max_steps:
                break
            raw_loss = _mlm_step(batch)
            if step % config.logging.log_every == 0:
                logger.info(
                    f"Étape {step:>6d} | loss={raw_loss:.4f} | "
                    f"lr={scheduler.get_last_lr()[0]:.2e}"
                )
            step += 1

    # Sauvegarde
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "pretrained_model.pt")
    logger.success(f"Modèle pré-entraîné sauvegardé dans {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pré-entraînement MLM")
    parser.add_argument("--config", required=True, help="Chemin vers le fichier TOML")
    parser.add_argument("--corpus", required=True, help="Chemin vers le corpus JSONL")
    parser.add_argument("--tokenizer", default=None, help="Chemin vers le tokenizer")
    parser.add_argument(
        "--output", default="checkpoints/pretrain", help="Répertoire de sortie"
    )
    args = parser.parse_args()

    config = ExperimentConfig.from_toml(args.config)
    train_mlm(config, args.corpus, args.tokenizer, Path(args.output))


if __name__ == "__main__":
    main()
