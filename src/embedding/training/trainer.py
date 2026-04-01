"""
src/embedding/training/trainer.py
====================================
Chapitre 2 et 13 — Boucle d'entraînement générique pour les modèles d'embedding.

La classe EmbeddingTrainer gère :
  - Précision mixte (bfloat16 / float16 / float32)
  - Accumulation de gradients (simulation de grands batches)
  - Gradient clipping (protection contre les explosions)
  - Logging structuré avec loguru
  - Checkpointing automatique sur la meilleure validation loss
  - Bilan par époque : train loss, val loss, Recall@1/5/10, MRR@10, LR
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from loguru import logger
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from embedding.model.config import TrainingConfig
from embedding.training.metrics import mean_reciprocal_rank, recall_at_k
from embedding.utils.device import DeviceManager


@dataclass
class TrainingState:
    """État interne de l'entraînement."""

    step: int = 0
    epoch: int = 0  # Époque courante (0 = mode étapes)
    best_val_loss: float = float("inf")
    train_loss_ema: float = 0.0
    ema_alpha: float = 0.1  # Facteur de lissage exponentiel


class EmbeddingTrainer:
    """
    Boucle d'entraînement générique pour les modèles d'embedding.

    Usage :
        trainer = EmbeddingTrainer(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=MNRLoss(temperature=0.05),
            config=config.training,
            dm=DeviceManager.setup(seed=42),
            grad_accum_steps=4,
        )
        trainer.train(train_loader, val_loader)
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: AdamW,
        scheduler: LambdaLR,
        criterion: nn.Module,
        config: TrainingConfig,
        dm: DeviceManager,
        grad_accum_steps: int = 1,
        checkpoint_dir: Optional[Path] = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.config = config
        self.dm = dm
        self.grad_accum_steps = grad_accum_steps
        self.checkpoint_dir = checkpoint_dir or Path("checkpoints")

        # GradScaler : uniquement nécessaire pour fp16 (pas pour bf16)
        # bf16 a la même plage dynamique que fp32 -> pas de débordement
        self.scaler = torch.amp.GradScaler(
            enabled=(config.mixed_precision == "fp16")
        )
        self.state = TrainingState()

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        eval_every: int = 500,
    ) -> None:
        """
        Lance la boucle d'entraînement principale.

        Deux modes selon config.num_epochs :
          - num_epochs > 0 : entraînement orienté époques — num_epochs passes
            complètes sur le dataset ; validation automatique en fin d'époque.
          - num_epochs == 0 : entraînement orienté étapes — s'arrête après
            config.max_steps mises à jour (comportement historique).

        Args:
            train_loader: DataLoader d'entraînement.
            val_loader:   DataLoader de validation (optionnel).
            eval_every:   Fréquence d'évaluation intermédiaire (en étapes).
        """
        self.model.train()
        self.optimizer.zero_grad()

        num_epochs = self.config.num_epochs

        if num_epochs > 0:
            # ------------------------------------------------------------------
            # Mode époques : num_epochs passes complètes sur le dataset
            # ------------------------------------------------------------------
            for epoch in range(1, num_epochs + 1):
                self.state.epoch = epoch
                logger.info(f"--- Époque {epoch}/{num_epochs} ---")
                epoch_loss_sum = 0.0
                epoch_steps = 0

                for batch in train_loader:
                    loss = self._train_step(batch)
                    epoch_loss_sum += loss
                    epoch_steps += 1

                    if self.state.step % 100 == 0:
                        logger.info(
                            f"Époque {epoch}/{num_epochs} | "
                            f"Étape {self.state.step:>6d} | "
                            f"loss={loss:.4f} (ema={self.state.train_loss_ema:.4f}) | "
                            f"lr={self.scheduler.get_last_lr()[0]:.2e}"
                        )

                    if (
                        val_loader
                        and self.state.step % eval_every == 0
                        and self.state.step > 0
                    ):
                        val_loss = self.evaluate(val_loader)
                        self._maybe_save_checkpoint(val_loss)
                        self.model.train()

                # Bilan de fin d'époque
                train_loss_avg = epoch_loss_sum / max(epoch_steps, 1)
                if val_loader:
                    metrics = self.evaluate_with_metrics(val_loader)
                    self._maybe_save_checkpoint(metrics["val_loss"])
                    self.model.train()
                else:
                    metrics = {}
                self._log_epoch_summary(epoch, num_epochs, train_loss_avg, metrics)
        else:
            # ------------------------------------------------------------------
            # Mode étapes : s'arrête après config.max_steps mises à jour
            # ------------------------------------------------------------------
            for batch in train_loader:
                if self.state.step >= self.config.max_steps:
                    break

                loss = self._train_step(batch)

                if self.state.step % 100 == 0:
                    logger.info(
                        f"Étape {self.state.step:>6d} | "
                        f"loss={loss:.4f} (ema={self.state.train_loss_ema:.4f}) | "
                        f"lr={self.scheduler.get_last_lr()[0]:.2e}"
                    )

                if (
                    val_loader
                    and self.state.step % eval_every == 0
                    and self.state.step > 0
                ):
                    val_loss = self.evaluate(val_loader)
                    self._maybe_save_checkpoint(val_loss)
                    self.model.train()

        logger.success(
            f"Entraînement terminé après {self.state.step} étapes. "
            f"Meilleure val_loss : {self.state.best_val_loss:.4f}"
        )

    def _train_step(self, batch: dict[str, torch.Tensor]) -> float:
        """
        Une étape d'entraînement avec accumulation de gradients.

        L'accumulation simule un grand batch :
          loss_effective = Σ_{i=1}^{K} loss_i / K
        La division par grad_accum_steps est cruciale pour normaliser
        la magnitude du gradient.

        Returns:
            Perte scalaire (avant normalisation par grad_accum_steps).
        """
        batch = {k: v.to(self.dm.device) for k, v in batch.items()}

        # ---- Passe forward avec précision mixte ----
        with self.dm.autocast_context:
            # Support paires (MNR) et triplets selon les clés du batch
            if "query_input_ids" in batch:
                q_emb = self.model(
                    batch["query_input_ids"], batch["query_attention_mask"]
                )
                p_emb = self.model(
                    batch["pos_input_ids"], batch["pos_attention_mask"]
                )
                loss = self.criterion(q_emb, p_emb)
            else:
                # Fallback générique
                emb = self.model(batch["input_ids"], batch["attention_mask"])
                loss = self.criterion(emb, batch)

            loss = loss / self.grad_accum_steps

        # ---- Passe backward ----
        if self.config.mixed_precision == "fp16":
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        # ---- Mise à jour des paramètres ----
        if (self.state.step + 1) % self.grad_accum_steps == 0:
            if self.config.mixed_precision == "fp16":
                self.scaler.unscale_(self.optimizer)

            # Gradient clipping : norme maximale des gradients
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.config.gradient_clip,
            )

            if self.config.mixed_precision == "fp16":
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            self.scheduler.step()
            self.optimizer.zero_grad()

        # ---- Mise à jour de l'état ----
        raw_loss = loss.item() * self.grad_accum_steps
        alpha = self.state.ema_alpha
        self.state.train_loss_ema = (
            alpha * raw_loss + (1 - alpha) * self.state.train_loss_ema
        )
        self.state.step += 1

        return raw_loss

    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader) -> float:
        """
        Calcule la perte de validation sur tout val_loader.

        Returns:
            Perte de validation moyenne.
        """
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        for batch in val_loader:
            batch = {k: v.to(self.dm.device) for k, v in batch.items()}
            with self.dm.autocast_context:
                if "query_input_ids" in batch:
                    q_emb = self.model(
                        batch["query_input_ids"], batch["query_attention_mask"]
                    )
                    p_emb = self.model(
                        batch["pos_input_ids"], batch["pos_attention_mask"]
                    )
                    loss = self.criterion(q_emb, p_emb)
                else:
                    emb = self.model(batch["input_ids"], batch["attention_mask"])
                    loss = self.criterion(emb, batch)
            total_loss += loss.item()
            n_batches += 1

        val_loss = total_loss / max(n_batches, 1)
        logger.info(f"  -> Validation loss : {val_loss:.4f}")
        return val_loss

    @torch.no_grad()
    def evaluate_with_metrics(self, val_loader: DataLoader) -> dict[str, float]:
        """
        Validation complète : loss + métriques de retrieval.

        Pour chaque paire (query, positive) du val_loader, on encode les deux
        côtés et on construit un corpus « positives ». Chaque requête i a
        exactement un document pertinent à l'indice i (diagonal parfait).

        Returns:
            Dict contenant val_loss, Recall@1, Recall@5, Recall@10, MRR@10.
        """
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        all_q_embs: list[torch.Tensor] = []
        all_p_embs: list[torch.Tensor] = []

        for batch in val_loader:
            batch = {k: v.to(self.dm.device) for k, v in batch.items()}
            with self.dm.autocast_context:
                if "query_input_ids" in batch:
                    q_emb = self.model(
                        batch["query_input_ids"], batch["query_attention_mask"]
                    )
                    p_emb = self.model(
                        batch["pos_input_ids"], batch["pos_attention_mask"]
                    )
                    loss = self.criterion(q_emb, p_emb)
                    all_q_embs.append(q_emb.cpu().float())
                    all_p_embs.append(p_emb.cpu().float())
                else:
                    emb = self.model(batch["input_ids"], batch["attention_mask"])
                    loss = self.criterion(emb, batch)
            total_loss += loss.item()
            n_batches += 1

        metrics: dict[str, float] = {"val_loss": total_loss / max(n_batches, 1)}

        if all_q_embs:
            queries = torch.cat(all_q_embs, dim=0)
            corpus = torch.cat(all_p_embs, dim=0)
            # Chaque requête i a son positif exactement à l'indice i
            relevant = [[i] for i in range(len(queries))]
            metrics["recall@1"] = recall_at_k(queries, corpus, relevant, k=1)
            metrics["recall@5"] = recall_at_k(queries, corpus, relevant, k=5)
            metrics["recall@10"] = recall_at_k(queries, corpus, relevant, k=10)
            metrics["mrr@10"] = mean_reciprocal_rank(queries, corpus, relevant, k=10)

        return metrics

    def _log_epoch_summary(
        self,
        epoch: int,
        num_epochs: int,
        train_loss_avg: float,
        metrics: dict[str, float],
    ) -> None:
        """Affiche un bilan structuré en fin d'époque."""
        lr = self.scheduler.get_last_lr()[0]
        sep = "─" * 50
        logger.info(f"┌{sep}")
        logger.info(f"│  Bilan — époque {epoch}/{num_epochs}")
        logger.info(f"│  {'train_loss':<12}: {train_loss_avg:.4f}")
        if "val_loss" in metrics:
            logger.info(f"│  {'val_loss':<12}: {metrics['val_loss']:.4f}")
        if "recall@1" in metrics:
            logger.info(f"│  {'Recall@1':<12}: {metrics['recall@1']:.4f}")
            logger.info(f"│  {'Recall@5':<12}: {metrics['recall@5']:.4f}")
            logger.info(f"│  {'Recall@10':<12}: {metrics['recall@10']:.4f}")
            logger.info(f"│  {'MRR@10':<12}: {metrics['mrr@10']:.4f}")
        logger.info(f"│  {'lr':<12}: {lr:.2e}")
        logger.info(f"└{sep}")

    def _maybe_save_checkpoint(self, val_loss: float) -> None:
        """Sauvegarde le modèle si val_loss est le meilleur observé."""
        if val_loss < self.state.best_val_loss:
            self.state.best_val_loss = val_loss
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            if self.state.epoch > 0:
                name = f"best_model_epoch{self.state.epoch}_step{self.state.step}.pt"
            else:
                name = f"best_model_step{self.state.step}.pt"
            path = self.checkpoint_dir / name
            torch.save(
                {
                    "step": self.state.step,
                    "epoch": self.state.epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "val_loss": val_loss,
                },
                path,
            )
            logger.success(
                f"  -> Nouveau meilleur modèle sauvegardé : {path.name} "
                f"(val_loss={val_loss:.4f})"
            )
