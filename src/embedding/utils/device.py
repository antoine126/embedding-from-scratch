"""
src/embedding/utils/device.py
==============================
Chapitre 1 — Gestion centralisée du device et de la reproductibilité.

La classe DeviceManager centralise :
  - La sélection du device (auto-détection ou choix explicite)
  - La fixation des seeds aléatoires pour la reproductibilité
  - La configuration du déterminisme CUDA
  - Le contexte de précision mixte (bfloat16 / float16 / float32)

Utilisation :
    dm = DeviceManager.setup(seed=42, mixed_precision="bf16", device="auto")
    dm = DeviceManager.setup(seed=42, mixed_precision="bf16", device="cuda:1")
    model = model.to(dm.device)
    with dm.autocast_context:
        output = model(batch)

Valeurs acceptées pour ``device`` :
    "auto"    → sélection automatique CUDA > MPS > CPU
    "cpu"     → force le CPU
    "cuda"    → premier GPU disponible (équivalent à "cuda:0")
    "cuda:N"  → GPU d'indice N (ex. "cuda:1" pour le second GPU)
    "mps"     → Apple Silicon (MPS)
"""
from __future__ import annotations

import os
import random
import warnings
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch

DeviceType = Literal["cuda", "mps", "cpu"]


@dataclass
class DeviceManager:
    """
    Gestion centralisée du device et de la reproductibilité.

    Ne pas instancier directement — utiliser DeviceManager.setup().
    """

    device: torch.device
    seed: int
    mixed_precision: Literal["bf16", "fp16", "fp32"]

    @classmethod
    def setup(
        cls,
        seed: int = 42,
        mixed_precision: Literal["bf16", "fp16", "fp32"] = "bf16",
        deterministic: bool = True,
        device: str = "auto",
    ) -> DeviceManager:
        """
        Configure le device, fixe les seeds et initialise le déterminisme.

        Args:
            seed:            Graine aléatoire pour la reproductibilité.
            mixed_precision: Format de précision mixte ("bf16", "fp16", "fp32").
            deterministic:   Active le déterminisme CUDA (légèrement plus lent).
            device:          Device cible. "auto" sélectionne CUDA > MPS > CPU.
                             Exemples : "cpu", "cuda", "cuda:0", "cuda:1", "mps".

        Returns:
            Instance configurée de DeviceManager.

        Raises:
            RuntimeError: Si le device demandé n'est pas disponible.
            ValueError:   Si l'indice GPU est hors des limites.
        """
        resolved_device = cls._resolve_device(device)
        cls._set_seeds(seed)

        # Déterminisme CUDA (Ampere+ recommandé)
        if deterministic and resolved_device.type == "cuda":
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # warn_only=True : avertissement plutôt qu'exception pour les
            # opérations sans implémentation déterministe
            torch.use_deterministic_algorithms(True, warn_only=True)

        # TF32 sur Ampere+ : légèrement moins précis, beaucoup plus rapide
        if resolved_device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # Validation de la précision mixte selon le device
        if mixed_precision == "bf16" and resolved_device.type == "cpu":
            warnings.warn(
                "bfloat16 non supporté sur CPU, repli sur fp32.",
                stacklevel=2,
            )
            mixed_precision = "fp32"
        if mixed_precision == "fp16" and resolved_device.type == "mps":
            warnings.warn(
                "float16 instable sur MPS (Apple Silicon), repli sur bf16.",
                stacklevel=2,
            )
            mixed_precision = "bf16"

        return cls(device=resolved_device, seed=seed, mixed_precision=mixed_precision)

    @staticmethod
    def _detect_device() -> torch.device:
        """Sélectionne automatiquement CUDA > MPS > CPU."""
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        """
        Résout et valide une chaîne de device.

        Args:
            device: "auto", "cpu", "cuda", "cuda:N" ou "mps".

        Returns:
            torch.device validé et prêt à l'emploi.

        Raises:
            RuntimeError: Device demandé non disponible sur cette machine.
            ValueError:   Indice GPU hors limites ou type inconnu.
        """
        if device == "auto":
            return DeviceManager._detect_device()

        resolved = torch.device(device)

        if resolved.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "Device 'cuda' demandé mais CUDA n'est pas disponible sur cette machine. "
                    f"Devices disponibles : {DeviceManager.available_devices()}"
                )
            index = resolved.index if resolved.index is not None else 0
            n_gpus = torch.cuda.device_count()
            if index >= n_gpus:
                raise ValueError(
                    f"GPU cuda:{index} demandé mais seulement {n_gpus} GPU(s) "
                    f"disponible(s) (indices valides : 0 à {n_gpus - 1})."
                )
            return torch.device(f"cuda:{index}")

        if resolved.type == "mps":
            if not torch.backends.mps.is_available():
                raise RuntimeError(
                    "Device 'mps' demandé mais MPS n'est pas disponible sur cette machine. "
                    f"Devices disponibles : {DeviceManager.available_devices()}"
                )
            return resolved

        if resolved.type == "cpu":
            return resolved

        raise ValueError(
            f"Type de device inconnu : {resolved.type!r}. "
            "Valeurs acceptées : 'auto', 'cpu', 'cuda', 'cuda:N', 'mps'."
        )

    @staticmethod
    def available_devices() -> list[str]:
        """
        Retourne la liste des devices disponibles sur cette machine.

        Returns:
            Ex. ["cpu", "cuda:0", "cuda:1"] ou ["cpu", "mps"].
        """
        devices = ["cpu"]
        if torch.cuda.is_available():
            devices += [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        if torch.backends.mps.is_available():
            devices.append("mps")
        return devices

    @staticmethod
    def _set_seeds(seed: int) -> None:
        """Fixe toutes les sources d'aléatoire."""
        random.seed(seed)
        np.random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def to(self, tensor: torch.Tensor) -> torch.Tensor:
        """Déplace un tenseur vers le device géré."""
        return tensor.to(self.device)

    @property
    def dtype(self) -> torch.dtype:
        """Retourne le dtype correspondant à la précision mixte configurée."""
        return {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }[self.mixed_precision]

    @property
    def autocast_context(self):
        """Retourne le contexte autocast pour la précision mixte."""
        if self.mixed_precision == "fp32":
            return torch.autocast(device_type=self.device.type, enabled=False)
        return torch.autocast(device_type=self.device.type, dtype=self.dtype)

    def __repr__(self) -> str:
        return (
            f"DeviceManager("
            f"device={self.device}, "
            f"seed={self.seed}, "
            f"precision={self.mixed_precision})"
        )
