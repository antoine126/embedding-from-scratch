"""
src/embedding/data/tokenizer.py
=================================
Chapitre 4 — Entraînement et chargement d'un tokenizer BPE.

Fonctions :
  - train_bpe_tokenizer : entraîne un tokenizer BPE sur un corpus
  - load_tokenizer      : charge un tokenizer sauvegardé

Tokens spéciaux utilisés dans tout le livre :
  [UNK]=0, [CLS]=1, [SEP]=2, [PAD]=3, [MASK]=4

Le post-processor ajoute automatiquement [CLS] et [SEP] autour
de chaque séquence (et de chaque paire pour les tâches bilingues).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.normalizers import NFD, Lowercase, Sequence, StripAccents
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer


def train_bpe_tokenizer(
    corpus_files: list[str],
    vocab_size: int = 32_000,
    save_path: Optional[Path] = None,
) -> Tokenizer:
    """
    Entraîne un tokenizer BPE sur les fichiers corpus fournis.

    Args:
        corpus_files: Chemins vers les fichiers texte (un document par ligne).
        vocab_size:   Taille cible du vocabulaire (incluant les tokens spéciaux).
        save_path:    Répertoire de sauvegarde (sauvegarde si fourni).

    Returns:
        Tokenizer entraîné et prêt à l'emploi.

    Exemple :
        tok = train_bpe_tokenizer(["data/corpus.txt"], vocab_size=32_000)
        enc = tok.encode("Bonjour le monde")
        print(enc.tokens)  # ['[CLS]', 'bonjour', 'le', 'monde', '[SEP]']
    """
    # --- Normalisation de l'entrée ---
    if isinstance(corpus_files, (str, Path)):
        files = [str(corpus_files)]
    else:
        files = [str(f) for f in corpus_files]

    # --- Vérification de l'existence des fichiers ---
    for f in files:
        if not Path(f).exists():
            raise FileNotFoundError(f"Le fichier corpus est introuvable : {f}")
    
    # --- Construction du tokenizer de base ---
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

    # Normalisation : décomposition unicode + minuscules + suppression accents
    tokenizer.normalizer = Sequence([NFD(), Lowercase(), StripAccents()])

    # Pré-tokenisation : split sur les espaces
    tokenizer.pre_tokenizer = Whitespace()

    # Tokens spéciaux
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
        min_frequency=2,  # Ignorer les sous-mots apparus moins de 2 fois
        show_progress=True,
    )

    # --- Entraînement ---
    tokenizer.train(files=corpus_files, trainer=trainer)

    # --- Post-processor : ajouter [CLS] et [SEP] automatiquement ---
    cls_id = tokenizer.token_to_id("[CLS]")
    sep_id = tokenizer.token_to_id("[SEP]")
    tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", cls_id), ("[SEP]", sep_id)],
    )

    # --- Sauvegarde ---
    if save_path is not None:
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        tokenizer.save(str(save_path / "tokenizer.json"))
        print(f"Tokenizer sauvegardé dans {save_path}/tokenizer.json")

    return tokenizer


def load_tokenizer(path: str | Path) -> Tokenizer:
    """
    Charge un tokenizer sauvegardé depuis un fichier JSON.

    Args:
        path: Chemin vers le fichier tokenizer.json.

    Returns:
        Tokenizer chargé.
    """
    return Tokenizer.from_file(str(path))
