"""
src/embedding/mining/hard_negative.py
========================================
Chapitre 14 — Mining de négatifs difficiles avec FAISS.

Le hard negative mining améliore la qualité des triplets d'entraînement.
Au lieu de sélectionner des négatifs aléatoires, on sélectionne les
exemples les plus difficiles : ceux qui sont proches de l'ancre dans
l'espace d'embedding mais qui ne sont pas pertinents.

Pipeline :
  1. Encoder tous les passages du corpus avec le modèle courant
  2. Indexer les embeddings avec FAISS (recherche ANN efficace)
  3. Pour chaque requête, retrouver les K plus proches voisins
  4. Filtrer les vrais positifs (faux négatifs) pour ne garder que
     les négatifs difficiles authentiques
  5. Générer les triplets (query, positive, hard_negative)

Ce processus est itératif : refaire le mining périodiquement avec
le modèle mis à jour (curriculum learning).
"""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


@dataclass
class HardNegativeMiner:
    """
    Mining de négatifs difficiles via index FAISS.

    Args:
        model:          Modèle d'embedding (EmbeddingModel ou compatible).
        tokenizer:      Tokenizer correspondant.
        k_candidates:   Nombre de candidats négatifs à récupérer par requête.
        similarity_threshold: Seuil cosinus au-delà duquel on considère un
                              voisin comme "trop proche" (potentiel faux négatif).
        device:         Device pour l'encodage.

    Exemple :
        miner = HardNegativeMiner(model, tokenizer, k_candidates=50)
        miner.build_index(corpus_texts)
        triplets = list(miner.mine(queries, positives_per_query))
    """

    model: nn.Module
    tokenizer: object
    k_candidates: int = 50
    similarity_threshold: float = 0.9
    device: str = "cpu"

    def __post_init__(self) -> None:
        self._index = None
        self._corpus_texts: list[str] = []
        self._corpus_embeddings: np.ndarray | None = None

    def build_index(
        self,
        corpus: list[str],
        batch_size: int = 256,
    ) -> None:
        """
        Encode le corpus et construit l'index FAISS.

        Utilise IndexFlatIP (produit scalaire) sur des embeddings
        normalisés L2 — équivalent à la similarité cosinus.

        Args:
            corpus:     Liste de textes à indexer.
            batch_size: Taille de batch pour l'encodage.
        """
        try:
            import faiss
        except ImportError as e:
            raise ImportError(
                "faiss est requis pour le negative mining. "
                "Installez-le avec : uv add faiss-cpu"
            ) from e

        self._corpus_texts = corpus
        embeddings = self._encode_corpus(corpus, batch_size)
        self._corpus_embeddings = embeddings

        # IndexFlatIP : recherche exacte par produit scalaire
        # (équivalent cosinus pour des embeddings normalisés L2)
        d = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(d)
        self._index.add(embeddings)  # type: ignore

    def mine(
        self,
        queries: list[str],
        positives: list[str],
        positive_indices: list[list[int]] | None = None,
    ) -> Iterator[dict[str, str]]:
        """
        Mine les négatifs difficiles pour chaque paire (query, positive).

        Args:
            queries:          Liste de requêtes.
            positives:        Positif correspondant à chaque requête.
            positive_indices: Indices dans le corpus des vrais positifs
                              (pour filtrer les faux négatifs).
                              Si None, on filtre par similarité seulement.

        Yields:
            Dicts {"query": str, "positive": str, "negative": str}
        """
        if self._index is None:
            raise RuntimeError("Appeler build_index() avant mine().")

        query_embeddings = self._encode_corpus(queries, batch_size=256)

        # Recherche des K plus proches voisins pour toutes les requêtes
        scores, indices = self._index.search(  # type: ignore
            query_embeddings, self.k_candidates
        )

        for i, (query, positive) in enumerate(zip(queries, positives)):
            true_positive_set = set(positive_indices[i]) if positive_indices else set()

            for rank, (score, corpus_idx) in enumerate(
                zip(scores[i], indices[i])
            ):
                # Exclure les vrais positifs (faux négatifs)
                if corpus_idx in true_positive_set:
                    continue

                # Exclure les passages trop similaires (potentiels faux négatifs)
                if score >= self.similarity_threshold:
                    continue

                # Candidat valide : c'est un négatif difficile
                hard_negative = self._corpus_texts[corpus_idx]
                yield {
                    "query": query,
                    "positive": positive,
                    "negative": hard_negative,
                }
                break  # Un négatif par requête (le plus difficile valide)

    def _encode_corpus(
        self, texts: list[str], batch_size: int = 256
    ) -> np.ndarray:
        """Encode une liste de textes en numpy array float32."""
        self.model.eval()
        all_embeddings: list[torch.Tensor] = []

        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]

                if hasattr(self.tokenizer, "encode_batch"):
                    # tokenizers.Tokenizer
                    enc = self.tokenizer.encode_batch(batch)
                    input_ids = torch.tensor(
                        [e.ids for e in enc], dtype=torch.long, device=self.device
                    )
                    mask = torch.tensor(
                        [e.attention_mask for e in enc],
                        dtype=torch.long,
                        device=self.device,
                    )
                else:
                    # transformers.AutoTokenizer
                    enc = self.tokenizer(
                        batch,
                        padding=True,
                        truncation=True,
                        return_tensors="pt",
                    )
                    input_ids = enc["input_ids"].to(self.device)
                    mask = enc["attention_mask"].to(self.device)

                emb = self.model(input_ids, mask)
                all_embeddings.append(emb.cpu())

        embeddings = torch.cat(all_embeddings, dim=0).numpy().astype(np.float32)
        return embeddings

    def save_triplets(
        self,
        queries: list[str],
        positives: list[str],
        output_path: str | Path,
        positive_indices: list[list[int]] | None = None,
    ) -> int:
        """
        Mine et sauvegarde les triplets dans un fichier JSONL.

        Returns:
            Nombre de triplets générés.
        """
        import json

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        n_triplets = 0
        with open(output_path, "w", encoding="utf-8") as f:
            for triplet in self.mine(queries, positives, positive_indices):
                f.write(json.dumps(triplet, ensure_ascii=False) + "\n")
                n_triplets += 1

        return n_triplets
