"""
src/embedding/rag/chunker.py
================================
Chapitre 17 — Stratégies de découpage de documents (chunking).

Le chunking transforme un document long en passages courts compatibles
avec la fenêtre de contexte du modèle d'embedding (généralement 512 tokens).

Stratégies :
  - NaiveChunker     : découpe par nombre de mots fixe
  - SentenceChunker  : découpe au niveau des phrases (avec overlap)
  - HierarchicalChunker : chunks avec contexte parent (titre + contenu)

Le choix de la stratégie impacte significativement les performances RAG :
  - Chunks trop courts : perte du contexte sémantique
  - Chunks trop longs : dilution du signal pertinent (lost-in-the-middle)
  - Overlap : améliore le rappel mais augmente la redondance
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class Chunk:
    """Représente un passage issu du découpage d'un document."""

    text: str
    doc_id: str
    chunk_id: int
    metadata: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.text.split())


@dataclass
class NaiveChunker:
    """
    Découpage naïf par nombre de mots avec overlap optionnel.

    Simple et rapide, mais ignore la structure du document
    (peut couper au milieu d'une phrase ou d'un paragraphe).

    Args:
        chunk_size: Nombre de mots par chunk.
        overlap:    Nombre de mots partagés entre chunks consécutifs.
    """

    chunk_size: int = 256
    overlap: int = 32

    def chunk(self, text: str, doc_id: str = "doc") -> list[Chunk]:
        """
        Découpe un texte en chunks de taille fixe.

        Args:
            text:   Texte à découper.
            doc_id: Identifiant du document source.

        Returns:
            Liste de Chunks.
        """
        words = text.split()
        chunks: list[Chunk] = []
        stride = self.chunk_size - self.overlap

        for i, start in enumerate(range(0, len(words), stride)):
            chunk_words = words[start : start + self.chunk_size]
            if not chunk_words:
                break
            chunks.append(
                Chunk(
                    text=" ".join(chunk_words),
                    doc_id=doc_id,
                    chunk_id=i,
                    metadata={"start_word": start},
                )
            )

        return chunks


@dataclass
class SentenceChunker:
    """
    Découpage au niveau des phrases avec accumulation et overlap.

    Respecte les frontières de phrases (délimitées par . ! ?),
    ce qui préserve la cohérence sémantique de chaque chunk.

    Args:
        max_words:  Nombre de mots maximum par chunk.
        overlap:    Nombre de phrases de recouvrement entre chunks.
    """

    max_words: int = 256
    overlap: int = 1

    def _split_sentences(self, text: str) -> list[str]:
        """Découpe un texte en phrases."""
        # Séparateurs : . ! ? suivi d'un espace et d'une majuscule
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-ZÀ-Ú])", text.strip())
        return [s.strip() for s in sentences if s.strip()]

    def chunk(self, text: str, doc_id: str = "doc") -> list[Chunk]:
        """
        Découpe un texte en chunks alignés sur les phrases.

        Args:
            text:   Texte à découper.
            doc_id: Identifiant du document source.

        Returns:
            Liste de Chunks.
        """
        sentences = self._split_sentences(text)
        chunks: list[Chunk] = []
        current_sentences: list[str] = []
        current_words = 0
        chunk_id = 0

        for sentence in sentences:
            n_words = len(sentence.split())

            if current_words + n_words > self.max_words and current_sentences:
                # Sauvegarder le chunk courant
                chunks.append(
                    Chunk(
                        text=" ".join(current_sentences),
                        doc_id=doc_id,
                        chunk_id=chunk_id,
                        metadata={"n_sentences": len(current_sentences)},
                    )
                )
                chunk_id += 1

                # Garder les dernières `overlap` phrases pour la continuité
                current_sentences = current_sentences[-self.overlap :]
                current_words = sum(len(s.split()) for s in current_sentences)

            current_sentences.append(sentence)
            current_words += n_words

        # Dernier chunk
        if current_sentences:
            chunks.append(
                Chunk(
                    text=" ".join(current_sentences),
                    doc_id=doc_id,
                    chunk_id=chunk_id,
                    metadata={"n_sentences": len(current_sentences)},
                )
            )

        return chunks


@dataclass
class HierarchicalChunker:
    """
    Découpage hiérarchique avec contexte parent.

    Chaque chunk contient :
      - Un extrait de contenu (chunk fils)
      - Le titre/section parente pour le contexte

    Cela améliore la récupération en préservant le contexte structurel
    (utile pour les documents techniques, les articles, les rapports).

    Args:
        child_chunk_words: Taille du chunk fils en mots.
        parent_chunk_words: Taille du chunk parent en mots (contexte).
    """

    child_chunk_words: int = 128
    parent_chunk_words: int = 512

    def chunk(
        self,
        text: str,
        doc_id: str = "doc",
        title: str = "",
    ) -> list[Chunk]:
        """
        Découpe un texte avec contexte hiérarchique.

        Args:
            text:   Texte à découper.
            doc_id: Identifiant du document source.
            title:  Titre ou section parente pour le contexte.

        Returns:
            Liste de Chunks avec contexte parent dans les métadonnées.
        """
        naiver = NaiveChunker(
            chunk_size=self.child_chunk_words, overlap=self.child_chunk_words // 8
        )
        child_chunks = naiver.chunk(text, doc_id)

        # Ajouter le contexte parent à chaque chunk fils
        context_prefix = f"[Document : {title}]\n\n" if title else ""
        enriched_chunks = []

        for chunk in child_chunks:
            enriched_text = context_prefix + chunk.text
            enriched_chunks.append(
                Chunk(
                    text=enriched_text,
                    doc_id=doc_id,
                    chunk_id=chunk.chunk_id,
                    metadata={
                        **chunk.metadata,
                        "title": title,
                        "has_context": bool(title),
                    },
                )
            )

        return enriched_chunks
