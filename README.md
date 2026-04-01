# embedding-from-scratch

Code source du livre **"Entraîner un modèle d'embedding — De l'architecture Transformer à la recherche hybride en production avec Python, UV et PyTorch"**.

Ce dépôt contient l'implémentation complète de tous les exemples présentés dans le livre. Chaque fichier correspond à un ou plusieurs chapitres et peut être exécuté directement, sans avoir à recopier le code depuis le PDF.

---

## Philosophie

Le livre suit une progression en chaîne : chaque concept s'appuie sur le précédent, de l'architecture Transformer jusqu'au pipeline RAG de production. Le code de ce dépôt suit la même logique.

Il n'est pas conçu comme une bibliothèque à importer clés en main, mais comme une **base de code pédagogique** que vous pouvez lire, modifier et réutiliser pour comprendre chaque décision de l'intérieur. Chaque module est documenté avec la référence au chapitre correspondant et l'intention derrière les choix d'implémentation.

---

## Prérequis

- Python 3.12+
- [UV](https://docs.astral.sh/uv/) — gestionnaire de paquets moderne (remplace pip + venv + poetry)

```bash
# Installation UV sur Linux / macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# Installation UV sur Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

---

## Installation

```bash
git clone https://github.com/antoine126/embedding-from-scratch
cd embedding-from-scratch

# Installe toutes les dépendances avec les versions exactes du fichier uv.lock
uv sync
```

`uv sync` crée automatiquement l'environnement virtuel `.venv/` et installe exactement les mêmes versions que celles utilisées lors de la rédaction du livre. Aucune activation manuelle du virtualenv n'est nécessaire — toutes les commandes s'exécutent via `uv run`.

> **GPU NVIDIA** : le `pyproject.toml` pointe par défaut sur l'index PyTorch CUDA 12.8. Sur CPU ou Apple Silicon, remplacez l'index `pytorch-cu128` par `pytorch-cpu` dans `pyproject.toml` puis relancez `uv sync`.

---

## Structure du projet

```
embedding-from-scratch/
├── pyproject.toml              # Dépendances et configuration (Ch. 1)
├── uv.lock                     # Versions exactes verrouillées — à commiter
├── configs/
│   ├── embedding_base.toml     # Modèle base : 768d, 12 têtes, 6 couches (Ch. 1)
│   └── embedding_large.toml    # Modèle large : 1024d, SwiGLU, RoPE (Ch. 1)
├── data/
│   ├── corpus.txt              # Corpus de démonstration pour le pré-entraînement
│   ├── pairs_train.jsonl       # Paires (query, positive) pour le fine-tuning
│   └── pairs_val.jsonl         # Paires de validation
├── scripts/
│   ├── pretrain.py             # Pré-entraînement MLM (Ch. 12)
│   ├── finetune.py             # Fine-tuning contrastif (Ch. 13)
│   └── demo_rag.py             # Démonstration RAG bout en bout (Ch. 16–20)
├── src/embedding/
│   ├── model/                  # Architecture Transformer encodeur (Ch. 3–8)
│   │   ├── config.py           #   Dataclasses de configuration
│   │   ├── embeddings.py       #   TokenEmbedding + PE sinusoïdal / appris / RoPE
│   │   ├── attention.py        #   MultiHeadAttention + Flash Attention
│   │   ├── layers.py           #   FeedForward (GELU/SwiGLU/GeGLU), TransformerBlock, Pooling
│   │   └── encoder.py          #   EmbeddingModel — assemblage complet
│   ├── data/                   # Tokenisation et datasets (Ch. 4, 12–13)
│   │   ├── tokenizer.py        #   Entraînement BPE avec HuggingFace tokenizers
│   │   ├── dataset.py          #   PairDataset, TripletDataset, MLMDataset
│   │   └── collators.py        #   PairCollator, TripletCollator, MLMDataCollator
│   ├── losses/                 # Fonctions de coût contrastives (Ch. 9)
│   │   ├── contrastive.py      #   Contrastive Loss (Hadsell 2006)
│   │   ├── triplet.py          #   Triplet Loss en espace cosinus
│   │   ├── mnr.py              #   MNR Loss + InfoNCE symétrique
│   │   └── matryoshka.py       #   Matryoshka Representation Learning
│   ├── training/               # Entraînement et évaluation (Ch. 2, 10–11)
│   │   ├── trainer.py          #   EmbeddingTrainer : bf16, accumulation, clipping
│   │   ├── optimizer.py        #   AdamW + cosine scheduler avec warmup
│   │   └── metrics.py          #   Recall@K, MRR, MAP, gradient norms
│   ├── mining/                 # Negative mining (Ch. 14)
│   │   └── hard_negative.py    #   HardNegativeMiner avec index FAISS
│   ├── rag/                    # Pipeline RAG (Ch. 16–20)
│   │   ├── chunker.py          #   Naive / Sentence / Hierarchical chunking
│   │   ├── retriever.py        #   DenseRetriever (FAISS) + BM25Retriever
│   │   ├── reranker.py         #   CrossEncoderReranker
│   │   └── pipeline.py         #   HybridRetriever (RRF) + RAGPipeline
│   └── utils/
│       └── device.py           # DeviceManager : device, seeds, autocast (Ch. 1–2)
└── tests/
    ├── test_model.py
    ├── test_losses.py
    └── test_chunker.py
```

---

## Correspondance chapitres / fichiers

| Chapitre | Sujet | Fichier(s) |
|---|---|---|
| Ch. 1 | UV, configuration, DeviceManager | `model/config.py`, `utils/device.py`, `configs/` |
| Ch. 2 | PyTorch, autograd, boucle d'entraînement | `training/trainer.py` |
| Ch. 4 | Tokenisation BPE, collators | `data/tokenizer.py`, `data/collators.py` |
| Ch. 5 | Token embedding, encodage positionnel | `model/embeddings.py` |
| Ch. 6 | Attention multi-têtes | `model/attention.py` |
| Ch. 7 | FFN, TransformerBlock, Pooling | `model/layers.py` |
| Ch. 8 | Modèle d'embedding complet | `model/encoder.py` |
| Ch. 9 | Contrastive, Triplet, MNR, Matryoshka | `losses/` |
| Ch. 10 | AdamW, cosine scheduler avec warmup | `training/optimizer.py` |
| Ch. 11 | Recall@K, MRR@K, MAP | `training/metrics.py` |
| Ch. 12 | Pré-entraînement MLM, MLMDataset | `data/dataset.py`, `scripts/pretrain.py` |
| Ch. 13 | Fine-tuning contrastif | `scripts/finetune.py` |
| Ch. 14 | Hard negative mining avec FAISS | `mining/hard_negative.py` |
| Ch. 17 | Stratégies de chunking | `rag/chunker.py` |
| Ch. 18 | Retrieval vectoriel avec FAISS | `rag/retriever.py` |
| Ch. 19 | Recherche hybride, RRF | `rag/pipeline.py` |
| Ch. 20 | Re-ranking cross-encoder | `rag/reranker.py`, `scripts/demo_rag.py` |

---

## Utilisation

### Lancer les tests

```bash
uv run pytest
```

### Démo RAG (sans entraînement préalable)

Illustre le pipeline RAG complet avec un modèle E5-small pré-entraîné depuis HuggingFace : chunking, indexation hybride dense + BM25, Reciprocal Rank Fusion.

```bash
uv run python scripts/demo_rag.py
```

### Pré-entraînement MLM (Chapitre 12)

Pré-entraîne un modèle sur un corpus brut via la tâche Masked Language Modeling (masquage 80/10/10).

```bash
uv run python scripts/pretrain.py \
    --config configs/embedding_base.toml \
    --corpus data/corpus.txt \
    --output checkpoints/pretrain/
```

Le corpus doit être un fichier JSONL avec une clé `"text"` par ligne :

```jsonl
{"text": "Les modèles d'embedding transforment le texte en vecteurs denses."}
{"text": "L'attention multi-têtes capture les dépendances à longue distance."}
```

### Fine-tuning contrastif (Chapitre 13)

Fine-tune le modèle sur des paires (query, positive) avec MNR Loss.

```bash
uv run python scripts/finetune.py \
    --config configs/embedding_base.toml \
    --tokenizer checkpoints/pretrain/tokenizer/ \
    --checkpoint checkpoints/pretrain/pretrained_model.pt \
    --output checkpoints/finetune/
```

Les données d'entraînement doivent être au format JSONL :

```jsonl
{"query": "Comment fonctionne l'attention ?", "positive": "L'attention calcule une moyenne pondérée des valeurs selon les scores query-key."}
{"query": "Qu'est-ce que la MNR Loss ?", "positive": "La Multiple Negative Ranking Loss utilise les autres exemples du batch comme négatifs."}
```

La fonction de coût est sélectionnée dans le fichier TOML :

```toml
[loss]
type = "mnr"          # "contrastive" | "triplet" | "mnr" | "matryoshka"
temperature = 0.05
```

---

## Principes d'architecture

### Configuration par fichiers TOML

Aucun hyperparamètre n'est hardcodé dans le code. Tout passe par des fichiers TOML chargés dans des dataclasses Python typées :

```python
from embedding.model.config import ExperimentConfig

config = ExperimentConfig.from_toml("configs/embedding_base.toml")
print(config.model.d_model)             # 768
print(config.training.mixed_precision)  # "bf16"
print(config.loss.type)                 # "mnr"
```

### Gestion du device et reproductibilité

`DeviceManager` centralise la détection du device (CUDA > MPS > CPU), la fixation des seeds et le contexte de précision mixte :

```python
from embedding.utils.device import DeviceManager

dm = DeviceManager.setup(seed=42, mixed_precision="bf16")
# DeviceManager(device=cuda, seed=42, precision=bf16)

model = model.to(dm.device)
with dm.autocast_context:
    embeddings = model(input_ids, attention_mask)
```

### Architecture Transformer modulaire

Chaque brique est un `nn.Module` indépendant. La configuration pilote les choix d'architecture :

```python
from embedding.model.config import ModelConfig
from embedding.model.encoder import EmbeddingModel

config = ModelConfig(
    d_model=768,
    n_heads=12,
    n_layers=6,
    pooling="mean",           # "mean" | "cls" | "weighted_mean"
    pos_encoding="learned",   # "learned" | "sinusoidal" | "rope"
    activation="gelu",        # "gelu" | "relu" | "swiglu" | "geglu"
)
model = EmbeddingModel(config)
# EmbeddingModel initialisé : 86.5M paramètres

# Les sorties sont normalisées L2 — similarité cosinus = produit scalaire
embeddings = model(input_ids, attention_mask)  # (B, 768)
```

### Fonctions de coût contrastives

```python
from embedding.losses import MNRLoss, TripletLoss, MatryoshkaLoss

# MNR Loss — standard industriel pour les paires (query, positive)
loss = MNRLoss(temperature=0.05)(query_embeddings, positive_embeddings)

# Matryoshka — embeddings utiles à plusieurs dimensions simultanément
loss = MatryoshkaLoss(
    dimensions=[64, 128, 256, 512, 768],
    temperature=0.05,
)(query_embeddings, positive_embeddings)
```

### Pipeline RAG hybride

```python
from embedding.rag.chunker import HierarchicalChunker
from embedding.rag.retriever import DenseRetriever, BM25Retriever
from embedding.rag.pipeline import HybridRetriever, RAGPipeline
from embedding.rag.reranker import CrossEncoderReranker

# 1. Découpage
chunks = HierarchicalChunker(child_chunk_words=128).chunk(
    text, doc_id="doc_1", title="Introduction aux embeddings"
)

# 2. Pipeline : hybride (dense + BM25) + re-ranking cross-encoder
pipeline = RAGPipeline(
    retriever=HybridRetriever(
        dense_retriever=DenseRetriever(model, tokenizer),
        bm25_retriever=BM25Retriever(),
    ),
    reranker=CrossEncoderReranker(),
    n_retrieval=20,
    n_final=5,
)
pipeline.index_documents(chunks)

# 3. Requête — retourne le prompt prêt pour le LLM
result = pipeline.query("Comment fonctionne le negative mining ?")
print(result["prompt"])
```

---

## Commandes UV essentielles

| Commande | Usage |
|---|---|
| `uv sync` | Synchronise l'environnement avec `uv.lock` |
| `uv add <pkg>` | Ajoute une dépendance principale |
| `uv add --dev <pkg>` | Ajoute une dépendance de développement |
| `uv run python <script>` | Exécute un script dans le venv |
| `uv run pytest` | Lance les tests |
| `uv run ruff check src/` | Lint du code source |
| `uv run ruff format src/` | Formatage du code |
| `uv run jupyter lab` | Lance Jupyter Lab |

Il n'est jamais nécessaire d'activer manuellement le virtualenv (`source .venv/bin/activate`). `uv run` s'en charge pour la durée de la commande.

---

## Dépendances principales

| Package | Version | Rôle |
|---|---|---|
| `torch` | ≥ 2.5.1 | Calcul tensoriel, autograd, Flash Attention |
| `tokenizers` | ≥ 0.21.0 | Entraînement BPE, tokenisation rapide |
| `transformers` | ≥ 4.47.0 | Modèles HuggingFace, cross-encoders |
| `faiss-cpu` | ≥ 1.9.0 | Index vectoriel ANN pour le retrieval |
| `rank-bm25` | ≥ 0.2.2 | Recherche lexicale BM25 |
| `sentence-transformers` | ≥ 3.3.0 | Cross-encoders pour le re-ranking |
| `loguru` | ≥ 0.7.2 | Logging structuré |
| `wandb` | ≥ 0.19.0 | Suivi des expériences |

Dépendances de développement (`uv sync --extra dev`) :

| Package | Rôle |
|---|---|
| `pytest` + `pytest-cov` | Tests unitaires avec couverture |
| `ruff` | Linting et formatage (remplace flake8 + isort + black) |
| `mypy` | Vérification statique des types |
| `jupyter` | Notebooks d'exploration |

---

## Notes de compatibilité

- Développé et testé avec **Python 3.12**, **PyTorch 2.5.1**, **CUDA 12.8**.
- Compatible CPU, GPU NVIDIA (CUDA) et Apple Silicon (MPS).
- Sur GPU NVIDIA Ampere+ (RTX 3000+, A100, H100) : `mixed_precision = "bf16"` recommandé — même plage dynamique que fp32, deux fois moins de mémoire.
- Flash Attention (`F.scaled_dot_product_attention`) activé automatiquement avec PyTorch ≥ 2.0.
- Pour l'évaluation sur les benchmarks MTEB/BEIR : `uv sync --extra eval`.
