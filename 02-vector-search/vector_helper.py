"""Общие утилиты для домашки модуля 2 (Vector Search).

Кеши на диске (создаются автоматически при первом запуске):
  - documents.json  — 72 страницы уроков, чтобы не дёргать GitHub
                      в каждом скрипте;
  - chunks_X.npy    — матрица эмбеддингов чанков.
Удали соответствующий кеш, если поменял коммит / размер чанков / модель.
"""
import json
from pathlib import Path

import numpy as np
from gitsource import GithubRepositoryDataReader, chunk_documents

COMMIT_ID = "8c1834d"


def load_documents(cache="documents.json"):
    """72 страницы уроков (ключи: filename, content). Кешируется локально."""
    cache = Path(cache)
    if cache.exists():
        with cache.open(encoding="utf-8") as f:
            return json.load(f)

    reader = GithubRepositoryDataReader(
        repo_owner="DataTalksClub",
        repo_name="llm-zoomcamp",
        commit_id=COMMIT_ID,
        allowed_extensions={"md"},
        filename_filter=lambda path: "/lessons/" in path,
    )
    documents = [file.parse() for file in reader.read()]

    with cache.open("w", encoding="utf-8") as f:
        json.dump(documents, f, ensure_ascii=False)
    return documents


def load_chunks(size=2000, step=1000):
    """Чанки с перекрытием (ключи: filename, start, content)."""
    return chunk_documents(load_documents(), size=size, step=step)


def embed_chunks(embedder, chunks, cache="chunks_X.npy"):
    """Матрица X (num_chunks, 384) с эмбеддингами content каждого чанка."""
    cache = Path(cache)
    if cache.exists():
        return np.load(cache)

    texts = [chunk["content"] for chunk in chunks]
    X = embedder.encode_batch(texts)
    np.save(cache, X)
    return X
