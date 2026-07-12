"""Shared helpers for Module 4 homework: Evaluation.

Data loading (with local caching), text/vector/hybrid search over
lesson chunks, and the evaluation metrics from the module lessons
(Hit Rate, MRR), adapted to `filename` labels.
"""

import json
import os

import numpy as np
from tqdm.auto import tqdm

from gitsource import GithubRepositoryDataReader, chunk_documents
from minsearch import Index, VectorSearch

DOCUMENTS_FILE = "documents.json"
EMBEDDINGS_FILE = "chunk_embeddings.npy"


# --- Data loading -----------------------------------------------------------

def load_documents():
    """Load the 72 lesson pages (commit 8c1834d), caching to documents.json.

    gitsource makes fresh GitHub API calls on every read, which is slow
    and can hit rate limits - so we cache the result locally.
    """
    if os.path.exists(DOCUMENTS_FILE):
        with open(DOCUMENTS_FILE) as f:
            return json.load(f)

    reader = GithubRepositoryDataReader(
        repo_owner="DataTalksClub",
        repo_name="llm-zoomcamp",
        commit_id="8c1834d",
        allowed_extensions={"md"},
        filename_filter=lambda path: "/lessons/" in path,
    )
    documents = [file.parse() for file in reader.read()]

    with open(DOCUMENTS_FILE, "w") as f:
        json.dump(documents, f)

    return documents


def get_chunks(documents):
    """Chunk the pages exactly as in homework 2: 295 chunks."""
    return chunk_documents(documents, size=2000, step=1000)


# --- Text search ------------------------------------------------------------

def build_text_index(chunks):
    index = Index(
        text_fields=["content"],
        keyword_fields=["filename"],
    )
    index.fit(chunks)
    return index


def make_text_search(index):
    def text_search(query, num_results=5):
        return index.search(query, num_results=num_results)

    return text_search


# --- Vector search ----------------------------------------------------------

def embed_chunks(chunks, embedder, batch_size=32):
    """Embed all chunk contents, caching the matrix to a .npy file."""
    if os.path.exists(EMBEDDINGS_FILE):
        return np.load(EMBEDDINGS_FILE)

    texts = [chunk["content"] for chunk in chunks]

    batches = []
    for i in tqdm(range(0, len(texts), batch_size), desc="embedding chunks"):
        batch = texts[i:i + batch_size]
        batches.append(embedder.encode_batch(batch))

    X = np.vstack(batches)
    np.save(EMBEDDINGS_FILE, X)
    return X


def build_vector_index(chunks, embedder):
    X = embed_chunks(chunks, embedder)
    vindex = VectorSearch()
    vindex.fit(X, chunks)
    return vindex


def make_vector_search(vindex, embedder):
    def vector_search(query, num_results=5):
        v = embedder.encode(query)
        return vindex.search(v, num_results=num_results)

    return vector_search


# --- Hybrid search (RRF), same as homework 2 --------------------------------

def rrf(result_lists, k=60, num_results=5):
    scores = {}
    docs = {}

    for results in result_lists:
        for rank, doc in enumerate(results):
            key = (doc["filename"], doc["start"])
            scores[key] = scores.get(key, 0) + 1 / (k + rank)
            docs[key] = doc

    ranked = sorted(scores, key=scores.get, reverse=True)
    return [docs[key] for key in ranked[:num_results]]


def make_hybrid_search(text_search, vector_search):
    def hybrid_search(query, k=60):
        text_results = text_search(query, num_results=10)
        vector_results = vector_search(query, num_results=10)
        return rrf([text_results, vector_results], k=k)

    return hybrid_search


# --- Evaluation metrics (from the module lessons) ---------------------------

def compute_relevance(q, search_function):
    """Relevance list for one ground truth record.

    A hit is a returned chunk whose `filename` matches the question's
    `filename` (the homework labels by page, not by document id).
    """
    target = q["filename"]
    results = search_function(query=q["question"])

    return [int(d["filename"] == target) for d in results]


def compute_relevance_total(ground_truth, search_function):
    relevance_total = []

    for q in tqdm(ground_truth):
        relevance_total.append(compute_relevance(q, search_function))

    return relevance_total


def hit_rate(relevance):
    cnt = 0

    for line in relevance:
        if 1 in line:
            cnt = cnt + 1

    return cnt / len(relevance)


def mrr(relevance):
    total_score = 0.0

    for line in relevance:
        for rank in range(len(line)):
            if line[rank] == 1:
                total_score = total_score + 1 / (rank + 1)
                break

    return total_score / len(relevance)


def evaluate(ground_truth, search_function):
    relevance_total = compute_relevance_total(ground_truth, search_function)

    return {
        "hit_rate": hit_rate(relevance_total),
        "mrr": mrr(relevance_total),
    }


def load_ground_truth(path="ground-truth.csv"):
    import pandas as pd

    df = pd.read_csv(path)
    return df.to_dict(orient="records")
