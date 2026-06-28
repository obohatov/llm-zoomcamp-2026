from embedder import Embedder
from minsearch import VectorSearch, Index
from vector_helper import load_chunks, embed_chunks


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


embedder = Embedder()

chunks = load_chunks(size=2000, step=1000)
X = embed_chunks(embedder, chunks)

vindex = VectorSearch()
vindex.fit(X, chunks)

tindex = Index(text_fields=["content"])
tindex.fit(chunks)

query = "How do I give the model access to tools?"

vector_results = vindex.search(embedder.encode(query), num_results=5)
text_results = tindex.search(query, num_results=5)

results = rrf([vector_results, text_results])
print(f"Первый после RRF: {results[0]['filename']}")
