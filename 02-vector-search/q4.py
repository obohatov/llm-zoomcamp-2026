from embedder import Embedder
from minsearch import VectorSearch
from vector_helper import load_chunks, embed_chunks

embedder = Embedder()

chunks = load_chunks(size=2000, step=1000)
X = embed_chunks(embedder, chunks)

vindex = VectorSearch()
vindex.fit(X, chunks)

query = "What metric do we use to evaluate a search engine?"
q = embedder.encode(query)

results = vindex.search(q, num_results=5)
print(f"Первый результат: {results[0]['filename']}")
