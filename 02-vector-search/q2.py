from embedder import Embedder
from vector_helper import load_documents

embedder = Embedder()

# тот же запрос, что и в Q1
query = "How does approximate nearest neighbor search work?"
v_query = embedder.encode(query)

documents = load_documents()

target = "02-vector-search/lessons/07-sqlitesearch-vector.md"
doc = next(d for d in documents if d["filename"] == target)

v_doc = embedder.encode(doc["content"])

# нормализованные векторы -> dot == косинус
cosine = float(v_query.dot(v_doc))
print(f"Косинусная близость с {target}: {cosine:.4f}")
