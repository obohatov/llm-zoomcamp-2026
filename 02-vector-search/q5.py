from embedder import Embedder
from minsearch import VectorSearch, Index
from vector_helper import load_chunks, embed_chunks

embedder = Embedder()

chunks = load_chunks(size=2000, step=1000)
X = embed_chunks(embedder, chunks)

# векторный индекс
vindex = VectorSearch()
vindex.fit(X, chunks)

# текстовый индекс (поле content)
tindex = Index(text_fields=["content"])
tindex.fit(chunks)

query = "How do I store vectors in PostgreSQL?"

vector_results = vindex.search(embedder.encode(query), num_results=5)
text_results = tindex.search(query, num_results=5)

v_files = [r["filename"] for r in vector_results]
t_files = [r["filename"] for r in text_results]

print("Векторный топ-5:")
for f in v_files:
    print("  ", f)
print("Текстовый топ-5:")
for f in t_files:
    print("  ", f)

only_in_vector = set(v_files) - set(t_files)
print(f"\nЕсть в векторном, но не в текстовом: {only_in_vector}")
