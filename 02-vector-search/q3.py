import numpy as np
from embedder import Embedder
from vector_helper import load_chunks, embed_chunks

embedder = Embedder()

query = "How does approximate nearest neighbor search work?"
v = embedder.encode(query)

chunks = load_chunks(size=2000, step=1000)
X = embed_chunks(embedder, chunks)          # (num_chunks, 384)

scores = X.dot(v)                           # один скор на чанк
best = int(np.argmax(scores))

print(f"Чанков всего: {len(chunks)}")
print(f"Максимальный скор: {scores[best]:.4f}")
print(f"Файл лучшего чанка: {chunks[best]['filename']}")
