from embedder import Embedder

embedder = Embedder()

query = "How does approximate nearest neighbor search work?"
v = embedder.encode(query)

print(f"Размерность вектора: {v.shape}")          # (384,)
print(f"Первое значение v[0]: {v[0]:.4f}")
