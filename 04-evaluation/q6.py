"""Q6. Tuning hybrid search: which RRF `k` gives the best MRR."""

from embedder import Embedder
from eval_helper import (
    build_text_index,
    build_vector_index,
    evaluate,
    get_chunks,
    load_documents,
    load_ground_truth,
    make_hybrid_search,
    make_text_search,
    make_vector_search,
)

K_VALUES = [1, 50, 100, 200]


def main():
    documents = load_documents()
    chunks = get_chunks(documents)

    index = build_text_index(chunks)
    text_search = make_text_search(index)

    embedder = Embedder()
    vindex = build_vector_index(chunks, embedder)
    vector_search = make_vector_search(vindex, embedder)

    hybrid_search = make_hybrid_search(text_search, vector_search)

    ground_truth = load_ground_truth()
    print(f"ground truth questions: {len(ground_truth)}")

    results = []

    for k in K_VALUES:
        print(f"\nevaluating hybrid search with k={k}...")
        metrics = evaluate(
            ground_truth,
            lambda query, k=k: hybrid_search(query, k=k),
        )
        print(f"k={k}: {metrics}")
        results.append((k, metrics["mrr"], metrics["hit_rate"]))

    print("\nsummary:")
    print(f"{'k':>5} {'mrr':>8} {'hit_rate':>10}")
    for k, m, h in results:
        print(f"{k:>5} {m:>8.4f} {h:>10.4f}")

    # best MRR; on ties, the smallest k (list is sorted ascending by k)
    best_k, best_mrr, _ = max(results, key=lambda r: r[1])
    print(f"\nQ6 answer -> best k = {best_k} (mrr = {best_mrr:.4f})")


if __name__ == "__main__":
    main()
