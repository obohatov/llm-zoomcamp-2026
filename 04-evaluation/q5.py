"""Q5. Evaluating vector search: MRR over the full ground truth."""

from embedder import Embedder
from eval_helper import (
    build_vector_index,
    evaluate,
    get_chunks,
    load_documents,
    load_ground_truth,
    make_vector_search,
)


def main():
    documents = load_documents()
    chunks = get_chunks(documents)

    embedder = Embedder()
    vindex = build_vector_index(chunks, embedder)
    vector_search = make_vector_search(vindex, embedder)

    ground_truth = load_ground_truth()
    print(f"ground truth questions: {len(ground_truth)}")

    metrics = evaluate(ground_truth, vector_search)
    print(f"\nvector search metrics: {metrics}")
    print(f"\nQ5 answer -> mrr = {metrics['mrr']:.4f}")


if __name__ == "__main__":
    main()
