"""Q3. First result with vector search for the same question."""

from embedder import Embedder
from eval_helper import (
    build_vector_index,
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
    q = ground_truth[0]["question"]
    print(f"question: {q}")
    print(f"expected page: {ground_truth[0]['filename']}")

    results = vector_search(q)
    print("\nvector search results:")
    for i, d in enumerate(results):
        print(f"  {i + 1}. {d['filename']} (start={d['start']})")

    print(f"\nQ3 answer -> {results[0]['filename']}")


if __name__ == "__main__":
    main()
