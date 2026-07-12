"""Q2. First result with text search for the first ground truth question."""

from eval_helper import (
    build_text_index,
    get_chunks,
    load_documents,
    load_ground_truth,
    make_text_search,
)


def main():
    documents = load_documents()
    chunks = get_chunks(documents)
    print(f"documents: {len(documents)}, chunks: {len(chunks)}")

    index = build_text_index(chunks)
    text_search = make_text_search(index)

    ground_truth = load_ground_truth()
    q = ground_truth[0]["question"]
    print(f"question: {q}")
    print(f"expected page: {ground_truth[0]['filename']}")

    results = text_search(q)
    print("\ntext search results:")
    for i, d in enumerate(results):
        print(f"  {i + 1}. {d['filename']} (start={d['start']})")

    print(f"\nQ2 answer -> {results[0]['filename']}")


if __name__ == "__main__":
    main()
