"""Q4. Evaluating text search: Hit Rate over the full ground truth."""

from eval_helper import (
    build_text_index,
    evaluate,
    get_chunks,
    load_documents,
    load_ground_truth,
    make_text_search,
)


def main():
    documents = load_documents()
    chunks = get_chunks(documents)

    index = build_text_index(chunks)
    text_search = make_text_search(index)

    ground_truth = load_ground_truth()
    print(f"ground truth questions: {len(ground_truth)}")

    metrics = evaluate(ground_truth, text_search)
    print(f"\ntext search metrics: {metrics}")
    print(f"\nQ4 answer -> hit_rate = {metrics['hit_rate']:.4f}")


if __name__ == "__main__":
    main()
