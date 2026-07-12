"""Q1. Generating questions for the first 3 lesson pages.

For each page, we ask the LLM to write 5 questions answered by that
page (structured output), and report the average input tokens across
the 3 calls.
"""

import json

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

from eval_helper import load_documents
from evaluation_utils import llm_structured

load_dotenv()
openai_client = OpenAI()


class Questions(BaseModel):
    questions: list[str]


data_gen_instructions = """
You emulate a student who is taking our LLM course.
You are given one lesson page from the course.
Formulate 5 questions this student might ask that are answered by this page.

Rules:
- The page should contain the answer to each question.
- Make the questions complete and not too short.
- Use as few words as possible from the page; don't copy its phrasing.
- The questions should resemble how people actually ask things online:
  not too formal, not too short, not too long.
- Ask about the content of the lesson, not about its formatting or filename.
""".strip()

TARGET_PAGES = [
    "01-agentic-rag/lessons/01-intro.md",
    "01-agentic-rag/lessons/02-environment.md",
    "01-agentic-rag/lessons/03-rag.md",
]


def main():
    documents = load_documents()
    doc_idx = {d["filename"]: d for d in documents}

    input_tokens = []

    for filename in TARGET_PAGES:
        doc = doc_idx[filename]
        user_prompt = json.dumps(
            {"filename": doc["filename"], "content": doc["content"]}
        )

        result, usage = llm_structured(
            openai_client,
            data_gen_instructions,
            user_prompt,
            Questions,
        )

        print(f"\n=== {filename} ===")
        print(f"input tokens: {usage.input_tokens}, "
              f"output tokens: {usage.output_tokens}")
        for q in result.questions:
            print(f"  - {q}")

        input_tokens.append(usage.input_tokens)

    avg = sum(input_tokens) / len(input_tokens)
    print(f"\ninput tokens per call: {input_tokens}")
    print(f"Q1 answer -> average input tokens: {avg:.1f}")


if __name__ == "__main__":
    main()
