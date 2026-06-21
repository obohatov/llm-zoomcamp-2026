from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
from gitsource import GithubRepositoryDataReader, chunk_documents
from minsearch import Index

# --- Готовим документы и чанки (как в Q4/Q5) ---
reader = GithubRepositoryDataReader(
    repo_owner="DataTalksClub",
    repo_name="llm-zoomcamp",
    commit_id="8c1834d",
    allowed_extensions={"md"},
    filename_filter=lambda path: "/lessons/" in path,
)

files = reader.read()
documents = [file.parse() for file in files]
chunks = chunk_documents(documents, size=2000, step=1000)

chunk_index = Index(
    text_fields=["content"],
    keyword_fields=["filename"],
)
chunk_index.fit(chunks)

# --- Считаем вызовы search ---
search_call_count = 0

def search(query: str) -> list[dict]:
    """
    Search the course lesson pages for relevant chunks of text.

    Args:
        query: the search query (keywords related to the question)

    Returns:
        a list of matching chunks, each with 'filename' and 'content'
    """
    global search_call_count
    search_call_count += 1
    return chunk_index.search(query, num_results=5)


# --- Собираем агента ---
from toyaikit.tools import Tools
from toyaikit.llm import OpenAIClient
from toyaikit.chat.runners import OpenAIResponsesRunner

tools = Tools()
tools.add_tool(search)

INSTRUCTIONS = """
You're a course teaching assistant. Answer the student's question using the
search tool. Make multiple searches with different keywords before answering.
"""

llm_client = OpenAIClient(client=OpenAI(), model="gpt-5.4-mini")

runner = OpenAIResponsesRunner(
    tools=tools,
    developer_prompt=INSTRUCTIONS,
    chat_interface=None,
    llm_client=llm_client,
)

question = "How does the agentic loop work, and how is it different from plain RAG?"
result = runner.loop(prompt=question)

print("ОТВЕТ:")
print(result.last_message)
print()
print(f"Количество вызовов search: {search_call_count}")
