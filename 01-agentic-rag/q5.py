from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
from gitsource import GithubRepositoryDataReader, chunk_documents
from minsearch import Index
from rag_helper import RAGBase

# --- Готовим документы и чанки ---
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

# --- Индекс по чанкам (та же схема полей, что и в Q2) ---
chunk_index = Index(
    text_fields=["content"],
    keyword_fields=["filename"],
)
chunk_index.fit(chunks)

# --- Тот же RAG-класс, что и в Q3 ---
class RAG(RAGBase):
    def search(self, query, num_results=5):
        return self.index.search(query, num_results=num_results)

    def build_context(self, search_results):
        lines = []
        for doc in search_results:
            lines.append('FILE: ' + doc['filename'])
            lines.append(doc['content'])
            lines.append('')
        return '\n'.join(lines).strip()

    def llm(self, prompt):
        input_messages = [
            {'role': 'developer', 'content': self.instructions},
            {'role': 'user', 'content': prompt}
        ]
        response = self.llm_client.responses.create(
            model=self.model,
            input=input_messages
        )
        return response

    def rag(self, query):
        search_results = self.search(query)
        prompt = self.build_prompt(query, search_results)
        response = self.llm(prompt)
        answer = response.output_text
        usage = response.usage
        return answer, usage


# --- Запускаем на индексе чанков ---
client = OpenAI()
rag = RAG(index=chunk_index, llm_client=client, model="gpt-5.4-mini")

query = "How does the agentic loop keep calling the model until it stops?"
answer, usage = rag.rag(query)

print("ОТВЕТ:")
print(answer)
print()
print("INPUT TOKENS (chunked):", usage.input_tokens)

# Сравнение с Q3
q3_tokens = 7126  # подставь СВОЁ число, полученное в Q3
ratio = q3_tokens / usage.input_tokens
print(f"В Q3 было {q3_tokens} токенов, теперь {usage.input_tokens} токенов")
print(f"Во сколько раз меньше: {ratio:.1f}x")
