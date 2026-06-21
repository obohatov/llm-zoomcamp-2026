from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
from gitsource import GithubRepositoryDataReader
from minsearch import Index
from rag_helper import RAGBase

# --- Готовим документы и индекс (как в Q1/Q2) ---
reader = GithubRepositoryDataReader(
    repo_owner="DataTalksClub",
    repo_name="llm-zoomcamp",
    commit_id="8c1834d",
    allowed_extensions={"md"},
    filename_filter=lambda path: "/lessons/" in path,
)

files = reader.read()
documents = [file.parse() for file in files]

index = Index(
    text_fields=["content"],
    keyword_fields=["filename"],
)
index.fit(documents)

# --- Наш RAG-класс под схему filename/content ---
class RAG(RAGBase):
    def search(self, query, num_results=5):
        # У нас нет boost_dict/filter_dict - наша схема проще
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
        return response  # возвращаем весь response, не только текст

    def rag(self, query):
        search_results = self.search(query)
        prompt = self.build_prompt(query, search_results)
        response = self.llm(prompt)
        answer = response.output_text
        usage = response.usage
        return answer, usage


# --- Запускаем ---
client = OpenAI()
rag = RAG(index=index, llm_client=client, model="gpt-5.4-mini")

query = "How does the agentic loop keep calling the model until it stops?"
answer, usage = rag.rag(query)

print("ОТВЕТ:")
print(answer)
print()
print("USAGE:", usage)
print("INPUT TOKENS:", usage.input_tokens)
