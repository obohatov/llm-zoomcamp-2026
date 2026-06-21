from dotenv import load_dotenv
load_dotenv()

from gitsource import GithubRepositoryDataReader
from minsearch import Index

reader = GithubRepositoryDataReader(
    repo_owner="DataTalksClub",
    repo_name="llm-zoomcamp",
    commit_id="8c1834d",
    allowed_extensions={"md"},
    filename_filter=lambda path: "/lessons/" in path,
)

files = reader.read()

documents = []
for file in files:
    doc = file.parse()
    documents.append(doc)

# Создаём индекс: content - текстовое поле, filename - keyword-поле
index = Index(
    text_fields=["content"],
    keyword_fields=["filename"],
)
index.fit(documents)

query = "How does the agentic loop keep calling the model until it stops?"
results = index.search(query)

print("Первый результат:")
print(results[0]["filename"])
