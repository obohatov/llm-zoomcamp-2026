from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from sqlite_exporter import SQLiteSpanExporter

provider = TracerProvider()
provider.add_span_processor(
    SimpleSpanProcessor(SQLiteSpanExporter("traces.db"))
)
trace.set_tracer_provider(provider)

from rag_traced import RAGTraced  # noqa: E402
from starter import index, client  # noqa: E402

rag = RAGTraced(index=index, llm_client=client)

query = "How does the agentic loop keep calling the model until it stops?"
answer = rag.rag(query)
print("ANSWER:", answer)

# Check what landed in the database:
#   uv run python -c "import sqlite3; \
#       print(sqlite3.connect('traces.db').execute('SELECT name FROM spans').fetchall())"
# Expected: rows for "rag", "search", and "llm" -> answer to Q4 is
# "rag, search, and llm".
