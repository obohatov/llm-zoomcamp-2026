from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

exporter = InMemorySpanExporter()
provider = TracerProvider()
provider.add_span_processor(SimpleSpanProcessor(exporter))
trace.set_tracer_provider(provider)

from rag_traced import RAGTraced  # noqa: E402
from starter import index, client  # noqa: E402

rag = RAGTraced(index=index, llm_client=client)

query = "How does the agentic loop keep calling the model until it stops?"
answer = rag.rag(query)
print("ANSWER:", answer)
print()

for span in exporter.get_finished_spans():
    duration_ms = (span.end_time - span.start_time) / 1_000_000
    print(f"{span.name:8s} {duration_ms:8.2f} ms")

# "search" (in-memory minsearch lookup) should be a couple of
# milliseconds. "llm" (a real network call to OpenAI + generation)
# should be dramatically higher - look at the printed number here to
# pick the bucket for Q3.
