from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

provider = TracerProvider()
provider.add_span_processor(
    SimpleSpanProcessor(ConsoleSpanExporter())
)
trace.set_tracer_provider(provider)

from rag_traced import RAGTraced  # noqa: E402  (input_tokens/output_tokens/cost
from starter import index, client  # noqa: E402   are already set in llm() span)

rag = RAGTraced(index=index, llm_client=client)

query = "How does the agentic loop keep calling the model until it stops?"
answer = rag.rag(query)
print("ANSWER:", answer)

# The console output above already contains a JSON block for the "llm" span
# with "input_tokens", "output_tokens", and "cost" under "attributes".
# Look at that block to answer Q2 (~7000 input tokens for this question,
# since the context is 5 lesson chunks, roughly 30k characters).
