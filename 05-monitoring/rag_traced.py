"""RAGBase subclass instrumented with OpenTelemetry spans.

Wraps rag(), search(), and llm() each in their own span, and attaches
token usage + cost as attributes on the llm span.
"""

from opentelemetry import trace

from rag_helper import RAGBase

tracer = trace.get_tracer("llm-zoomcamp")


def calculate_cost(model, usage):
    cost = 0
    if "gpt-5.4-mini" in model:
        cost = (usage.input_tokens * 0.15 + usage.output_tokens * 0.60) / 1_000_000
    return cost


class RAGTraced(RAGBase):

    def search(self, query, num_results=5):
        with tracer.start_as_current_span("search") as span:
            results = super().search(query, num_results=num_results)
            span.set_attribute("num_results", len(results))
            return results

    def llm(self, prompt):
        with tracer.start_as_current_span("llm") as span:
            response = super().llm(prompt)
            usage = response.usage
            cost = calculate_cost(self.model, usage)

            span.set_attribute("input_tokens", usage.input_tokens)
            span.set_attribute("output_tokens", usage.output_tokens)
            span.set_attribute("cost", cost)
            return response

    def rag(self, query):
        with tracer.start_as_current_span("rag") as span:
            span.set_attribute("question", query)
            # self.search() and self.llm() below resolve to the
            # overridden methods above, so this call still nests
            # a "search" span and an "llm" span inside "rag".
            return super().rag(query)
