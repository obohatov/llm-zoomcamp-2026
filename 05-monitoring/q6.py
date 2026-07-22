import sqlite3
import pandas as pd

conn = sqlite3.connect("traces.db")
df = pd.read_sql("SELECT * FROM spans WHERE name = 'llm'", conn)

print(df[["input_tokens", "output_tokens", "cost"]])

tokens = df["input_tokens"]
print()
print("min:", tokens.min(), "max:", tokens.max())
print("relative spread:", (tokens.max() - tokens.min()) / tokens.mean())

# Run this after calling q1.py/q4.py (or the same script) 4 times total
# with the SAME query, so the "spans" table has 4 "llm" rows to compare.
#
# minsearch's text search (BM25-style scoring) is deterministic: the
# same query against the same, unchanged index always returns the same
# ranked documents in the same order. That means build_prompt() produces
# byte-for-byte the same prompt every time, so the tokenizer sees the
# same input and input_tokens comes out identical across all 4 runs.
