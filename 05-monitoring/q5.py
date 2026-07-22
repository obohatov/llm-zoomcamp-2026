import sqlite3
import pandas as pd

conn = sqlite3.connect("traces.db")
df = pd.read_sql("SELECT * FROM spans", conn)

# start_time / end_time are nanoseconds since epoch (OTel convention)
df["duration_ms"] = (df["end_time"] - df["start_time"]) / 1_000_000

totals = (
    df[df["name"] != "rag"]
    .groupby("name")["duration_ms"]
    .sum()
    .sort_values(ascending=False)
)

print(totals)
print()
print("Span with the most total time:", totals.index[0])
