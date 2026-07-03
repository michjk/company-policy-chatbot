from __future__ import annotations

import json
import os
import random
from typing import Any

import dspy
import psycopg

from app.config import settings
from eval.config import DATASET_MAX_CHUNKS, DATASET_QA_PER_CHUNK, configure_dspy_lm


class GenerateQA(dspy.Signature):  # type: ignore[misc]
    """Generate a realistic employee question and a faithful answer based only on the given policy excerpt."""

    excerpt: str = dspy.InputField()
    question: str = dspy.OutputField()
    answer: str = dspy.OutputField()


def fetch_chunks(max_chunks: int) -> list[dict[str, Any]]:
    """Query pgvector directly (sync psycopg) to get one chunk per document."""
    with psycopg.connect(settings.postgres_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT DISTINCT ON (e.cmetadata->>'doc_id')
                    e.document, e.cmetadata
                FROM langchain_pg_embedding e
                JOIN langchain_pg_collection c ON e.collection_id = c.uuid
                WHERE c.name = %s
                ORDER BY e.cmetadata->>'doc_id', e.id
                LIMIT %s
                """,
                (settings.pgvector_collection, max_chunks),
            )
            return [{"document": row[0], "metadata": row[1]} for row in cur.fetchall()]


def generate_pairs(
    chunks: list[dict[str, Any]], n_per_chunk: int
) -> list[dict[str, str]]:
    """Call the LLM once per (chunk, n) to produce Q&A pairs."""
    generator = dspy.Predict(GenerateQA)
    pairs: list[dict[str, str]] = []
    for chunk in chunks:
        for _ in range(n_per_chunk):
            result = generator(excerpt=chunk["document"])
            pairs.append(
                {
                    "question": result.question,
                    "expected_answer": result.answer,
                    "context": chunk["document"],
                }
            )
    return pairs


def main() -> None:
    configure_dspy_lm()
    chunks = fetch_chunks(DATASET_MAX_CHUNKS)
    if not chunks:
        print(
            "No chunks found in pgvector. Have you ingested documents? Run: make up && ingest your docs first."
        )
        return
    pairs = generate_pairs(chunks, DATASET_QA_PER_CHUNK)
    random.shuffle(pairs)
    split = int(len(pairs) * 0.8)
    dataset: dict[str, list[dict[str, str]]] = {
        "train": pairs[:split],
        "dev": pairs[split:],
    }
    os.makedirs("eval/data", exist_ok=True)
    with open("eval/data/synthetic_dataset.json", "w") as f:
        json.dump(dataset, f, indent=2)
    print(
        f"Generated {len(pairs)} examples ({split} train, {len(pairs) - split} dev) → eval/data/synthetic_dataset.json"
    )


if __name__ == "__main__":
    main()
