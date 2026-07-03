import json
import os
from unittest.mock import MagicMock, patch

import dspy
import pytest


@pytest.fixture(autouse=True)
def dummy_lm():
    # Provide enough responses for the test with the most LLM calls (4 calls: 2 chunks × 2 per_chunk)
    lm = dspy.utils.DummyLM(
        [{"question": "What is PTO?", "answer": "15 days per year."}] * 10
    )
    dspy.configure(lm=lm)


def _make_mock_conn(rows):
    mock_cursor = MagicMock()
    mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
    mock_cursor.__exit__ = MagicMock(return_value=False)
    mock_cursor.fetchall.return_value = rows

    mock_conn = MagicMock()
    mock_conn.__enter__ = MagicMock(return_value=mock_conn)
    mock_conn.__exit__ = MagicMock(return_value=False)
    mock_conn.cursor.return_value = mock_cursor
    return mock_conn


def test_fetch_chunks_returns_dicts():
    rows = [("Employees get 15 days PTO.", {"doc_id": "abc", "filename": "pto.md"})]
    with patch("psycopg.connect", return_value=_make_mock_conn(rows)):
        from eval.generate_dataset import fetch_chunks

        result = fetch_chunks(10)

    assert len(result) == 1
    assert result[0]["document"] == "Employees get 15 days PTO."
    assert result[0]["metadata"]["filename"] == "pto.md"


def test_fetch_chunks_empty():
    with patch("psycopg.connect", return_value=_make_mock_conn([])):
        from eval.generate_dataset import fetch_chunks

        result = fetch_chunks(10)

    assert result == []


def test_generate_pairs_count():
    chunks = [
        {"document": "Employees get 15 days PTO.", "metadata": {}},
        {"document": "Remote work is allowed.", "metadata": {}},
    ]
    from eval.generate_dataset import generate_pairs

    pairs = generate_pairs(chunks, n_per_chunk=2)
    assert len(pairs) == 4
    for p in pairs:
        assert "question" in p
        assert "expected_answer" in p
        assert "context" in p


def test_main_writes_json(tmp_path, monkeypatch):
    rows = [("Policy text.", {"doc_id": "x", "filename": "policy.md"})]
    monkeypatch.chdir(tmp_path)
    os.makedirs("eval/data", exist_ok=True)

    with patch("psycopg.connect", return_value=_make_mock_conn(rows)):
        from eval import generate_dataset

        # reload so monkeypatched chdir takes effect
        import importlib

        importlib.reload(generate_dataset)
        generate_dataset.main()

    with open("eval/data/synthetic_dataset.json") as f:
        data = json.load(f)

    assert "train" in data
    assert "dev" in data
    assert len(data["train"]) + len(data["dev"]) == 3  # 1 chunk × 3 per chunk
