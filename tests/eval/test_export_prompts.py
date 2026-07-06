from unittest.mock import MagicMock, patch

import dspy
import pytest


@pytest.fixture(autouse=True)
def dummy_lm():
    dspy.configure(lm=dspy.utils.DummyLM([]))
    yield
    dspy.settings.configure(lm=None)


def _make_mock_program(
    rag_instr: str = "You are a policy assistant.",
    query_instr: str = "Reformulate the question.",
    cite_instr: str = "List sources.",
) -> MagicMock:
    """Build a mock compiled PolicyRAG whose module instructions are readable."""

    def _sig(instructions: str) -> MagicMock:
        sig = MagicMock()
        sig.instructions = instructions
        return sig

    reformulate = MagicMock()
    reformulate.signature = _sig(query_instr)

    generate = MagicMock()
    generate.predict = MagicMock()
    generate.predict.signature = _sig(rag_instr)
    # Prevent .signature from matching (ChainOfThought path only)
    del generate.signature

    cite = MagicMock()
    cite.signature = _sig(cite_instr)

    program = MagicMock()
    program.reformulate = reformulate
    program.generate = generate
    program.cite = cite
    return program


def test_dry_run_prints_diff(capsys):
    mock_program = _make_mock_program()

    with patch("eval.export_prompts._load_compiled_program", return_value=mock_program):
        from eval.export_prompts import export

        export(apply=False, compiled_path="dummy.json")

    captured = capsys.readouterr()
    assert "RAG_SYSTEM_PROMPT" in captured.out
    assert "QUERY_REFORMULATION_PROMPT" in captured.out
    assert "CITATION_EXTRACTION_PROMPT" in captured.out


def test_apply_writes_prompts_py(tmp_path):
    mock_program = _make_mock_program(
        rag_instr="Optimized RAG prompt.",
        query_instr="Optimized query prompt.",
        cite_instr="Optimized citation prompt.",
    )
    output_path = tmp_path / "prompts.py"

    with patch("eval.export_prompts._load_compiled_program", return_value=mock_program):
        from eval.export_prompts import export

        export(apply=True, compiled_path="dummy.json", output_path=str(output_path))

    content = output_path.read_text()
    assert "Optimized RAG prompt." in content
    assert "Optimized query prompt." in content
    assert "Optimized citation prompt." in content
    assert "RAG_SYSTEM_PROMPT" in content
    assert "QUERY_REFORMULATION_PROMPT" in content
    assert "CITATION_EXTRACTION_PROMPT" in content
