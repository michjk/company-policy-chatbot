from __future__ import annotations

import argparse
from typing import Any
from unittest.mock import MagicMock


from eval.rag_program import PolicyRAG

_DEFAULT_OUTPUT = "app/prompts.py"
_DEFAULT_COMPILED = "eval/compiled/optimized_rag.json"


def _get_instructions(module: Any) -> str:
    """Extract optimized instruction string from a compiled DSPy Predict/ChainOfThought module."""
    for getter in [
        lambda m: m.extended_signature.instructions,
        lambda m: m.signature.instructions,
        lambda m: m.predict.signature.instructions,  # ChainOfThought wraps a Predict
    ]:
        try:
            result = getter(module)
            if isinstance(result, str) and result:
                return result
        except AttributeError:
            continue
    return ""


def _build_prompts_content(rag: str, query: str, citation: str) -> str:
    return (
        f"RAG_SYSTEM_PROMPT = {repr(rag)}\n\n"
        f"QUERY_REFORMULATION_PROMPT = {repr(query)}\n\n"
        f"CITATION_EXTRACTION_PROMPT = {repr(citation)}\n"
    )


def _load_compiled_program(compiled_path: str) -> PolicyRAG:
    dummy_retriever = MagicMock()
    program = PolicyRAG(retriever=dummy_retriever)
    program.load(compiled_path)
    return program


def export(
    apply: bool = False,
    compiled_path: str = _DEFAULT_COMPILED,
    output_path: str = _DEFAULT_OUTPUT,
) -> None:
    program = _load_compiled_program(compiled_path)

    rag_instr = _get_instructions(program.generate)
    query_instr = _get_instructions(program.reformulate)
    citation_instr = _get_instructions(program.cite)

    new_content = _build_prompts_content(rag_instr, query_instr, citation_instr)

    print("=== Proposed app/prompts.py ===")
    print(new_content)

    if not apply:
        print("(Dry run — pass --apply to write the file.)")
        return

    with open(output_path, "w") as f:
        f.write(new_content)
    print(f"Written → {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export GEPA-optimized prompts to app/prompts.py"
    )
    parser.add_argument(
        "--apply", action="store_true", help="Write file (default: dry-run)"
    )
    parser.add_argument(
        "--compiled", default=_DEFAULT_COMPILED, help="Path to compiled JSON"
    )
    parser.add_argument(
        "--output", default=_DEFAULT_OUTPUT, help="Output path for prompts.py"
    )
    args = parser.parse_args()
    export(apply=args.apply, compiled_path=args.compiled, output_path=args.output)


if __name__ == "__main__":
    main()
