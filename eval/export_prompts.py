from __future__ import annotations

import argparse
from typing import Any
from unittest.mock import MagicMock


from eval.rag_program import PolicyRAG

_PROMPTS_PY_TEMPLATE = '''\
RAG_SYSTEM_PROMPT = """\\\n{rag}\\\n"""

QUERY_REFORMULATION_PROMPT = """\\\n{query}\\\n"""

CITATION_EXTRACTION_PROMPT = """\\\n{citation}\\\n"""
'''

_DEFAULT_OUTPUT = "app/prompts.py"
_DEFAULT_COMPILED = "eval/compiled/optimized_rag.json"


def _get_instructions(module: Any) -> str:
    """Extract optimized instruction string from a compiled DSPy Predict/ChainOfThought module."""
    for attr in ("extended_signature", "signature"):
        sig = getattr(module, attr, None)
        if sig is not None:
            instr = getattr(sig, "instructions", None)
            if isinstance(instr, str) and instr:
                return instr
    return ""


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

    new_content = _PROMPTS_PY_TEMPLATE.format(
        rag=rag_instr,
        query=query_instr,
        citation=citation_instr,
    )

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
