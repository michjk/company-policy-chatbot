from dataclasses import dataclass, field
from typing import Any

from pydantic_ai import Agent, RunContext

from .prompts import RAG_SYSTEM_PROMPT
from .pydantic_ai_llm import build_pydantic_model


@dataclass
class Citation:
    source: str
    chunk: int
    doc_id: str


@dataclass
class AgentDeps:
    retriever: Any
    citations: list[Citation] = field(default_factory=list)


def build_pydantic_agent() -> Agent[AgentDeps, str]:
    # output_type=str (not a Pydantic model) keeps streaming natural — the AG-UI protocol
    # expects plain text tokens, not JSON. The API response type safety is enforced at the
    # ChatResponse layer in api.py. Citations come from deps.citations (accurate), not LLM output.
    agent: Agent[AgentDeps, str] = Agent(
        build_pydantic_model(),
        deps_type=AgentDeps,
        output_type=str,
        system_prompt=RAG_SYSTEM_PROMPT,
    )

    @agent.tool
    async def retrieve_context(ctx: RunContext[AgentDeps], query: str) -> str:
        """Retrieve relevant company policy excerpts for a query."""
        docs = await ctx.deps.retriever.ainvoke(query)
        ctx.deps.citations.extend(
            Citation(
                source=d.metadata.get("filename", "unknown"),
                chunk=d.metadata.get("chunk_index", 0),
                doc_id=d.metadata.get("doc_id", ""),
            )
            for d in docs
        )
        return "\n\n".join(
            f"<context>\n[source: {d.metadata.get('filename', 'unknown')}#{d.metadata.get('chunk_index', 0)}]\n{d.page_content}\n</context>"
            for d in docs
        )

    return agent
