from dataclasses import dataclass, field
from typing import Annotated, Any

from fastapi import Depends
from langchain_postgres import PGVector
from langgraph.graph.state import CompiledStateGraph


@dataclass
class AppResources:
    vectorstore: PGVector | None = field(default=None)
    graph: CompiledStateGraph | None = field(default=None)
    retriever: Any = field(default=None)
    pydantic_agent: Any = field(default=None)
    session_store: Any = field(default=None)


resources = AppResources()


def get_vectorstore() -> PGVector:
    return resources.vectorstore  # type: ignore[return-value]


def get_graph() -> CompiledStateGraph:
    return resources.graph  # type: ignore[return-value]


VectorstoreDep = Annotated[PGVector, Depends(get_vectorstore)]
GraphDep = Annotated[CompiledStateGraph, Depends(get_graph)]
