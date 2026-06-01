from ag_ui_langgraph import LangGraphAgent, add_langgraph_fastapi_endpoint
from fastapi import FastAPI
from langgraph.graph.state import CompiledStateGraph


def register_streaming_endpoint(
    app: FastAPI, graph: CompiledStateGraph, path: str = "/chat/stream"
) -> None:
    agent = LangGraphAgent(
        name="company-policy-rag",
        description="RAG chatbot that answers questions about company policies.",
        graph=graph,
    )
    add_langgraph_fastapi_endpoint(app, agent, path=path)
