from ag_ui_langgraph import LangGraphAgent, add_langgraph_fastapi_endpoint


def register_streaming_endpoint(app, graph, path: str = "/chat/stream") -> None:
    """Register the AG-UI streaming endpoint backed by the compiled LangGraph."""
    agent = LangGraphAgent(
        name="company-policy-rag",
        graph=graph,
        description="RAG chatbot that answers questions about company policies.",
    )
    add_langgraph_fastapi_endpoint(app, agent, path=path)
