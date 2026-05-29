from ag_ui_langgraph import LangGraphAgent, add_langgraph_fastapi_endpoint
from copilotkit import CopilotKitRemoteEndpoint, LangGraphAGUIAgent
from copilotkit.integrations.fastapi import add_fastapi_endpoint
from fastapi import FastAPI
from langgraph.graph.state import CompiledStateGraph


def register_streaming_endpoint(
    app: FastAPI, graph: CompiledStateGraph, path: str = "/chat/stream"
) -> None:
    agent = LangGraphAgent(
        name="company-policy-rag",
        graph=graph,
        description="RAG chatbot that answers questions about company policies.",
    )
    add_langgraph_fastapi_endpoint(app, agent, path=path)


def register_copilotkit_endpoint(
    app: FastAPI, graph: CompiledStateGraph, path: str = "/copilotkit"
) -> None:
    sdk = CopilotKitRemoteEndpoint(
        agents=[
            LangGraphAGUIAgent(
                name="company-policy-rag",
                description="RAG chatbot that answers questions about company policies.",
                graph=graph,
            )
        ]
    )
    add_fastapi_endpoint(app, sdk, path)
