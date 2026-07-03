from typing import Annotated

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.retrievers import BaseRetriever
from langchain_core.tools import tool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from typing_extensions import TypedDict

from .llm import build_chat_model
from .prompts import RAG_SYSTEM_PROMPT

RETRIEVE_TOOL_NAME = "retrieve_context"  # must match the decorated function name below


class RAGState(TypedDict):
    messages: Annotated[list, add_messages]
    context: list[Document]


def build_rag_graph(
    retriever: BaseRetriever, checkpointer: BaseCheckpointSaver
) -> CompiledStateGraph:
    llm = build_chat_model()

    @tool(response_format="content_and_artifact")
    async def retrieve_context(query: str) -> tuple[str, list[Document]]:
        """Retrieve relevant company policy excerpts for a query."""
        docs = await retriever.ainvoke(query)
        serialized = "\n\n".join(
            f"<context>\n[source: {d.metadata.get('filename', 'unknown')}#{d.metadata.get('chunk_index', 0)}]\n{d.page_content}\n</context>"
            for d in docs
        )
        return serialized, docs

    retrieve_tool = retrieve_context
    llm_with_tools = llm.bind_tools([retrieve_tool])

    async def agent(state: RAGState) -> dict:
        messages = [SystemMessage(content=RAG_SYSTEM_PROMPT)] + list(state["messages"])
        response = await llm_with_tools.ainvoke(messages)
        return {"messages": [response]}

    async def call_tools(state: RAGState) -> dict:
        last_msg = state["messages"][-1]
        all_docs = list(state.get("context", []))
        tool_messages = []
        for tc in last_msg.tool_calls:
            if tc["name"] == RETRIEVE_TOOL_NAME:
                result = await retrieve_tool.ainvoke(tc)
                tool_messages.append(result)
                if hasattr(result, "artifact") and result.artifact:
                    all_docs.extend(result.artifact)
        return {"messages": tool_messages, "context": all_docs}

    async def finalize(state: RAGState) -> dict:
        docs = state.get("context", [])
        citations = [
            {
                "source": meta.get("filename", "unknown"),
                "chunk": meta.get("chunk_index", 0),
                "doc_id": meta.get("doc_id", ""),
            }
            for d in docs
            for meta in [
                d.metadata if hasattr(d, "metadata") else d.get("metadata", {})
            ]
        ]
        last_msg = state["messages"][-1]
        updated = AIMessage(
            id=last_msg.id,
            content=last_msg.content,
            response_metadata={"citations": citations},
        )
        return {"messages": [updated]}

    def should_continue(state: RAGState) -> str:
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        return "finalize"

    graph = StateGraph(RAGState)
    graph.add_node("agent", agent)
    graph.add_node("tools", call_tools)
    graph.add_node("finalize", finalize)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent", should_continue, {"tools": "tools", "finalize": "finalize"}
    )
    graph.add_edge("tools", "agent")
    graph.add_edge("finalize", END)

    return graph.compile(checkpointer=checkpointer)
