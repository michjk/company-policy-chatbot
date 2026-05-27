from typing import Annotated, Any

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from .llm import build_chat_model

SYSTEM_PROMPT = """\
You are a helpful company policy assistant. Answer the user's question using \
only the provided policy excerpts. If the answer is not in the excerpts, say so.

Policy excerpts:
{context}
"""


class RAGState(TypedDict):
    messages: Annotated[list, add_messages]
    context: list[Document]


def build_rag_graph(retriever: Any, checkpointer: BaseCheckpointSaver):
    llm = build_chat_model()

    async def retrieve(state: RAGState) -> dict:
        last_human = next(
            (
                m
                for m in reversed(state["messages"])
                if hasattr(m, "type") and m.type == "human"
            ),
            None,
        )
        query = last_human.content if last_human else ""
        docs = await retriever.ainvoke(query)
        return {"context": docs}

    async def generate(state: RAGState) -> dict:
        docs = state.get("context", [])
        context_text = "\n\n".join(
            f"[source: {d.metadata.get('filename', 'unknown')}#{d.metadata.get('chunk_index', 0)}]\n{d.page_content}"
            for d in docs
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=SYSTEM_PROMPT.format(context=context_text)),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
        chain = prompt | llm
        response = await chain.ainvoke({"messages": state["messages"]})
        citations = [
            {
                "source": d.metadata.get("filename", "unknown"),
                "chunk": d.metadata.get("chunk_index", 0),
                "doc_id": d.metadata.get("doc_id", ""),
            }
            for d in docs
        ]
        ai_msg = AIMessage(
            content=response.content,
            response_metadata={"citations": citations},
        )
        return {"messages": [ai_msg]}

    graph = StateGraph(RAGState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    return graph.compile(checkpointer=checkpointer)
