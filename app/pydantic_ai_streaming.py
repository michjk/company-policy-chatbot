from typing import Any

from fastapi import FastAPI, Request
from pydantic_ai import Agent
from starlette.responses import Response

from .pydantic_ai_agent import AgentDeps


def register_pydantic_streaming(
    app: FastAPI,
    agent: Agent[AgentDeps, str],
    retriever: Any,
    path: str = "/chat/stream",
) -> None:
    from pydantic_ai.ui.ag_ui import AGUIAdapter

    @app.post(path)
    async def stream(request: Request) -> Response:
        deps = AgentDeps(retriever=retriever)
        # NOTE: deps.citations are populated during the agent run but AGUIAdapter
        # does not expose a hook to attach them to the SSE stream. Clients that need
        # citation metadata should use POST /chat (synchronous) instead.
        return await AGUIAdapter.dispatch_request(request, agent=agent, deps=deps)
