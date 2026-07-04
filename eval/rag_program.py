from __future__ import annotations

import asyncio

import dspy
from langchain_core.retrievers import BaseRetriever


class ReformulateQuery(dspy.Signature):  # type: ignore[misc]
    """Reformulate the user question into an optimal search query for retrieving relevant policy documents."""

    question: str = dspy.InputField()
    reformulated_query: str = dspy.OutputField()


class GenerateAnswer(dspy.Signature):  # type: ignore[misc]
    """Answer the question using only the retrieved policy excerpts. If the answer is not in the excerpts, say so."""

    question: str = dspy.InputField()
    context: str = dspy.InputField(desc="Retrieved policy excerpts")
    answer: str = dspy.OutputField()


class ExtractCitations(dspy.Signature):  # type: ignore[misc]
    """List the source filenames from the context that directly support the answer."""

    context: str = dspy.InputField()
    answer: str = dspy.InputField()
    citations: str = dspy.OutputField(
        desc="Comma-separated source filenames, e.g. 'handbook.md, pto.md'"
    )


class PolicyRAG(dspy.Module):  # type: ignore[misc]
    def __init__(self, retriever: BaseRetriever) -> None:
        self.retriever = retriever
        self.reformulate = dspy.Predict(ReformulateQuery)
        self.generate = dspy.ChainOfThought(GenerateAnswer)
        self.cite = dspy.Predict(ExtractCitations)

    def forward(self, question: str) -> dspy.Prediction:  # type: ignore[override]
        reformulated = self.reformulate(question=question).reformulated_query

        loop = asyncio.new_event_loop()
        try:
            docs = loop.run_until_complete(self.retriever.ainvoke(reformulated))
        finally:
            loop.close()

        context = "\n\n".join(
            f"[source: {d.metadata.get('filename', 'unknown')}]\n{d.page_content}"
            for d in docs
        )
        pred = self.generate(question=question, context=context)
        citations = self.cite(context=context, answer=pred.answer).citations
        return dspy.Prediction(answer=pred.answer, context=context, citations=citations)
