import os
import asyncio

import pytest
from openai import AsyncOpenAI
from ragas.embeddings.base import embedding_factory
from ragas.llms import llm_factory
from ragas.metrics.collections import AnswerRelevancy

import RequestsUtils

@pytest.mark.asyncio
async def test_evaluate_metrics_altogether():
    # Use AsyncOpenAI for async usage
    client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])

    # Create ragas LLM + embeddings with the async client
    wrapped_llm = llm_factory("gpt-4o-mini", client=client)
    embeddings = embedding_factory("openai", model="text-embedding-3-small", client=client)

    # Use the collections metric
    answer_relevancy = AnswerRelevancy(llm=wrapped_llm, embeddings=embeddings)

    # If RequestsUtils.call_... is blocking (likely), run it in a thread so we don't block the event loop
    response = await asyncio.to_thread(RequestsUtils.call_rahulshetty_rag_system_with_no_history)

    # Call metric.ascore -> returns MetricResult, use .value and convert to float
    result = await answer_relevancy.ascore(
        user_input="How many articles are there in selenium python course?",
        response=response["answer"],
    )

    score = float(result.value)
    print(f"Response Relevancy Score: {score}")

    assert score > 0.8
