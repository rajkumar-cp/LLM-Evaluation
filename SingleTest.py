import os

import pytest
from openai import OpenAI
from ragas import SingleTurnSample
from ragas.embeddings.base import embedding_factory
from ragas.llms import llm_factory
from ragas.metrics.collections import AnswerRelevancy

import RequestsUtils

@pytest.mark.asyncio
async def test_evaluate_metrics_altogether():
    llm_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    wrapped_llm = llm_factory('gpt-4o-mini', client=llm_client)
    embeddings = embedding_factory("openai", model="text-embedding-3-small", client=llm_client)
    answer_relevancy = AnswerRelevancy(llm=wrapped_llm, embeddings=embeddings)
    response = RequestsUtils.call_rahulshetty_rag_system_with_no_history()
    sample = {
        "user_input": "How many articles are there in selenium python course?",
    "response" : response["answer"]
    }
    score = await answer_relevancy.ascore(user_input= sample.get("user_input"), response=sample.get("response"))
    print(f"Response Relevancy Score: {score}")
    assert score > 0.8
