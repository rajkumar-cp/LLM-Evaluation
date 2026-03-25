import asyncio
import os

import pytest
import requests
from openai import OpenAI, AsyncOpenAI
from ragas import SingleTurnSample
from ragas.llms import llm_factory
from ragas.metrics.collections import ContextPrecisionWithoutReference, ContextRecall, Faithfulness

import RequestsUtils


@pytest.mark.asyncio
async def test_context_precision_without_reference():
    # Initialize the async OpenAI client
    llm_client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    wrapped_llm = llm_factory('gpt-4o-mini', client=llm_client)

    # Define the question and make the RAG system request
    question = "How many articles are there in selenium python course?"
    response_data = await asyncio.to_thread(RequestsUtils.call_rahulshetty_rag_system_with_no_history)

    # Extract the retrieved contexts from the response
    retrieved_contexts = [doc["page_content"] for doc in response_data["retrieved_docs"]]

    # Initialize the ContextPrecisionWithoutReference metric
    context_precision = ContextPrecisionWithoutReference(llm=wrapped_llm)

    # Call the ascore method with keyword arguments
    result = await context_precision.ascore(
        user_input=question,
        response=response_data["answer"],
        retrieved_contexts=retrieved_contexts
    )

    # Extract the score and assert
    score = float(result.value)  # MetricResult.value is a float
    print(f"Context Precision Without Reference Score: {score}")
    assert score > 0.9


async def test_context_recall():
    llm_client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    llm_wrapper = llm_factory('gpt-4o-mini', client=llm_client)
    question = "How many articles are there in selenium python course?"
    response = RequestsUtils.call_rahulshetty_rag_system_with_no_history();
    retrieved_contexts = [doc["page_content"] for doc in response["retrieved_docs"]]
    context_recall = ContextRecall(llm=llm_wrapper)
    score = await context_recall.ascore(user_input=question,retrieved_contexts=retrieved_contexts,  reference="23")
    assert score.value > 0.9

async  def test_faithfulness():
    llm = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    wrapped_llm = llm_factory('gpt-4o-mini', client=llm)
    question = "How many articles are there in selenium python course?"
    response = requests.post(url="https://rahulshettyacademy.com/rag-llm/ask",json={
        "question": question,
        "chat_history": [
        ]
    }
    ).json()
    retrieved_docs_length = len(response["retrieved_docs"])
    sample = SingleTurnSample(
        user_input=question,
        response=response["answer"],
        retrieved_contexts=[response["retrieved_docs"][i]["page_content"] for i in range(retrieved_docs_length)]
    )
    faithfulness = Faithfulness(llm=wrapped_llm)
    score = await faithfulness.ascore(sample)
    assert score > 0.9
