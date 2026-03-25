import os

import pytest
import requests
from openai import OpenAI
from ragas import SingleTurnSample
from ragas.llms import llm_factory
from ragas.metrics import LLMContextPrecisionWithoutReference, LLMContextRecall, Faithfulness


@pytest.mark.asyncio
async def test_context_precision_without_reference():
    llm_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    #client = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    #wrapped_llm = LangchainLLMWrapper(client)
    wrapped_llm = llm_factory('gpt-4o-mini', client=llm_client)
    question = "How many articles are there in selenium python course?"
    response = requests.post(url="https://rahulshettyacademy.com/rag-llm/ask", json={
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
    context_precision = LLMContextPrecisionWithoutReference(llm=wrapped_llm)
    score = await context_precision.single_turn_ascore(sample)
    assert  score > 0.9


async def test_context_recall():
    llm_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    llm_wrapper = llm_factory('gpt-4o-mini', client=llm_client)
    question = "How many articles are there in selenium python course?"
    response = requests.post(url="https://rahulshettyacademy.com/rag-llm/ask", json={
        "question": question,
        "chat_history": [
        ]
    }
    ).json()
    retrieved_docs_length = len(response["retrieved_docs"])
    sample = SingleTurnSample(
        user_input=question,
        retrieved_contexts=[response["retrieved_docs"][i]["page_content"] for i in range(retrieved_docs_length)],
        reference="23"
    )
    context_recall = LLMContextRecall(llm=llm_wrapper)
    score = await context_recall.single_turn_ascore(sample)
    assert score > 0.9

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
    score = await faithfulness.single_turn_ascore(sample)
    assert score > 0.9
