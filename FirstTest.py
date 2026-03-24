import os

import pytest
from instructor.cli.jobs import client
from langchain_openai import ChatOpenAI
from openai import OpenAI
from ragas.llms import llm_factory, LangchainLLMWrapper
from ragas import SingleTurnSample
from ragas.metrics import LLMContextPrecisionWithoutReference


@pytest.mark.asyncio
async def test_context_precision_without_reference():
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    #client = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    #wrapped_llm = LangchainLLMWrapper(client)
    wrapped_llm = llm_factory('gpt-4o-mini', client=client)
    sample = SingleTurnSample(
        user_input="How many articles are there in selenium python course?",
        response="There are 23 articles in the course",
        retrieved_contexts=["Complete Understanding on Selenium Python API Methods with real time Scenarios on LIVE "
                            "Websites\n\"Last but not least\" you can clear any Interview and can Lead Entire "
                            "Selenium Python Projects from Design Stage\nThis course includes:\n17.5 hours on-demand "
                            "video\nAssignments\n23 articles\n9 downloadable resources\nAccess on mobile and "
                            "TV\nCertificate of completion\nRequirements",
                            "What you'll learn\nAt the end of this course, You will get complete knowledge on Python "
                            "Automation using Selenium WebDriver\nYou will be able to implement Python Test "
                            "Automation Frameworks from Scratch with all latest Technlogies\nComplete Understanding "
                            "of Python Basics with many practise Examples to gain a solid exposure\nYou will be "
                            "learning Python Unit Test Frameworks like PyTest which will helpful for Unit and "
                            "Integration Testing,"
                            "Wish you all the Best! See you all in the course with above topics :)\n\nWho this course "
                            "is for:\nAutomation Engineers\nSoftware Engineers\nManual testers\nSoftware developers",
                            "So what makes this course Unique in the Market?\nWe assume that students have no "
                            "experience in automation / coding and start every topic from scratch and "
                            "basics.\nExamples are taken from  REAL TIME HOSTED WEB APPLICATIONS  to understand how "
                            "different components can be automated.\n  Topics includes: \nPython Basics\nPython "
                            "Programming examples\nPython Data types\nPython OOPS Examples\nSelenium "
                            "Locators\nSelenium Multi Browser Execution\nPython Selenium API Methods\nAdvanced "
                            "Selenium User interactions"]
    )
    context_precision = LLMContextPrecisionWithoutReference(llm=wrapped_llm)
    score = await context_precision.single_turn_ascore(sample)
    print(score)
