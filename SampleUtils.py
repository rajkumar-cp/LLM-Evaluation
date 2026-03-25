from ragas import SingleTurnSample

import RequestsUtils


def create_single_turn_sample():
    response = RequestsUtils.call_rahulshetty_rag_system_with_no_history()
    sample = SingleTurnSample(
        user_input="How many articles are there in selenium python course?",
        response=response["answer"],
        retrieved_contexts=[response["retrieved_docs"][i]["page_content"] for i in range(len(response["retrieved_docs"]))],
        reference="There are 23 articles in this course."
    )
    return sample