import requests


def call_rahulshetty_rag_system_with_no_history():
    response = requests.post(url="https://rahulshettyacademy.com/rag-llm/ask",json={
        "question": "How many articles are there in selenium python course?",
        "chat_history": [
        ]
    }).json()
    return response