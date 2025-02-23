import os
from mistralai import Mistral

# Set up the Mistral chat completion client.
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "r6cJTcdQ93jseOPcR8jlSgWDC87PmREn")
MODEL_NAME = "mistral-large-latest" 
client = Mistral(api_key=MISTRAL_API_KEY)

def generate_answer(context: str, query: str) -> str:
    """
    Generates an answer using Mistral's chat completion API.
    If context is provided, it is included in the prompt.
    
    :param context: Retrieved context (may be empty for trivial queries).
    :param query: The original user query.
    :return: The generated answer.
    """
    if context:
        prompt = (
            f"Context information is below.\n"
            f"---------------------\n"
            f"{context}\n"
            f"---------------------\n"
            f"Given the context above, answer the following question:\n"
            f"{query}\n"
            f"Answer:"
        )
    else:
        prompt = query

    messages = [
        {"role": "user", "content": prompt}
    ]
    
    try:
        chat_response = client.chat.complete(
            model=MODEL_NAME,
            messages=messages
        )
        answer = chat_response.choices[0].message.content
        return answer
    except Exception as e:
        print(f"Error generating answer: {e}")
        return "Sorry, I could not generate an answer at this time."