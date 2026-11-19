import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("NAVIGATELAB_API_KEY"),
    base_url="https://nexusapi.navigatelabs.ai"
)

def generate_answer(prompt):
    response = client.chat.completions.create(
        model="nova-micro",
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=500
    )

    return response.choices[0].message.content