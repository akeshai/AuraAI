# To run this code you need to install the following dependencies:
# pip install google-genai

import os
from google import genai
from google.genai import types

from dotenv import load_dotenv
load_dotenv()
api_key=os.environ.get("GEMINI_API_KEY")
print(api_key)
def generate():
    client = genai.Client(
        api_key=api_key,
    )

    model = "gemini-3.1-flash-lite"
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text="""INSERT_INPUT_HERE"""),
            ],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(
            thinking_level="MINIMAL",
        ),
    )

    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        if text := chunk.text:
            print(text, end="")

if __name__ == "__main__":
    generate()


