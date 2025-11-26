# To run this code you need to install the following dependencies:
# pip install google-genai

import base64
import os
from google import genai
from google.generativeai import types
from dotenv import load_dotenv
load_dotenv()

def generate():
    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )

    model = "gemini-flash-lite-latest"
    print (client.models.generate_content(model=model,contents = 'HIe'))
generate()