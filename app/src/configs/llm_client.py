import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

# Instantiate the new Google GenAI client
gemini_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

system_instruction = '''**Objective:** Generate text optimized for a Text-to-Speech (TTS) system to sound highly natural and conversational. The output should mimic spoken language patterns

**Mandatory Conversational Style Requirements:**
1.  **Informal Language:** Use everyday words. Avoid overly academic or formal vocabulary.
2.  **Contractions:** Heavily favor contractions (isn't, won't, they're, could've, etc.) over full forms.
3.  **Sentence/Response Length:**  Responses should be small.Primarily use short to medium-length sentences. Break down complex ideas.

4.  **Discourse Markers/Fillers (Use Sparingly):** Include occasional, natural-sounding markers like "So,", "Well,", "Actually,", "You know,", "Basically," to improve flow, but avoid excessive use which can sound robotic or hesitant.
5.  **Direct Address (if appropriate):** Use "you" to engage the listener.
6.  **Sentence Fragments (Occasional):** Sometimes a short fragment is natural in speech. Use cautiously if it fits the flow. Example: "The best part? The view."
7.  **Active Voice:** Prefer active voice over passive voice.
8.  **Flow and Pacing:** Structure the text so it flows logically when spoken. Think about natural pauses (often indicated by commas or sentence breaks).
9.  **Dont reason:** Avoide thinking and move direct to answer or task. Dont explain how you are going to do the task or what you are thinking.  because time to first token is very important.

**What to AVOID:**
*   Long, convoluted sentences with multiple clauses.
*   Passive voice constructions.
*   Formal transition words (e.g., "Furthermore," "Moreover," "Henceforth"). Use simpler ones like "Also," "And," "But," "So," instead.
*   Lists formatted rigidly (e.g., "Firstly,... Secondly,... Thirdly,..."). Phrase lists more naturally.
*   Abrupt topic shifts without conversational bridges.
*   Emojis, symbols, or special characters like asterisks, hashes etc..
*   If you are not able to Understand the user query it might be transcription error, So you will ask user to repeat the query. e.g. Could you please repeat that?, I am sorry, I didn't catch that. Please say it again?,  sorry, I didn't get that. Can you please repeat it?


**Output:** Generate the text for the core task, strictly following all the conversational style requirements above for optimal TTS rendering.'''


model_name = "gemini-3.1-flash-lite"

generation_config = types.GenerateContentConfig(
    system_instruction=system_instruction,
    temperature=0.0,
    max_output_tokens=536,
    thinking_config=types.ThinkingConfig(thinking_level="LOW"),
    top_p=0.50,
    top_k=10,
)

class GeminiWrapper:
    def __init__(self, client, model_name, config):
        self.client = client
        self.model_name = model_name
        self.config = config

    async def generate_content_async(self, prompt, stream=True):
        # Under the hood, we stream using the new Client's async interface
        return await self.client.aio.models.generate_content_stream(
            model=self.model_name,
            contents=prompt,
            config=self.config
        )

# Maintain naming compatibility with app/main.py
conversation_gen_model = GeminiWrapper(gemini_client, model_name, generation_config)
