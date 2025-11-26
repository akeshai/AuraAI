# Project Memory: Audio2Audio Conversational AI

This document serves as a comprehensive log of the development process for the Audio2Audio conversational AI application, detailing the initial state, problems encountered, solutions implemented, and potential future work.

## Project Context

The Audio2Audio project aims to create a real-time conversational AI system. It leverages FastAPI for the backend, WebSockets for audio streaming, `sherpa-onnx` for Automatic Speech Recognition (ASR), `kokoro` for Text-to-Speech (TTS), and Google's Generative AI (Gemini) for conversational responses. The frontend is a simple HTML page with JavaScript for audio capture, visualization, and playback.

## Initial State

Upon starting the analysis, the `app` folder contained the core FastAPI application, with separate modules for configurations, speech generation (TTS), transcription (ASR and VAD), and utility functions. The frontend (`static/index_speech_to_text.html`) was set up for basic audio capture and WebSocket communication.

## Problems Encountered and Solutions Implemented

The development process involved addressing several issues to improve the robustness, responsiveness, and naturalness of the conversational flow.

### 1. Initial Code Analysis and File Access

*   **Problem:** Initial attempts to analyze the `app` folder using `read_many_files` failed to locate files.
*   **Solution:** Used `glob` to correctly identify all Python files within the `app` directory structure, then explicitly listed these paths for `read_many_files` to ensure all relevant code was read for analysis.

### 2. Voice Activity Detection (VAD) Robustness and Errors

*   **Problem:** Encountered `webrtcvad.Error: Error while processing frame`. This indicated that the audio chunks being fed to `webrtcvad` did not meet its strict requirements for frame size and sample rate. Additionally, the VAD was not robust enough in detecting speech reliably across varying conditions.
*   **Solution (VAD Function Simplification):**
    *   **Change:** Refactored `app/src/transcription/voice_activity_detection.py`. The `detect_speech_in_chunk` function was renamed to `is_speech` and simplified to only perform the `webrtcvad` check, assuming the input audio is already 16kHz, 16-bit PCM. Resampling logic was removed from this function.
    *   **Rationale:** Decoupled VAD logic from audio preprocessing, making the VAD function more focused and efficient.
*   **Solution (VAD Framing in `app/main.py`):**
    *   **Change:** Implemented explicit audio framing in `app/main.py` within the `websocket_audio_endpoint`. Incoming 44.1kHz audio is now resampled to 16kHz, converted to `int16`, and then segmented into 30ms frames before being passed to `is_speech`.
    *   **Rationale:** Ensured `webrtcvad` receives correctly formatted audio frames, resolving the `webrtcvad.Error`.
*   **Solution (Energy-Based VAD Enhancement):**
    *   **Change:** Combined the `webrtcvad` output with an energy-based threshold (`AUDIO_ENERGY_SPEAKING_THRESHOLD`) in `app/main.py`. `speech_detected` is now `True` if either `webrtcvad` detects speech OR the audio energy exceeds the threshold.
    *   **Rationale:** Increased VAD robustness by accounting for both acoustic features (from `webrtcvad`) and loudness, making it more resilient to quiet speech or noisy environments.

### 3. Utterance Segmentation and Short Utterances

*   **Problem:** The system was either cutting off user utterances prematurely or discarding short, valid phrases. This was due to overly strict `max_allowed_silence` and `min_allowed_speaking` parameters.
*   **Solution (Parameter Tuning):**
    *   **Change:** Adjusted `max_allowed_silence` from 10 to 20 chunks and `min_allowed_speaking` from 30 to 10 chunks in `app/main.py`.
    *   **Rationale:** Provided more natural pauses within an utterance and allowed shorter, but meaningful, user inputs to be processed.
*   **Solution (Correct `speaking_chunk_count` Implementation):**
    *   **Change:** Correctly initialized `speaking_chunk_count = 0` at the start of the WebSocket handler, incremented it within the `if speech_detected:` block, and reset it after a turn was processed (or discarded). A check was added to ensure `speaking_chunk_count` meets `min_allowed_speaking` before proceeding with transcription.
    *   **Rationale:** Ensured the `min_allowed_speaking` logic functions as intended, preventing very brief, unintentional sounds from triggering a full AI response.

### 4. Missing Last Words of Transcription

*   **Problem:** The ASR model was sometimes cutting off the last few words of a user's utterance, leading to incomplete transcriptions. This happened because the system detected the end of speech too quickly.
*   **Solution (Post-Speech Buffer):**
    *   **Change:** Implemented a `collections.deque` named `post_speech_buffer` in `app/main.py` with a `maxlen` of 10 chunks. All incoming `raw_audio` chunks are appended to this buffer. When `silence_chunk_count` exceeds `max_allowed_silence`, the contents of this `post_speech_buffer` are fed to the ASR stream *before* `stream_state.input_finished()` is called. The buffer is cleared after each turn.
    *   **Rationale:** Allowed for a small "tail" of audio to be captured and processed even after the primary speech detection ends, ensuring trailing words are included in the final transcription.

### 5. Real-time Transcription Display on Frontend

*   **Problem:** Interim transcription results were not being displayed in real-time on the frontend, only the final transcription appeared after a pause.
*   **Solution (Backend - Frequent Interim Sending):**
    *   **Change:** Modified `app/main.py` to retrieve and send `interim_text` from the ASR stream *after every audio chunk is processed* (within the `if speech_detected:` block), regardless of whether `recognizer_en.is_ready` was true. Only non-empty `interim_text` is sent.
    *   **Rationale:** Ensured the backend pushes updates as frequently as the ASR model can provide them, even if they are minor or partial.
*   **Solution (Frontend - Interim Message Handling):**
    *   **Change:** Modified `static/index_speech_to_text.html` to include an `else if (data.type === 'interim_transcription')` block in the WebSocket `onmessage` handler. This block updates the `transcriptElement.textContent` with the incoming interim text. The `llmResponseElement` is also cleared when a final `transcription` message arrives.
    *   **Rationale:** Enabled the frontend to receive and display the frequent interim updates, providing a more dynamic and real-time user experience.

### 6. Interruption Mechanism (Barge-in) Not Working

*   **Problem:** The AI's audio playback was not stopping when the user interrupted, leading to overlapping speech.
*   **Solution (Backend - Cancellation Logging):**
    *   **Change:** Added debug logs in `app/main.py` to confirm when `ai_audio_task.cancel()` is called and to verify if the `asyncio.CancelledError` is caught within the `send_ai_audio` function. Also added a log within the `send_ai_audio` loop to check if the task is still attempting to send data after being cancelled.
    *   **Rationale:** Provided crucial visibility into the backend's cancellation process to confirm if the signal is being sent and received by the task.
*   **Solution (Frontend - Separate AudioContexts):**
    *   **Change:** Refactored `static/index_speech_to_text.html` to use two distinct `AudioContext` variables: `playbackAudioContext` for playing AI responses and `recordingAudioContext` for capturing user audio. All relevant functions (`getPlaybackAudioContext`, `startRecording`, `stopRecording`, `playAudioQueue`, `stopAudioPlayback`) were updated to use the correct context.
    *   **Rationale:** This was the critical fix. The previous single `audioContext` was being closed by `stopRecording` (for user mic) or `stopAudioPlayback` (for AI voice), but if the *other* operation was active, its context remained open and continued playing/recording. Separating them ensures that closing one does not affect the other, allowing for proper interruption.
*   **Solution (Frontend - Misplaced Function Definition):**
    *   **Problem:** The `stopAudioPlayback` function was inadvertently defined outside the main `<script>` tags in `static/index_speech_to_text.html`, causing its code to be displayed on the webpage.
    *   **Solution:** Manually removed the misplaced `stopAudioPlayback` function definition from outside the `<script>` tags. (This was a manual step performed by the user based on my instruction).
    *   **Rationale:** Corrected a display bug and ensured the JavaScript is syntactically correct and properly encapsulated. **(Verified as corrected)**

## Current State

The application now features:
*   More robust Voice Activity Detection, combining `webrtcvad` with energy thresholds.
*   Improved utterance segmentation, handling natural pauses and short phrases better.
*   Real-time interim transcription display on the frontend.
*   A functional "barge-in" mechanism, where the AI's speech is interrupted when the user starts speaking, achieved by separating recording and playback `AudioContext` instances on the frontend.
*   Enhanced logging on both backend and frontend for better debugging.

## Future Work / Next Steps

1.  **Further VAD Refinement:**
    *   **Adaptive Thresholds:** Implement dynamic adjustment of `AUDIO_ENERGY_SPEAKING_THRESHOLD` based on ambient noise levels.
    *   **More Advanced VAD Models:** Explore integrating more sophisticated VAD models (e.g., Silero VAD) if `webrtcvad` and energy thresholds prove insufficient in very challenging acoustic environments.
2.  **LLM Response Streaming:**
    *   Currently, the LLM generates the full response text before TTS begins. Implement token-by-token streaming from the LLM to the TTS, and then to the client, to reduce perceived latency for AI responses.
3.  **Client-Side Audio Processing Improvements:**
    *   **Web Audio API for Recording:** Transition from `ScriptProcessorNode` (deprecated) to `AudioWorkletNode` for more performant and flexible client-side audio processing.
    *   **Noise Reduction:** Implement client-side noise reduction to improve ASR accuracy in noisy environments.
4.  **Error Handling and Resilience:**
    *   Implement more comprehensive error handling for network drops, model loading failures, and unexpected data formats.
    *   Add retry mechanisms for WebSocket connections or API calls.
5.  **UI/UX Enhancements:**
    *   **Visual Feedback:** Provide more intuitive visual feedback for VAD (e.g., a "speaking" indicator).
    *   **Status Messages:** Refine status messages to be more user-friendly and informative.
    *   **Settings:** Allow users to adjust VAD parameters or microphone sensitivity from the UI.
6.  **Backend Scalability:**
    *   For production, consider using a message queue (e.g., RabbitMQ, Kafka) for handling audio chunks and ASR/TTS requests to improve scalability and resilience.
    *   Explore deploying ASR/TTS models as separate microservices.
