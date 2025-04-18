import asyncio
import threading
import pyaudio
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import queue

app = FastAPI()

# Audio Configuration
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024

class AudioBridge:
    def __init__(self):
        self.active_connections = set()
        self.input_queue = queue.Queue()  # From clients to broadcast
        self.output_queues: dict[WebSocket, queue.Queue] = {}  # Individual queues per client
        self.pyaudio = pyaudio.PyAudio()
        self.lock = threading.Lock()
        
    async def register(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)
        self.output_queues[websocket] = queue.Queue()
        return websocket

    async def unregister(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        del self.output_queues[websocket]

    def _process_audio(self, data: bytes) -> bytes:
        """Basic audio processing (noise gate example)"""
        audio_data = np.frombuffer(data, dtype=np.int16)
        # Simple noise gate - mute quiet sounds
        if np.max(np.abs(audio_data)) < 500:
            return np.zeros_like(audio_data).tobytes()
        return audio_data.tobytes()

    async def handle_incoming(self, websocket: WebSocket):
        """Receive audio from client and broadcast to others"""
        try:
            while True:
                data = await websocket.receive_bytes()
                processed = self._process_audio(data)
                # Broadcast to all other clients
                for conn in self.active_connections:
                    if conn != websocket:
                        self.output_queues[conn].put(processed)
                        
        except WebSocketDisconnect:
            await self.unregister(websocket)

    # async def handle_outgoing(self, websocket: WebSocket):
    #     """Send audio to client from their output queue"""
    #     try:
    #         while True:
    #             # Get audio data from queue (blocking)
    #             data = await asyncio.get_event_loop().run_in_executor(
    #                 None,
    #                 self.output_queues[websocket].get
    #             )
    #             await websocket.send_bytes(data)
    #     except Exception as e:
    #         print(f"Output error: {e}")

bridge = AudioBridge()

@app.websocket("/ws/audio")
async def websocket_endpoint(websocket: WebSocket):
    ws = await bridge.register(websocket)
    try:
        await asyncio.gather(
            bridge.handle_incoming(ws),
            # bridge.handle_outgoing(ws)
        )
    except Exception as e:
        await bridge.unregister(ws)
        print(f"Connection error: {e}")

@app.get("/")
async def get():
    return HTMLResponse("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Two-Way Audio</title>
        <style>
            body { font-family: Arial; max-width: 600px; margin: 0 auto; padding: 20px; }
            button { padding: 10px; margin: 5px; }
            #status { margin: 10px 0; padding: 10px; background: #f0f0f0; }
        </style>
    </head>
    <body>
        <h1>Two-Way Audio Communication</h1>
        <div id="status">Disconnected</div>
        <button id="connectBtn">Connect</button>
        <button id="disconnectBtn" disabled>Disconnect</button>
        <button id="startMicBtn" disabled>Start Microphone</button>
        <button id="stopMicBtn" disabled>Stop Microphone</button>
        
        <script>
            let ws;
            let audioContext;
            let source;
            let processor;
            let mediaStream;
            const statusEl = document.getElementById('status');
            
            // WebSocket Connection
            document.getElementById('connectBtn').addEventListener('click', connect);
            document.getElementById('disconnectBtn').addEventListener('click', disconnect);
            
            // Microphone Control
            document.getElementById('startMicBtn').addEventListener('click', startMicrophone);
            document.getElementById('stopMicBtn').addEventListener('click', stopMicrophone);
            
            function updateStatus(text) {
                statusEl.textContent = text;
                console.log(text);
            }
            
            async function connect() {
                ws = new WebSocket(`ws://${window.location.host}/ws/audio`);
                
                ws.onopen = () => {
                    updateStatus("Connected");
                    document.getElementById('connectBtn').disabled = true;
                    document.getElementById('disconnectBtn').disabled = false;
                    document.getElementById('startMicBtn').disabled = false;
                };
                
                ws.onmessage = async (event) => {
                    if (!audioContext) {
                        audioContext = new AudioContext();
                        await audioContext.resume();
                    }
                    
                    const audioData = new Int16Array(await event.data.arrayBuffer());
                    const float32 = new Float32Array(audioData.length);
                    for (let i = 0; i < audioData.length; i++) {
                        float32[i] = audioData[i] / 32768.0;
                    }
                    
                    if (source) source.disconnect();
                    source = audioContext.createBufferSource();
                    const buffer = audioContext.createBuffer(1, float32.length, 44100);
                    buffer.getChannelData(0).set(float32);
                    source.buffer = buffer;
                    source.connect(audioContext.destination);
                    source.start();
                };
                
                ws.onclose = () => {
                    updateStatus("Disconnected");
                    resetUI();
                    stopMicrophone();
                };
                
                ws.onerror = (error) => {
                    updateStatus(`Error: ${error.message}`);
                    resetUI();
                };
            }
            
            function disconnect() {
                if (ws) ws.close();
            }
            
            async function startMicrophone() {
                try {
                    mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
                    audioContext = new AudioContext();
                    const input = audioContext.createMediaStreamSource(mediaStream);
                    
                    processor = audioContext.createScriptProcessor(4096, 1, 1);
                    processor.onaudioprocess = (e) => {
                        if (ws && ws.readyState === WebSocket.OPEN) {
                            const audioData = e.inputBuffer.getChannelData(0);
                            const int16 = new Int16Array(audioData.length);
                            for (let i = 0; i < audioData.length; i++) {
                                int16[i] = Math.min(32767, Math.max(-32768, audioData[i] * 32768));
                            }
                            ws.send(int16.buffer);
                        }
                    };
                    
                    input.connect(processor);
                    processor.connect(audioContext.destination);
                    
                    updateStatus("Microphone Active");
                    document.getElementById('startMicBtn').disabled = true;
                    document.getElementById('stopMicBtn').disabled = false;
                } catch (err) {
                    updateStatus(`Microphone Error: ${err.message}`);
                }
            }
            
            function stopMicrophone() {
                if (mediaStream) {
                    mediaStream.getTracks().forEach(track => track.stop());
                    mediaStream = null;
                }
                if (processor) {
                    processor.disconnect();
                    processor = null;
                }
                updateStatus("Microphone Inactive");
                document.getElementById('startMicBtn').disabled = false;
                document.getElementById('stopMicBtn').disabled = true;
            }
            
            function resetUI() {
                document.getElementById('connectBtn').disabled = false;
                document.getElementById('disconnectBtn').disabled = true;
                document.getElementById('startMicBtn').disabled = true;
                document.getElementById('stopMicBtn').disabled = true;
            }
        </script>
    </body>
    </html>
    """)