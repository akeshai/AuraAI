from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import json

app = FastAPI()

class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    async def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            print("Client disconnected")

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)  # Parse JSON from frontend

            # Handle events from frontend
            if message.get("event") == "message":
                print(f"Message received: {message['message']}")
                await manager.broadcast(f"Broadcast: {message['message']}")

            elif message.get("event") == "disconnect":
                print("Client requested disconnect")
                await manager.disconnect(websocket)
                await websocket.close(code=1000, reason="Client requested disconnect")
                break  # Exit the loop to close connection
                
    except WebSocketDisconnect:
        await manager.disconnect(websocket)
        print("Client disconnected abruptly")

@app.get("/")
async def get():
    return HTMLResponse("""
    <!DOCTYPE html>
    <html>
    <body>
        <h1>WebSocket Control via Python</h1>
        <button onclick="sendEvent('connect')">Connect</button>
        <button onclick="sendEvent('disconnect')">Disconnect</button>
        <input type="text" id="messageInput" placeholder="Type message">
        <button onclick="sendEvent('message')">Send Message</button>
        <div id="responses"></div>

        <script>
            let ws = null;

            function sendEvent(eventType) {
                const messageInput = document.getElementById("messageInput");
                const data = {
                    event: eventType,
                    message: eventType === "message" ? messageInput.value : null
                };

                if (eventType === "connect" && (!ws || ws.readyState === WebSocket.CLOSED)) {
                    ws = new WebSocket("ws://localhost:8000/ws");
                    ws.onmessage = (event) => {
                        document.getElementById("responses").innerHTML += `<p>${event.data}</p>`;
                    };
                } 
                else if (eventType === "disconnect" && ws && ws.readyState === WebSocket.OPEN) {
                    ws.send(JSON.stringify(data));
                } 
                else if (ws && ws.readyState === WebSocket.OPEN) {
                    ws.send(JSON.stringify(data));
                    if (eventType === "message") messageInput.value = "";
                }
            }
        </script>
    </body>
    </html>
    """)