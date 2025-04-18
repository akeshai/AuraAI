from fastapi import WebSocket
import json

class ConnectionManager:
    def __init__(self):
        self.active_connections = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_message(self, websocket: WebSocket, message_type: str, text: str):
        message = {"type": message_type, "text": text}
        await websocket.send_text(json.dumps(message))
