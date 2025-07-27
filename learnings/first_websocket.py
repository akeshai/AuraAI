from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse



app = FastAPI()

html = """
<!DOCTYPE html>
<html>
<head>
    <title>Chat</title>
</head>
<body>
    <h1>WebSocket with FastAPI</h1>
    <form action="" onsubmit="sendMessage(event)">
        <input type="text" id="messageText" autocomplete="off" />
        <button>Send</button>
    </form>
    <ul id='messages'>
    </ul>
<script>
    var ws = new WebSocket(`ws://localhost:8000/communicate`);
    
    ws.onmessage = function(event) {
        var messages = document.getElementById('messages')
        var message = document.createElement('li')
        var content = document.createTextNode(event.data)
        message.appendChild(content)
        messages.appendChild(message)
    };

    function sendMessage(event) {
        var input = document.getElementById("messageText")
        ws.send(input.value)
        input.value = ''
        event.preventDefault()
    }


</script>


</body>
</html>
"""

'''    // Function to manually disconnect
    function disconnectWebSocket() {
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.close(); // Close WebSocket connection
            console.log("Disconnected from WebSocket");
        }
    }
    <!-- Add a disconnect button -->
    <button onclick="disconnectWebSocket()">Disconnect</button>
    
    '''

@app.get("/")
async def get():
    return HTMLResponse(html)

class ConnectionManager:
    """Class defining socket events"""
    def __init__(self):
        """init method, keeping track of connections"""
        self.active_connections = []
    
    async def connect(self, websocket: WebSocket):
        """connect event"""
        await websocket.accept()
        self.active_connections.append(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        """Direct Message"""
        await websocket.send_text(message)
    
    def disconnect(self, websocket: WebSocket):
        """disconnect event"""
        print("Disconnecting")
        self.active_connections.remove(websocket)


manager = ConnectionManager()

@app.websocket("/communicate")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    while True:
        data = await websocket.receive_text()
        print('Data',data)
        if data == "Bye":
            break
        # await manager.send_personal_message("Hello",websocket)
        await manager.send_personal_message(f"Received:{data}",websocket)
    await manager.send_personal_message("Bye!!!",websocket)
    manager.disconnect(websocket)

