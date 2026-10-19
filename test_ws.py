import asyncio
import websockets
import json

async def test_ws():
    uri = "ws://localhost:8000/ws/sign-language"
    async with websockets.connect(uri) as websocket:
        print("Connected.")
        
        for i in range(10):
            frame = {
                "type": "frame",
                "vector": [0.0]*126,
                "hands_visible": True,
                "lang": "vi"
            }
            await websocket.send(json.dumps(frame))
            print(f"Sent frame {i+1}")
            
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=0.5)
                print(f"Received: {response}")
            except asyncio.TimeoutError:
                pass

asyncio.run(test_ws())
