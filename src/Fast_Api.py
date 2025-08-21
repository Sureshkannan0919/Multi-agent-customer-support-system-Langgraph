from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware
import random

app = FastAPI(title="FastAPI Chatbot for Next.js")

# Enable CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://192.168.1.8:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Pydantic model for request and response
class ChatRequest(BaseModel):
    message: str
    userId: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    user_id: Optional[str] = None

# Simulated conversational AI (replace with real LLM if needed)
def generate_response(message: str) -> str:
    responses = [
        f"Got it! You said: {message}. What's next?",
        "Interesting point! Tell me more about that.",
        f"You mentioned '{message}'. Want to dive deeper?",
        "Cool, let's keep chatting! What's up?"
    ]
    return random.choice(responses)

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    # Generate response
    bot_response = generate_response(request.message)
    print(f"User ID: {request.userId}")
    
    return ChatResponse(response=bot_response, user_id=request.userId)

@app.get("/")
async def root():
    return {"message": "FastAPI Chatbot for Next.js. Use POST /chat to interact."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)