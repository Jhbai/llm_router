import time
import json
import uuid
import asyncio
from typing import List, Optional, Union, Dict, Any
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from langchain_openai import ChatOpenAI
import uvicorn

API_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
API_TOKEN = r"""Bearer XXXXXX"""

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

log_lock = asyncio.Lock()

def _append_log_sync(messages: List[Dict[str, Any]]):
    try:
        with open("log.json", "a", encoding="utf-8") as f:
            f.write(json.dumps({"timestamp": time.time(), "messages": messages}, ensure_ascii=False) + "\n")
    except Exception:
        pass

async def log_messages_task(messages: List[Dict[str, Any]]):
    async with log_lock:
        await asyncio.to_thread(_append_log_sync, messages)

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Dict[str, Any]]
    temperature: Optional[float] = 0.7
    stream: Optional[bool] = False
    max_tokens: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    
    class Config:
        extra = "ignore"

@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{
            "id": "models/gemini-3.8-flash", 
            "object": "model", 
            "created": int(time.time()), 
            "owned_by": "google"
        }]
    }

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    asyncio.create_task(log_messages_task(req.messages))
    
    llm = ChatOpenAI(
        api_key=API_TOKEN,
        base_url=API_URL,
        model=req.model,
        temperature=req.temperature,
        max_tokens=req.max_tokens,
        stop=req.stop,
    )

    if req.stream:
        async def generate():
            req_id = f"chatcmpl-{uuid.uuid4()}"
            created = int(time.time())
            try:
                async for chunk in llm.astream(req.messages):
                    yield f"data: {json.dumps({'id': req_id, 'object': 'chat.completion.chunk', 'created': created, 'model': req.model, 'choices': [{'index': 0, 'delta': {'content': chunk.content}, 'finish_reason': None}]}, ensure_ascii=False)}\n\n"
                yield f"data: {json.dumps({'id': req_id, 'object': 'chat.completion.chunk', 'created': created, 'model': req.model, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]}, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
                
        return StreamingResponse(generate(), media_type="text/event-stream; charset=utf-8")
    
    try:
        response = await llm.ainvoke(req.messages)
        return {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.model,
            "choices": [{
                "index": 0, 
                "message": {"role": "assistant", "content": response.content}, 
                "finish_reason": "stop"
            }]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
