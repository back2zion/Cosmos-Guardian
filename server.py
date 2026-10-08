from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
import shutil
import os
from pathlib import Path
import json
try:
    from cosmos_guardian.core.agent import CosmosGuardianAgent
except ImportError:
    from core.agent import CosmosGuardianAgent

# Initialize Agent
agent = CosmosGuardianAgent()

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model on startup with fine-tuned LoRA adapter
    from core.agent import DEFAULT_ADAPTER_PATH
    print("Pre-loading Cosmos Reason 2 model into VRAM...")
    agent.load_model(adapter_path=DEFAULT_ADAPTER_PATH)
    yield
    # Clean up if needed
    print("Shutting down...")

app = FastAPI(title="Cosmos Guardian API", lifespan=lifespan)

# Enable CORS for React
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

from fastapi.responses import StreamingResponse
import asyncio

@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    context: str = Form("General Safety")
):
    file_path = UPLOAD_DIR / file.filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    async def event_generator():
        from anyio import to_thread
        # Create the generator
        generator = agent.analyze_media(str(file_path.absolute()), safety_context=context)
        
        while True:
            try:
                # Execute next(generator) in a thread to keep FastAPI reactive
                update = await to_thread.run_sync(next, generator)
                yield f"data: {json.dumps(update)}\n\n"
                # Small sleep to ensure the stream feels smooth on the frontend
                await asyncio.sleep(0.05)
            except StopIteration:
                break
            except Exception as e:
                print(f"Streaming error: {e}")
                yield f"data: {json.dumps({'stage': 'error', 'detail': str(e)})}\n\n"
                break

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/health")
async def health():
    return {"status": "ok", "model": agent.model_id}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8888)
