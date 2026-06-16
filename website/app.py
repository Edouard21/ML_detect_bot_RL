"""RL Bot Detector backend for Hugging Face Spaces.

This API only exposes backend endpoints and is intended to be called through
the Vercel serverless proxy (/api/analyze).
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import asyncio
import os
import shutil
import tempfile

from analyzer_utils import extract_players_from_replay
from eval_player_game import load_inference_model, evaluate_csv

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------
CKPT_PATH = "best.pt"
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB
MAX_CONCURRENT_JOBS = max(1, int(os.environ.get("MAX_CONCURRENT_JOBS", "2")))
QUEUE_WAIT_SECONDS = max(1, int(os.environ.get("QUEUE_WAIT_SECONDS", "8")))
PROCESSING_TIMEOUT_SECONDS = max(10, int(os.environ.get("PROCESSING_TIMEOUT_SECONDS", "180")))

model_glob = None
meta_glob = None
mean_glob = None
std_glob = None
device_glob = None
inference_semaphore = asyncio.Semaphore(MAX_CONCURRENT_JOBS)


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_glob, meta_glob, mean_glob, std_glob, device_glob
    try:
        model_glob, meta_glob, mean_glob, std_glob, device_glob = load_inference_model(CKPT_PATH)
        print(f"✅ Model loaded from {CKPT_PATH}")
    except Exception as e:
        print(f"⚠️  Could not load model: {e}")
    yield


app = FastAPI(title="RL Bot Detector API", lifespan=lifespan)

allowed_origin = os.environ.get("ALLOWED_ORIGIN", "http://localhost:3000")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[allowed_origin],
    allow_credentials=False,
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["Content-Type", "X-API-Key"],
)


# ---------------------------------------------------------------------------
# API endpoint
# ---------------------------------------------------------------------------
async def verify_api_key(x_api_key: str = Header(None)):
    expected_key = os.environ.get("HF_API_KEY")
    if not expected_key:
        raise HTTPException(status_code=500, detail="Server misconfiguration: missing HF_API_KEY.")
    if x_api_key != expected_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API Key")

@app.post("/analyze", dependencies=[Depends(verify_api_key)])
async def analyze_replay(file: UploadFile = File(...)):
    """Upload a .replay file and get bot/real predictions for each player."""

    if model_glob is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please try again later.")

    if not file.filename or not file.filename.lower().endswith(".replay"):
        raise HTTPException(status_code=400, detail="Only .replay files are accepted.")

    # Create a unique temporary directory for this request
    tmp_dir = tempfile.mkdtemp(prefix="rl_analyze_")
    replay_path = os.path.join(tmp_dir, file.filename)
    csv_dir = os.path.join(tmp_dir, "csvs")
    os.makedirs(csv_dir, exist_ok=True)
    semaphore_acquired = False

    try:
        # Stream upload to disk and enforce max size without loading full file in memory.
        total_size = 0
        with open(replay_path, "wb") as f:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                total_size += len(chunk)
                if total_size > MAX_FILE_SIZE:
                    raise HTTPException(status_code=413, detail=f"File too large. Maximum size is {MAX_FILE_SIZE // (1024*1024)} MB.")
                f.write(chunk)

        # Bound backend pressure: only a small number of analyses run concurrently.
        try:
            await asyncio.wait_for(inference_semaphore.acquire(), timeout=QUEUE_WAIT_SECONDS)
            semaphore_acquired = True
        except asyncio.TimeoutError:
            raise HTTPException(status_code=429, detail="Server is busy. Please retry in a few seconds.")

        def run_inference_pipeline():
            extracted_local = extract_players_from_replay(replay_path, csv_dir)
            local_results = []
            for player in extracted_local:
                csv_path = player["csv_path"]
                name = player["nom_joueur"]
                label, conf = evaluate_csv(csv_path, model_glob, meta_glob, mean_glob, std_glob, device_glob)
                local_results.append({
                    "name": name,
                    "prediction": label,
                    "confidence": round(conf * 100, 2),
                })
            return local_results

        try:
            results = await asyncio.wait_for(
                asyncio.to_thread(run_inference_pipeline),
                timeout=PROCESSING_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="Analysis timeout. Please try another replay.")

        return {"replay_name": file.filename, "players": results}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing replay: {str(e)}")
    finally:
        if semaphore_acquired:
            inference_semaphore.release()
        # Always clean up temp files
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
