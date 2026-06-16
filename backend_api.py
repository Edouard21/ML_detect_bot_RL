from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import shutil
from typing import List
from contextlib import asynccontextmanager

# Import logique
from analyzer_utils import extract_players_from_replay
from eval_player_game import load_inference_model, evaluate_csv

# Constantes
TMP_DIR = "tmp_replays"
OUT_CSV_DIR = "tmp_csvs"
CKPT_PATH = "best.pt"
TRAIN_BOTS_DIR = os.path.join("data", "train", "bots")
TRAIN_REAL_DIR = os.path.join("data", "train", "real")

os.makedirs(TMP_DIR, exist_ok=True)
os.makedirs(OUT_CSV_DIR, exist_ok=True)
os.makedirs(TRAIN_BOTS_DIR, exist_ok=True)
os.makedirs(TRAIN_REAL_DIR, exist_ok=True)

# Charger le modèle globalement
model_glob = None
meta_glob = None
mean_glob = None
std_glob = None
device_glob = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_glob, meta_glob, mean_glob, std_glob, device_glob
    try:
        model_glob, meta_glob, mean_glob, std_glob, device_glob = load_inference_model(CKPT_PATH)
        print("Modèle chargé avec succès au démarrage.")
    except Exception as e:
        print(f"Attention, impossible de charger {CKPT_PATH}. Détails: {e}")
    yield

app = FastAPI(title="RL Bot Detector API", lifespan=lifespan)

# Setup CORS for the React Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ImportRequest(BaseModel):
    csv_path: str
    target_label: str # "bots" ou "real"

@app.get("/")
def read_root():
    return {"status": "Backend FastAPI running"}

@app.post("/upload")
async def upload_replay(file: UploadFile = File(...)):
    if model_glob is None:
        raise HTTPException(status_code=500, detail="Modèle non chargé côté serveur.")
        
    replay_path = os.path.join(TMP_DIR, file.filename)
    
    with open(replay_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    # Extraire CSVs
    try:
        extracted = extract_players_from_replay(replay_path, OUT_CSV_DIR)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur d'extraction : {str(e)}")
        
    results = []
    for player in extracted:
        csv_path = player["csv_path"]
        name = player["nom_joueur"]
        label, conf = evaluate_csv(csv_path, model_glob, meta_glob, mean_glob, std_glob, device_glob)
        results.append({
            "name": name,
            "csv_path": csv_path,
            "prediction": label,
            "confidence": round(conf * 100, 2)
        })
        
    return {"players": results}

@app.post("/import")
async def import_csv(req: ImportRequest):
    if not os.path.exists(req.csv_path):
        raise HTTPException(status_code=404, detail="Fichier CSV original introuvable.")
        
    filename = os.path.basename(req.csv_path)
    
    if req.target_label == "bots":
        dest_dir = TRAIN_BOTS_DIR
    elif req.target_label == "real":
        dest_dir = TRAIN_REAL_DIR
    else:
        raise HTTPException(status_code=400, detail="Label invalide (bots | real attendu).")
        
    dest_path = os.path.join(dest_dir, filename)
    
    # Déplacement: ça écrasera automatiquement s'il existe (comportement shutil.move avec même système ou os.replace)
    # Pour s'assurer de l'écrasement :
    shutil.copy(req.csv_path, dest_path)
    os.remove(req.csv_path)
        
    return {"status": "success", "destination": dest_path}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
