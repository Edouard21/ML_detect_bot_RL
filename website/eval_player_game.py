import torch
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional

try:
    from src.model import build_model
except ImportError:
    pass # Permet l'exécution si on le run dans un autre chemin
import torch.nn.functional as F

def load_inference_model(ckpt_path: str, device: str = "cpu") -> Tuple[torch.nn.Module, Dict[str, Any], Any, Any, torch.device]:
    device_torch = torch.device(device)
    try:
        ckpt = torch.load(ckpt_path, map_location=device_torch, weights_only=True)
    except:
        ckpt = torch.load(ckpt_path, map_location=device_torch, weights_only=False)
        
    meta = ckpt.get("meta", {})
    model_type = meta.get("model", "tcnn")
    feature_columns = meta.get("feature_columns")
    
    model = build_model(model_type, num_features=len(feature_columns), num_classes=2).to(device_torch)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    std_pack = ckpt.get("standardizer")
    mean = std = None
    if std_pack is not None:
        mean = torch.as_tensor(np.asarray(std_pack.get("mean"), dtype=np.float32), device=device_torch)
        std = torch.as_tensor(np.asarray(std_pack.get("std"), dtype=np.float32), device=device_torch)
        
    return model, meta, mean, std, device_torch

def evaluate_csv(csv_path: str, model: torch.nn.Module, meta: Dict[str, Any], mean: Any, std: Any, device: torch.device) -> Tuple[str, float]:
    feature_columns = meta.get("feature_columns")
    seq_len = int(meta.get("seq_len", 90))
    stride = seq_len # Non-overlapping windows pour l'évaluation globale
    
    df = pd.read_csv(csv_path)
    
    # Check if all columns exist
    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        for c in missing:
            df[c] = 0.0

    df = df[feature_columns]
    arr = df.fillna(0.0).to_numpy(dtype=np.float32)
    
    # Build sequences
    num_frames = arr.shape[0]
    num_sequences = max(1, (max(0, num_frames - 1)) // stride + 1)
    
    windows = []
    for j in range(num_sequences):
        start = j * stride
        end = start + seq_len
        window = arr[start:end]
        if window.shape[0] < seq_len:
            out = np.full((seq_len, arr.shape[1]), 0.0, dtype=np.float32)
            out[:window.shape[0]] = window
            window = out
        windows.append(window)
        
    if not windows:
        return "real", 0.0
        
    X = torch.from_numpy(np.stack(windows)).to(device)
    if mean is not None and std is not None:
        X = (X - mean) / std
        
    with torch.no_grad():
        logits = model(X) # (batch, num_classes)
        probs = F.softmax(logits, dim=-1) # (batch, 2)
        
    # Agregation: moyenne des probabilités
    avg_probs = probs.mean(dim=0).cpu().numpy()
    
    # Labels attendus: Real = 0, Bots = 1
    bot_prob = float(avg_probs[1])
    is_bot = bot_prob > 0.5
    label = "bots" if is_bot else "real"
    confidence = bot_prob if is_bot else (1.0 - bot_prob)
    
    return label, confidence

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="Chemin vers le fichier csv joueur")
    parser.add_argument("--ckpt", type=str, default="best.pt", help="Chemin vers le checkpoint")
    args = parser.parse_args()
    
    import sys
    sys.path.append(".") # Ensure we can import src module
    
    model, meta, mean, std, device = load_inference_model(args.ckpt)
    label, conf = evaluate_csv(args.csv, model, meta, mean, std, device)
    print(f"Result for {args.csv}:")
    print(f"[{label.upper()}] Confidence: {conf * 100:.2f}%")
