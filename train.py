"""Train a temporal CNN on the sequence CSV dataset.

This script is designed to be compatible with `eval.py`.

Example:
    python train.py --epochs 10 --batch-size 256 --out checkpoints/best.pt
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Optional, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--model", type=str, default="tcnn", help="Model type (default: tcnn)")

    p.add_argument("--seq-len", type=int, default=360)
    p.add_argument("--stride", type=int, default=30)

    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)

    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])

    p.add_argument(
        "--no-standardize",
        action="store_true",
        help="Disable feature standardization (mean/std) computed from train split",
    )

    p.add_argument(
        "--out",
        type=str,
        default="checkpoints/best.pt",
        help="Where to save the best checkpoint",
    )

    p.add_argument(
        "--save-every",
        type=int,
        default=0,
        help="If >0, also save checkpoints every N epochs (in same folder as --out)",
    )

    return p.parse_args()


def choose_device(requested: str):
    import torch

    def _cuda_usable() -> bool:
        if not torch.cuda.is_available():
            return False
        try:
            major, minor = torch.cuda.get_device_capability(0)
            device_arch = f"sm_{major}{minor}"
            return device_arch in set(torch.cuda.get_arch_list())
        except Exception:
            return False

    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not _cuda_usable():
            major, minor = torch.cuda.get_device_capability(0)
            device_arch = f"sm_{major}{minor}"
            raise RuntimeError(
                "--device cuda was requested but this PyTorch build does not support your GPU architecture. "
                f"GPU arch={device_arch}, torch arch list={torch.cuda.get_arch_list()}"
            )
        return torch.device("cuda")
    return torch.device("cuda" if _cuda_usable() else "cpu")


def compute_standardizer(train_loader, eps: float = 1e-6) -> Tuple["torch.Tensor", "torch.Tensor"]:
    """Compute per-feature mean/std across (batch,time) dims on CPU."""

    import torch

    with torch.no_grad():
        sum_x = None
        sum_x2 = None
        count = 0

        for x, _y in train_loader:
            # x: (B, T, F)
            x = x.float().cpu()
            b, t, f = x.shape
            flat_count = b * t
            x_sum = x.sum(dim=(0, 1), dtype=torch.float64)  # (F,)
            x2_sum = (x * x).sum(dim=(0, 1), dtype=torch.float64)  # (F,)

            if sum_x is None:
                sum_x = x_sum
                sum_x2 = x2_sum
            else:
                sum_x += x_sum
                sum_x2 += x2_sum
            count += flat_count

    if sum_x is None or sum_x2 is None or count <= 0:
        raise RuntimeError("Could not compute standardizer (empty loader?)")

    mean = (sum_x / float(count)).to(dtype=torch.float32)
    var = (sum_x2 / float(count)).to(dtype=torch.float32) - mean * mean
    var = torch.clamp(var, min=0.0)
    std = torch.sqrt(var + float(eps))
    return mean, std


def run_epoch(
    *,
    model,
    loader,
    device,
    optimizer=None,
    mean=None,
    std=None,
) -> Dict[str, float]:
    import torch
    import torch.nn.functional as F

    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        if mean is not None and std is not None:
            x = (x - mean) / std

        logits = model(x)
        loss = F.cross_entropy(logits, y)

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        total_loss += float(loss.detach().cpu()) * x.shape[0]
        preds = torch.argmax(logits, dim=-1)
        total_correct += int((preds == y).sum().detach().cpu())
        total += int(x.shape[0])

    avg_loss = total_loss / max(1, total)
    acc = total_correct / max(1, total)
    return {"loss": float(avg_loss), "acc": float(acc)}


def main() -> None:
    args = parse_args()

    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    from src.dataset import RocketLeagueSequenceDataset
    from src.model import build_model

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = choose_device(args.device)

    train_ds = RocketLeagueSequenceDataset(
        data_dir=args.data_dir,
        split="train",
        seq_len=args.seq_len,
        stride=args.stride,
        drop_last=True,
        as_torch=True,
        return_metadata=False,
    )

    # Reuse the exact same columns in val.
    feature_columns = train_ds.feature_columns

    val_ds = RocketLeagueSequenceDataset(
        data_dir=args.data_dir,
        split="val",
        seq_len=args.seq_len,
        stride=args.seq_len,  # evaluation on non-overlapping windows
        drop_last=True,
        as_torch=True,
        return_metadata=False,
        columns=feature_columns,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = build_model(args.model, num_features=len(feature_columns), num_classes=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.no_standardize:
        mean = std = None
        std_pack: Optional[Dict[str, object]] = None
    else:
        # Compute on CPU, then move to device for fast broadcasting.
        mean_cpu, std_cpu = compute_standardizer(train_loader)
        mean = mean_cpu.to(device)
        std = std_cpu.to(device)
        std_pack = {
            "mean": mean_cpu.detach().cpu().numpy().astype(np.float32).tolist(),
            "std": std_cpu.detach().cpu().numpy().astype(np.float32).tolist(),
        }

    best_acc = -math.inf

    meta = {
        "model": str(args.model).lower().strip(),
        "seq_len": int(args.seq_len),
        "stride": int(args.stride),
        "feature_columns": list(feature_columns),
        "num_features": int(len(feature_columns)),
        "num_classes": 2,
    }

    print("Device:", device)
    print("Train:", train_ds.describe())
    print("Val:", val_ds.describe())
    print("Model:", meta["model"], "num_features=", meta["num_features"], "seq_len=", meta["seq_len"])
    if mean is not None:
        print("Standardization: enabled")
    else:
        print("Standardization: disabled")

    for epoch in range(1, int(args.epochs) + 1):
        tr = run_epoch(model=model, loader=train_loader, device=device, optimizer=optimizer, mean=mean, std=std)
        va = run_epoch(model=model, loader=val_loader, device=device, optimizer=None, mean=mean, std=std)

        print(
            f"epoch={epoch:03d} "
            f"train_loss={tr['loss']:.4f} train_acc={tr['acc']:.4f} "
            f"val_loss={va['loss']:.4f} val_acc={va['acc']:.4f}"
        )

        is_best = va["acc"] > best_acc
        if is_best:
            best_acc = float(va["acc"])
            ckpt = {
                "model_state": model.state_dict(),
                "meta": meta,
                "standardizer": std_pack,
            }
            torch.save(ckpt, out_path)

        if args.save_every and args.save_every > 0 and (epoch % int(args.save_every) == 0):
            ep_path = out_path.with_name(f"{out_path.stem}_epoch{epoch:03d}{out_path.suffix}")
            ckpt = {
                "model_state": model.state_dict(),
                "meta": meta,
                "standardizer": std_pack,
            }
            torch.save(ckpt, ep_path)

    print(f"Best val_acc={best_acc:.4f}")
    print("Saved best checkpoint to:", str(out_path))


if __name__ == "__main__":
    # Keep this for a clearer error message if torch isn't installed.
    try:
        import torch  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise RuntimeError("PyTorch is required to run train.py") from e

    main()
