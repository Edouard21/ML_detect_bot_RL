"""Evaluate a saved checkpoint on the validation split.

Run:
    python eval.py --ckpt checkpoints/best.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--ckpt", type=str, default="checkpoints/best.pt")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])

    # Confusion matrix plot
    p.add_argument(
        "--cm-out",
        type=str,
        default="plots/confusion_matrix.png",
        help="Where to save the confusion matrix PNG",
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
            arch_list = set(torch.cuda.get_arch_list())
            # If torch was built without this architecture, kernels will fail at runtime.
            return device_arch in arch_list
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
    # auto
    return torch.device("cuda" if _cuda_usable() else "cpu")


def main() -> None:
    args = parse_args()

    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    from src.dataset import RocketLeagueSequenceDataset
    from src.model import build_model

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    device = choose_device(args.device)

    # PyTorch 2.6 defaults torch.load(weights_only=True). Older checkpoints that
    # include non-tensor objects (e.g., numpy arrays) may fail to load.
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except TypeError:
        # Older torch without weights_only arg.
        ckpt = torch.load(ckpt_path, map_location="cpu")
    except Exception:
        # Fallback for locally-created checkpoints that contain non-tensor objects.
        # Only do this if you trust the checkpoint source.
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    meta = ckpt.get("meta", {})
    model_type = meta.get("model", "tcnn")
    feature_columns = meta.get("feature_columns")
    seq_len = int(meta.get("seq_len", 90))

    if not feature_columns:
        raise RuntimeError("Checkpoint meta missing 'feature_columns' (was it saved by train.py?)")

    val_ds = RocketLeagueSequenceDataset(
        data_dir=args.data_dir,
        split="val",
        seq_len=seq_len,
        stride=seq_len,
        drop_last=True,
        as_torch=True,
        columns=feature_columns,
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = build_model(model_type, num_features=len(feature_columns), num_classes=2).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    std_pack = ckpt.get("standardizer")
    if std_pack is not None:
        mean_val = std_pack.get("mean")
        std_val = std_pack.get("std")
        mean = torch.as_tensor(np.asarray(mean_val, dtype=np.float32), device=device)
        std = torch.as_tensor(np.asarray(std_val, dtype=np.float32), device=device)
    else:
        mean = std = None

    total_loss = 0.0
    total_correct = 0
    total = 0

    # Confusion matrix for binary classification (rows=true, cols=pred)
    # [[TN, FP],
    #  [FN, TP]]
    cm = np.zeros((2, 2), dtype=np.int64)

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device)

            if mean is not None and std is not None:
                x = (x - mean) / std

            logits = model(x)
            loss = F.cross_entropy(logits, y)

            total_loss += float(loss.detach().cpu()) * x.shape[0]
            preds = torch.argmax(logits, dim=-1)
            total_correct += int((preds == y).sum().detach().cpu())
            total += int(x.shape[0])

            # Update confusion matrix
            y_cpu = y.detach().to("cpu").numpy().astype(np.int64)
            p_cpu = preds.detach().to("cpu").numpy().astype(np.int64)
            for yt, yp in zip(y_cpu, p_cpu):
                if yt in (0, 1) and yp in (0, 1):
                    cm[yt, yp] += 1

    avg_loss = total_loss / max(1, total)
    acc = total_correct / max(1, total)

    print("Checkpoint:", str(ckpt_path))
    print("Device:", device)
    print("Val:", val_ds.describe())
    print(f"val_loss={avg_loss:.4f} val_acc={acc:.4f}")

    # Save confusion matrix plot
    cm_out = Path(args.cm_out)
    cm_out.parent.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(4.5, 4.0), dpi=160)
        ax = fig.add_subplot(1, 1, 1)
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")

        ax.set_title("Confusion Matrix (val)")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["real (0)", "bot (1)"])
        ax.set_yticklabels(["real (0)", "bot (1)"])

        # Annotate counts
        for i in range(2):
            for j in range(2):
                ax.text(
                    j,
                    i,
                    str(int(cm[i, j])),
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=10,
                )

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(cm_out)
        plt.close(fig)
        print("Saved confusion matrix plot:", str(cm_out))
    except ModuleNotFoundError:
        print(
            "matplotlib is not installed, so the confusion matrix plot was not saved. "
            "Install it with: pip install matplotlib"
        )
        print("Confusion matrix counts (rows=true, cols=pred):")
        print(cm)


if __name__ == "__main__":
    main()
