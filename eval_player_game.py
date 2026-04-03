"""Evaluate a whole player game (single CSV) with a trained model.

This script runs sliding-window inference over a single player's frame CSV and:
- outputs a per-window "bot probability" time series
- highlights windows that look bot-like
- estimates whether the player is a bot after considering the whole game

Example:
    python eval_player_game.py --csv data/val/real/some_player.csv --ckpt checkpoints/best.pt
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, required=True, help="Path to a single player CSV")
    p.add_argument("--ckpt", type=str, default="checkpoints/best.pt", help="Checkpoint path")

    p.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Inference device",
    )

    p.add_argument(
        "--stride",
        type=int,
        default=0,
        help="Window stride (frames). Default: checkpoint meta stride if available, else seq_len",
    )

    p.add_argument(
        "--pad-value",
        type=float,
        default=0.0,
        help="Padding value when including the last partial window",
    )
    p.add_argument(
        "--drop-last",
        action="store_true",
        help="Drop the last partial window (default: keep it with padding)",
    )

    p.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Inference batch size (windows)",
    )

    p.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Frames-per-second for the x-axis (seconds). Default assumes 30 FPS.",
    )

    p.add_argument(
        "--window-threshold",
        type=float,
        default=0.5,
        help="A window is considered bot-like if P(bot) >= this threshold",
    )
    p.add_argument(
        "--game-threshold",
        type=float,
        default=0.5,
        help="Classify as bot if the aggregated game score >= this threshold",
    )

    p.add_argument(
        "--out",
        type=str,
        default="",
        help="Where to save the PNG plot (default: plots/player_eval_<csv_stem>.png)",
    )
    p.add_argument(
        "--show",
        action="store_true",
        help="Show the plot interactively (also saves it if --out is set)",
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


@dataclass(frozen=True)
class WindowSpec:
    seq_len: int
    stride: int
    drop_last: bool
    pad_value: float


def _load_csv_array(*, csv_path: Path, feature_columns: Sequence[str], pad_value: float):
    import numpy as np
    import pandas as pd

    header_df = pd.read_csv(csv_path, nrows=0)
    available = set(header_df.columns)
    missing = [c for c in feature_columns if c not in available]
    if missing:
        raise ValueError(
            f"CSV is missing required columns from checkpoint meta: {missing}. "
            f"First columns in file: {list(header_df.columns)[:10]}"
        )

    df = pd.read_csv(csv_path, usecols=list(feature_columns))

    # Defensive: coerce numeric and fill missing.
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    arr = df.fillna(float(pad_value)).to_numpy(dtype=np.float32, copy=False)
    if arr.ndim != 2 or arr.shape[1] != len(feature_columns):
        raise RuntimeError(
            f"Unexpected array shape from CSV: {arr.shape}, expected (N, {len(feature_columns)})"
        )
    return arr


def _iter_window_starts(num_frames: int, spec: WindowSpec) -> List[int]:
    if num_frames <= 0:
        return []

    if spec.drop_last:
        if num_frames < spec.seq_len:
            return []
        last_start = num_frames - spec.seq_len
        return list(range(0, last_start + 1, spec.stride))

    # Include a final (possibly partial) window.
    starts = list(range(0, num_frames, spec.stride))
    if not starts:
        starts = [0]
    return starts


def _make_window(arr, start: int, spec: WindowSpec):
    import numpy as np

    end = start + spec.seq_len
    if end <= arr.shape[0]:
        return arr[start:end]

    out = np.full((spec.seq_len, arr.shape[1]), float(spec.pad_value), dtype=arr.dtype)
    available = max(0, arr.shape[0] - start)
    if available > 0:
        out[:available] = arr[start : start + available]
    return out


def _batched(iterable: Sequence[int], batch_size: int) -> Iterable[List[int]]:
    batch: List[int] = []
    for item in iterable:
        batch.append(int(item))
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _load_checkpoint(ckpt_path: Path):
    import torch

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location="cpu")
    except Exception:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    meta = ckpt.get("meta", {})
    model_type = meta.get("model", "tcnn")
    feature_columns = meta.get("feature_columns")
    seq_len = int(meta.get("seq_len", 90))
    stride_meta = int(meta.get("stride", seq_len))

    if not feature_columns:
        raise RuntimeError("Checkpoint meta missing 'feature_columns' (was it saved by train.py?)")

    return ckpt, meta, str(model_type), list(feature_columns), int(seq_len), int(stride_meta)


def _get_standardizer_tensors(ckpt, device, num_features: int):
    import numpy as np
    import torch

    std_pack = ckpt.get("standardizer")
    if std_pack is None:
        return None, None

    mean_val = std_pack.get("mean")
    std_val = std_pack.get("std")
    if mean_val is None or std_val is None:
        return None, None

    mean_np = np.asarray(mean_val, dtype=np.float32)
    std_np = np.asarray(std_val, dtype=np.float32)

    if mean_np.shape != (num_features,) or std_np.shape != (num_features,):
        print(
            "Warning: checkpoint standardizer shape mismatch; ignoring standardization. "
            f"mean shape={mean_np.shape}, std shape={std_np.shape}, expected ({num_features},)"
        )
        return None, None

    mean = torch.as_tensor(mean_np, device=device)
    std = torch.as_tensor(std_np, device=device)
    return mean, std


def _infer_window_bot_probs(
    *,
    model,
    arr,
    starts: Sequence[int],
    spec: WindowSpec,
    device,
    mean=None,
    std=None,
    batch_size: int = 512,
) -> List[float]:
    import numpy as np
    import torch
    import torch.nn.functional as F

    probs: List[float] = []
    model.eval()

    with torch.no_grad():
        for batch_starts in _batched(starts, batch_size=batch_size):
            windows = [_make_window(arr, s, spec) for s in batch_starts]
            x = torch.from_numpy(np.stack(windows, axis=0)).to(device)  # (B, T, F)
            if mean is not None and std is not None:
                x = (x - mean) / std
            logits = model(x)
            p = F.softmax(logits, dim=-1)[:, 1]  # class 1 = bot
            probs.extend([float(v) for v in p.detach().to("cpu").numpy().tolist()])

    return probs


def _aggregate_game_score(bot_probs: Sequence[float], *, window_threshold: float) -> Tuple[float, float]:
    """Return (mean_prob, fraction_windows_ge_threshold)."""

    if not bot_probs:
        return 0.0, 0.0
    mean_prob = float(sum(bot_probs) / len(bot_probs))
    thr = float(window_threshold)
    frac_bot = float(sum(1 for p in bot_probs if p >= thr) / len(bot_probs))
    return mean_prob, frac_bot


def _save_plot(
    *,
    out_path: Path,
    starts: Sequence[int],
    bot_probs: Sequence[float],
    fps: float,
    window_threshold: float,
    title: str,
    show: bool,
) -> None:
    try:
        import matplotlib

        if not show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        out_path.parent.mkdir(parents=True, exist_ok=True)

        xs = [(float(s) / float(fps)) for s in starts]  # seconds
        ys = list(bot_probs)

        fig = plt.figure(figsize=(10.5, 4.5), dpi=160)
        ax = fig.add_subplot(1, 1, 1)

        ax.plot(xs, ys, linewidth=1.6)
        ax.axhline(window_threshold, linestyle="--", linewidth=1.0)

        # Shade bot-like windows.
        ax.fill_between(
            xs,
            0.0,
            1.0,
            where=[p >= window_threshold for p in ys],
            alpha=0.15,
            interpolate=True,
        )

        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("P(bot) per window")
        ax.set_title(title)

        fig.tight_layout()

        if str(out_path):
            fig.savefig(out_path)
            print("Saved plot:", str(out_path))

        if show:
            plt.show()
        plt.close(fig)

    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "matplotlib is not installed, so the plot cannot be created. "
            "Install it with: pip install matplotlib"
        )


def main() -> None:
    args = parse_args()

    import torch

    from src.model import build_model

    csv_path = Path(args.csv)
    ckpt_path = Path(args.ckpt)

    ckpt, meta, model_type, feature_columns, seq_len, stride_meta = _load_checkpoint(ckpt_path)

    stride = int(args.stride) if int(args.stride) > 0 else int(stride_meta) if int(stride_meta) > 0 else int(seq_len)
    spec = WindowSpec(
        seq_len=int(seq_len),
        stride=int(stride),
        drop_last=bool(args.drop_last),
        pad_value=float(args.pad_value),
    )

    device = choose_device(args.device)

    arr = _load_csv_array(csv_path=csv_path, feature_columns=feature_columns, pad_value=spec.pad_value)
    starts = _iter_window_starts(arr.shape[0], spec)
    if not starts:
        raise RuntimeError(
            f"Not enough frames for evaluation. frames={arr.shape[0]}, seq_len={spec.seq_len}, drop_last={spec.drop_last}"
        )

    model = build_model(model_type, num_features=len(feature_columns), num_classes=2).to(device)
    model.load_state_dict(ckpt["model_state"])

    mean, std = _get_standardizer_tensors(ckpt, device=device, num_features=len(feature_columns))

    bot_probs = _infer_window_bot_probs(
        model=model,
        arr=arr,
        starts=starts,
        spec=spec,
        device=device,
        mean=mean,
        std=std,
        batch_size=int(args.batch_size),
    )

    mean_prob, frac_bot05 = _aggregate_game_score(bot_probs, window_threshold=float(args.window_threshold))
    pred = "BOT" if mean_prob >= float(args.game_threshold) else "REAL"

    print("CSV:", str(csv_path))
    print("Checkpoint:", str(ckpt_path))
    print("Device:", device)
    print(f"Frames: {arr.shape[0]} | Features: {arr.shape[1]}")
    print(f"Windowing: seq_len={spec.seq_len} stride={spec.stride} windows={len(starts)} drop_last={spec.drop_last}")
    print(f"Window threshold: {float(args.window_threshold):.3f} | Game threshold: {float(args.game_threshold):.3f}")
    print(f"Game bot score (mean P(bot)): {mean_prob:.4f}")
    print(
        f"Fraction windows with P(bot) >= {float(args.window_threshold):.3f}: {frac_bot05:.4f}"
    )
    print(f"Predicted: {pred}")

    out = Path(args.out) if args.out else Path("plots") / f"player_eval_{csv_path.stem}.png"
    title = f"{csv_path.stem} | mean P(bot)={mean_prob:.3f} | pred={pred}"
    _save_plot(
        out_path=out,
        starts=starts,
        bot_probs=bot_probs,
        fps=float(args.fps),
        window_threshold=float(args.window_threshold),
        title=title,
        show=bool(args.show),
    )


if __name__ == "__main__":
    main()
