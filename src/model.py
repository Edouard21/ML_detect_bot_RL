"""PyTorch models for bot-vs-real sequence classification.

The dataset (`RocketLeagueSequenceDataset`) yields windows shaped:

- single item: (seq_len, num_features)
- batch: (batch, seq_len, num_features)

This module provides a small temporal CNN (1D conv over time) that consumes
those tensors directly.

Public API expected by `eval.py`:
- `build_model(model: str, num_features: int, num_classes: int)`
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional


try:
	import torch
	import torch.nn as nn
	import torch.nn.functional as F

	_TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
	torch = None
	nn = None  # type: ignore
	F = None  # type: ignore
	_TORCH_AVAILABLE = False


ModelType = Literal["tcnn"]


def _require_torch() -> None:
	if not _TORCH_AVAILABLE:
		raise RuntimeError("PyTorch is required to use src.model")


@dataclass(frozen=True)
class TemporalCNNConfig:
	"""Config for `TemporalCNN`.

	Defaults are intentionally modest to train quickly.
	"""

	channels: int = 64
	depth: int = 4
	kernel_size: int = 5
	dropout: float = 0.1
	use_batchnorm: bool = True
	pool: Literal["avg", "max"] = "avg"


class _TemporalBlock(nn.Module):
	def __init__(
		self,
		in_ch: int,
		out_ch: int,
		kernel_size: int,
		dropout: float,
		use_batchnorm: bool,
	) -> None:
		super().__init__()
		padding = kernel_size // 2
		self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=padding)
		self.bn = nn.BatchNorm1d(out_ch) if use_batchnorm else None
		self.drop = nn.Dropout(p=dropout) if dropout and dropout > 0 else None

		self.proj = None
		if in_ch != out_ch:
			self.proj = nn.Conv1d(in_ch, out_ch, kernel_size=1)

	def forward(self, x):
		res = x
		x = self.conv(x)
		if self.bn is not None:
			x = self.bn(x)
		x = F.relu(x)
		if self.drop is not None:
			x = self.drop(x)
		if self.proj is not None:
			res = self.proj(res)
		return x + res


class TemporalCNN(nn.Module):
	"""Temporal 1D-CNN over frames.

	Input shapes supported:
	- (B, T, F) where T=seq_len and F=num_features (dataset default)
	- (B, F, T)
	- (T, F) (single sample) will be expanded to (1, T, F)
	"""

	def __init__(
		self,
		num_features: int,
		num_classes: int = 2,
		config: Optional[TemporalCNNConfig] = None,
	) -> None:
		super().__init__()
		if num_features <= 0:
			raise ValueError(f"num_features must be > 0, got {num_features}")
		if num_classes <= 1:
			raise ValueError(f"num_classes must be > 1, got {num_classes}")

		cfg = config or TemporalCNNConfig()
		if cfg.depth <= 0:
			raise ValueError(f"depth must be > 0, got {cfg.depth}")
		if cfg.kernel_size <= 0 or cfg.kernel_size % 2 == 0:
			raise ValueError(
				f"kernel_size must be an odd positive integer (for same padding), got {cfg.kernel_size}"
			)

		layers = []
		in_ch = int(num_features)
		ch = int(cfg.channels)
		for _ in range(int(cfg.depth)):
			layers.append(
				_TemporalBlock(
					in_ch=in_ch,
					out_ch=ch,
					kernel_size=int(cfg.kernel_size),
					dropout=float(cfg.dropout),
					use_batchnorm=bool(cfg.use_batchnorm),
				)
			)
			in_ch = ch
		self.backbone = nn.Sequential(*layers)

		if cfg.pool == "avg":
			self.pool = nn.AdaptiveAvgPool1d(1)
		elif cfg.pool == "max":
			self.pool = nn.AdaptiveMaxPool1d(1)
		else:
			raise ValueError(f"Unknown pool: {cfg.pool}")

		self.head = nn.Linear(in_ch, int(num_classes))

	def forward(self, x):
		# Normalize to (B, F, T)
		if x.dim() == 2:
			# (T, F)
			x = x.unsqueeze(0)
		if x.dim() != 3:
			raise ValueError(f"Expected input dim=2 or 3, got shape={tuple(x.shape)}")

		# Common case: (B, T, F)
		if x.shape[1] != self._num_features() and x.shape[2] == self._num_features():
			x = x.transpose(1, 2)  # (B, F, T)
		elif x.shape[1] == self._num_features():
			# Already (B, F, T)
			pass
		else:
			raise ValueError(
				"Input must be (B, T, F) or (B, F, T) with F=num_features. "
				f"Got shape={tuple(x.shape)}, num_features={self._num_features()}"
			)

		x = x.float()
		x = self.backbone(x)
		x = self.pool(x).squeeze(-1)  # (B, C)
		logits = self.head(x)
		return logits

	def _num_features(self) -> int:
		# First conv's input channels.
		first = self.backbone[0]
		return int(first.conv.in_channels)


def build_model(model: str, num_features: int, num_classes: int = 2):
	"""Factory used by training/evaluation scripts.

	Currently supported:
	- "tcnn": temporal CNN over time (Conv1d)
	"""

	_require_torch()
	model_l = str(model).lower().strip()
	if model_l in {"tcnn", "cnn", "temporal_cnn", "temporalcnn"}:
		return TemporalCNN(num_features=num_features, num_classes=num_classes)
	raise ValueError(f"Unknown model '{model}'. Supported: tcnn")

