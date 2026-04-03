"""Dataset utilities for bot-vs-real sequence classification.

This project stores per-player time-series as CSV files under:

	data/
	  train/
		bots/
		real/
	  val/
		bots/
		real/

Each CSV row is a frame (e.g., 30 FPS). This module provides a dataset that
returns fixed-length sliding windows (sequences) such as 90 frames (~3s).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


try:
	import torch
	from torch.utils.data import Dataset as TorchDataset

	_TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
	torch = None
	TorchDataset = object  # type: ignore
	_TORCH_AVAILABLE = False


Split = Literal["train", "val"]
ClassDir = Literal["bots", "real"]


@dataclass(frozen=True)
class SequenceIndex:
	"""Pointer to a (file, start_row) window."""

	file_idx: int
	start: int


@dataclass(frozen=True)
class FileInfo:
	path: Path
	label: int
	num_frames: int
	num_sequences: int
	player_id: str


def _count_data_rows(csv_path: Path) -> int:
	"""Fast-ish count of data rows (excludes header)."""

	# Count '\n' in binary; works for typical CSVs.
	# We subtract 1 for the header line if file isn't empty.
	newline_count = 0
	with csv_path.open("rb") as f:
		while True:
			chunk = f.read(1024 * 1024)
			if not chunk:
				break
			newline_count += chunk.count(b"\n")

	# If the file ends without a newline, count(b"\n") is still correct for lines,
	# but the last line won't be counted. We'll fall back to pandas for that rare case.
	if newline_count == 0:
		return 0

	# Typical CSV has header + N data lines.
	# If there's only one line, it is just the header.
	data_rows = max(0, newline_count - 1)
	return data_rows


def _default_label_map() -> Dict[str, int]:
	return {"bots": 1, "real": 0}


def _infer_player_id(csv_path: Path) -> str:
	# Filenames look like "<player>_<uuid>.csv" or "_<uuid>.csv".
	return csv_path.stem


class RocketLeagueSequenceDataset(TorchDataset):
	"""Sequence window dataset over per-player CSV time-series.

	- Each item is a fixed-length window: shape (seq_len, num_features)
	- Labels are inferred from folder name (`bots`=1, `real`=0 by default)

	Notes on performance:
	- Files are loaded on demand, with an LRU cache of decoded arrays.
	- Increase `cache_max_files` if you have enough RAM and want faster iteration.
	"""

	def __init__(
		self,
		data_dir: Union[str, Path] = "data",
		split: Split = "train",
		class_dirs: Sequence[ClassDir] = ("bots", "real"),
		seq_len: int = 90,
		stride: int = 90,
		columns: Optional[Sequence[str]] = None,
		label_map: Optional[Dict[str, int]] = None,
		drop_last: bool = True,
		pad_value: float = 0.0,
		return_metadata: bool = False,
		as_torch: bool = False,
		cache_max_files: int = 8,
		dtype: Union[np.dtype, str] = np.float32,
	) -> None:
		if seq_len <= 0:
			raise ValueError(f"seq_len must be > 0, got {seq_len}")
		if stride <= 0:
			raise ValueError(f"stride must be > 0, got {stride}")
		if cache_max_files <= 0:
			raise ValueError(f"cache_max_files must be > 0, got {cache_max_files}")

		self.data_dir = Path(data_dir)
		self.split: Split = split
		self.class_dirs: Tuple[ClassDir, ...] = tuple(class_dirs)
		self.seq_len = int(seq_len)
		self.stride = int(stride)
		self.columns = list(columns) if columns is not None else None
		self.label_map = label_map or _default_label_map()
		self.drop_last = bool(drop_last)
		self.pad_value = float(pad_value)
		self.return_metadata = bool(return_metadata)
		self.as_torch = bool(as_torch)
		self.dtype = np.dtype(dtype)

		if self.as_torch and not _TORCH_AVAILABLE:
			raise RuntimeError(
				"as_torch=True requires PyTorch, but torch could not be imported."
			)

		self._files: List[FileInfo] = []
		self._cumulative: List[int] = []
		self._index: List[SequenceIndex] = []
		self._feature_columns: Optional[List[str]] = None

		# Make a per-instance cached loader with configurable maxsize.
		self._load_array = lru_cache(maxsize=cache_max_files)(self._load_array_uncached)

		self._build_index()

	@property
	def feature_columns(self) -> List[str]:
		if self._feature_columns is None:
			raise RuntimeError("Dataset is not initialized correctly (no columns inferred)")
		return list(self._feature_columns)

	def _build_index(self) -> None:
		root = self.data_dir / self.split
		if not root.exists():
			raise FileNotFoundError(f"Split folder not found: {root}")

		files: List[FileInfo] = []

		for class_dir in self.class_dirs:
			class_path = root / class_dir
			if not class_path.exists():
				continue

			label = int(self.label_map.get(str(class_dir), 0))
			for csv_path in sorted(class_path.glob("*.csv")):
				num_frames = _count_data_rows(csv_path)
				if num_frames <= 0:
					continue

				if self.drop_last:
					num_sequences = max(0, (num_frames - self.seq_len) // self.stride + 1)
				else:
					# At least one sequence if there is data.
					num_sequences = max(1, (max(0, num_frames - 1)) // self.stride + 1)

				if num_sequences <= 0:
					continue

				files.append(
					FileInfo(
						path=csv_path,
						label=label,
						num_frames=num_frames,
						num_sequences=num_sequences,
						player_id=_infer_player_id(csv_path),
					)
				)

		if not files:
			raise RuntimeError(
				f"No CSV files found under {root} for class dirs {self.class_dirs}."
			)

		# Infer columns once (from first file), and validate requested columns.
		first_df = pd.read_csv(files[0].path, nrows=1)
		all_cols = list(first_df.columns)
		if self.columns is None:
			self._feature_columns = all_cols
		else:
			missing = [c for c in self.columns if c not in all_cols]
			if missing:
				raise ValueError(
					f"Requested columns not found in CSV header: {missing}. "
					f"Available columns include: {all_cols[:10]}{'...' if len(all_cols) > 10 else ''}"
				)
			self._feature_columns = list(self.columns)

		# Build flat index of all (file_idx, start) windows.
		index: List[SequenceIndex] = []
		cumulative: List[int] = []
		running = 0
		for file_idx, info in enumerate(files):
			for j in range(info.num_sequences):
				start = j * self.stride
				index.append(SequenceIndex(file_idx=file_idx, start=start))
			running += info.num_sequences
			cumulative.append(running)

		self._files = files
		self._index = index
		self._cumulative = cumulative

	def __len__(self) -> int:
		return len(self._index)

	def _load_array_uncached(self, csv_path: Path) -> np.ndarray:
		df = pd.read_csv(csv_path)
		df = df[self.feature_columns]
		# Defensive: coerce numeric and fill missing.
		for col in df.columns:
			df[col] = pd.to_numeric(df[col], errors="coerce")
		arr = df.fillna(self.pad_value).to_numpy(dtype=self.dtype, copy=False)
		return arr

	def _get_window(self, arr: np.ndarray, start: int) -> np.ndarray:
		end = start + self.seq_len
		if end <= arr.shape[0]:
			return arr[start:end]

		if self.drop_last:
			# Should not happen due to indexing logic.
			raise IndexError("Window exceeds file length; check indexing logic")

		# Pad to seq_len.
		out = np.full((self.seq_len, arr.shape[1]), self.pad_value, dtype=self.dtype)
		available = max(0, arr.shape[0] - start)
		if available > 0:
			out[:available] = arr[start : start + available]
		return out

	def __getitem__(self, idx: int):
		if idx < 0:
			idx = len(self) + idx
		if idx < 0 or idx >= len(self):
			raise IndexError(idx)

		ptr = self._index[idx]
		info = self._files[ptr.file_idx]
		arr = self._load_array(info.path)
		x = self._get_window(arr, ptr.start)
		y = info.label

		if self.as_torch:
			x_t = torch.from_numpy(np.asarray(x))
			y_t = torch.tensor(y, dtype=torch.long)
			if self.return_metadata:
				return {
					"x": x_t,
					"y": y_t,
					"player_id": info.player_id,
					"path": str(info.path),
					"start": int(ptr.start),
				}
			return x_t, y_t

		if self.return_metadata:
			return {
				"x": x,
				"y": y,
				"player_id": info.player_id,
				"path": str(info.path),
				"start": int(ptr.start),
			}
		return x, y

	def clear_cache(self) -> None:
		"""Clear the file decoding cache."""

		self._load_array.cache_clear()  # type: ignore[attr-defined]

	def describe(self) -> Dict[str, object]:
		"""Small summary useful for sanity checks."""

		bots = sum(1 for f in self._files if f.label == self.label_map.get("bots", 1))
		real = sum(1 for f in self._files if f.label == self.label_map.get("real", 0))
		return {
			"split": self.split,
			"data_dir": str(self.data_dir),
			"num_files": len(self._files),
			"num_files_bots": bots,
			"num_files_real": real,
			"num_sequences": len(self),
			"seq_len": self.seq_len,
			"stride": self.stride,
			"num_features": len(self.feature_columns),
			"feature_columns": list(self.feature_columns),
		}


def make_torch_dataloader(
	dataset: RocketLeagueSequenceDataset,
	batch_size: int = 32,
	shuffle: bool = True,
	num_workers: int = 0,
):
	"""Convenience wrapper; requires torch."""

	if not _TORCH_AVAILABLE:
		raise RuntimeError("PyTorch is not installed; cannot create a DataLoader")
	if not dataset.as_torch:
		raise ValueError("dataset.as_torch must be True to use DataLoader")

	from torch.utils.data import DataLoader

	return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


if __name__ == "__main__":
	ds = RocketLeagueSequenceDataset(
		data_dir="data",
		split="train",
		seq_len=90,
		stride=90,
		drop_last=True,
		return_metadata=True,
		as_torch=False,
	)
	print(ds.describe())
	sample = ds[0]
	print("sample keys:", list(sample.keys()))
	print("x shape:", sample["x"].shape, "y:", sample["y"], "player_id:", sample["player_id"])
