"""Run logging: CSV files (source of truth for analysis) plus optional TensorBoard scalars."""
from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Dict, List


class RunLogger:
    def __init__(self, run_dir: Path, tensorboard: bool = True) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.fields: Dict[str, List[str]] = {}
        self.tb = None
        if tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self.tb = SummaryWriter(str(self.run_dir / "tb"))
            except ImportError:
                print("TensorBoard is not installed; logging to CSV only.")

    def log(self, name: str, step: int, row: Dict[str, float]) -> None:
        row = {"step": step, **row}
        path = self.run_dir / f"{name}.csv"
        if name not in self.fields:
            if path.exists() and path.stat().st_size > 0:
                with path.open(newline="") as f:
                    self.fields[name] = next(csv.reader(f))
            else:
                self.fields[name] = list(row)
                with path.open("w", newline="") as f:
                    csv.writer(f).writerow(self.fields[name])
        with path.open("a", newline="") as f:
            csv.DictWriter(f, fieldnames=self.fields[name], extrasaction="ignore").writerow(row)
        if self.tb is not None:
            for key, value in row.items():
                if key != "step" and isinstance(value, (int, float)) and math.isfinite(value):
                    self.tb.add_scalar(f"{name}/{key}", value, step)
            self.tb.flush()

    def truncate_after(self, step: int) -> None:
        """Drop rows logged after ``step`` (used when resuming from an earlier checkpoint)."""
        for path in self.run_dir.glob("*.csv"):
            with path.open(newline="") as f:
                rows = list(csv.reader(f))
            if not rows:
                continue
            kept = [rows[0]] + [r for r in rows[1:] if r and float(r[0]) <= step]
            with path.open("w", newline="") as f:
                csv.writer(f).writerows(kept)

    def close(self) -> None:
        if self.tb is not None:
            self.tb.close()
