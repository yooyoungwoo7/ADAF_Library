#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import csv
import json
import numpy as np


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(data, path):
    def convert(v):
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, (np.float16, np.float32, np.float64)):
            return float(v)
        if isinstance(v, (np.int16, np.int32, np.int64, np.int_)):
            return int(v)
        if isinstance(v, Path):
            return str(v)
        return v

    if isinstance(data, dict):
        data = {k: convert(v) for k, v in data.items()}

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_csv_rows(rows, path, header=None):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if header is not None:
            writer.writerow(header)
        writer.writerows(rows)
