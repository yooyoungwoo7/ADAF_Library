#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


def l2_rel(pred, ref, eps=1e-12):
    pred = np.asarray(pred, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    return float(np.linalg.norm(pred - ref) / (np.linalg.norm(ref) + eps))


def l1_rel(pred, ref, eps=1e-12):
    pred = np.asarray(pred, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    return float(np.sum(np.abs(pred - ref)) / (np.sum(np.abs(ref)) + eps))


def statewise_l2_rel(pred, ref, eps=1e-12):
    pred = np.asarray(pred, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    out = []
    for j in range(ref.shape[-1]):
        out.append(float(np.linalg.norm(pred[..., j] - ref[..., j]) / (np.linalg.norm(ref[..., j]) + eps)))
    return np.array(out, dtype=np.float64)


def statewise_l1_rel(pred, ref, eps=1e-12):
    pred = np.asarray(pred, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    out = []
    for j in range(ref.shape[-1]):
        out.append(float(np.sum(np.abs(pred[..., j] - ref[..., j])) / (np.sum(np.abs(ref[..., j])) + eps)))
    return np.array(out, dtype=np.float64)


def final_state_error(pred, ref):
    pred = np.asarray(pred, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    return float(np.linalg.norm(pred[-1] - ref[-1]))


def residual_stats(residual):
    residual = np.asarray(residual, dtype=np.float64)
    sq = np.sum(residual ** 2, axis=-1)
    return {
        "mean_residual_norm_sq": float(np.mean(sq)),
        "max_residual_norm_sq": float(np.max(sq)),
    }
