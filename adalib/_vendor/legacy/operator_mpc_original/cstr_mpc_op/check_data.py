#!/usr/bin/env python3
import os
os.environ["PROBLEM"] = "cstr_mpc"
os.environ["BASIS"]   = "lpa"
import numpy as np

arr = np.load("data_files/cstr_mpc/cstr_mpc_train_segments.npz")
X = arr["X"]
Y = arr["Y_ref_seg"]

print("=== Segment X stats ===")
print(f"shape: {X.shape}")
print(f"X[:,0] C_A : min={X[:,0].min():.3f}  max={X[:,0].max():.3f}  mean={X[:,0].mean():.3f}")
print(f"X[:,1] C_B : min={X[:,1].min():.3f}  max={X[:,1].max():.3f}  mean={X[:,1].mean():.3f}")
print(f"X[:,2] T_R : min={X[:,2].min():.2f}  max={X[:,2].max():.2f}  mean={X[:,2].mean():.2f}")
print(f"X[:,3] T_K : min={X[:,3].min():.2f}  max={X[:,3].max():.2f}  mean={X[:,3].mean():.2f}")
print(f"X[:,4] Q_k : min={X[:,4].min():.0f}  max={X[:,4].max():.0f}  mean={X[:,4].mean():.0f}")

print(f"\n=== Segment Y (reference trajectory) stats ===")
print(f"shape: {Y.shape}")
print(f"Y T_R: min={Y[:,:,2].min():.2f}  max={Y[:,:,2].max():.2f}  mean={Y[:,:,2].mean():.2f}")
print(f"Y T_K: min={Y[:,:,3].min():.2f}  max={Y[:,:,3].max():.2f}  mean={Y[:,:,3].mean():.2f}")

print(f"\n=== 샘플 5개 ===")
for i in range(5):
    print(f"  X[{i}] = C_A={X[i,0]:.3f} C_B={X[i,1]:.3f} "
          f"T_R={X[i,2]:.1f} T_K={X[i,3]:.1f} Q={X[i,4]:.0f}"
          f"  →  Y_end T_R={Y[i,-1,2]:.2f} T_K={Y[i,-1,3]:.2f}")

# normalization stats 확인
if "X_mean" in arr:
    print(f"\n=== Normalization stats ===")
    print(f"X_mean = {arr['X_mean']}")
    print(f"X_std  = {arr['X_std']}")
else:
    print("\n[!] X_mean/X_std 없음 — learner가 직접 계산")
    print(f"computed mean = {X.mean(axis=0)}")
    print(f"computed std  = {X.std(axis=0).clip(1e-6)}")
