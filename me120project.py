"""
ME120 — Tumor Growth PINN
=========================
Self-contained local script. Run with:

    python me120project.py

Sections (run in order, all in one process):
  1. Config & Parameters
  2. PDE Solver & Dataset Generation
  3. PDE Validation (no neural network needed)
  4. PINN Training & Evaluation

Outputs written to:  ./tumor_project_data/   and   ./pinn_output/
"""
# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — saves files, no GUI popup
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import json, os, time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — CONFIG & PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR = "./tumor_project_data/"
OUT_DIR  = "./pinn_output/"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUT_DIR,  exist_ok=True)

FIXED_PARAMS = {
    "K":      1.0,
    "D_n":    0.01,
    "alpha":  0.02,      # was 0.10 — reduced to prevent nutrient starvation
    "N":      64,
    "dx":     1.0 / 64,
    "dt":     0.001,
    "T":      10,
    "sigma":  6.0,
    "c0_max": 0.8,
}

PARAM_SWEEP = [
    {"D_c": 1.0e-3, "rho": 0.015, "alpha": 0.01, "label": "LGG Low-D / Low-rho"},
    {"D_c": 5.0e-3, "rho": 0.030, "alpha": 0.02, "label": "LGG Med-D / Low-rho"},
    {"D_c": 1.0e-2, "rho": 0.036, "alpha": 0.01, "label": "LGG High-D / Low-rho"},
    {"D_c": 1.0e-2, "rho": 0.090, "alpha": 0.02, "label": "GBM Low-D / Med-rho"},
    {"D_c": 2.0e-2, "rho": 0.120, "alpha": 0.02, "label": "GBM Med-D / High-rho"},
    {"D_c": 3.4e-2, "rho": 0.150, "alpha": 0.02, "label": "GBM High-D / High-rho"},
    {"D_c": 3.0e-3, "rho": 0.045, "alpha": 0.01, "label": "Mid Low-D / Low-rho"},
    {"D_c": 8.0e-3, "rho": 0.060, "alpha": 0.02, "label": "Mid Med-D / Med-rho"},
    {"D_c": 1.5e-2, "rho": 0.075, "alpha": 0.01, "label": "Mid High-D / Med-rho"},
    {"D_c": 2.5e-2, "rho": 0.105, "alpha": 0.02, "label": "Mid High-D / High-rho"},
    {"D_c": 1.0e-3, "rho": 0.150, "alpha": 0.02, "label": "Low-D / High-rho"},
    {"D_c": 3.4e-2, "rho": 0.015, "alpha": 0.01, "label": "High-D / Low-rho"},
]

VALIDATION_PARAMS = [
    {"D_c": 4.0e-3, "rho": 0.025, "alpha": 0.010, "label": "Val-1 LGG (interpolated)"},
    {"D_c": 1.8e-2, "rho": 0.045, "alpha": 0.020, "label": "Val-2 GBM (interpolated)"},
    {"D_c": 1.2e-2, "rho": 0.050, "alpha": 0.015, "label": "Val-3 Mid (near boundary)"},
]

# PINN config — mirrors FIXED_PARAMS where applicable
CFG = {
    "train_path": os.path.join(DATA_DIR, "dataset.npz"),
    "val_path":   os.path.join(DATA_DIR, "val_dataset.npz"),
    "out_dir":    OUT_DIR,
    "N":          FIXED_PARAMS["N"],
    "dx":         FIXED_PARAMS["dx"],
    "T":          float(FIXED_PARAMS["T"]),
    "K":          FIXED_PARAMS["K"],
    "D_n":        FIXED_PARAMS["D_n"],
    "D_c_min":    1e-3,  "D_c_max":  3.4e-2,
    "rho_min": 0.015,  "rho_max": 0.150, 
    "alpha_min": 0.01,  "alpha_max": 0.02,
    "enc_channels":      [1, 32, 64, 64, 128],
    "dec_channels":      [128, 64, 32, 16, 1],
    "film_hidden":       64,
    "n_params":          3,
    "epochs":            300,
    "batch_size":        6,
    "lr":                3e-4,
    "lr_decay":          0.95,
    "lr_decay_step":     20,
    "lambda_data":       1.0,
    "lambda_pde":        0.01,
    "lambda_bc":         0.05,
    "pde_warmup_epochs": 75,
    "device":            "cuda" if torch.cuda.is_available() else "cpu",
    "seed":              42,
}

# Save config for reference
config = {"DATA_DIR": DATA_DIR, "FIXED_PARAMS": FIXED_PARAMS,
          "PARAM_SWEEP": PARAM_SWEEP, "VALIDATION_PARAMS": VALIDATION_PARAMS}
with open(os.path.join(DATA_DIR, "config.json"), "w") as f:
    json.dump(config, f, indent=2)

torch.manual_seed(CFG["seed"])
np.random.seed(CFG["seed"])

# Unpack FIXED_PARAMS into local variables for convenience
FP     = FIXED_PARAMS
N      = FP["N"]
dx     = FP["dx"]
dt     = FP["dt"]
T      = FP["T"]
K      = FP["K"]
D_n    = FP["D_n"]
sigma  = FP["sigma"]
c0_max = FP["c0_max"]

print("=" * 62)
print("  ME120 — Tumor Growth PINN")
print("=" * 62)
print(f"  Grid: {N}×{N}  |  dx={dx:.4f}  |  dt={dt}  |  T={T} days")
print(f"  Training sims: {len(PARAM_SWEEP)}  |  Val sims: {len(VALIDATION_PARAMS)}")
print(f"  Device: {CFG['device']}")

# CFL stability check
print("\n── CFL Stability Check ──────────────────────────────────────────")
for p in PARAM_SWEEP:
    cfl_c = p["D_c"] * dt / dx**2
    cfl_n = D_n * dt / dx**2
    status = "✓ STABLE" if max(cfl_c, cfl_n) < 0.25 else "✗ UNSTABLE"
    print(f"  {p['label']:30s}  CFL_c={cfl_c:.5f}  CFL_n={cfl_n:.5f}  {status}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — PDE SOLVER & DATASET GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def make_initial_conditions(N, sigma, c0_max, nutrient_val=1.0):
    """Gaussian tumor seed centered in domain; uniform nutrient field."""
    x = np.linspace(0, 1, N)
    X, Y = np.meshgrid(x, x)
    c = c0_max * np.exp(-((X - 0.5)**2 + (Y - 0.5)**2) / (2 * (sigma / N)**2))
    n = np.ones((N, N)) * nutrient_val
    return c, n


def laplacian_2d(u, dx):
    """5-point FD Laplacian with Neumann (zero-flux) BCs."""
    lap = np.zeros_like(u)
    lap[1:-1, 1:-1] = (
        u[2:,   1:-1] + u[:-2,  1:-1] +
        u[1:-1, 2:]   + u[1:-1, :-2]  -
        4 * u[1:-1, 1:-1]
    ) / dx**2
    return lap


def run_pde(D_c, rho, alpha=None, K=None, D_n=None,
            N=None, T=None, dt=None, sigma=None, c0_max=None,
            save_every=200):
    """
    Solve coupled tumor + nutrient reaction-diffusion system.

      ∂c/∂t = D_c · ∇²c  +  ρ · c · n · (1 − c/K)
      ∂n/∂t = D_n · ∇²n  −  α · c · n

    All None arguments fall back to FIXED_PARAMS.
    """
    alpha  = alpha  if alpha  is not None else FP["alpha"]
    K      = K      if K      is not None else FP["K"]
    D_n    = D_n    if D_n    is not None else FP["D_n"]
    N      = N      if N      is not None else FP["N"]
    T      = T      if T      is not None else FP["T"]
    dt     = dt     if dt     is not None else FP["dt"]
    sigma  = sigma  if sigma  is not None else FP["sigma"]
    c0_max = c0_max if c0_max is not None else FP["c0_max"]

    dx_val = 1.0 / N
    cfl_c  = D_c * dt / dx_val**2
    cfl_n  = D_n * dt / dx_val**2
    if max(cfl_c, cfl_n) >= 0.25:
        raise ValueError(f"CFL violated: CFL_c={cfl_c:.3f}, CFL_n={cfl_n:.3f}")

    c, n = make_initial_conditions(N=N, sigma=sigma, c0_max=c0_max)
    snapshots, times, tumor_area = {}, [], []

    steps = int(T / dt)
    for step in range(steps + 1):
        t = round(step * dt, 6)
        if step % save_every == 0:
            snapshots[t] = (c.copy(), n.copy())
            times.append(t)
            tumor_area.append(np.sum(c > 0.01) * dx_val**2)
        if step == steps:
            break
        lap_c = laplacian_2d(c, dx_val)
        lap_n = laplacian_2d(n, dx_val)
        c = np.clip(c + dt * (D_c * lap_c + rho * c * n * (1.0 - c / K)), 0.0, K)
        n = np.clip(n + dt * (D_n * lap_n - alpha * c * n), 0.0, 1.0)

    return snapshots, np.array(times), np.array(tumor_area)


print("\n── Running PDE Simulations ─────────────────────────────────────────")
train_c_initial, train_c_final, train_n_final = [], [], []
train_params, train_areas, train_labels = [], [], []

t_start = time.time()
for i, p in enumerate(PARAM_SWEEP):
    snaps, times_arr, areas_arr = run_pde(
        D_c=p["D_c"], rho=p["rho"], alpha=p["alpha"],
        K=K, D_n=D_n, N=N, T=T, dt=dt, sigma=sigma, c0_max=c0_max,
        save_every=200)
    sk = sorted(snaps.keys())
    SNAPSHOT_TIMES = [5, 10, 20, 30]
    for t_target in SNAPSHOT_TIMES:
        # Find the closest saved snapshot to the target time
        closest_key = min(snaps.keys(), key=lambda t: abs(t - t_target))
        train_c_initial.append(snaps[sk[0]][0])          # always t=0 as input
        train_c_final.append(snaps[closest_key][0])      # target at this timestep
        train_n_final.append(snaps[closest_key][1])
        train_params.append([p["D_c"], p["rho"], p["alpha"]])
        train_areas.append(np.sum(snaps[closest_key][0] > 0.01) * dx**2)
        train_labels.append(f"{p['label']} t={t_target}")
    snap_c = np.array([snaps[k][0] for k in sk])
    snap_n = np.array([snaps[k][1] for k in sk])
    np.savez_compressed(os.path.join(DATA_DIR, f"snapshots_sim{i:02d}.npz"),
                        c=snap_c, n=snap_n, times=np.array(sk),
                        params=np.array([p["D_c"], p["rho"], p["alpha"]]))
    print(f"  [{i+1:2d}/{len(PARAM_SWEEP)}]  {p['label']:35s}"
          f"  area={areas_arr[-1]:.4f}  max_c={snaps[sk[-1]][0].max():.3f}")

train_c_initial = np.array(train_c_initial, dtype=np.float32)
train_c_final   = np.array(train_c_final,   dtype=np.float32)
train_n_final   = np.array(train_n_final,   dtype=np.float32)
train_params    = np.array(train_params,    dtype=np.float32)
train_areas     = np.array(train_areas,     dtype=np.float32)
print(f"\n[✓] Training dataset complete in {time.time()-t_start:.1f}s")

print("\n── Running Validation Simulations ──────────────────────────────────")
val_c_initial, val_c_final, val_n_final, val_params = [], [], [], []
for i, p in enumerate(VALIDATION_PARAMS):
    snaps, _, areas = run_pde(D_c=p["D_c"], rho=p["rho"], alpha=p["alpha"],
                               K=K, D_n=D_n, N=N, T=T, dt=dt,
                               sigma=sigma, c0_max=c0_max)
    sk = sorted(snaps.keys())
    val_c_initial.append(snaps[sk[0]][0])
    val_c_final.append(snaps[sk[-1]][0])
    val_n_final.append(snaps[sk[-1]][1])
    val_params.append([p["D_c"], p["rho"], p["alpha"]])
    print(f"  [{i+1}] {p['label']:35s}  area={areas[-1]:.4f}")

val_c_initial = np.array(val_c_initial, dtype=np.float32)
val_c_final   = np.array(val_c_final,   dtype=np.float32)
val_n_final   = np.array(val_n_final,   dtype=np.float32)
val_params    = np.array(val_params,    dtype=np.float32)

np.savez_compressed(os.path.join(DATA_DIR, "dataset.npz"),
                    c_initial=train_c_initial, c_final=train_c_final,
                    n_final=train_n_final, params=train_params, areas=train_areas)
np.savez_compressed(os.path.join(DATA_DIR, "val_dataset.npz"),
                    c_initial=val_c_initial, c_final=val_c_final,
                    n_final=val_n_final, params=val_params)
print(f"\n[✓] Datasets saved to {DATA_DIR}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — PDE VALIDATION (no neural network)
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 62)
print("  PDE SOLVER VALIDATION")
print("=" * 62)

train_data = np.load(os.path.join(DATA_DIR, "dataset.npz"))
val_data   = np.load(os.path.join(DATA_DIR, "val_dataset.npz"))
c_initial  = train_data["c_initial"]
c_final    = train_data["c_final"]
areas      = train_data["areas"]
vc_final   = val_data["c_final"]

# Test 1: Field bounds
print("\n── TEST 1: Field Bounds ─────────────────────────────────────")
all_pass = True
for i in range(len(PARAM_SWEEP)):
    snap  = np.load(os.path.join(DATA_DIR, f"snapshots_sim{i:02d}.npz"))
    c_all = snap["c"]; n_all = snap["n"]
    c_ok  = (c_all.min() >= -1e-6) and (c_all.max() <= K + 1e-6)
    n_ok  = (n_all.min() >= -1e-6) and (n_all.max() <= 1.0 + 1e-6)
    status = "PASS" if (c_ok and n_ok) else "FAIL"
    if not (c_ok and n_ok): all_pass = False
    print(f"  Sim {i+1:2d}  c∈[{c_all.min():.4f},{c_all.max():.4f}]  "
          f"n∈[{n_all.min():.4f},{n_all.max():.4f}]  [{status}]")
print(f"\n  Result: {'ALL PASS ✓' if all_pass else 'FAILURES DETECTED ✗'}")

# TEST 2: Total mass should be monotonically non-decreasing
# Mass = sum of all cell densities × dx² (no threshold)
print("\n── TEST 2: Total Tumor Mass (non-decreasing) ────────────────")
all_pass = True
for i in range(len(PARAM_SWEEP)):
    snap      = np.load(os.path.join(DATA_DIR, f"snapshots_sim{i:02d}.npz"))
    c_snaps   = snap["c"]
    times     = snap["times"]

    # Total mass at each timestep — no threshold
    mass_seq  = [c_snaps[j].sum() * dx**2 for j in range(len(times))]
    diffs     = np.diff(mass_seq)
    max_drop  = min(diffs) if len(diffs) > 0 else 0

    # Mass should never decrease by more than 0.1% (numerical noise tolerance)
    ok        = max_drop >= -1e-4
    if not ok: all_pass = False
    print(f"  Sim {i+1:2d}  mass: {mass_seq[0]:.4f} → {mass_seq[-1]:.4f}"
          f"  max_drop={max_drop:+.6f}  [{'PASS' if ok else 'FAIL'}]")

print(f"\n  Result: {'ALL PASS ✓' if all_pass else 'FAILURES DETECTED ✗'}")

# Test 3: Zero-flux BCs
# Test 3: Boundary conditions
print("\n── TEST 3: Zero-Flux Boundary Conditions ────────────────────")
all_pass = True
for i in range(len(PARAM_SWEEP)):
    snap    = np.load(os.path.join(DATA_DIR, f"snapshots_sim{i:02d}.npz"))
    c_snaps = snap["c"]

    c_t0 = c_snaps[0]
    c_tT = c_snaps[-1]

    edge_t0 = max(c_t0[0,  :].max(), c_t0[-1, :].max(),
                  c_t0[:,  0].max(), c_t0[:, -1].max())
    edge_tT = max(c_tT[0,  :].max(), c_tT[-1, :].max(),
                  c_tT[:,  0].max(), c_tT[:, -1].max())

    # Boundary density should not grow significantly over time
    ok = edge_tT <= edge_t0 + 0.01
    if not ok: all_pass = False

    print(f"  Sim {i+1:2d}  edge t=0: {edge_t0:.4f}  edge t=T: {edge_tT:.4f}"
          f"  delta={edge_tT - edge_t0:+.4f}  [{'PASS' if ok else 'FAIL'}]")

print(f"\n  Result: {'ALL PASS ✓' if all_pass else 'FAILURES DETECTED ✗'}")

# Test 4: Nutrient anti-correlation
print("\n── TEST 4: Nutrient–Tumor Anti-Correlation ──────────────────")
all_pass = True
n_final_arr = train_data["n_final"]
for i in range(len(PARAM_SWEEP)):
    corr   = np.corrcoef(c_final[i].flatten(), n_final_arr[i].flatten())[0, 1]
    ok     = corr < -0.3
    if not ok: all_pass = False
    print(f"  Sim {i+1:2d}  Pearson r = {corr:+.4f}  [{'PASS' if ok else 'FAIL'}]")
print(f"\n  Result: {'ALL PASS ✓' if all_pass else 'SOME WEAK CORRELATIONS ✗'}")

# Validation figures
tumor_cmap = LinearSegmentedColormap.from_list(
    "tumor", ["#0d0d1a","#1a1a4e","#3d1c8e","#8b2fc9","#e0529c","#ffa07a","#fff5e6"])
nutr_cmap  = LinearSegmentedColormap.from_list(
    "nutr",  ["#0d1117","#0a3d62","#1a6b8a","#2eb8b8","#7fffd4","#f0fff0"])
colors_p   = ["#e0529c","#4fc3f7","#81c784","#ffb74d","#ce93d8",
               "#80cbc4","#f48fb1","#aed6f1","#ff8a65","#b39ddb","#4db6ac","#fff176"]

val_areas = [np.sum(vc_final[i] > 0.005) * dx**2 for i in range(len(VALIDATION_PARAMS))]

# Figure A: growth curves
fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor="#0d1117")
fig.suptitle("PDE Validation — Growth Curves & Final Areas",
             color="white", fontsize=14, fontweight="bold")
for ax in axes:
    ax.set_facecolor("#111827"); ax.tick_params(colors="#8892a4")
    for sp in ax.spines.values(): sp.set_edgecolor("#2a2a4a")
    ax.grid(alpha=0.15, color="#3a3a5a")
for i in range(len(PARAM_SWEEP)):
    snap     = np.load(os.path.join(DATA_DIR, f"snapshots_sim{i:02d}.npz"))
    area_seq = [np.sum(snap["c"][j] > 0.1) * dx**2 for j in range(len(snap["times"]))]
    axes[0].plot(snap["times"], area_seq, color=colors_p[i], lw=2,
                 label=PARAM_SWEEP[i]["label"][:20], alpha=0.9)
for i in range(len(VALIDATION_PARAMS)):
    axes[0].axhline(val_areas[i], color="white", lw=1.2, ls=":",
                    alpha=0.5, label=f"Val{i+1} final" if i == 0 else "")
axes[0].set_xlabel("Time (days)", color="#8892a4")
axes[0].set_ylabel("Tumor Area Fraction", color="#8892a4")
axes[0].set_title("All Training Simulations", color="#aed6f1")
axes[0].legend(fontsize=6, facecolor="#0d1117", labelcolor="white",
               framealpha=0.6, ncol=2, loc="upper left")
rho_arr     = np.array([p["rho"] for p in PARAM_SWEEP])
dc_arr      = np.array([p["D_c"] * 1e4 for p in PARAM_SWEEP])
areas_final = np.array([
    train_areas[i] for i, label in enumerate(train_labels)
    if "t=30" in label
])
sc = axes[1].scatter(dc_arr, areas_final, c=rho_arr, cmap="plasma",
                     s=160, zorder=5, edgecolors="white", lw=0.8)
axes[1].scatter([p["D_c"]*1e4 for p in VALIDATION_PARAMS], val_areas,
                marker="*", s=280, color="#00ff88", zorder=6,
                edgecolors="white", lw=0.5, label="Validation")
cb = plt.colorbar(sc, ax=axes[1])
cb.set_label("ρ", color="white"); cb.ax.tick_params(colors="white")
plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")
axes[1].set_xlabel("D_c (×10⁻⁴)", color="#8892a4")
axes[1].set_ylabel("Final Tumor Area Fraction", color="#8892a4")
axes[1].set_title("Final Area vs D_c  (color=ρ, ★=val)", color="#aed6f1")
axes[1].legend(facecolor="#0d1117", labelcolor="white", fontsize=9, framealpha=0.7)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(OUT_DIR, "val_figA_growth.png"),
            dpi=150, bbox_inches="tight", facecolor="#0d1117")
plt.close()
print("\n[✓] Saved val_figA_growth.png")

# Figure B: initial vs final fields
fig = plt.figure(figsize=(20, 8), facecolor="#0d1117")
fig.suptitle("PDE Fields — Initial → Final  (all training simulations)",
             color="white", fontsize=13, fontweight="bold")
for i in range(12):
    ax  = fig.add_subplot(3, 8, i // 4 * 8 + (i % 4) * 2 + 1)
    ax2 = fig.add_subplot(3, 8, i // 4 * 8 + (i % 4) * 2 + 2)
    ax.imshow(c_initial[i],  origin="lower", cmap=tumor_cmap, vmin=0, vmax=1)
    ax2.imshow(c_final[i],   origin="lower", cmap=tumor_cmap, vmin=0, vmax=1)
    ax.set_xticks([]); ax.set_yticks([])
    ax2.set_xticks([]); ax2.set_yticks([])
    ax.set_title(f"S{i+1} t=0",   color="#7fb3d3", fontsize=7)
    ax2.set_title(f"S{i+1} t={T}", color="#e0529c", fontsize=7)
    for sp in list(ax.spines.values()) + list(ax2.spines.values()):
        sp.set_edgecolor("#2a2a4a")
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(OUT_DIR, "val_figB_fields.png"),
            dpi=150, bbox_inches="tight", facecolor="#0d1117")
plt.close()
print("[✓] Saved val_figB_fields.png")

# Figure C: validation hold-outs
fig, axes = plt.subplots(2, 3, figsize=(12, 7), facecolor="#0d1117")
fig.suptitle("Validation Hold-Out Simulations",
             color="white", fontsize=13, fontweight="bold")
for i in range(len(VALIDATION_PARAMS)):
    for row, (field, t_label) in enumerate([
        (val_c_initial[i], "t = 0"),
        (val_c_final[i],   f"t = {T}d"),
    ]):
        ax = axes[row, i]
        ax.set_facecolor("#111827")
        im = ax.imshow(field, origin="lower", cmap=tumor_cmap, vmin=0, vmax=1)
        ax.contour(field, levels=[0.1, 0.5], colors="white", alpha=0.4, linewidths=0.7)
        ax.set_xticks([]); ax.set_yticks([])
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cb.ax.tick_params(colors="white", labelsize=7)
        plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")
        title = (f"{VALIDATION_PARAMS[i]['label']}\n{t_label}" if row == 0
                 else f"area={val_areas[i]:.4f}  |  {t_label}")
        ax.set_title(title, color="#aed6f1" if row == 0 else "#e0529c", fontsize=8)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(OUT_DIR, "val_figC_holdout.png"),
            dpi=150, bbox_inches="tight", facecolor="#0d1117")
plt.close()
print("[✓] Saved val_figC_holdout.png")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — PINN ARCHITECTURE
# ─────────────────────────────────────────────────────────────────────────────

class TumorDataset(Dataset):
    def __init__(self, npz_path, cfg, augment=False):
        data         = np.load(npz_path)
        self.c_init  = torch.from_numpy(data["c_initial"]).float()
        self.c_final = torch.from_numpy(data["c_final"]).float()
        self.n_final = torch.from_numpy(data["n_final"]).float()
        raw_params   = torch.from_numpy(data["params"]).float()
        self.augment = augment

        D   = raw_params[:, 0]
        rho = raw_params[:, 1]
        alp = raw_params[:, 2]
        D_norm   = (torch.log10(D)   - np.log10(cfg["D_c_min"])) / \
                   (np.log10(cfg["D_c_max"]) - np.log10(cfg["D_c_min"]))
        rho_norm = (rho - cfg["rho_min"])   / (cfg["rho_max"]   - cfg["rho_min"])
        alp_norm = (alp - cfg["alpha_min"]) / (cfg["alpha_max"] - cfg["alpha_min"])
        self.params_norm = torch.stack([D_norm, rho_norm, alp_norm], dim=1)
        self.params_raw  = raw_params

    def __len__(self):
        return len(self.c_init)

    def __getitem__(self, idx):
        ci = self.c_init[idx].unsqueeze(0)
        cf = self.c_final[idx].unsqueeze(0)
        nf = self.n_final[idx].unsqueeze(0)
        if self.augment:
            k = torch.randint(0, 4, (1,)).item()
            ci = torch.rot90(ci, k, dims=[1, 2])
            cf = torch.rot90(cf, k, dims=[1, 2])
            nf = torch.rot90(nf, k, dims=[1, 2])
            if torch.rand(1) > 0.5:
                ci = torch.flip(ci, dims=[2])
                cf = torch.flip(cf, dims=[2])
                nf = torch.flip(nf, dims=[2])
        return {"c_init": ci, "c_final": cf, "n_final": nf,
                "params_norm": self.params_norm[idx],
                "params_raw":  self.params_raw[idx]}


class FiLM(nn.Module):
    """Feature-wise Linear Modulation: conditions spatial features on PDE params."""
    def __init__(self, n_params, n_channels, hidden=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(n_params, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden),   nn.SiLU(),
            nn.Linear(hidden, 2 * n_channels),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        self.mlp[-1].bias.data[:n_channels] = 1.0  # gamma starts at 1 (identity)

    def forward(self, x, params):
        gb = self.mlp(params)
        gamma, beta = gb.chunk(2, dim=1)
        return gamma.unsqueeze(-1).unsqueeze(-1) * x + beta.unsqueeze(-1).unsqueeze(-1)


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, residual=False):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, out_ch), out_ch), nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, out_ch), out_ch), nn.SiLU(),
        )
        self.residual = residual
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if (residual and in_ch != out_ch) else None

    def forward(self, x):
        out = self.conv(x)
        if self.residual:
            out = out + (self.skip(x) if self.skip is not None else x)
        return out


class TumorPINN(nn.Module):
    """
    U-Net encoder-decoder with FiLM parameter conditioning.
    Input:  c_initial (B,1,64,64) + params_norm (B,3)
    Output: c_final   (B,1,64,64)
    """
    def __init__(self, cfg):
        super().__init__()
        enc = cfg["enc_channels"]   # [1, 32, 64, 64, 128]
        dec = cfg["dec_channels"]   # [128, 64, 32, 16, 1]
        n_p = cfg["n_params"]
        fh  = cfg["film_hidden"]

        # Encoder
        self.enc0 = ConvBlock(enc[0], enc[1])
        self.enc1 = ConvBlock(enc[1], enc[2], residual=True)
        self.enc2 = ConvBlock(enc[2], enc[3], residual=True)
        self.enc3 = ConvBlock(enc[3], enc[4], residual=True)
        self.pool = nn.MaxPool2d(2)

        # FiLM at each encoder scale
        self.film0 = FiLM(n_p, enc[1], fh)
        self.film1 = FiLM(n_p, enc[2], fh)
        self.film2 = FiLM(n_p, enc[3], fh)
        self.film3 = FiLM(n_p, enc[4], fh)

        # Bottleneck
        self.bottleneck = ConvBlock(enc[4], enc[4], residual=True)
        self.film_bn    = FiLM(n_p, enc[4], fh)

        # Decoder — each dec block receives upsample_out + skip
        # Channels: up3_out=dec[1], skip_e3=enc[3]  → dec[1]+enc[3]
        self.up3  = nn.ConvTranspose2d(enc[4],  dec[1], 2, stride=2)
        self.dec3 = ConvBlock(dec[1] + enc[4],  dec[1], residual=True)

        self.up2  = nn.ConvTranspose2d(dec[1],  dec[2], 2, stride=2)
        self.dec2 = ConvBlock(dec[2] + enc[3],  dec[2], residual=True)

        self.up1  = nn.ConvTranspose2d(dec[2],  dec[3], 2, stride=2)
        self.dec1 = ConvBlock(dec[3] + enc[2],  dec[3], residual=True)

        self.head = nn.Sequential(
            nn.Conv2d(dec[3], dec[3], 3, padding=1), nn.SiLU(),
            nn.Conv2d(dec[3], 1, 1), nn.Sigmoid(),
        )

    def forward(self, c_init, params_norm):
        p  = params_norm
        e0 = self.film0(self.enc0(c_init),          p)   # (B, 32, 64, 64)
        e1 = self.film1(self.enc1(self.pool(e0)),    p)   # (B, 64, 32, 32)
        e2 = self.film2(self.enc2(self.pool(e1)),    p)   # (B, 64, 16, 16)
        e3 = self.film3(self.enc3(self.pool(e2)),    p)   # (B,128,  8,  8)
        bn = self.film_bn(self.bottleneck(self.pool(e3)), p)  # (B,128, 4, 4)

        d3 = self.dec3(torch.cat([self.up3(bn), e3], dim=1))  # (B, 64,  8,  8)
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))  # (B, 32, 16, 16)
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))  # (B, 16, 32, 32)
        d0 = F.interpolate(d1, scale_factor=2, mode="bilinear", align_corners=False)
        return self.head(d0)                                   # (B,  1, 64, 64)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — PHYSICS LOSSES
# ─────────────────────────────────────────────────────────────────────────────

def laplacian_2d_torch(u, dx):
    """5-point FD Laplacian with reflect padding (zero-flux BC). u: (B,1,H,W)"""
    u_pad = F.pad(u, (1, 1, 1, 1), mode="reflect")
    return (u_pad[:, :, 2:,  1:-1] + u_pad[:, :, :-2, 1:-1] +
            u_pad[:, :, 1:-1, 2:] + u_pad[:, :, 1:-1, :-2] -
            4 * u) / (dx ** 2)


def pde_residual_loss(c_pred, c_init, params_raw, cfg):
    """
    PDE residual: R = ∂c/∂t - D_c·∇²c - ρ·c·n·(1-c/K)
    Approximates n via quasi-steady: n ≈ clamp(1 - (α/D_n)·c, 0, 1)
    """
    B     = c_pred.shape[0]
    D_c   = params_raw[:, 0].view(B, 1, 1, 1)
    rho   = params_raw[:, 1].view(B, 1, 1, 1)
    alpha = params_raw[:, 2].view(B, 1, 1, 1)
    dc_dt    = (c_pred - c_init) / cfg["T"]
    lap_c    = laplacian_2d_torch(c_pred, cfg["dx"])
    n_approx = torch.clamp(1.0 - (alpha / cfg["D_n"]) * c_pred, 0.0, 1.0)
    residual = dc_dt - D_c * lap_c - rho * c_pred * n_approx * (1.0 - c_pred / cfg["K"])
    return (residual ** 2).mean()


def boundary_loss(c_pred):
    """Zero normal gradient at all 4 edges."""
    return (((c_pred[:, :, 0,  :] - c_pred[:, :, 1,  :]) ** 2).mean() +
            ((c_pred[:, :, -1, :] - c_pred[:, :, -2, :]) ** 2).mean() +
            ((c_pred[:, :, :,  0] - c_pred[:, :, :,  1]) ** 2).mean() +
            ((c_pred[:, :, :, -1] - c_pred[:, :, :, -2]) ** 2).mean())


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def train(cfg):
    device = torch.device(cfg["device"])

    train_ds     = TumorDataset(cfg["train_path"], cfg, augment=True)
    val_ds       = TumorDataset(cfg["val_path"],   cfg, augment=False)
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"],
                              shuffle=True, drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=len(val_ds),
                              shuffle=False, drop_last=False)

    model     = TumorPINN(cfg).to(device)

    checkpoint_path = os.path.join(cfg["out_dir"], "best_model.pt")
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"[✓] Warm-started from {checkpoint_path}")
    else:
        print("[i] No checkpoint found — training from scratch")

    n_params  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=cfg["lr_decay_step"], gamma=cfg["lr_decay"])

    print(f"\n[✓] Model parameters: {n_params:,}")
    print(f"\n{'Epoch':>6}  {'Train':>10}  {'Data':>9}  {'PDE':>9}  "
          f"{'BC':>8}  {'Val MSE':>10}  {'LR':>8}")
    print("─" * 72)

    history      = {k: [] for k in
                    ["train_total","train_data","train_pde","train_bc","val_mse","val_ssim"]}
    best_val_mse = float("inf")
    best_epoch   = 0

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        ep_total = ep_data = ep_pde = ep_bc = 0.0
        n_batches = 0
        pde_weight = cfg["lambda_pde"] * min(1.0, epoch / cfg["pde_warmup_epochs"])

        for batch in train_loader:
            c_init  = batch["c_init"].to(device)
            c_final = batch["c_final"].to(device)
            p_norm  = batch["params_norm"].to(device)
            p_raw   = batch["params_raw"].to(device)

            optimizer.zero_grad()
            c_pred = model(c_init, p_norm)
            l_data = F.mse_loss(c_pred, c_final)
            l_pde  = pde_residual_loss(c_pred, c_init, p_raw, cfg)
            l_bc   = boundary_loss(c_pred)
            loss   = cfg["lambda_data"] * l_data + pde_weight * l_pde + cfg["lambda_bc"] * l_bc
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            ep_total += loss.item(); ep_data += l_data.item()
            ep_pde   += l_pde.item(); ep_bc   += l_bc.item()
            n_batches += 1

        scheduler.step()
        ep_total /= n_batches; ep_data /= n_batches
        ep_pde   /= n_batches; ep_bc   /= n_batches

        model.eval()
        val_mse = val_ssim = 0.0
        with torch.no_grad():
            for batch in val_loader:
                ci = batch["c_init"].to(device)
                cf = batch["c_final"].to(device)
                pn = batch["params_norm"].to(device)
                cp = model(ci, pn)
                val_mse  += F.mse_loss(cp, cf).item()
                c_p = cp.view(cp.shape[0], -1)
                c_t = cf.view(cf.shape[0], -1)
                cov  = ((c_p - c_p.mean(1, keepdim=True)) *
                        (c_t - c_t.mean(1, keepdim=True))).mean(1)
                val_ssim += (cov / (c_p.std(1).clamp(1e-8) * c_t.std(1).clamp(1e-8))).mean().item()

        history["train_total"].append(ep_total); history["train_data"].append(ep_data)
        history["train_pde"].append(ep_pde);     history["train_bc"].append(ep_bc)
        history["val_mse"].append(val_mse);      history["val_ssim"].append(val_ssim)

        if val_mse < best_val_mse:
            best_val_mse = val_mse; best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(cfg["out_dir"], "best_model.pt"))

        if epoch % 10 == 0 or epoch == 1:
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"{epoch:>6}  {ep_total:>10.6f}  {ep_data:>9.6f}  "
                  f"{ep_pde:>9.6f}  {ep_bc:>8.6f}  {val_mse:>10.6f}  {lr_now:>8.2e}")

    print(f"\n[✓] Best val MSE: {best_val_mse:.6f} at epoch {best_epoch}")
    return model, history, val_ds


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — EVALUATION & PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_and_plot(model, val_ds, history, cfg):
    device = torch.device(cfg["device"])
    model.eval()

    BG, PBG = "#0d1117", "#111827"
    TC, MC  = "#e2e8f0", "#6b7280"
    tcmap   = LinearSegmentedColormap.from_list("tumor",
        ["#0d0d1a","#1a1a4e","#3d1c8e","#8b2fc9","#e0529c","#ffa07a","#fff5e6"])
    plt.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": PBG,
        "text.color": TC, "axes.labelcolor": MC,
        "xtick.color": MC, "ytick.color": MC,
        "axes.edgecolor": "#2a2a4a", "grid.color": "#1f2937",
        "axes.grid": True, "grid.alpha": 0.25,
        "font.family": "monospace",
    })

    # Training curves
    fig, axes = plt.subplots(1, 3, figsize=(16, 4), facecolor=BG)
    fig.suptitle("PINN Training History", color=TC, fontsize=13, fontweight="bold")
    ep_range = range(1, len(history["train_total"]) + 1)
    axes[0].semilogy(ep_range, history["train_total"], color="#60a5fa", lw=2,  label="Total")
    axes[0].semilogy(ep_range, history["train_data"],  color="#34d399", lw=1.5, ls="--", label="Data")
    axes[0].semilogy(ep_range, history["train_pde"],   color="#fbbf24", lw=1.5, ls="--", label="PDE")
    axes[0].semilogy(ep_range, history["train_bc"],    color="#f472b6", lw=1.5, ls="--", label="BC")
    axes[0].set_title("Training Losses", color=TC)
    axes[0].legend(fontsize=9, framealpha=0.2, facecolor=BG, labelcolor=TC)
    axes[1].semilogy(ep_range, history["val_mse"], color="#f87171", lw=2)
    axes[1].set_title("Validation MSE", color=TC)
    axes[2].plot(ep_range, history["val_ssim"], color="#a78bfa", lw=2)
    axes[2].set_title("Validation Spatial Correlation", color=TC)
    for ax in axes:
        ax.set_xlabel("Epoch")
    plt.tight_layout()
    plt.savefig(os.path.join(cfg["out_dir"], "training_curves.png"),
                dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print("[✓] Saved training_curves.png")

    # Predictions vs ground truth
    domain     = cfg["N"] * cfg["dx"]
    VAL_LABELS = ["Val-1 LGG", "Val-2 GBM", "Val-3 Mid"]
    all_pred, all_true = [], []

    fig = plt.figure(figsize=(18, 14), facecolor=BG)
    fig.suptitle("PINN Predictions vs. Ground Truth — Validation Set",
                 color=TC, fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.45, wspace=0.3)

    with torch.no_grad():
        for i in range(len(val_ds)):
            s       = val_ds[i]
            ci      = s["c_init"].unsqueeze(0).to(device)
            pn      = s["params_norm"].unsqueeze(0).to(device)
            pr      = s["params_raw"]
            cf_true = s["c_final"].squeeze().cpu().numpy()
            c_pred  = model(ci, pn).squeeze().cpu().numpy()
            all_pred.append(c_pred); all_true.append(cf_true)
            vmax    = max(cf_true.max(), c_pred.max(), 0.01)
            mse_i   = np.mean((c_pred - cf_true) ** 2)
            err     = np.abs(c_pred - cf_true)

            for row, (field, title, cmap_, v0, v1) in enumerate([
                (s["c_init"].squeeze().numpy(), f"{VAL_LABELS[i]}\nc_initial",
                 tcmap, 0, 1),
                (cf_true,  f"Ground truth\nD={pr[0]:.4f} ρ={pr[1]:.4f}",
                 tcmap, 0, vmax),
                (c_pred,   f"PINN prediction\nMSE={mse_i:.5f}",
                 tcmap, 0, vmax),
                (err,      f"Absolute error\nmax={err.max():.4f}",
                 "hot", 0, err.max()),
            ]):
                ax = fig.add_subplot(gs[row, i])
                im = ax.imshow(field, origin="lower", cmap=cmap_, vmin=v0, vmax=v1,
                               extent=[0, domain, 0, domain])
                ax.set_title(title, color=["#93c5fd","#34d399","#fbbf24","#f87171"][row],
                             fontsize=9)
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02).ax.tick_params(
                    colors="white", labelsize=7)

    plt.savefig(os.path.join(cfg["out_dir"], "predictions.png"),
                dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print("[✓] Saved predictions.png")

    # Pixel scatter
    fig, ax = plt.subplots(figsize=(7, 6), facecolor=BG)
    for i, (p_arr, t_arr) in enumerate(zip(all_pred, all_true)):
        ax.scatter(t_arr.flatten()[::4], p_arr.flatten()[::4],
                   s=1.5, alpha=0.3, color=["#60a5fa","#34d399","#f472b6"][i],
                   label=f"Val {i+1}")
    lo = min(p.min() for p in all_pred + all_true)
    hi = max(p.max() for p in all_pred + all_true)
    ax.plot([lo, hi], [lo, hi], "w--", lw=1.5, alpha=0.6, label="Perfect")
    ax.set_xlabel("True c_final"); ax.set_ylabel("Predicted c_final")
    ax.set_title("Pixel-Level Accuracy (every 4th pixel)", color=TC)
    ax.legend(fontsize=9, framealpha=0.2, facecolor=BG, labelcolor=TC, markerscale=6)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg["out_dir"], "scatter.png"),
                dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print("[✓] Saved scatter.png")
    print(f"\n[✓] All outputs in {cfg['out_dir']}")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n── Training PINN ────────────────────────────────────────────────")

    # Warm-start from existing checkpoint if available
    CFG["epochs"] = 400
    CFG["lr"]     = 3e-4   # lower than before — fine-tuning on clean data

    model, history, val_ds = train(CFG)

    print("\n── Loading best checkpoint ──────────────────────────────────────")
    model.load_state_dict(torch.load(
        os.path.join(CFG["out_dir"], "best_model.pt"),
        map_location=CFG["device"],
        weights_only=True))   # suppresses the FutureWarning in newer torch

    evaluate_and_plot(model, val_ds, history, CFG)

    # ── Final metrics summary ─────────────────────────────────────
    print("\n── Final Validation Metrics ─────────────────────────────────────")
    device = torch.device(CFG["device"])
    model.eval()
    val_mses, val_maxes = [], []
    with torch.no_grad():
        for i in range(len(val_ds)):
            s      = val_ds[i]
            ci     = s["c_init"].unsqueeze(0).to(device)
            pn     = s["params_norm"].unsqueeze(0).to(device)
            cf     = s["c_final"].squeeze().numpy()
            cp     = model(ci, pn).squeeze().cpu().numpy()
            mse    = float(np.mean((cp - cf)**2))
            maxerr = float(np.abs(cp - cf).max())
            area_true = float(np.sum(cf > 0.01) * dx**2)
            area_pred = float(np.sum(cp > 0.01) * dx**2)
            val_mses.append(mse)
            val_maxes.append(maxerr)
            print(f"  Val {i+1}  MSE={mse:.5f}  MaxErr={maxerr:.4f}"
                  f"  Area_true={area_true:.4f}  Area_pred={area_pred:.4f}"
                  f"  ΔArea={abs(area_pred-area_true):.4f}")
    print(f"\n  Mean MSE : {np.mean(val_mses):.5f}")
    print(f"  Mean MaxErr: {np.mean(val_maxes):.4f}")
    print("\n[✓] Done.")