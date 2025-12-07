#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vol_surface_seq2seq_experiments.py — encoder–decoder Transformer for SPX volatility surface
with feature ablation experiments.

We train multiple models (experiments) differing only in INPUT FEATURES:

Examples (configured in EXPERIMENTS):
  - E0: log returns only
  - E1: log returns + RVOL
  - E2: log returns + VIX daily variance
  - E3: log returns + RVOL + VIX daily variance
  - E4: E3 + past volatility surface as encoder input

For each experiment we:
  - Train on next-day IV surface (residual on log-IV)
  - Compute R^2 per (days, delta) bucket (IS & OOS)
  - Summarise mean R^2 (IS, OOS) across the surface
  - Plot:
      * Bar chart of mean R^2 per experiment
      * ΔOOS R^2 heatmaps vs a chosen baseline experiment

Data:
  - Daily file (Vol_Index.csv), e.g.
        Date,VIX,RVOL,SPX,SPX_OX,RVOL_5min
  - Options file with surface, e.g. columns
        secid,date,days,delta,impl_volatility,impl_strike,impl_premiu,...

All plots saved to: ~/Desktop/Transformer_images_surface
Model weights saved to: ./outputs_ed_surface
"""

import math, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# ---------- Matplotlib ----------
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.dates as mdates

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 14,
    "mathtext.default": "regular"
})

# ===== Color theme =====
C_PRED        = "r"   # model prediction
C_TRUE        = "k"   # ground truth
GRID_ALPHA    = 0.35

# ---------------- CONFIG ----------------
# Daily SPX/VIX/RVOL data
CSV_PATH       = Path("Vol_Index.csv")

# Options surface data (date, days, delta, impl_volatility, ...)
SURF_CSV_PATH  = Path("ihp3jscbmyr4llbs.csv")    # your surface file

OUT_DIR  = Path("outputs_ed_surface")      # model weights
IMG_DIR  = Path.home() / "Desktop" / "Transformer_images_surface"

DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED     = 42

# Data / model hyperparameters
BATCH_SIZE   = 128
BLOCK_SIZE   = 21         # encoder context length (past days)
HORIZON      = 1          # predict next-day surface (can set >1 if desired)
EMBED_DIM    = 32
N_HEADS      = 4
N_LAYERS_ENC = 2
N_LAYERS_DEC = 2
DROPOUT      = 0.3
LR_SCALE     = 1.0
WARMUP_STEPS = 4000
EPOCHS       = 200
GRAD_CLIP    = 1.0
PATIENCE     = 1000
WEIGHT_DECAY = 1e-3       # AdamW

# Residual target on log-IV (recommended True)
USE_RESIDUAL_TARGET = True

# Split for IS/OOS
SPLIT_DATE = pd.Timestamp("2009-01-01")

# Annualisation constants (for VIX daily variance feature)
ANN_SQRT = math.sqrt(252.0)

# Vol surface grid
# If set to None, we infer ALL unique 'days' and 'delta' values from the surface file.
SURF_DAYS   = None      # e.g. [10, 30, 60, 91, 182, 365] if you want to restrict
SURF_DELTAS = None      # e.g. [-80, -60, -40, -20, 20, 40, 60, 80]

# ---- Experiments: feature-set ablations ----
EXPERIMENTS = [
    {
        "name": "E0_ret",
        "label": "log ret",
        "use_rvol": False,
        "use_vixd": False,
        "include_past_surface": False,
    },
    {
        "name": "E1_ret_rvol",
        "label": "log ret + RVOL",
        "use_rvol": True,
        "use_vixd": False,
        "include_past_surface": False,
    },
    {
        "name": "E2_ret_vix",
        "label": "log ret + VIX var",
        "use_rvol": False,
        "use_vixd": True,
        "include_past_surface": False,
    },
    {
        "name": "E3_ret_rvol_vix",
        "label": "log ret + RVOL + VIX var",
        "use_rvol": True,
        "use_vixd": True,
        "include_past_surface": False,
    },
    {
        "name": "E4_ret_rvol_vix_pastSurf",
        "label": "E3 + past surface",
        "use_rvol": True,
        "use_vixd": True,
        "include_past_surface": True,
    },
]

# Choose which experiment is the baseline for ΔR² heatmaps
BASELINE_EXP_NAME = "E3_ret_rvol_vix"

# Make dirs
OUT_DIR.mkdir(parents=True, exist_ok=True)
IMG_DIR.mkdir(parents=True, exist_ok=True)
torch.manual_seed(SEED)
np.random.seed(SEED)

# ---------------- Utils ----------------
EPS = 1e-12
YEAR_FMT = mdates.DateFormatter('%Y')

def r2_score(y_true, y_pred):
    ss_res = float(((y_true - y_pred)**2).sum())
    ss_tot = float(((y_true - y_true.mean())**2).sum())
    return float(1 - ss_res/ss_tot) if ss_tot > 0 else float("nan")

def savefig(fname, dpi=300):
    path = IMG_DIR / fname
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"[saved] {path}")

# ---------------- Generic CSV loader with Date column ----------------
def load_with_Date_col(path):
    """
    Read a CSV and ensure it has a 'Date' column parsed as datetime.
    Accepts either 'Date' or 'date' in the raw file.
    """
    df = pd.read_csv(path)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
    elif "date" in df.columns:
        df["Date"] = pd.to_datetime(df["date"])
        df = df.drop(columns=["date"])
    else:
        raise ValueError(f"No Date/date column in {path}")
    df.sort_values("Date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

# ---------------- Surface construction ----------------
def build_surface_df(opt_path=SURF_CSV_PATH,
                     days_grid=SURF_DAYS,
                     delta_grid=SURF_DELTAS):
    """
    Build a per-date volatility surface on a fixed (days, delta) grid.

    If days_grid or delta_grid is None, we infer all unique values from the data.

    Returns:
        surf_df: DataFrame with columns
                 Date, IV_d{days}_D{delta} for each grid point
        days_grid, delta_grid: lists actually used
    """
    opt = load_with_Date_col(opt_path)

    if "days" not in opt.columns or "delta" not in opt.columns:
        raise ValueError("Options file must contain 'days' and 'delta' columns.")
    if "impl_volatility" not in opt.columns:
        raise ValueError("Options file must contain 'impl_volatility' column.")

    if days_grid is None:
        days_grid = sorted(opt["days"].unique().tolist())
    if delta_grid is None:
        delta_grid = sorted(opt["delta"].unique().tolist())

    opt_sub = opt[opt["days"].isin(days_grid) & opt["delta"].isin(delta_grid)].copy()

    pivot = opt_sub.pivot_table(
        index="Date",
        columns=["days", "delta"],
        values="impl_volatility",
        aggfunc="mean"
    )

    cols = []
    for d in days_grid:
        for k in delta_grid:
            cols.append((d, k))
    pivot = pivot.reindex(columns=cols)

    new_cols = [f"IV_d{d}_D{int(delta)}" for (d, delta) in pivot.columns]
    pivot.columns = new_cols

    surf_df = pivot.reset_index()
    surf_df.sort_values("Date", inplace=True)
    surf_df.reset_index(drop=True, inplace=True)

    surf_df[new_cols] = surf_df[new_cols].ffill().bfill()

    return surf_df, days_grid, delta_grid

# ---------------- Daily data + surface merge ----------------
def load_prepare_surface(csv_path=CSV_PATH, opt_path=SURF_CSV_PATH):
    """
    Returns:
      df_all     : merged daily + surface dataframe
      surf_cols  : list of IV_* columns (surface)
      y_cols     : list of target columns y_IV_*
      surf_days, surf_deltas: grids used for plotting
    """
    df = load_with_Date_col(csv_path)

    df["SPX"] = pd.to_numeric(df["SPX"], errors="coerce")
    df["log_ret"] = np.log(df["SPX"] / df["SPX"].shift(1))

    if "RVOL_5min" in df.columns:
        df["rvol"] = pd.to_numeric(df["RVOL_5min"], errors="coerce").clip(lower=EPS)
    elif "RVOL" in df.columns:
        df["rvol"] = pd.to_numeric(df["RVOL"], errors="coerce").clip(lower=EPS)
    else:
        raise ValueError("Need RVOL_5min or RVOL column in daily CSV")

    if "VIX" in df.columns:
        df["VIX"] = pd.to_numeric(df["VIX"], errors="coerce")
        df["vixd"] = (df["VIX"] / (100.0 * ANN_SQRT)) ** 2
    else:
        df["vixd"] = np.nan
    df["vixd"] = df["vixd"].ffill().bfill().astype(float).clip(lower=EPS)

    df["rvol_feat"] = df["rvol"].astype(float)
    df["vixd_feat"] = df["vixd"].astype(float)

    surf_df, surf_days, surf_deltas = build_surface_df(opt_path)

    df_all = pd.merge(df, surf_df, on="Date", how="inner")
    df_all.sort_values("Date", inplace=True)
    df_all.reset_index(drop=True, inplace=True)

    surf_cols = [c for c in df_all.columns if c.startswith("IV_d")]
    if not surf_cols:
        raise RuntimeError("No IV_* columns found after merging surface data.")

    log_iv_today = np.log(df_all[surf_cols].astype(float).clip(lower=EPS))
    log_iv_next  = log_iv_today.shift(-1)

    y_cols = []
    for c in surf_cols:
        y_name = f"y_{c}"
        if USE_RESIDUAL_TARGET:
            df_all[y_name] = log_iv_next[c] - log_iv_today[c]
        else:
            df_all[y_name] = log_iv_next[c]
        y_cols.append(y_name)

    df_all.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_all.dropna(inplace=True)
    df_all.reset_index(drop=True, inplace=True)

    return df_all, surf_cols, y_cols, surf_days, surf_deltas

# ---------------- Standardisation ----------------
def make_std_views_surface(train_df, other_df_list,
                           feat_cols, surf_cols, y_cols,
                           block_size=BLOCK_SIZE, horizon=HORIZON):
    """
    Standardise features and targets using TRAIN statistics.
    """
    all_feat = feat_cols + surf_cols

    feat_mu = {c: float(train_df[c].mean()) for c in all_feat}
    feat_sd = {c: float(train_df[c].std() + 1e-12) for c in all_feat}

    y_mu = train_df[y_cols].mean().to_numpy(dtype=np.float32)
    y_sd = (train_df[y_cols].std() + 1e-12).to_numpy(dtype=np.float32)

    def std_one(df_in):
        cols = ["Date"] + all_feat + y_cols
        tmp = df_in[cols].copy()

        for c in all_feat:
            tmp[c] = (tmp[c] - feat_mu[c]) / feat_sd[c]

        y_arr = df_in[y_cols].to_numpy(dtype=np.float32)
        y_std = (y_arr - y_mu) / y_sd
        for i, c in enumerate(y_cols):
            tmp[c] = y_std[:, i]

        return tmp

    outs = [std_one(d) for d in other_df_list]
    train_std = std_one(train_df)

    stats = {
        "feat_cols": feat_cols,
        "surf_cols": surf_cols,
        "y_cols": y_cols,
        "feat_mu": feat_mu,
        "feat_sd": feat_sd,
        "y_mu": y_mu,
        "y_sd": y_sd,
        "block_size": block_size,
        "horizon": horizon,
        "use_residual": USE_RESIDUAL_TARGET
    }
    return train_std, outs, stats

# ---------------- Dataset ----------------
class VolSurfSeq2SeqDataset(Dataset):
    """
    Returns (X_src, y_prev, y_tgt):

      X_src : (T, enc_in_dim) standardised features
      y_prev: (1, D) previous standardised surface target (seed)
      y_tgt : (H, D) H-step standardised target surfaces
    """
    def __init__(self, df_std, y_cols, feat_cols, surf_cols,
                 block_size, horizon, include_past_surface=False):
        super().__init__()
        self.df = df_std.reset_index(drop=True)
        self.T = block_size
        self.H = horizon
        self.y_cols = y_cols
        self.D = len(y_cols)
        self.feat_cols = feat_cols
        self.surf_cols = surf_cols if include_past_surface else []
        self.include_past_surface = include_past_surface

        self.X = self.df[self.feat_cols + self.surf_cols].to_numpy(dtype=np.float32)
        self.Y = self.df[self.y_cols].to_numpy(dtype=np.float32)

        self.N = len(self.df)
        self.length = max(0, self.N - self.T - self.H + 1)
        if self.length <= 0:
            raise RuntimeError("Not enough rows for chosen BLOCK_SIZE and HORIZON")

        self.enc_in_dim = len(self.feat_cols + self.surf_cols)

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        x = self.X[i:i+self.T, :]
        j = i + self.T - 1
        y_seq  = self.Y[j:j+self.H, :]
        if j-1 >= 0:
            y_prev = self.Y[j-1, :][None, :]
        else:
            y_prev = np.zeros((1, self.D), dtype=np.float32)
        return (
            torch.from_numpy(x),
            torch.from_numpy(y_prev),
            torch.from_numpy(y_seq)
        )

# ---------------- Model ----------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=20000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) *
                        (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

def causal_mask(L, device):
    return torch.triu(torch.ones(L, L, dtype=torch.bool, device=device), diagonal=1)

class EDTransformer(nn.Module):
    """
    Encoder–decoder Transformer for vector-valued targets (vol surface).
    """
    def __init__(self, d_model=64, n_heads=4, n_enc=4, n_dec=4,
                 d_ff=None, dropout=0.1, H=1, enc_in_dim=3, surf_dim=1):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.H = H
        self.surf_dim = surf_dim

        self.x_proj = nn.Linear(enc_in_dim, d_model)
        self.pos    = PositionalEncoding(d_model)

        enc = nn.TransformerEncoderLayer(d_model, n_heads, d_ff,
                                         dropout=dropout, batch_first=True)
        dec = nn.TransformerDecoderLayer(d_model, n_heads, d_ff,
                                         dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, n_enc)
        self.decoder = nn.TransformerDecoder(dec, n_dec)

        self.y_proj    = nn.Linear(surf_dim, d_model)
        self.bos_embed = nn.Parameter(torch.zeros(1, 1, d_model))
        self.head      = nn.Linear(d_model, surf_dim)

    def forward(self, x_src, y_prev, y_tgt):
        B, H, D = y_tgt.size()
        mem = self.encoder(self.pos(self.x_proj(x_src)))

        bos  = self.bos_embed.expand(B, 1, -1)
        y0   = self.y_proj(y_prev)
        toks = torch.cat([bos, y0], dim=1)
        if H > 1:
            toks = torch.cat([toks, self.y_proj(y_tgt[:, :-1, :])], dim=1)

        toks = self.pos(toks)
        mask = causal_mask(toks.size(1), x_src.device)
        dec  = self.decoder(tgt=toks, memory=mem, tgt_mask=mask)
        out  = self.head(dec[:, -H:, :])
        return out

    @torch.no_grad()
    def greedy_decode(self, x_src, H=None, first_val_std=None):
        H = H or self.H
        B = x_src.size(0)
        device = x_src.device

        mem = self.encoder(self.pos(self.x_proj(x_src)))

        bos = self.bos_embed.expand(B, 1, -1)

        if first_val_std is None:
            first_val_std = torch.zeros(B, self.surf_dim, device=device)
        else:
            first_val_std = first_val_std.to(device)
            if first_val_std.dim() == 1:
                first_val_std = first_val_std.unsqueeze(0)
            if first_val_std.size(0) != B:
                first_val_std = first_val_std.expand(B, -1)

        y_prev = first_val_std.unsqueeze(1)
        prev_emb = self.y_proj(y_prev)
        hist_emb = torch.zeros(B, 0, prev_emb.size(-1), device=device)

        preds = []
        for _ in range(H):
            toks = torch.cat([bos, prev_emb, hist_emb], dim=1)
            toks = self.pos(toks)
            mask = causal_mask(toks.size(1), device)
            dec  = self.decoder(tgt=toks, memory=mem, tgt_mask=mask)
            next_std = self.head(dec[:, -1:, :])
            preds.append(next_std)
            hist_emb = torch.cat([hist_emb, self.y_proj(next_std)], dim=1)

        return torch.cat(preds, dim=1)

# ---------------- Noam Scheduler ----------------
class NoamLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, d_model, warmup_steps=4000, scale=1.0, last_epoch=-1):
        self.d_model = d_model
        self.warmup = warmup_steps
        self.scale = scale
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = max(1, self._step_count)
        factor = self.scale * (self.d_model ** -0.5) * \
                 min(step ** -0.5, step * (self.warmup ** -1.5))
        return [factor for _ in self.optimizer.param_groups]

# ---------------- Train / Eval helpers ----------------
def train_epoch(model, dl, optimizer, scheduler, loss_fn):
    model.train()
    total = 0.0
    nobs = 0
    for xb, y0b, yb in dl:
        xb, y0b, yb = xb.to(DEVICE), y0b.to(DEVICE), yb.to(DEVICE)
        pred = model(xb, y0b, yb)
        loss = loss_fn(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        scheduler.step()
        bs = xb.size(0)
        total += float(loss.item()) * bs
        nobs += bs
    return total / max(1, nobs)

@torch.no_grad()
def eval_epoch(model, dl, loss_fn):
    model.eval()
    total = 0.0
    nobs = 0
    for xb, y0b, yb in dl:
        xb, y0b, yb = xb.to(DEVICE), y0b.to(DEVICE), yb.to(DEVICE)
        pred = model(xb, y0b, yb)
        loss = loss_fn(pred, yb)
        bs = xb.size(0)
        total += float(loss.item()) * bs
        nobs += bs
    return total / max(1, nobs)

@torch.no_grad()
def roll_forecast_surface(model, df_std, df_raw, stats,
                          include_past_surface=False):
    """
    Rolling forecasts for the volatility surface.

    Returns:
      preds: (N,H,D) predicted IV levels
      trues: (N,H,D) true IV levels
      anchor_dates: (N,) dates of first horizon (t+1) for each window
    """
    T = stats["block_size"]
    H = stats["horizon"]
    feat_cols = stats["feat_cols"]
    surf_cols = stats["surf_cols"]
    y_cols = stats["y_cols"]
    y_mu = stats["y_mu"]
    y_sd = stats["y_sd"]
    use_resid = bool(stats.get("use_residual", False))

    X_cols = feat_cols + (surf_cols if include_past_surface else [])
    X = df_std[X_cols].to_numpy(dtype=np.float32)
    Y_std = df_std[y_cols].to_numpy(dtype=np.float32)
    dates = df_raw["Date"].to_numpy()

    base_iv = df_raw[surf_cols].astype(float).to_numpy()
    log_base = np.log(np.clip(base_iv, EPS, None))

    N, D = Y_std.shape
    length = max(0, N - T - H + 1)
    if length <= 0:
        return None, None, None

    preds = np.zeros((length, H, D), dtype=np.float64)
    trues = np.zeros_like(preds)
    anchor_dates = []

    for i in range(length):
        x_win = X[i:i+T, :].reshape(1, T, X.shape[1])
        xb = torch.tensor(x_win, dtype=torch.float32, device=DEVICE)
        j = i + T - 1

        if j-1 >= 0:
            first_std = Y_std[j-1, :]
        else:
            first_std = np.zeros(D, dtype=np.float32)
        first_std_t = torch.tensor(first_std, dtype=torch.float32, device=DEVICE)

        y_pred_std = model.greedy_decode(
            xb, H=H, first_val_std=first_std_t
        ).squeeze(0).cpu().numpy()
        y_true_std = Y_std[j:j+H, :]

        y_pred_log = y_pred_std * y_sd + y_mu
        y_true_log = y_true_std * y_sd + y_mu

        if use_resid:
            y_pred_log += log_base[j:j+H, :]
            y_true_log += log_base[j:j+H, :]

        preds[i, :, :] = np.exp(y_pred_log)
        trues[i, :, :] = np.exp(y_true_log)
        anchor_dates.append(dates[j+1])

    return preds, trues, np.array(anchor_dates)

# ---------------- Plot helpers ----------------
FIGSIZE_MAIN = (16, 10)
LABEL_FONTSIZE = 16
TICK_FONTSIZE  = 14
TITLE_FONTSIZE = 18
LEGEND_FONTSIZE= 14

def plot_loss_curve(train_losses, val_losses, title, fname):
    epochs = np.arange(1, len(train_losses) + 1)
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.plot(epochs, train_losses, linewidth=1.0, linestyle="-.", label="Train loss")
    ax.plot(epochs, val_losses, linewidth=1.0, linestyle="-.", label="Validation loss")
    ax.set_title(title, fontsize=TITLE_FONTSIZE)
    ax.set_xlabel("Epoch", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Smooth L1 loss", fontsize=LABEL_FONTSIZE)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=GRID_ALPHA)
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
    ax.legend(fontsize=LEGEND_FONTSIZE, frameon=False, loc="upper right")
    plt.tight_layout()
    savefig(fname)
    plt.close(fig)

def plot_r2_heatmap(
    r2_matrix,
    days,
    deltas,
    title,
    fname,
    vmin=None,
    vmax=None,
    cbar_label="R²",
    annotate=False,
    fmt="0.2f",
):
    """
    r2_matrix: shape (len(days), len(deltas))

    - Uses a bright 'turbo' (thermal / ROYGBIV-style) colormap for visibility.
    - Rotates x-axis tick labels so they don't overlap.
    - Optionally annotates each cell with its numeric value.
    """
    days = list(days)
    deltas = list(deltas)
    r2_matrix = np.array(r2_matrix, dtype=float)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    im = ax.imshow(
        r2_matrix,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
        cmap="turbo",   # bright ROYGBIV-like colormap
    )

    ax.set_title(title, fontsize=TITLE_FONTSIZE)
    ax.set_xlabel("Delta", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Days to maturity", fontsize=LABEL_FONTSIZE)

    ax.set_xticks(np.arange(len(deltas)))
    ax.set_xticklabels(deltas, rotation=45, ha="right")  # avoid overlap
    ax.set_yticks(np.arange(len(days)))
    ax.set_yticklabels(days)

    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label, fontsize=LABEL_FONTSIZE)

    # --- Optional numeric labels inside cells ---
    if annotate:
        n_cells = r2_matrix.size
        ANNOTATE_MAX_CELLS = 150  # avoid unreadable clutter on huge grids
        if n_cells <= ANNOTATE_MAX_CELLS and np.isfinite(r2_matrix).any():
            # Handle auto vmin/vmax if None
            _vmin = np.nanmin(r2_matrix) if vmin is None else vmin
            _vmax = np.nanmax(r2_matrix) if vmax is None else vmax
            scale = _vmax - _vmin if _vmax > _vmin else 1.0

            for i in range(len(days)):
                for j in range(len(deltas)):
                    val = r2_matrix[i, j]
                    if not np.isfinite(val):
                        continue
                    # normalised position in colormap to pick text color (simple heuristic)
                    norm_val = (val - _vmin) / scale
                    # darker text on bright colors, white text on dark colors
                    text_color = "black" if norm_val > 0.6 else "white"
                    ax.text(
                        j,
                        i,
                        format(val, fmt),
                        ha="center",
                        va="center",
                        color=text_color,
                        fontsize=8,
                    )

    plt.tight_layout()
    savefig(fname)
    plt.close(fig)

def plot_experiment_r2_bars(exp_results, title, fname):
    names  = [e.get("label", e["name"]) for e in exp_results]
    mean_in  = [e["mean_r2_in"] for e in exp_results]
    mean_out = [e["mean_r2_out"] for e in exp_results]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(1, 1, figsize=(max(6, 1.5*len(names)), 5))
    ax.bar(x - width/2, mean_in,  width, label="IS R²")
    ax.bar(x + width/2, mean_out, width, label="OOS R²")

    ax.set_title(title, fontsize=TITLE_FONTSIZE)
    ax.set_ylabel("Mean R² (across surface)", fontsize=LABEL_FONTSIZE)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=GRID_ALPHA)
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    ax.legend(fontsize=LEGEND_FONTSIZE, frameon=False, loc="best")

    plt.tight_layout()
    savefig(fname)
    plt.close(fig)

def plot_delta_r2_vs_baseline(exp_results, baseline_name, surf_days, surf_deltas, prefix="SURF"):
    """
    For each experiment, plot ΔOOS R² heatmap vs baseline:
       ΔR² = R²_exp - R²_baseline
    """
    baseline = None
    for e in exp_results:
        if e["name"] == baseline_name:
            baseline = e
            break
    if baseline is None:
        print(f"[WARN] Baseline experiment '{baseline_name}' not found; skipping ΔR² heatmaps.")
        return

    base_mat = baseline["r2_out_mat"]
    for e in exp_results:
        if e["name"] == baseline_name:
            continue
        delta = e["r2_out_mat"] - base_mat
        max_abs = float(np.nanmax(np.abs(delta))) if np.isfinite(delta).any() else 1.0
        vmin, vmax = -max_abs, max_abs
        title = f"{e.get('label', e['name'])} — ΔOOS R² vs {baseline.get('label', baseline['name'])}"
        fname = f"{prefix}_dR2_OOS_{baseline['name']}_vs_{e['name']}.png"
        plot_r2_heatmap(
            delta,
            surf_days,
            surf_deltas,
            title,
            fname,
            vmin=vmin,
            vmax=vmax,
            cbar_label="ΔR²",
            annotate=True,          # <-- now labels each bucket (if grid not huge)
            fmt="0.2f",
        )

# ---------------- Pipeline for one configuration ----------------
def fit_and_evaluate_surface(df_all, surf_cols, y_cols,
                             surf_days, surf_deltas,
                             tag, feat_cols,
                             include_past_surface=False):
    """
    Train & evaluate one model (with specific feature set and with/without past surfaces).
    """
    print(f"[{tag}] feature cols: {feat_cols}, include_past_surface={include_past_surface}")
    print(f"[{tag}] total rows: {len(df_all)}")

    df_pre  = df_all[df_all["Date"] < SPLIT_DATE].reset_index(drop=True)
    df_post = df_all[df_all["Date"] >= SPLIT_DATE].reset_index(drop=True)
    print(f"[{tag}] rows pre: {len(df_pre)}, post: {len(df_post)}")

    if len(df_pre) <= BLOCK_SIZE + HORIZON:
        raise RuntimeError(f"[{tag}] Not enough In-Sample rows for chosen BLOCK_SIZE/HORIZON")

    usable = len(df_pre) - BLOCK_SIZE - HORIZON + 1
    train_usable = int(0.8 * usable)
    train_end = train_usable + BLOCK_SIZE + HORIZON - 1

    train_df = df_pre.iloc[:train_end].reset_index(drop=True)
    val_df   = df_pre.iloc[train_usable:].reset_index(drop=True)

    train_std, [val_std, pre_std, post_std], stats = make_std_views_surface(
        train_df, [val_df, df_pre, df_post],
        feat_cols=feat_cols, surf_cols=surf_cols, y_cols=y_cols
    )

    train_ds = VolSurfSeq2SeqDataset(
        train_std, y_cols, feat_cols, surf_cols,
        BLOCK_SIZE, HORIZON, include_past_surface=include_past_surface
    )
    val_ds = VolSurfSeq2SeqDataset(
        val_std, y_cols, feat_cols, surf_cols,
        BLOCK_SIZE, HORIZON, include_past_surface=include_past_surface
    )

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE,
                          shuffle=True, drop_last=True)
    val_dl   = DataLoader(val_ds, batch_size=BATCH_SIZE,
                          shuffle=False, drop_last=False)

    enc_in_dim = train_ds.enc_in_dim
    surf_dim = len(y_cols)

    model = EDTransformer(
        d_model=EMBED_DIM, n_heads=N_HEADS,
        n_enc=N_LAYERS_ENC, n_dec=N_LAYERS_DEC,
        d_ff=4*EMBED_DIM, dropout=DROPOUT,
        H=HORIZON, enc_in_dim=enc_in_dim, surf_dim=surf_dim
    ).to(DEVICE)

    opt = torch.optim.AdamW(
        model.parameters(),
        betas=(0.9, 0.98), eps=1e-8, weight_decay=WEIGHT_DECAY
    )
    sch = NoamLR(opt, d_model=EMBED_DIM,
                 warmup_steps=WARMUP_STEPS, scale=LR_SCALE)
    loss_fn = nn.SmoothL1Loss()

    tr_hist, va_hist = [], []
    best_val = float("inf")
    stall = 0
    ckpt = OUT_DIR / f"best_{tag}.pt"

    print(f"[{tag}] training ...")
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        tr = train_epoch(model, train_dl, opt, sch, loss_fn)
        va = eval_epoch(model, val_dl, loss_fn)
        tr_hist.append(tr)
        va_hist.append(va)
        print(f"[{tag}] Epoch {epoch:03d}  train={tr:.6e}  val={va:.6e}  {time.time()-t0:.1f}s")

        if va + 1e-9 < best_val:
            best_val = va
            stall = 0
            torch.save(model.state_dict(), ckpt)
        else:
            stall += 1
            if stall >= PATIENCE:
                print(f"[{tag}] Early stopping at epoch {epoch}.")
                break

    plot_loss_curve(tr_hist, va_hist,
                    f"{tag} — Transformer Loss (train vs val)",
                    f"{tag}_loss.png")

    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    model.eval()

    preds_pre, trues_pre, dates_pre = roll_forecast_surface(
        model, pre_std, df_pre[["Date"] + surf_cols],
        stats, include_past_surface=include_past_surface
    )
    preds_post, trues_post, dates_post = roll_forecast_surface(
        model, post_std, df_post[["Date"] + surf_cols],
        stats, include_past_surface=include_past_surface
    )

    if preds_pre is None or preds_post is None:
        print(f"[{tag}] Not enough data for rolling forecast.")
        return None

    D = preds_pre.shape[-1]
    H = preds_pre.shape[1]
    if H < 1:
        raise RuntimeError("HORIZON must be >= 1 for R^2 computation")

    true_in  = trues_pre[:, 0, :]
    pred_in  = preds_pre[:, 0, :]
    true_out = trues_post[:, 0, :]
    pred_out = preds_post[:, 0, :]

    r2_in_vec  = np.array([r2_score(true_in[:, d],  pred_in[:, d])  for d in range(D)])
    r2_out_vec = np.array([r2_score(true_out[:, d], pred_out[:, d]) for d in range(D)])

    n_days = len(surf_days)
    n_deltas = len(surf_deltas)
    assert D == n_days * n_deltas, "Surface dimension does not match days × deltas grid."

    r2_in_mat  = r2_in_vec.reshape(n_days, n_deltas)
    r2_out_mat = r2_out_vec.reshape(n_days, n_deltas)

    mean_r2_in  = float(np.nanmean(r2_in_vec))
    mean_r2_out = float(np.nanmean(r2_out_vec))

    print(f"[{tag}] mean R² (IS):  {mean_r2_in:.4f}")
    print(f"[{tag}] mean R² (OOS): {mean_r2_out:.4f}")

    return {
        "feat_cols": feat_cols,
        "include_past_surface": include_past_surface,
        "r2_in_mat": r2_in_mat,
        "r2_out_mat": r2_out_mat,
        "mean_r2_in": mean_r2_in,
        "mean_r2_out": mean_r2_out,
    }

# ---------------- Main ----------------
def main():
    df_all, surf_cols, y_cols, surf_days, surf_deltas = load_prepare_surface(
        CSV_PATH, SURF_CSV_PATH
    )

    print("Merged daily + surface dataframe head():")
    print(df_all.head())

    all_results = []

    for exp in EXPERIMENTS:
        base_feat_cols = ["log_ret"]
        if exp.get("use_rvol", False):
            base_feat_cols.append("rvol_feat")
        if exp.get("use_vixd", False):
            base_feat_cols.append("vixd_feat")

        tag = exp["name"]
        res = fit_and_evaluate_surface(
            df_all, surf_cols, y_cols, surf_days, surf_deltas,
            tag=tag,
            feat_cols=base_feat_cols,
            include_past_surface=exp.get("include_past_surface", False)
        )
        if res is None:
            continue

        exp_record = {
            "name": exp["name"],
            "label": exp.get("label", exp["name"]),
            "use_rvol": exp.get("use_rvol", False),
            "use_vixd": exp.get("use_vixd", False),
            "include_past_surface": exp.get("include_past_surface", False),
        }
        exp_record.update(res)
        all_results.append(exp_record)

    if not all_results:
        print("No successful experiments; exiting.")
        return

    plot_experiment_r2_bars(
        all_results,
        title="Mean IS/OOS R² vs Feature Set (next-day IV surface)",
        fname="experiments_mean_R2_bar.png"
    )

    plot_delta_r2_vs_baseline(
        all_results,
        baseline_name=BASELINE_EXP_NAME,
        surf_days=surf_days,
        surf_deltas=surf_deltas,
        prefix="SURF"
    )

if __name__ == "__main__":
    main()
