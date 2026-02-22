#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 01:34:14 2025

@author: svosve


vol_seq2seq_encoder_decoder.py — multi-input encoder–decoder Transformer

- Encoder inputs (T × 3): [log_ret, rvol_feat, vixd_feat]
  * log_ret = log(SPX_t / SPX_{t-1})
  * rvol_feat = realized daily variance (level, no log)
  * vixd_feat = VIX daily variance ( (VIX/100/√252)^2, level, no log )

- Target (H × 1): standardized log-variance next day.
  * By default we learn a **residual**: y = log(var_{t+1}) - log(var_t)
    (works well OOS). Turn off by setting USE_RESIDUAL_TARGET=False.

- Two models trained with same pipeline:
  A) RVOL target (next realized variance)
  B) VIX target  (next VIX daily variance)

- All plots saved to: ~/Desktop/Transformer_images
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
from matplotlib.ticker import MaxNLocator, ScalarFormatter
import matplotlib.dates as mdates

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 14,
    "mathtext.default": "regular"
})

# ===== Color theme (your requested consistent palette) =====
C_PRED        = "r"   # model prediction (lines / IS scatter)
C_TRUE        = "k"   # ground truth (dark gray)
C_SPX         = "b"   # SPX line (blue)
C_SCATTER_IS  = "r"   # in-sample scatter
C_SCATTER_OOS = "b"   # out-of-sample scatter
C_DIAG        = "k"   # 45° line in scatters
GRID_ALPHA    = 0.35

# ---------------- CONFIG ----------------
CSV_PATH = Path("Vol_Index.csv")
OUT_DIR  = Path("outputs_ed")  # model weights
IMG_DIR  = Path.home() / "Desktop" / "Transformer_images_10yr_pred"  # <— requested
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED     = 42

# Data / model hyperparameters
BATCH_SIZE   = 128
BLOCK_SIZE   = 21         # encoder context length (past steps)
HORIZON      = 1          # decoder length (future steps)
EMBED_DIM    = 8
N_HEADS      = 4
N_LAYERS_ENC = 2
N_LAYERS_DEC = 2
DROPOUT      = 0.3
LR_SCALE     = 1.0
WARMUP_STEPS = 4000
EPOCHS       = 100
GRAD_CLIP    = 1.0
PATIENCE     = 1000
WEIGHT_DECAY = 1e-3       # AdamW

# Inputs to encoder: returns + RVOL(level) + VIX_dailyVar(level)
ENC_IN_DIM   = 3

# Target choice (recommended True for OOS)
USE_RESIDUAL_TARGET = True   # y = log(var_{t+1}) - log(var_t)

# Split for IS/OOS figures
SPLIT_DATE = pd.Timestamp("2009-01-01")

# Annualisation constants
ANN_SQRT = math.sqrt(252.0)
ANN_PCT  = 100.0 * ANN_SQRT  # multiply sqrt(variance) by this to get VIX-like % level

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

def annualise_from_variance(var_arr):
    """variance -> annualised % level (VIX-style, σ * 100 * √252)"""
    return np.sqrt(np.maximum(var_arr, 0.0)) * ANN_PCT

def savefig(fname, dpi=300):
    path = IMG_DIR / fname
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"[saved] {path}")

# ---------------- Data preparation ----------------
def load_prepare(csv_path=CSV_PATH):
    """
    Returns a DataFrame with:
      Date, SPX, log_ret, rvol, rvol_next, vixd, vixd_next,
      rvol_feat, vixd_feat (levels), and
      y_rvol (target), y_vix (target) according to USE_RESIDUAL_TARGET.
    """
    df = pd.read_csv(csv_path, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)

    # SPX & returns (no look-ahead)
    df["SPX"] = pd.to_numeric(df["SPX"], errors="coerce")
    df["log_ret"] = np.log(df["SPX"] / df["SPX"].shift(1))

    # Realized variance (daily)
    df["rvol"] = pd.to_numeric(df["RVOL_5min"], errors="coerce").clip(lower=EPS)
    df["rvol_next"] = df["rvol"].shift(-1)

    # VIX daily variance
    if "VIX" in df.columns:
        df["VIX"] = pd.to_numeric(df["VIX"], errors="coerce")
        df["vixd"] = (df["VIX"] / (100.0 * ANN_SQRT)) ** 2  # daily variance
    else:
        df["vixd"] = np.nan
    df["vixd"] = df["vixd"].ffill().bfill().astype(float).clip(lower=EPS)  # keep as level
    df["vixd_next"] = df["vixd"].shift(-1)

    # Encoder features (level, as requested — no logs)
    df["rvol_feat"] = df["rvol"].astype(float)
    df["vixd_feat"] = df["vixd"].astype(float)

    # Targets (log-variance)
    log_rv_next  = np.log(df["rvol_next"].astype(float).clip(lower=EPS))
    log_rv_today = np.log(df["rvol"].astype(float).clip(lower=EPS))

    log_vx_next  = np.log(df["vixd_next"].astype(float).clip(lower=EPS))
    log_vx_today = np.log(df["vixd"].astype(float).clip(lower=EPS))

    if USE_RESIDUAL_TARGET:
        df["y_rvol"] = log_rv_next - log_rv_today
        df["y_vix"]  = log_vx_next - log_vx_today
    else:
        df["y_rvol"] = log_rv_next
        df["y_vix"]  = log_vx_next

    # Clean
    need = ["Date","SPX","log_ret","rvol","rvol_next","vixd","vixd_next",
            "rvol_feat","vixd_feat","y_rvol","y_vix"]
    df = df[need]
    df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    return df

def make_std_views(train_df, other_df_list, y_col, block_size=BLOCK_SIZE, horizon=HORIZON):
    """
    Standardize encoder features (log_ret, rvol_feat, vixd_feat) and the chosen y_col.
    """
    feat_cols = ["log_ret", "rvol_feat", "vixd_feat"]
    feat_mu = {c: float(train_df[c].mean()) for c in feat_cols}
    feat_sd = {c: float(train_df[c].std() + 1e-12) for c in feat_cols}

    y_mu = float(train_df[y_col].mean())
    y_sd = float(train_df[y_col].std() + 1e-12)

    def std_one(df_in):
        tmp = df_in[["Date", "rvol", "vixd", "log_ret", "rvol_feat", "vixd_feat", y_col]].copy()
        for c in feat_cols:
            tmp[c] = (tmp[c] - feat_mu[c]) / feat_sd[c]
        tmp["y"] = (tmp[y_col] - y_mu) / y_sd
        return tmp.drop(columns=[y_col])

    outs = [std_one(d) for d in other_df_list]
    train_std = std_one(train_df)

    stats = {
        "feat_mu": feat_mu, "feat_sd": feat_sd,
        "y_mu": y_mu, "y_sd": y_sd,
        "block_size": block_size, "horizon": horizon,
        "use_residual": USE_RESIDUAL_TARGET
    }
    return train_std, outs, stats

# ---------------- Dataset ----------------
class VolSeq2SeqDataset(Dataset):
    """
    Returns (X_src, y_prev, y_tgt):
      X_src : (T, 3) standardized features
      y_prev: (1,1)  previous standardized target (seed)
      y_tgt : (H,1)  standardized target(s)
    """
    def __init__(self, df_std, block_size, horizon):
        super().__init__()
        self.df = df_std.reset_index(drop=True)
        self.T = block_size
        self.H = horizon

        self.X = self.df[["log_ret", "rvol_feat", "vixd_feat"]].to_numpy(dtype=np.float32)
        self.y = self.df["y"].to_numpy(dtype=np.float32)

        self.N = len(self.df)
        self.length = max(0, self.N - self.T - self.H + 1)
        if self.length <= 0:
            raise RuntimeError("Not enough rows for chosen BLOCK_SIZE and HORIZON")

    def __len__(self): return self.length

    def __getitem__(self, i):
        x = self.X[i:i+self.T, :]                 # (T,3)
        j = i + self.T - 1
        y_seq  = self.y[j:j+self.H].reshape(self.H, 1)             # (H,1)
        y_prev = np.array([[ self.y[j-1] if (j-1) >= 0 else 0.0 ]], dtype=np.float32)  # (1,1)
        return torch.from_numpy(x), torch.from_numpy(y_prev), torch.from_numpy(y_seq)

# ---------------- Model ----------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=20000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x):  # x: (B,L,d)
        return x + self.pe[:, :x.size(1), :]

def causal_mask(L, device):
    return torch.triu(torch.ones(L, L, dtype=torch.bool, device=device), diagonal=1)

class EDTransformer(nn.Module):
    def __init__(self, d_model=64, n_heads=4, n_enc=4, n_dec=4, d_ff=None, dropout=0.1, H=1, enc_in_dim=3):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.H = H

        self.x_proj = nn.Linear(enc_in_dim, d_model)
        self.pos    = PositionalEncoding(d_model)

        enc = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout=dropout, batch_first=True)
        dec = nn.TransformerDecoderLayer(d_model, n_heads, d_ff, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, n_enc)
        self.decoder = nn.TransformerDecoder(dec, n_dec)

        self.y_proj   = nn.Linear(1, d_model)
        self.bos_embed= nn.Parameter(torch.zeros(1,1,d_model))
        self.head     = nn.Linear(d_model, 1)

    def forward(self, x_src, y_prev, y_tgt):
        B, H = x_src.size(0), y_tgt.size(1)
        mem = self.encoder(self.pos(self.x_proj(x_src)))   # (B,T,d)

        bos  = self.bos_embed.expand(B,1,-1)
        y0   = self.y_proj(y_prev)                         # (B,1,d)
        toks = torch.cat([bos, y0], dim=1)                 # (B,2,d)
        if H > 1:
            toks = torch.cat([toks, self.y_proj(y_tgt[:,:-1,:])], dim=1)

        toks = self.pos(toks)
        mask = causal_mask(toks.size(1), x_src.device)
        dec  = self.decoder(tgt=toks, memory=mem, tgt_mask=mask)
        out  = self.head(dec[:, -H:, :])                   # (B,H,1)
        return out

    @torch.no_grad()
    def greedy_decode(self, x_src, H=None, first_val_std=0.0):
        H = H or self.H
        B = x_src.size(0); device = x_src.device
        mem = self.encoder(self.pos(self.x_proj(x_src)))

        bos = self.bos_embed.expand(B,1,-1)
        y_prev = torch.full((B,1,1), float(first_val_std), device=device)
        prev_emb = self.y_proj(y_prev)
        hist_emb = torch.zeros(B,0,prev_emb.size(-1), device=device)
        preds = []
        for _ in range(H):
            toks = torch.cat([bos, prev_emb, hist_emb], dim=1)
            toks = self.pos(toks)
            mask = causal_mask(toks.size(1), device)
            dec  = self.decoder(tgt=toks, memory=mem, tgt_mask=mask)
            next_std = self.head(dec[:,-1:,:])   # (B,1,1)
            preds.append(next_std)
            hist_emb = torch.cat([hist_emb, self.y_proj(next_std)], dim=1)
        return torch.cat(preds, dim=1)  # (B,H,1)

# ---------------- Noam Scheduler ----------------
class NoamLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, d_model, warmup_steps=4000, scale=1.0, last_epoch=-1):
        self.d_model = d_model; self.warmup = warmup_steps; self.scale = scale
        super().__init__(optimizer, last_epoch)
    def get_lr(self):
        step = max(1, self._step_count)
        factor = self.scale * (self.d_model ** -0.5) * min(step ** -0.5, step * (self.warmup ** -1.5))
        return [factor for _ in self.optimizer.param_groups]

# ---------------- Train / Eval helpers ----------------
def train_epoch(model, dl, optimizer, scheduler, loss_fn):
    model.train(); total = 0.0; nobs = 0
    for xb, y0b, yb in dl:
        xb, y0b, yb = xb.to(DEVICE), y0b.to(DEVICE), yb.to(DEVICE)
        pred = model(xb, y0b, yb)
        loss = loss_fn(pred, yb)
        optimizer.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step(); scheduler.step()
        bs = xb.size(0); total += float(loss.item()) * bs; nobs += bs
    return total / max(1, nobs)

@torch.no_grad()
def eval_epoch(model, dl, loss_fn):
    model.eval(); total = 0.0; nobs = 0
    for xb, y0b, yb in dl:
        xb, y0b, yb = xb.to(DEVICE), y0b.to(DEVICE), yb.to(DEVICE)
        pred = model(xb, y0b, yb)
        loss = loss_fn(pred, yb)
        bs = xb.size(0); total += float(loss.item()) * bs; nobs += bs
    return total / max(1, nobs)

@torch.no_grad()
def roll_forecast(model, df_std, df_raw, stats, base_var_col):
    """
    Rolling forecasts.
    Returns:
      preds_var: (N,H) variance predictions
      trues_var: (N,H) variance truths
      anchor_dates: (N,)
    """
    T = stats["block_size"]; H = stats["horizon"]
    y_mu, y_sd = stats["y_mu"], stats["y_sd"]
    use_resid  = bool(stats.get("use_residual", False))

    X = df_std[["log_ret","rvol_feat","vixd_feat"]].to_numpy(dtype=np.float32)
    y_std = df_std["y"].to_numpy(dtype=np.float32)
    dates = df_raw["Date"].to_numpy()
    log_base = np.log(df_raw[base_var_col].astype(float).clip(lower=EPS)).to_numpy()

    N = len(df_std)
    length = max(0, N - T - H + 1)
    if length <= 0: return None, None, None

    preds_var = np.zeros((length, H), dtype=np.float64)
    trues_var = np.zeros((length, H), dtype=np.float64)
    anchor_dates = []

    for i in range(length):
        x_win = X[i:i+T,:].reshape(1, T, ENC_IN_DIM)
        xb = torch.tensor(x_win, dtype=torch.float32, device=DEVICE)
        j = i + T - 1

        first_std = float(y_std[j-1]) if (j-1) >= 0 else 0.0
        y_pred_std = model.greedy_decode(xb, H=H, first_val_std=first_std).squeeze(0).cpu().numpy().reshape(H)

        # unstandardize
        y_pred_logv = y_pred_std * y_sd + y_mu
        y_true_logv = (y_std[j:j+H]) * y_sd + y_mu

        if use_resid:
            y_pred_logv += log_base[j]
            y_true_logv += log_base[j]

        preds_var[i,:] = np.exp(y_pred_logv)
        trues_var[i,:] = np.exp(y_true_logv)
        anchor_dates.append(dates[j+1])

    return preds_var, trues_var, np.array(anchor_dates)

# ---------------- Plot helpers ----------------
FIGSIZE_MAIN = (16, 10)
LABEL_FONTSIZE = 16
TICK_FONTSIZE  = 14
LEGEND_FONTSIZE= 14
TITLE_FONTSIZE = 18
SUPTITLE_FONTSIZE = 20
LINE_WIDTH = 1.0
XTICK_COUNT = 7

def plot_spx_and_series(df_pre_price, df_post_price,
                        df_pre_var, r2_pre, df_post_var, r2_post, y_label, suptitle, fname):
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE_MAIN)

    # SPX (top row)
    for ax, dfp, title in zip(
        [axes[0,0], axes[0,1]],
        [df_pre_price, df_post_price],
        ["SPX Price (In-Sample)", "SPX Price (Out-of-Sample)"]
    ):
        ax.plot(dfp["Date"], dfp["SPX"], color=C_SPX, linewidth=LINE_WIDTH, alpha=0.5,)
        ax.set_title(title, fontsize=TITLE_FONTSIZE)
        ax.set_xlabel("Date", fontsize=LABEL_FONTSIZE)
        ax.set_ylabel("SPX", fontsize=LABEL_FONTSIZE)
        ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=GRID_ALPHA)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=XTICK_COUNT))
        ax.xaxis.set_major_formatter(YEAR_FMT)

    # Series (bottom row)
    def one(ax, dfv, r2_here, title):
        ax.plot(dfv["Date"], dfv["pred_lvl"], label=rf"Transformer pred.  ($R^2={r2_here:.4f}$)",
                color=C_PRED, linewidth=LINE_WIDTH)
        ax.plot(dfv["Date"], dfv["true_lvl"], label="Oxford Man Institute (OMI) RVOL", color=C_TRUE, alpha=0.4, linewidth=LINE_WIDTH)
        ax.set_title(title, fontsize=TITLE_FONTSIZE)
        ax.set_xlabel("Date", fontsize=LABEL_FONTSIZE)
        ax.set_ylabel(y_label, fontsize=LABEL_FONTSIZE)
        ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=GRID_ALPHA)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=XTICK_COUNT))
        ax.xaxis.set_major_formatter(YEAR_FMT)
        ax.legend(fontsize=LEGEND_FONTSIZE, frameon=False, loc="upper center")

    one(axes[1,0], df_pre_var,  r2_pre,  "RVOL vs Transformer pred. (In-Sample)")
    one(axes[1,1], df_post_var, r2_post, "RVOL vs Transformer pred. (Out-of-Sample)")

    fig.suptitle(suptitle, fontsize=SUPTITLE_FONTSIZE)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    savefig(fname)
    plt.close(fig)

def plot_scatter(logx_true, logy_pred, r2_in, logx_true_oos, logy_pred_oos, r2_out, x_label, y_label, fname):
    xlim = [min(logx_true.min(), logx_true_oos.min()) - 0.2,
            max(logx_true.max(), logx_true_oos.max()) + 0.2]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    # In-sample
    axes[0].scatter(logx_true, logy_pred, color=C_SCATTER_IS, alpha=0.35, s=5, label=fr"$R^2={r2_in:.4f}$")
    axes[0].plot(xlim, xlim, linestyle="--", color=C_DIAG, lw=1)
    axes[0].set_title("In-Sample", fontsize=TITLE_FONTSIZE)
    axes[0].set_xlabel(x_label, fontsize=LABEL_FONTSIZE); axes[0].set_ylabel(y_label, fontsize=LABEL_FONTSIZE)
    axes[0].set_xlim(xlim); axes[0].set_ylim(xlim)
    axes[0].tick_params(axis="both", labelsize=TICK_FONTSIZE)
    axes[0].grid(True, linestyle="--", alpha=GRID_ALPHA)
    axes[0].legend(fontsize=LEGEND_FONTSIZE, frameon=False, loc="upper left")

    # Out-of-sample
    axes[1].scatter(logx_true_oos, logy_pred_oos, color=C_SCATTER_OOS, alpha=0.35, s=5, label=fr"$R^2={r2_out:.4f}$")
    axes[1].plot(xlim, xlim, linestyle="--", color=C_DIAG, lw=1)
    axes[1].set_title("Out-of-Sample", fontsize=TITLE_FONTSIZE)
    axes[1].set_xlabel(x_label, fontsize=LABEL_FONTSIZE); axes[1].set_ylabel(y_label, fontsize=LABEL_FONTSIZE)
    axes[1].set_xlim(xlim); axes[1].set_ylim(xlim)
    axes[1].tick_params(axis="both", labelsize=TICK_FONTSIZE)
    axes[1].grid(True, linestyle="--", alpha=GRID_ALPHA)
    axes[1].legend(fontsize=LEGEND_FONTSIZE, frameon=False, loc="upper left")

    plt.tight_layout()
    savefig(fname)
    plt.close(fig)

def plot_loss_curve(train_losses, val_losses, title, fname):
    epochs = np.arange(1, len(train_losses) + 1)
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.plot(epochs, train_losses, linewidth=1.0, linestyle="-.", label="Train loss", color="b")
    ax.plot(epochs, val_losses,   linewidth=1.0, linestyle="-.", label="Validation loss", color ='r')
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

# ---------------- Pipeline for one target ----------------
def fit_and_evaluate(df, y_col, base_var_col, tag):
    # Split
    df_pre  = df[df["Date"] < SPLIT_DATE].reset_index(drop=True)
    df_post = df[df["Date"] >= SPLIT_DATE].reset_index(drop=True)
    print(f"[{tag}] rows pre: {len(df_pre)}, post: {len(df_post)}")

    if len(df_pre) <= BLOCK_SIZE + HORIZON:
        raise RuntimeError(f"[{tag}] Not enough In-Sample rows for chosen BLOCK_SIZE/HORIZON")

    usable = len(df_pre) - BLOCK_SIZE - HORIZON + 1
    train_usable = int(0.8 * usable)
    train_df = df_pre.iloc[: train_usable + BLOCK_SIZE + HORIZON - 1].reset_index(drop=True)
    val_df   = df_pre.iloc[train_usable :].reset_index(drop=True)

    train_std, [val_std], stats = make_std_views(train_df, [val_df], y_col=y_col)

    train_ds = VolSeq2SeqDataset(train_std, BLOCK_SIZE, HORIZON)
    val_ds   = VolSeq2SeqDataset(val_std,   BLOCK_SIZE, HORIZON)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  drop_last=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    model = EDTransformer(
        d_model=EMBED_DIM, n_heads=N_HEADS,
        n_enc=N_LAYERS_ENC, n_dec=N_LAYERS_DEC,
        d_ff=4*EMBED_DIM, dropout=DROPOUT, H=HORIZON, enc_in_dim=ENC_IN_DIM
    ).to(DEVICE)

    opt = torch.optim.AdamW(model.parameters(), betas=(0.9, 0.98), eps=1e-8, weight_decay=WEIGHT_DECAY)
    sch = NoamLR(opt, d_model=EMBED_DIM, warmup_steps=WARMUP_STEPS, scale=LR_SCALE)
    loss_fn = nn.SmoothL1Loss()

    # histories
    tr_hist, va_hist = [], []
    best_val = float("inf"); stall = 0
    ckpt = OUT_DIR / f"best_{tag}.pt"
    print(f"[{tag}] training ...")
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        tr = train_epoch(model, train_dl, opt, sch, loss_fn)
        va = eval_epoch(model, val_dl, loss_fn)
        tr_hist.append(tr); va_hist.append(va)
        if epoch % 1 == 0:
            print(f"[{tag}] Epoch {epoch:03d}  train={tr:.6e}  val={va:.6e}  {time.time()-t0:.1f}s")
        if va + 1e-9 < best_val:
            best_val = va; stall = 0; torch.save(model.state_dict(), ckpt)
        else:
            stall += 1
            if stall >= PATIENCE:
                print(f"[{tag}] Early stopping at epoch {epoch}."); break

    plot_loss_curve(tr_hist, va_hist, f"{tag} — Transformer Loss (train vs val)", f"{tag}_loss.png")

    # Reload best and forecast
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    model.eval()

    _, [pre_std],  _ = make_std_views(train_df, [df_pre],  y_col=y_col)
    _, [post_std], _ = make_std_views(train_df, [df_post], y_col=y_col)

    preds_pre,  trues_pre,  dates_pre  = roll_forecast(model, pre_std,  df_pre[["Date", base_var_col]],  stats, base_var_col)
    preds_post, trues_post, dates_post = roll_forecast(model, post_std, df_post[["Date", base_var_col]], stats, base_var_col)

    # R^2 on sqrt scale (H=1)
    r2_in  = r2_score(np.sqrt(trues_pre[:,0]),  np.sqrt(preds_pre[:,0]))  if preds_pre  is not None else float("nan")
    r2_out = r2_score(np.sqrt(trues_post[:,0]), np.sqrt(preds_post[:,0])) if preds_post is not None else float("nan")

    # Time-series figures (annualised %)
    df_pre_price  = df_pre[["Date","SPX"]].copy()
    df_post_price = df_post[["Date","SPX"]].copy()

    df_pre_plot  = pd.DataFrame({
        "Date": dates_pre,
        "pred_lvl": annualise_from_variance(preds_pre[:,0]),
        "true_lvl": annualise_from_variance(trues_pre[:,0])
    })
    df_post_plot = pd.DataFrame({
        "Date": dates_post,
        "pred_lvl": annualise_from_variance(preds_post[:,0]),
        "true_lvl": annualise_from_variance(trues_post[:,0])
    })

    if tag.upper().startswith("RVOL"):
        ylab = "OMI RVOL (annualised %)"
        supt = ""
    else:
        ylab = "VIX"
        supt = ""

    plot_spx_and_series(df_pre_price, df_post_price, df_pre_plot, r2_in, df_post_plot, r2_out, ylab, supt, f"{tag}_timeseries.png")

    # Scatter on log(annualised %) for stability/visibility
    log_true_in   = np.log(df_pre_plot["true_lvl"].values  + EPS)
    log_pred_in   = np.log(df_pre_plot["pred_lvl"].values  + EPS)
    log_true_out  = np.log(df_post_plot["true_lvl"].values + EPS)
    log_pred_out  = np.log(df_post_plot["pred_lvl"].values + EPS)

    xlab = "True " + ("RVOL" if tag.upper().startswith("RVOL") else "VIX") + " (log scale)"
    ylab = "Transformer Pred. (log scale)"
    plot_scatter(log_true_in, log_pred_in, r2_in, log_true_out, log_pred_out, r2_out, xlab, ylab, f"{tag}_scatter.png")

# ---------------- Main ----------------
def main():
    df = load_prepare(CSV_PATH)

    # ----- Model A: RVOL -----
    fit_and_evaluate(df, y_col="y_rvol", base_var_col="rvol", tag="RVOL")

    # ----- Model B: VIX (rescaled daily variance) -----
    # keep only rows where VIX feature is available (already ffilled/bfilled)
    fit_and_evaluate(df, y_col="y_vix", base_var_col="vixd", tag="VIX")

if __name__ == "__main__":
    main()
