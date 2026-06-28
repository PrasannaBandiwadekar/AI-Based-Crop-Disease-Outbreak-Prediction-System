"""
generate_pipeline_figure.py
----------------------------
Generates figure3_pipeline.png — a compact, publication-quality diagram
of the KG-CTCN end-to-end pipeline for inclusion in the IEEE paper.

Output: figure3_pipeline.png  (600 dpi, ~3.5 inches wide when placed at
        0.35\textwidth in a two-column IEEE layout)

Requirements:
    pip install matplotlib
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os

# ── Output path ───────────────────────────────────────────────────────────────
OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "figure3_pipeline.png")

# ── Colour palette ────────────────────────────────────────────────────────────
C_DATA   = "#1a5276"   # dark blue  — data sources
C_FEAT   = "#117a65"   # dark green — feature engineering
C_MODEL  = "#6c3483"   # purple     — model / inference
C_OUT    = "#784212"   # brown      — output / advisory
C_STORE  = "#515a5a"   # grey       — storage
C_ARROW  = "#2c3e50"   # near-black — arrows
C_BG     = "#fdfefe"   # off-white  — background

# ── Canvas ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(4.2, 7.8))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")
fig.patch.set_facecolor(C_BG)
ax.set_facecolor(C_BG)


# ── Helper functions ──────────────────────────────────────────────────────────

def box(ax, cx, cy, w, h, color, label, sublabel=None, fontsize=7.5):
    """Draw a rounded rectangle with centred label."""
    x, y = cx - w / 2, cy - h / 2
    rect = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.012",
        linewidth=0.8,
        edgecolor=color,
        facecolor=color + "22",   # 13% opacity fill
    )
    ax.add_patch(rect)
    ax.text(cx, cy + (0.010 if sublabel else 0), label,
            ha="center", va="center",
            fontsize=fontsize, fontweight="bold", color=color)
    if sublabel:
        ax.text(cx, cy - 0.022, sublabel,
                ha="center", va="center",
                fontsize=5.8, color=color, style="italic")


def arrow(ax, x1, y1, x2, y2):
    """Draw a thin downward arrow."""
    ax.annotate("",
                xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=C_ARROW,
                    lw=0.9,
                    mutation_scale=8,
                ))


def side_arrow(ax, x1, y1, x2, y2, label=""):
    """Horizontal / diagonal connector with optional label."""
    ax.annotate("",
                xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=C_STORE,
                    lw=0.8,
                    linestyle="dashed",
                    mutation_scale=7,
                ))
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx + 0.02, my, label,
                fontsize=5.5, color=C_STORE, va="center")


def section_label(ax, y, text):
    ax.text(0.02, y, text, fontsize=5.5, color="#7f8c8d",
            va="center", fontstyle="italic")


# ── Pipeline stages (centre-x=0.50, evenly spaced top→bottom) ────────────────

BW, BH = 0.56, 0.072   # standard box width / height
CX = 0.50              # horizontal centre

stages = [
    # (cy,   color,    label,              sublabel)
    (0.930, C_DATA,   "NASA POWER API",   "Daily meteorological fetch"),
    (0.820, C_FEAT,   "Z-Score Normalisation", "90-day rolling window"),
    (0.710, C_FEAT,   "Sequence Builder", "30-day sliding window"),
    (0.595, C_MODEL,  "KG-CTCN Encoder",  "Causal TCN + KG-Fusion"),
    (0.480, C_MODEL,  "Temperature Scaling", "Calibrated probability"),
    (0.365, C_MODEL,  "KG Biological Gate", "RH & Rain threshold"),
    (0.245, C_OUT,    "Risk Classifier",  "Low / Medium / High"),
    (0.120, C_OUT,    "Advisory Output",  "JSON + SQLite log"),
]

for cy, color, label, sublabel in stages:
    box(ax, CX, cy, BW, BH, color, label, sublabel)

# ── Vertical arrows between consecutive stages ────────────────────────────────
for i in range(len(stages) - 1):
    cy_top = stages[i][0] - BH / 2
    cy_bot = stages[i + 1][0] + BH / 2
    arrow(ax, CX, cy_top, CX, cy_bot)

# ── SQLite side-store connector (from Advisory Output) ───────────────────────
store_cx, store_cy = 0.88, 0.120
box(ax, store_cx, store_cy, 0.19, 0.065, C_STORE, "SQLite DB", "Persistence")
side_arrow(ax, CX + BW / 2, 0.120, store_cx - 0.095, store_cy, "persist")

# ── Agronomic DB side-feed (feeds into KG-CTCN Encoder) ─────────────────────
agro_cx, agro_cy = 0.10, 0.595
box(ax, agro_cx, agro_cy, 0.155, 0.065, C_STORE, "Agro DB", "Variety / Ratoon")
side_arrow(ax, agro_cx + 0.078, agro_cy, CX - BW / 2, 0.595, "state")

# ── Title ─────────────────────────────────────────────────────────────────────
ax.text(CX, 0.985, "KG-CTCN Inference Pipeline",
        ha="center", va="center",
        fontsize=8.5, fontweight="bold", color="#1c2833")

# ── Legend ────────────────────────────────────────────────────────────────────
legend_items = [
    (C_DATA,  "Data Ingestion"),
    (C_FEAT,  "Feature Engineering"),
    (C_MODEL, "Model / Inference"),
    (C_OUT,   "Output"),
    (C_STORE, "Storage"),
]
handles = [
    mpatches.Patch(facecolor=c + "33", edgecolor=c, linewidth=0.7, label=lbl)
    for c, lbl in legend_items
]
ax.legend(handles=handles, loc="lower center",
          bbox_to_anchor=(0.5, -0.01),
          ncol=3, fontsize=5.5,
          frameon=True, framealpha=0.9,
          edgecolor="#cccccc",
          handlelength=1.2, handleheight=0.9,
          borderpad=0.5, labelspacing=0.3)

# ── Save ──────────────────────────────────────────────────────────────────────
plt.tight_layout(pad=0.3)
plt.savefig(OUT_PATH, dpi=600, bbox_inches="tight",
            facecolor=C_BG, format="png")
plt.close()
print(f"Saved: {OUT_PATH}")