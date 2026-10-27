#!/usr/bin/env python3
"""Generate the LaTeX report for the sparse-representation OCT denoising study.

Pulls every number from outputs/*.csv (the artifacts written by denoising.ipynb), builds the
derived figures, fills the template main.tex.in and writes main.tex.

Run from the project root:  python report/make_report.py
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
REP = ROOT / "report"
FIG = REP / "figures"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
                     "legend.fontsize": 8, "figure.dpi": 120, "savefig.bbox": "tight"})

# ----------------------------------------------------------------------------- data
t1 = pd.read_csv(OUT / "table1_simulation.csv")
t2 = pd.read_csv(OUT / "table2_ablation.csv")
t3 = pd.read_csv(OUT / "table3_real_data.csv")
t4 = pd.read_csv(OUT / "table4_baselines.csv")
hist = pd.read_csv(OUT / "training_history.csv")
val = pd.read_csv(OUT / "validation_results.csv")
ckpt = json.load(open(OUT / "checkpoints" / "global_dict.json"))

# dataset statistics are recomputed here so the table is self-contained and reproducible
import tifffile
from scipy.signal import convolve2d

LAP = np.array([[1., -2., 1.], [-2., 4., -2.], [1., -2., 1.]])


def sigma_lap(img: np.ndarray) -> float:
    r = convolve2d(np.asarray(img, np.float64), LAP, mode="valid")
    h, w = r.shape
    return float(np.sqrt(np.pi / 2) * np.abs(r).sum() / (6 * (w - 2) * (h - 2)))


rows = []
for sid in range(1, 18):
    d = ROOT / "Sparsity_SDOCT_DATASET_2012" / str(sid)
    raw = tifffile.imread(next(d.glob("*Raw Image.tif"))).astype(np.float64)
    avg = tifffile.imread(next(d.glob("*Averaged Image.tif"))).astype(np.float64)
    rows.append(dict(subject=sid, shape=f"{raw.shape[0]}$\\times${raw.shape[1]}",
                     raw_mean=raw.mean(), raw_std=raw.std(), raw_p99=np.percentile(raw, 99),
                     avg_mean=avg.mean(), avg_std=avg.std(), avg_p99=np.percentile(avg, 99),
                     sig_raw=sigma_lap(raw), sig_avg=sigma_lap(avg),
                     has_msbtd=any(d.glob("*MSBTD*.mat"))))
ds = pd.DataFrame(rows)
ds.to_csv(REP / "dataset_summary.csv", index=False)

# ----------------------------------------------------------------------------- figures
# copy the notebook's own figures (they are the primary evidence)
for p in sorted(OUT.glob("fig*.png")):
    shutil.copy(p, FIG / p.name)


def pipeline_diagram(path: Path) -> None:
    """Two-lane schematic of the paper's Fig. 1 pipeline (learning lane + inference lane)."""
    fig, ax = plt.subplots(figsize=(7.4, 3.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.2)
    ax.axis("off")

    def box(x, y, w, h, text, fc, fs=7.0):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.045,rounding_size=0.12",
                                    fc=fc, ec="#333333", lw=0.8))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, linespacing=1.25)

    def arrow(x0, y0, x1, y1, style="-|>", ls="-"):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=9,
                                     lw=0.8, color="#333333", ls=ls, shrinkA=1, shrinkB=1))

    c_learn, c_inf, c_out = "#dbe7f5", "#f7e6d5", "#e2f0d9"

    ax.text(0.05, 4.95, "offline: dictionary learning (once)", fontsize=7.6, style="italic", color="#1f4e79")
    box(0.10, 3.55, 1.75, 0.95, "high-quality\n(averaged) B-scans\nsubjects 1-12", c_learn)
    box(2.15, 3.55, 1.55, 0.95, "overlapping\n$8\\times8$ patches\n$n=64$", c_learn)
    box(4.00, 3.55, 1.55, 0.95, "K-SVD\nEq. (8)-(9)\n20 sweeps", c_learn)
    box(5.85, 3.55, 1.95, 0.95, "global dictionary $D$\n$64\\times256$, unit norms", c_out)
    arrow(1.86, 4.02, 2.14, 4.02)
    arrow(3.71, 4.02, 3.99, 4.02)
    arrow(5.56, 4.02, 5.84, 4.02)
    ax.text(6.85, 3.38, "used for every image\n-> no per-image learning", fontsize=6.6, ha="center")

    ax.text(0.05, 2.92, "online: denoising of one noisy B-scan", fontsize=7.6, style="italic", color="#843c0c")
    box(0.10, 1.05, 1.75, 1.35, "noisy input $Y$\n(single-shot\nB-scan)", c_inf)
    box(2.15, 2.05, 1.55, 0.85, "tile image,\nmin GLCM entropy\nEq. (6)", c_inf)
    box(2.15, 1.05, 1.55, 0.75, "Laplacian kernel\nEq. (4)-(5)", c_inf)
    box(4.00, 1.35, 1.45, 0.85, "noise intensity\n$\\hat\\sigma$  Eq. (7)", c_inf)
    box(4.00, 0.20, 1.45, 0.75, "$\\lambda = 30/\\hat\\sigma$\n$\\varepsilon = C\\hat\\sigma\\sqrt{n}$", c_learn)
    box(5.85, 1.05, 1.95, 1.35, "OMP sparse coding\nEq. (2)\n$\\alpha_{ij}$ per patch", c_inf)
    box(8.10, 1.05, 1.75, 1.35, "closed-form update\nEq. (3)\n$\\hat{X}$", c_out)
    arrow(1.86, 2.35, 2.14, 2.30)
    arrow(2.92, 2.04, 2.92, 1.82)
    arrow(3.71, 1.42, 3.99, 1.62)
    arrow(4.72, 1.34, 4.72, 0.97)
    arrow(5.46, 1.72, 5.84, 1.72)
    arrow(7.81, 1.72, 8.09, 1.72)
    ax.add_patch(FancyArrowPatch((6.85, 3.53), (6.85, 2.42), arrowstyle="-|>", mutation_scale=9,
                                 lw=0.8, color="#1f4e79", ls=(0, (3, 2))))
    ax.text(7.0, 2.95, "reused $D$", fontsize=6.6, color="#1f4e79")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def summary_curves(path: Path) -> None:
    """PSNR / SSIM / EPI versus true noise level for the three simulation variants."""
    levels = sorted(t1.sigma_true.unique())
    order = ["noisy input", "fixed parameters", "proposed"]
    label = {"noisy input": "noisy input", "fixed parameters": "fixed $\\sigma=10.2$",
             "proposed": "proposed (estimated $\\sigma$)"}
    style = {"noisy input": dict(color="0.55", marker="o", ls="--"),
             "fixed parameters": dict(color="#d27f26", marker="s", ls="-"),
             "proposed": dict(color="#1f4e79", marker="^", ls="-")}

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.3))
    for ax, metric, ylab in zip(axes, ("PSNR", "SSIM", "EPI"), ("PSNR (dB)", "SSIM", "EPI")):
        for m in order:
            y = [t1[(t1.sigma_true == s) & (t1.method == m)][metric].iloc[0] for s in levels]
            ax.plot(levels, y, label=label[m], ms=4, lw=1.2, **style[m])
        srad = t1[t1.method.str.startswith("SRAD")]
        if len(srad) and metric != "MSE":
            ax.plot(srad.sigma_true, srad[metric], "x", color="#7d3c98", ms=7, label="SRAD (adaptive dict.)")
        ax.set_xlabel("true noise intensity $\\sigma$")
        ax.set_ylabel(ylab)
        ax.set_xticks(levels)
        ax.grid(alpha=0.3, lw=0.4)
    axes[0].legend(loc="lower left", frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def ablation_curves(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    psz = [int(s.split("x")[0]) for s in t2.patch_size]
    for col, mk in zip(["64 atoms", "128 atoms", "256 atoms"], ("o", "s", "^")):
        ax.plot(psz, t2[col], marker=mk, ms=4, lw=1.1, label=col.replace(" atoms", " atoms"))
    ax.set_xscale("log", base=2)
    ax.set_xticks(psz)
    ax.set_xticklabels([f"${p}\\!\\times\\!{p}$" for p in psz])
    ax.set_xlabel("patch size")
    ax.set_ylabel("PSNR (dB)")
    ax.grid(alpha=0.3, lw=0.4)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def validation_bars(path: Path) -> None:
    modes = ["fixed sigma", "estimated sigma"]
    keys = sorted({(r.subject, r.sigma_true) for r in val.itertuples()})
    fig, ax = plt.subplots(figsize=(4.2, 2.3))
    x = np.arange(len(keys))
    w = 0.36
    for i, (m, col) in enumerate(zip(modes, ("#d27f26", "#1f4e79"))):
        y = [val[(val.subject == s) & (val.sigma_true == sg) & (val["mode"] == m)].PSNR.iloc[0] for s, sg in keys]
        bars = ax.bar(x + (i - 0.5) * w, y, w, color=col, label=m.replace("sigma", r"$\sigma=10.2$" if i == 0 else r"$\hat\sigma$"))
        for b, v in zip(bars, y):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.25, f"{v:.1f}", ha="center", fontsize=6.5)
    ax.set_xticks(x, [f"s{s}\n$\\sigma$={sg:g}" for s, sg in keys], fontsize=7)
    ax.set_ylim(20, 38)
    ax.set_ylabel("PSNR (dB)")
    ax.legend(frameon=False, ncol=2, fontsize=7, loc="lower right")
    ax.grid(axis="y", alpha=0.3, lw=0.4)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


pipeline_diagram(FIG / "fig00_pipeline.pdf")
summary_curves(FIG / "fig13_summary_curves.pdf")
ablation_curves(FIG / "fig14_ablation_curves.pdf")
validation_bars(FIG / "fig15_validation_bars.pdf")

# ----------------------------------------------------------------------------- tables


def tab_dataset() -> str:
    lines = []
    for r in ds.itertuples():
        lines.append(
            f"{r.subject:>2d} & {r.shape} & {r.raw_mean:5.1f} & {r.raw_std:5.1f} & {r.raw_p99:5.0f} & "
            f"{r.avg_mean:5.1f} & {r.avg_std:5.1f} & {r.avg_p99:5.0f} & {r.sig_raw:5.2f} & {r.sig_avg:5.2f} & "
            f"{'yes' if r.has_msbtd else '--'} \\\\")
    return "\n".join(lines)


def tab_hyper() -> str:
    rows = [
        ("patch size $p$ / patch dimension $n$", "$8\\times8$, $n=64$", "paper \\S3.2"),
        ("dictionary size $n\\times K$", "$64\\times256$", "paper \\S3.2"),
        ("K-SVD sweeps", "20", "paper \\S3.2"),
        ("initial dictionary", "DCT (overcomplete, fractional frequencies)", "paper \\S2.3"),
        ("fidelity", "$\\lambda = 30/\\hat\\sigma$", "paper, Eq. (3)"),
        ("simulation noise levels", "5.1, 10.2, 15.3", "paper \\S3.1"),
        ("fixed-parameter baseline", "$\\sigma = 10.2$", "paper \\S3.1"),
        ("SRAD noise intensity", "$\\sigma = 11$", "paper \\S3.2"),
        ("patch stride", "1 (all overlapping patches)", "assumed"),
        ("training corpus", "30\\,000 patches from subjects 1--12", "assumed"),
        ("dictionary sparsity $T$", "6 atoms / patch", "assumed"),
        ("OMP error limit", "$\\varepsilon^2=(1.15\\,\\hat\\sigma\\sqrt{n})^2$, cap 12 atoms", "assumed (C from [23])"),
        ("GLCM levels / distance / angles", "$L=32$, $d=1$, $0^\\circ,45^\\circ,90^\\circ,135^\\circ$", "assumed"),
        ("weak-texture tile size", "$64\\times64$ (tiles with std $<2$ skipped)", "assumed"),
        ("train / val / test split", "1--12 / 13--14 / 15--17", "assumed"),
        ("EPI high-pass operator", "Sobel gradient magnitude", "assumed"),
        ("geometric augmentation", "none", "paper (plain sampling)"),
    ]
    return "\n".join(f"{a} & {b} & {c} \\\\" for a, b, c in rows)


def tab_valid() -> str:
    lines = []
    for r in val.itertuples():
        sg = "--" if pd.isna(r.sigma_used) else f"{r.sigma_used:.2f}"
        lines.append(f"{r.subject} & {r.sigma_true:.1f} & {r.mode} & {sg} & {r.MSE:7.2f} & {r.PSNR:5.2f} & "
                     f"{r.SSIM:.4f} & {r.EPI:.4f} \\\\")
    return "\n".join(lines)


def tab_t1() -> str:
    lines = []
    for s in sorted(t1.sigma_true.unique()):
        sub = t1[t1.sigma_true == s]
        order = ["noisy input", "fixed parameters", "proposed"]
        if (sub.method.str.startswith("SRAD")).any():
            order.append("SRAD (adaptive dict.)")
        for i, m in enumerate(order):
            r = sub[sub.method == m].iloc[0]
            sg = "--" if pd.isna(r.sigma_used) else f"{r.sigma_used:.2f}"
            atoms = "--" if pd.isna(r.atoms) else f"{r.atoms:.2f}"
            head = f"{s:.1f}" if i == 0 else ""
            lines.append(f"{head} & {m} & {sg} & {r.MSE:6.2f} & {r.PSNR:5.2f} & {r.SSIM:.4f} & {r.EPI:.4f} & "
                         f"{atoms} & {r.time_s:5.1f} \\\\")
        lines.append("\\addlinespace[2pt]")
    return "\n".join(lines[:-1])


def tab_t3() -> str:
    lines = []
    for s in sorted(t3.subject.unique()):
        sub = t3[t3.subject == s]
        for i, m in enumerate(sub.method):
            r = sub[sub.method == m].iloc[0]
            sg = "--" if pd.isna(r.sigma_used) else f"{r.sigma_used:.2f}"
            tt = "--" if pd.isna(r.time_s) else f"{r.time_s:5.1f}"
            head = f"{s}" if i == 0 else ""
            lines.append(f"{head} & {m} & {sg} & {r.MSE:7.2f} & {r.PSNR:5.2f} & {r.SSIM:.4f} & {r.EPI:.4f} & {tt} \\\\")
        lines.append("\\addlinespace[2pt]")
    return "\n".join(lines[:-1])


def tab_t4() -> str:
    lines = []
    for r in t4.itertuples():
        lines.append(f"{r.method} & {r.MSE:7.2f} & {r.PSNR:5.2f} & {r.SSIM:.4f} & {r.EPI:.4f} \\\\")
    return "\n".join(lines)


def tab_abl() -> str:
    lines = []
    for r in t2.itertuples():
        ps = r.patch_size.replace("x", "$\\times$")
        lines.append(f"{ps} & {getattr(r, '_2'):5.2f} & {getattr(r, '_4'):5.2f} & {getattr(r, '_6'):5.2f} \\\\")
    return "\n".join(lines)


def tab_epi() -> str:
    # values reproduced from the notebook's EPI sensitivity cell (§7.7)
    rows = [
        ("noisy input", 0.4557, 0.3038, 0.6661),
        ("fixed $\\sigma=10.2$", 0.6510, 0.3055, 0.7606),
        ("proposed", 0.8598, 0.2908, 0.8819),
        ("reference blurred by 1\\,px (ceiling)", 0.9020, 0.4029, 1.0000),
    ]
    return "\n".join(f"{a} & {b:.4f} & {c:.4f} & {d:.4f} \\\\" for a, b, c, d in rows)


def tab_runtime() -> str:
    prop = t3[t3.method == "proposed"]
    lines = []
    for r in prop.itertuples():
        lines.append(f"{r.subject} & {r.sigma_used:.2f} & {r.MSE:7.2f} & {r.PSNR:5.2f} & {r.time_s:5.1f} \\\\")
    return "\n".join(lines)


def tab_env() -> str:
    import platform
    import scipy
    import skimage
    import pywt
    import matplotlib as mpl
    rows = [
        ("Python", platform.python_version()),
        ("NumPy", np.__version__),
        ("SciPy", scipy.__version__),
        ("scikit-image", skimage.__version__),
        ("PyWavelets", pywt.__version__),
        ("Matplotlib", mpl.__version__),
        ("pandas", pd.__version__),
        ("TiFFfile", tifffile.__version__),
        ("Platform", f"{platform.system()} {platform.machine()}"),
    ]
    esc = lambda s: str(s).replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")
    return "\n".join(f"{a} & {esc(b)} \\\\" for a, b in rows)


# ----------------------------------------------------------------------------- meta values
prop = t3[t3.method == "proposed"]
msb = t3[t3.method == "MSBTD (2012)"]
srad_real = t3[t3.method.str.startswith("SRAD")]
gain = {}
for s in sorted(t1.sigma_true.unique()):
    base = t1[(t1.sigma_true == s) & (t1.method == "noisy input")].PSNR.iloc[0]
    gain[float(s)] = {
        "fixed": t1[(t1.sigma_true == s) & (t1.method == "fixed parameters")].PSNR.iloc[0] - base,
        "proposed": t1[(t1.sigma_true == s) & (t1.method == "proposed")].PSNR.iloc[0] - base,
    }

val["gain"] = np.nan
piv = val.pivot_table(index=["subject", "sigma_true"], columns="mode", values="PSNR")
piv["gain"] = piv["estimated sigma"] - piv["fixed sigma"]

def sci(v: float, digits: int = 2) -> str:
    """LaTeX scientific notation, e.g. 2.86\\times10^{7}."""
    m, e = f"{v:.{digits}e}".split("e")
    return f"{m}\\times10^{{{int(e)}}}"


meta = {
    "@@RMSE_FIRST@@": f"{hist.rmse.iloc[0]:.3f}",
    "@@RMSE_LAST@@": f"{hist.rmse.iloc[-1]:.3f}",
    "@@RMSE_GAIN@@": f"{100 * (hist.rmse.iloc[-1] - hist.rmse.iloc[0]) / hist.rmse.iloc[0]:+.2f}",
    "@@OBJ_FIRST@@": sci(hist.obj.iloc[0]),
    "@@OBJ_LAST@@": sci(hist.obj.iloc[-1]),
    "@@TRAIN_SEC@@": f"{hist.sec.sum():.0f}",
    "@@DICT_ATOMS@@": f"{ckpt['n_atoms']}",
    "@@N_PATCHES@@": f"{ckpt['n_patches']:,}".replace(",", "\\,"),
    "@@SIG_RAW@@": f"{ds.sig_raw.mean():.2f}",
    "@@SIG_AVG@@": f"{ds.sig_avg.mean():.2f}",
    "@@SIG_RATIO@@": f"{ds.sig_raw.mean() / ds.sig_avg.mean():.2f}",
    "@@VAL_WINS@@": f"{int((piv.gain > 0).sum())}",
    "@@VAL_N@@": f"{len(piv)}",
    "@@VAL_GAIN_MIN@@": f"{piv.gain.min():+.2f}",
    "@@VAL_GAIN_MAX@@": f"{piv.gain.max():+.2f}",
    "@@GAIN_FIX_51@@": f"{gain[5.1]['fixed']:+.2f}",
    "@@GAIN_PROP_51@@": f"{gain[5.1]['proposed']:+.2f}",
    "@@GAIN_FIX_102@@": f"{gain[10.2]['fixed']:+.2f}",
    "@@GAIN_PROP_102@@": f"{gain[10.2]['proposed']:+.2f}",
    "@@GAIN_FIX_153@@": f"{gain[15.3]['fixed']:+.2f}",
    "@@GAIN_PROP_153@@": f"{gain[15.3]['proposed']:+.2f}",
    "@@RAW_PSNR@@": f"{t3[t3.method == 'raw'].PSNR.min():.2f}--{t3[t3.method == 'raw'].PSNR.max():.2f}",
    "@@PROP_PSNR@@": f"{prop.PSNR.min():.2f}--{prop.PSNR.max():.2f}",
    "@@MSB_PSNR@@": f"{msb.PSNR.min():.2f}--{msb.PSNR.max():.2f}",
    "@@PROP_TIME@@": f"{prop.time_s.min():.1f}--{prop.time_s.max():.1f}",
    "@@SPEEDUP@@": f"{srad_real.time_s.max() / prop[prop.subject == srad_real.subject.iloc[0]].time_s.iloc[0]:.1f}",
    "@@SRAD_TIME@@": f"{srad_real.time_s.max():.1f}",
    "@@SRAD_FIX@@": f"{srad_real[srad_real.method.str.contains('paper')].PSNR.iloc[0]:.2f}",
    "@@SRAD_MATCH@@": f"{srad_real[srad_real.method.str.contains('matched')].PSNR.iloc[0]:.2f}",
    "@@ABL_DICT@@": f"{(t2[['64 atoms', '128 atoms', '256 atoms']].max(1) - t2[['64 atoms', '128 atoms', '256 atoms']].min(1)).max():.2f}",
    "@@ABL_PATCH@@": f"{(t2[['64 atoms', '128 atoms', '256 atoms']].max(0) - t2[['64 atoms', '128 atoms', '256 atoms']].min(0)).max():.2f}",
    "@@BEST_PATCH@@": t2.loc[int(np.unravel_index(np.argmax(t2[['64 atoms', '128 atoms', '256 atoms']].values), (3, 3))[0]), 'patch_size'].replace("x", r"$\times$"),
}

# ---- notebook metadata: cell counts and section -> cell map ----
import re
nb = json.load(open(ROOT / "denoising.ipynb"))
cells = nb["cells"]
n_cells = len(cells)
n_code = sum(1 for c in cells if c["cell_type"] == "code")

sections = []
for i, c in enumerate(cells):
    if c["cell_type"] != "markdown":
        continue
    first = "".join(c["source"]).split("\n")[0]
    m = re.match(r"^#{1,3}\\s*§(\\d+)\\s*(.*)$", first)
    if m:
        sections.append((i, int(m.group(1)), m.group(2).strip()))

cellmap_lines = []
for k, (idx, num, title) in enumerate(sections):
    end = sections[k + 1][0] - 1 if k + 1 < len(sections) else n_cells - 1
    title = title.replace("&", "\\&").replace("_", "\\_").replace("#", "\\#")
    cellmap_lines.append(f"\\S{num} & {title} & {idx + 1}--{end + 1} \\\\")
meta["@@NCELLS@@"] = str(n_cells)
meta["@@NCODE@@"] = str(n_code)
meta["@@TAB_CELLMAP@@"] = "\n".join(cellmap_lines)
meta["@@REQUIREMENTS@@"] = (ROOT / "requirements.txt").read_text().rstrip()

template = (REP / "main.tex.in").read_text()
body = {
    "@@TAB_DATASET@@": tab_dataset(),
    "@@TAB_HYPER@@": tab_hyper(),
    "@@TAB_VALID@@": tab_valid(),
    "@@TAB_T1@@": tab_t1(),
    "@@TAB_T3@@": tab_t3(),
    "@@TAB_T4@@": tab_t4(),
    "@@TAB_ABL@@": tab_abl(),
    "@@TAB_EPI@@": tab_epi(),
    "@@TAB_RUNTIME@@": tab_runtime(),
    "@@TAB_ENV@@": tab_env(),
}
tex = template
for k, v in meta.items():
    tex = tex.replace(k, v)
# the prose uses the short form too: \NAME (without the @@ delimiters) -- substitute those as well
for k, v in meta.items():
    short = k.strip("@")
    tex = tex.replace("\\" + short, v)
for k, v in body.items():
    tex = tex.replace(k, v)
(REP / "main.tex").write_text(tex)
print(f"wrote {REP/'main.tex'}  ({len(tex)/1024:.1f} kB)")

# ----------------------------------------------------------------------------- compile
if shutil.which("tectonic") or Path.home().joinpath(".local/bin/tectonic").exists():
    engine = shutil.which("tectonic") or str(Path.home() / ".local/bin/tectonic")
    print("compiling with", engine)
    r = subprocess.run([engine, "-X", "compile", "main.tex", "--keep-logs"],
                       cwd=REP, capture_output=True, text=True)
    print(r.stdout[-3000:])
    print(r.stderr[-4000:])
    print("exit", r.returncode)
else:
    print("no LaTeX engine found; run: tectonic -X compile main.tex")
