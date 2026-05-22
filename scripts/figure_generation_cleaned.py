"""Cleaned public figure-generation script derived from the raw notebook export.

This file keeps only the final main-figure or panel-generation blocks identified
from the raw figure notebook export. Personal paths, notebook cell markers, and
comment-only notebook annotations were removed. Running this script can recreate
figures from existing analysis outputs; it does not train models by itself.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"
SUPPLEMENTARY_DIR = PROJECT_ROOT / "supplementary"

SAVE_DIR = str(RESULTS_DIR)

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
from matplotlib.gridspec import GridSpec




def _resolve_base_df():
    if "df_eda" in globals() and isinstance(globals()["df_eda"], pd.DataFrame):
        return globals()["df_eda"].copy()

    if "df" in globals() and isinstance(globals()["df"], pd.DataFrame):
        return globals()["df"].copy()

    has_train = "df_train" in globals() and isinstance(globals()["df_train"], pd.DataFrame)
    has_test = "df_test" in globals() and isinstance(globals()["df_test"], pd.DataFrame)

    if has_train and has_test:
        return pd.concat(
            [globals()["df_train"].copy(), globals()["df_test"].copy()],
            axis=0,
            ignore_index=True
        )

    raise ValueError("text df_eda / df / df_train+df_test, text ")

def _resolve_save_dir():
    if "SAVE_DIR" in globals():
        return globals()["SAVE_DIR"]
    return "figures_panel"

df_panel = _resolve_base_df()
SAVE_DIR = _resolve_save_dir()






PREVIEW_MODE = True

if PREVIEW_MODE:
    FIG_W, FIG_H = 18, 13
    FIG_DPI = 150
    SAVE_DPI = 300
    EXPORT_PNG = True
    EXPORT_SVG = False
    EXPORT_PDF = False
    OUT_NAME = "Figure1_panel_2x3_plus_1x2_preview_colored_box"
else:
    FIG_W, FIG_H = 19, 13.5
    FIG_DPI = 180
    SAVE_DPI = 600
    EXPORT_PNG = True
    EXPORT_SVG = True
    EXPORT_PDF = True
    OUT_NAME = "Figure1_panel_2x3_plus_1x2_final_colored_box"







FONT_CANDIDATES = [
    r"C:/Windows/Fonts/simhei.ttf",
    r"C:/Windows/Fonts/msyh.ttc",
    r"C:/Windows/Fonts/arial.ttf",
    r"/System/Library/Fonts/PingFang.ttc",
    r"/System/Library/Fonts/STHeiti Light.ttc",
    r"/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    r"/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    r"/usr/share/fonts/truetype/arphic/ukai.ttc",
    r"/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]

found_font = None
for fp in FONT_CANDIDATES:
    if os.path.exists(fp):
        found_font = fp
        break

if found_font:
    font_prop = fm.FontProperties(fname=found_font)
    plt.rcParams["font.family"] = font_prop.get_name()
else:
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]

plt.rcParams["axes.unicode_minus"] = False

sns.set_theme(style="whitegrid", context="paper")

plt.rcParams.update({
    "figure.dpi": FIG_DPI,
    "savefig.dpi": SAVE_DPI,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.labelweight": "bold",
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})




PANEL_DIR = os.path.join(SAVE_DIR, "05_panel")
os.makedirs(PANEL_DIR, exist_ok=True)





BAR_COLOR = "#4C72B0"


BOX_COLOR_G = "#BFD7EA"
BOX_COLOR_H = "#E8D3B0"
BOX_EDGE = "#7A7A7A"
MEDIAN_COLOR = "#4A4A4A"




def clean_cat_series(s):
    s = s.copy()
    s = s.fillna("Unknown").astype(str).str.strip()
    s = s.replace(r"^\s*others%\s*$", "Other", regex=True)
    s = s.replace("", "Unknown")
    return s

def add_panel_label(ax, label):
    ax.text(
        -0.04, 1.02, label,
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        ha="left",
        va="top"
    )

def get_top_counts(series, topk=8):
    vc = clean_cat_series(series).value_counts()
    if len(vc) > topk:
        head = vc.iloc[:topk].copy()
        tail_sum = vc.iloc[topk:].sum()
        if "Other" in head.index:
            head["Other"] += tail_sum
        else:
            head["Other"] = tail_sum
        vc = head
    return vc

def plot_categorical_bar(ax, df, col, topk=8):
    s = clean_cat_series(df[col])
    vc = get_top_counts(s, topk=topk)
    vc = vc.sort_values(ascending=True)
    pct = vc / vc.sum() * 100

    bars = ax.barh(
        vc.index,
        pct.values,
        color=BAR_COLOR,
        edgecolor="black",
        linewidth=0.7
    )

    for bar, p in zip(bars, pct.values):
        ax.text(
            p + 0.50,
            bar.get_y() + bar.get_height() / 2,
            f"{p:.1f}%",
            va="center",
            ha="left",
            fontsize=7.5
        )

    ax.set_title(col, fontsize=12, fontweight="bold", pad=4)
    ax.set_xlabel("Percentage (%)", fontsize=10, fontweight="bold")
    ax.set_ylabel("")
    ax.tick_params(axis="y", labelsize=9)
    ax.tick_params(axis="x", labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="x", linestyle="--", linewidth=0.45, alpha=0.28)
    ax.grid(False, axis="y")

def filter_top_groups(df, group_col="CT", topk=8):
    d = df.copy()
    d[group_col] = clean_cat_series(d[group_col])
    keep = d[group_col].value_counts().head(topk).index.tolist()
    d = d[d[group_col].isin(keep)].copy()
    return d, keep

def plot_box_by_group(ax, df, y_col, title_text, y_axis_label,
                      group_col="CT", order=None, topk=8, box_color="#BFD7EA"):
    d, keep = filter_top_groups(df, group_col=group_col, topk=topk)
    d[y_col] = pd.to_numeric(d[y_col], errors="coerce")
    d = d.dropna(subset=[y_col])

    if d.empty:
        ax.axis("off")
        ax.text(0.5, 0.5, f"No valid data for {y_col}", ha="center", va="center")
        return

    if order is None:
        order = [x for x in keep if x in d[group_col].unique()]
    else:
        order = [x for x in order if x in d[group_col].unique()]

    sns.boxplot(
        data=d,
        x=group_col,
        y=y_col,
        order=order,
        ax=ax,
        color=box_color,
        linewidth=1.0,
        fliersize=2.2,
        boxprops=dict(edgecolor=BOX_EDGE, alpha=0.95),
        whiskerprops=dict(color=BOX_EDGE, linewidth=1.0),
        capprops=dict(color=BOX_EDGE, linewidth=1.0),
        medianprops=dict(color=MEDIAN_COLOR, linewidth=1.4),
        flierprops=dict(
            marker='o',
            markersize=2.2,
            markerfacecolor=BOX_EDGE,
            markeredgecolor=BOX_EDGE,
            alpha=0.55
        )
    )

    ax.set_title(title_text, fontsize=12, fontweight="bold", pad=4)
    ax.set_xlabel(group_col, fontsize=10, fontweight="bold")
    ax.set_ylabel(y_axis_label, fontsize=10, fontweight="bold")
    ax.tick_params(axis="x", rotation=25, labelsize=8.2)
    ax.tick_params(axis="y", labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.45, alpha=0.28)
    ax.grid(False, axis="x")




if "Size_nm" not in df_panel.columns and "Size" in df_panel.columns:
    s = pd.to_numeric(df_panel["Size"], errors="coerce")
    if s.notna().mean() > 0.5:
        ok_lo = (s.dropna() >= -1).mean() if len(s.dropna()) else 0
        ok_hi = (s.dropna() <= 6).mean() if len(s.dropna()) else 0
        if (ok_lo > 0.8) and (ok_hi > 0.8):
            df_panel["Size_nm"] = 10 ** s





cat_vars = ["Type", "MAT", "Shape", "CT", "TM", "TS"]



continuous_vars = [
    ("Size_nm", "Size by CT", "Size, nm"),
    ("Zeta Potential", "Zeta potential by CT", "Zeta potential")
]

group_col = "CT"
CAT_TOPK = 8
CT_TOPK = 8


if group_col in df_panel.columns:
    ct_order = clean_cat_series(df_panel[group_col]).value_counts().head(CT_TOPK).index.tolist()
else:
    ct_order = None




fig = plt.figure(figsize=(FIG_W, FIG_H))
gs = GridSpec(
    nrows=3,
    ncols=6,
    figure=fig,
    hspace=0.35,
    wspace=1.20
)


axA = fig.add_subplot(gs[0, 0:2])
axB = fig.add_subplot(gs[0, 2:4])
axC = fig.add_subplot(gs[0, 4:6])

axD = fig.add_subplot(gs[1, 0:2])
axE = fig.add_subplot(gs[1, 2:4])
axF = fig.add_subplot(gs[1, 4:6])


axG = fig.add_subplot(gs[2, 0:3])
axH = fig.add_subplot(gs[2, 3:6])

axes_map = {
    "A": axA, "B": axB, "C": axC,
    "D": axD, "E": axE, "F": axF,
    "G": axG, "H": axH
}




for label, col in zip(list("ABCDEF"), cat_vars):
    ax = axes_map[label]
    if col in df_panel.columns:
        plot_categorical_bar(ax, df_panel, col, topk=CAT_TOPK)
        add_panel_label(ax, label)
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, f"{col} not found", ha="center", va="center", fontsize=10)




for label, (y_col, title_text, y_axis_label) in zip(["G", "H"], continuous_vars):
    ax = axes_map[label]
    if (y_col in df_panel.columns) and (group_col in df_panel.columns):
        this_color = BOX_COLOR_G if label == "G" else BOX_COLOR_H

        plot_box_by_group(
            ax=ax,
            df=df_panel,
            y_col=y_col,
            title_text=title_text,
            y_axis_label=y_axis_label,
            group_col=group_col,
            order=ct_order,
            topk=CT_TOPK,
            box_color=this_color
        )
        add_panel_label(ax, label)
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, f"{y_col} or {group_col} not found", ha="center", va="center", fontsize=10)




out_base = os.path.join(PANEL_DIR, OUT_NAME)

if EXPORT_PNG:
    fig.savefig(out_base + ".png", dpi=SAVE_DPI, bbox_inches="tight", facecolor="white")

if EXPORT_SVG:
    fig.savefig(out_base + ".svg", bbox_inches="tight", facecolor="white")

if EXPORT_PDF:
    fig.savefig(out_base + ".pdf", bbox_inches="tight", facecolor="white")

plt.show()
plt.close(fig)

print("text: ")
if EXPORT_PNG:
    print(out_base + ".png")
if EXPORT_SVG:
    print(out_base + ".svg")
if EXPORT_PDF:
    print(out_base + ".pdf")


















import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.gridspec import GridSpec




if "SAVE_DIR" not in globals():
    SAVE_DIR = "figures_panel"

if "TARGET_DE" not in globals():
    TARGET_DE = "DE_tumor"

os.makedirs(SAVE_DIR, exist_ok=True)

PANEL_DIR = os.path.join(SAVE_DIR, "Figure2_combined")
os.makedirs(PANEL_DIR, exist_ok=True)


if "df_plot" not in globals():
    if ("df_train" not in globals()) or ("df_test" not in globals()):
        raise ValueError("text df_plot, text df_train / df_test, text ")

    df_train_plot = df_train.copy()
    df_train_plot["split"] = "train"

    df_test_plot = df_test.copy()
    df_test_plot["split"] = "test"

    df_plot = pd.concat(
        [df_train_plot, df_test_plot],
        axis=0,
        ignore_index=True
    )


if "y" not in df_plot.columns:
    if "thr" not in globals():
        thr = float(pd.to_numeric(df_train[TARGET_DE], errors="coerce").quantile(0.75))

    df_plot["y"] = (
        pd.to_numeric(df_plot[TARGET_DE], errors="coerce") >= thr
    ).astype(int)


if "log_DE_tumor" not in df_plot.columns:
    df_plot["log_DE_tumor"] = np.log10(
        pd.to_numeric(df_plot[TARGET_DE], errors="coerce") + 0.01
    )


if "thr_log" not in globals():
    if "thr" not in globals():
        thr = float(pd.to_numeric(df_train[TARGET_DE], errors="coerce").quantile(0.75))
    thr_log = np.log10(thr + 0.01)


Q75_TEXT = r"Q$_{0.75}$ = 1.773"


if "df" in globals() and isinstance(df, pd.DataFrame):
    df_for_all = df.copy()
else:
    df_for_all = df_plot.copy()


if "Zeta Potential" in df_for_all.columns:
    ZETA_COL_ALL = "Zeta Potential"
elif "Zeta potential" in df_for_all.columns:
    ZETA_COL_ALL = "Zeta potential"
else:
    ZETA_COL_ALL = None


if "base_features" not in globals():
    base_features = []
    for c in ["Type", "MAT", "TS", "CT", "TM", "Shape", "Size"]:
        if c in df_for_all.columns:
            base_features.append(c)

    if ZETA_COL_ALL is not None:
        base_features.append(ZETA_COL_ALL)

    if "Admin" in df_for_all.columns:
        base_features.append("Admin")




FONT_CANDIDATES = [
    r"C:/Windows/Fonts/arial.ttf",
    r"C:/Windows/Fonts/ARIAL.TTF",
    r"C:/Windows/Fonts/msyh.ttc",
    r"C:/Windows/Fonts/simhei.ttf",
    r"/System/Library/Fonts/Supplemental/Arial.ttf",
    r"/System/Library/Fonts/PingFang.ttc",
    r"/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]

found_font = None
for fp in FONT_CANDIDATES:
    if os.path.exists(fp):
        found_font = fp
        break

if found_font:
    font_prop = fm.FontProperties(fname=found_font)
    plt.rcParams["font.family"] = font_prop.get_name()
else:
    plt.rcParams["font.family"] = "Arial"

plt.rcParams.update({
    "figure.dpi": 180,
    "savefig.dpi": 900,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.titleweight": "normal",
    "axes.labelsize": 9,
    "axes.labelweight": "normal",
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.linewidth": 1.0,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "axes.unicode_minus": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none"
})




COLOR_TRAIN = "#9FB8D8"
COLOR_TEST = "#E7B5B8"
COLOR_HIGH = "#B6423F"
COLOR_NONHIGH = "#AEBBE0"
COLOR_TEAL = "#4398A0"
COLOR_MISSING = "#B6423F"
COLOR_BOX = "#D7E5E7"
COLOR_BOX_EDGE = "#2F2F2F"
COLOR_DOT = "#4A4A4A"




DISPLAY_LABEL_MAP = {
    "TM": {
        "Xenograft Heterotopic": "XH",
        "Allograft Heterotopic": "AH",
        "Xenograft Orthotopic": "XO",
        "Allograft Orthotopic": "AO",
        "XH": "XH",
        "AH": "AH",
        "XO": "XO",
        "AO": "AO",
    },
    "Type": {
        "Organic": "ONM",
        "Inorganic": "INM",
        "Organic Nanomaterial": "ONM",
        "Inorganic Nanomaterial": "INM",
        "ONM": "ONM",
        "INM": "INM",
    }
}

TITLE_LABEL_MAP = {
    "Type": "Type",
    "MAT": "Material",
    "Shape": "Shape",
    "CT": "Cancer type",
    "TM": "Tumor model",
    "TS": "Targeting strategy"
}

MISSING_LABEL_MAP = {
    "Zeta potential": "Zeta Potential",
    "Zeta Potential": "Zeta Potential",
    "Size": "Size",
    "Admin": "Admin"
}

def clean_cat_series(s):
    s = s.copy()
    s = s.fillna("Unknown").astype(str).str.strip()
    s = s.replace(r"^\s*others%\s*$", "Other", regex=True)
    s = s.replace("", "Unknown")
    return s

def apply_display_abbreviation(s, col):
    if col in DISPLAY_LABEL_MAP:
        return s.replace(DISPLAY_LABEL_MAP[col])
    return s

def add_panel_label(ax, label, x=-0.16, y=1.06):
    label_text = str(label)
    if not label_text.startswith("("):
        label_text = f"({label_text})"

    ax.text(
        x, y, label_text,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        ha="left",
        va="top"
    )

def add_group_title(fig, axes, title, y_offset=0.012, fontsize=11):
    bboxes = [ax.get_position() for ax in axes]
    x0 = min(b.x0 for b in bboxes)
    x1 = max(b.x1 for b in bboxes)
    y1 = max(b.y1 for b in bboxes)

    fig.text(
        (x0 + x1) / 2,
        y1 + y_offset,
        title,
        ha="center",
        va="bottom",
        fontsize=fontsize,
        fontweight="bold"
    )

def remove_top_right_spines(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)




def get_top_counts(series, topk=8):
    vc = series.value_counts()

    if len(vc) > topk:
        head = vc.iloc[:topk].copy()
        tail_sum = vc.iloc[topk:].sum()

        if "Other" in head.index:
            head["Other"] += tail_sum
        else:
            head["Other"] = tail_sum

        vc = head

    return vc

def plot_categorical_bar(ax, df_obj, col, topk=8):
    s = clean_cat_series(df_obj[col])
    s = apply_display_abbreviation(s, col)

    vc = get_top_counts(s, topk=topk)
    vc = vc.sort_values(ascending=True)
    pct = vc / vc.sum() * 100

    bars = ax.barh(
        vc.index,
        pct.values,
        color=COLOR_TEAL,
        edgecolor="0.25",
        linewidth=0.6,
        height=0.72
    )

    x_max = max(float(pct.max()), 1.0)

    for bar, p in zip(bars, pct.values):
        ax.text(
            p + x_max * 0.025,
            bar.get_y() + bar.get_height() / 2,
            f"{p:.0f}%",
            va="center",
            ha="left",
            fontsize=7.3,
            color="#1F1F1F"
        )

    ax.set_title(
        TITLE_LABEL_MAP.get(col, col),
        pad=4,
        fontweight="normal"
    )
    ax.set_xlabel("Percentage (%)")
    ax.set_ylabel("")
    ax.set_xlim(0, x_max * 1.30)

    remove_top_right_spines(ax)
    ax.grid(True, axis="x", linestyle="--", linewidth=0.45, alpha=0.20)
    ax.grid(False, axis="y")

def filter_top_groups(df_obj, group_col="CT", topk=8):
    d = df_obj.copy()
    d[group_col] = clean_cat_series(d[group_col])
    keep = d[group_col].value_counts().head(topk).index.tolist()
    d = d[d[group_col].isin(keep)].copy()


    if "Other" in d[group_col].unique():
        d = d[d[group_col] != "Other"].copy()
        keep = [x for x in keep if x != "Other"]

    return d, keep

def get_value_label(y_col):
    if y_col == "Size_nm":
        return "Size (nm)"
    if y_col in ["Zeta Potential", "Zeta potential"]:
        return "Zeta Potential (mV)"
    return y_col

def simple_boxplot_vertical(ax, data, group_col, value_col, order=None):
    if order is None:
        order = data[group_col].dropna().unique().tolist()

    values = []
    labels = []

    for g in order:
        vals = pd.to_numeric(
            data.loc[data[group_col] == g, value_col],
            errors="coerce"
        ).dropna().values

        if len(vals) > 0:
            values.append(vals)
            labels.append(g)

    if len(values) == 0:
        ax.axis("off")
        ax.text(0.5, 0.5, f"No valid data for {value_col}", ha="center", va="center")
        return

    bp = ax.boxplot(
        values,
        labels=labels,
        patch_artist=True,
        showfliers=True,
        widths=0.62,
        medianprops=dict(color="black", linewidth=1.0),
        boxprops=dict(color=COLOR_BOX_EDGE, linewidth=0.8),
        whiskerprops=dict(color=COLOR_BOX_EDGE, linewidth=0.8),
        capprops=dict(color=COLOR_BOX_EDGE, linewidth=0.8),
        flierprops=dict(
            marker="o",
            markerfacecolor="white",
            markeredgecolor=COLOR_DOT,
            markersize=2.5,
            linestyle="none",
            markeredgewidth=0.5
        )
    )

    for patch in bp["boxes"]:
        patch.set_facecolor(COLOR_BOX)

    ax.tick_params(axis="x", rotation=25)

def simple_boxplot_horizontal(ax, data, group_col, value_col, order=None):
    if order is None:
        order = data[group_col].dropna().unique().tolist()

    values = []
    labels = []

    for g in order:
        vals = pd.to_numeric(
            data.loc[data[group_col] == g, value_col],
            errors="coerce"
        ).dropna().values

        if len(vals) > 0:
            values.append(vals)
            labels.append(g)

    if len(values) == 0:
        ax.axis("off")
        ax.text(0.5, 0.5, f"No valid data for {value_col}", ha="center", va="center")
        return

    bp = ax.boxplot(
        values,
        vert=False,
        labels=labels,
        patch_artist=True,
        showfliers=True,
        widths=0.58,
        medianprops=dict(color="black", linewidth=1.0),
        boxprops=dict(color=COLOR_BOX_EDGE, linewidth=0.8),
        whiskerprops=dict(color=COLOR_BOX_EDGE, linewidth=0.8),
        capprops=dict(color=COLOR_BOX_EDGE, linewidth=0.8),
        flierprops=dict(
            marker="o",
            markerfacecolor="white",
            markeredgecolor=COLOR_DOT,
            markersize=2.3,
            linestyle="none",
            markeredgewidth=0.5
        )
    )

    for patch in bp["boxes"]:
        patch.set_facecolor(COLOR_BOX)

    ax.invert_yaxis()
    ax.margins(x=0.05)

def plot_box_by_group(
    ax,
    df_obj,
    y_col,
    title_text,
    group_col="CT",
    order=None,
    topk=8,
    orientation="vertical"
):
    d, keep = filter_top_groups(df_obj, group_col=group_col, topk=topk)
    d[y_col] = pd.to_numeric(d[y_col], errors="coerce")
    d = d.dropna(subset=[y_col])

    if d.empty:
        ax.axis("off")
        ax.text(0.5, 0.5, f"No valid data for {y_col}", ha="center", va="center")
        return

    if order is None:
        order = [x for x in keep if x in d[group_col].unique()]
    else:
        order = [x for x in order if x in d[group_col].unique() and x != "Other"]

    if orientation == "horizontal":
        simple_boxplot_horizontal(
            ax=ax,
            data=d,
            group_col=group_col,
            value_col=y_col,
            order=order
        )

        ax.set_title(
            title_text,
            pad=4,
            fontweight="normal"
        )
        ax.set_ylabel("Cancer type")
        ax.set_xlabel(get_value_label(y_col))

        remove_top_right_spines(ax)
        ax.grid(True, axis="x", linestyle="--", linewidth=0.45, alpha=0.20)
        ax.grid(False, axis="y")

    else:
        simple_boxplot_vertical(
            ax=ax,
            data=d,
            group_col=group_col,
            value_col=y_col,
            order=order
        )

        ax.set_title(
            title_text,
            pad=4,
            fontweight="normal"
        )
        ax.set_xlabel("Cancer type")
        ax.set_ylabel(get_value_label(y_col))

        remove_top_right_spines(ax)
        ax.grid(True, axis="y", linestyle="--", linewidth=0.45, alpha=0.20)
        ax.grid(False, axis="x")




df_panel = df_for_all.copy()

if "Size_nm" not in df_panel.columns and "Size" in df_panel.columns:
    s = pd.to_numeric(df_panel["Size"], errors="coerce")

    if s.notna().mean() > 0.5:
        valid = s.dropna()
        if len(valid) > 0:
            ok_lo = (valid >= -1).mean()
            ok_hi = (valid <= 6).mean()

            if (ok_lo > 0.8) and (ok_hi > 0.8):
                df_panel["Size_nm"] = 10 ** s
            else:
                df_panel["Size_nm"] = s

if "Zeta Potential" in df_panel.columns:
    ZETA_COL = "Zeta Potential"
elif "Zeta potential" in df_panel.columns:
    ZETA_COL = "Zeta potential"
else:
    ZETA_COL = None




def make_figure2(box_orientation="vertical"):
    if box_orientation not in ["vertical", "horizontal"]:
        raise ValueError("box_orientation text 'vertical' text 'horizontal'")

    if box_orientation == "horizontal":
        fig_h = 10.8
        sub_hspace = 0.62
        e_title_offset = 0.020
        out_suffix = "horizontal_boxplot"
    else:
        fig_h = 10.8
        sub_hspace = 0.62
        e_title_offset = 0.020
        out_suffix = "vertical_boxplot"

    fig = plt.figure(figsize=(12.6, fig_h))

    outer = GridSpec(
        nrows=4,
        ncols=12,
        figure=fig,
        height_ratios=[1.05, 1.15, 1.15, 1.15],
        hspace=0.72,
        wspace=1.05
    )


    ax_a = fig.add_subplot(outer[0, 0:4])
    ax_b = fig.add_subplot(outer[0, 4:6])
    ax_c = fig.add_subplot(outer[0, 7:11])


    sub_lower = outer[1:4, 0:12].subgridspec(
        nrows=3,
        ncols=6,
        hspace=sub_hspace,
        wspace=1.12
    )


    d_axes_cat = [
        fig.add_subplot(sub_lower[0, 0:2]),
        fig.add_subplot(sub_lower[0, 2:4]),
        fig.add_subplot(sub_lower[0, 4:6]),
        fig.add_subplot(sub_lower[1, 0:2]),
        fig.add_subplot(sub_lower[1, 2:4]),
        fig.add_subplot(sub_lower[1, 4:6]),
    ]


    e_axes_cont = [
        fig.add_subplot(sub_lower[2, 0:3]),
        fig.add_subplot(sub_lower[2, 3:6]),
    ]




    train_values = df_plot.loc[df_plot["split"] == "train", "log_DE_tumor"].dropna()
    test_values = df_plot.loc[df_plot["split"] == "test", "log_DE_tumor"].dropna()

    all_values = df_plot["log_DE_tumor"].dropna()
    bins = np.linspace(all_values.min(), all_values.max(), 22)

    ax_a.hist(
        train_values,
        bins=bins,
        density=True,
        alpha=0.75,
        color=COLOR_TRAIN,
        edgecolor="none",
        label="train"
    )

    ax_a.hist(
        test_values,
        bins=bins,
        density=True,
        alpha=0.70,
        color=COLOR_TEST,
        edgecolor="none",
        label="test"
    )

    ax_a.axvline(
        thr_log,
        color="black",
        linestyle="--",
        linewidth=1.2
    )

    ymax = ax_a.get_ylim()[1]
    ax_a.text(
        thr_log + 0.05,
        ymax * 0.92,
        Q75_TEXT,
        rotation=0,
        va="top",
        ha="left",
        fontsize=7.6,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.65, pad=0.5)
    )

    ax_a.set_xlabel(r"log$_{10}$(DE$_{tumor}$ + 0.01)")
    ax_a.set_ylabel("Density")

    ax_a.legend(
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(0.12, 0.98),
        fontsize=8,
        handlelength=1.4,
        borderaxespad=0.0
    )

    add_panel_label(ax_a, "a", x=-0.16, y=1.08)
    remove_top_right_spines(ax_a)




    summary = (
        df_plot
        .groupby(["split", "y"])
        .size()
        .unstack(fill_value=0)
        .reindex(["train", "test"])
    )

    for cls in [0, 1]:
        if cls not in summary.columns:
            summary[cls] = 0

    summary = summary[[0, 1]]
    fraction = summary.div(summary.sum(axis=1), axis=0)

    x = np.arange(len(summary.index))
    bar_width = 0.65

    ax_b.bar(
        x,
        fraction[0],
        width=bar_width,
        color=COLOR_NONHIGH,
        edgecolor="white",
        linewidth=0.8,
        label="Non-high"
    )

    ax_b.bar(
        x,
        fraction[1],
        width=bar_width,
        bottom=fraction[0],
        color=COLOR_HIGH,
        edgecolor="white",
        linewidth=0.8,
        label="High"
    )

    for i, split_name in enumerate(summary.index):
        nonhigh_count = int(summary.loc[split_name, 0])
        high_count = int(summary.loc[split_name, 1])

        nonhigh_frac = float(fraction.loc[split_name, 0])
        high_frac = float(fraction.loc[split_name, 1])

        if nonhigh_count > 0:
            ax_b.text(
                x[i],
                nonhigh_frac / 2,
                str(nonhigh_count),
                ha="center",
                va="center",
                fontsize=8,
                color="#1F1F1F"
            )

        if high_count > 0:
            ax_b.text(
                x[i],
                nonhigh_frac + high_frac / 2,
                str(high_count),
                ha="center",
                va="center",
                fontsize=8,
                color="white"
            )

    ax_b.set_xticks(x)
    ax_b.set_xticklabels(summary.index)
    ax_b.set_ylabel("Fraction")
    ax_b.set_ylim(0, 1.05)
    ax_b.set_yticks([0, 0.5, 1.0])

    handles, labels = ax_b.get_legend_handles_labels()
    label_to_handle = dict(zip(labels, handles))

    ax_b.legend(
        [label_to_handle["High"], label_to_handle["Non-high"]],
        ["High", "Non-high"],
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.55, 1.28),
        ncol=2,
        handlelength=1.0,
        columnspacing=0.8
    )

    add_panel_label(ax_b, "b", x=-0.48, y=1.12)
    remove_top_right_spines(ax_b)




    missing_df = pd.DataFrame({
        "feature": base_features,
        "missing_n": [int(df_for_all[col].isna().sum()) for col in base_features],
        "missing_pct": [float(df_for_all[col].isna().mean() * 100) for col in base_features]
    })

    missing_df = missing_df.sort_values("missing_pct", ascending=False).reset_index(drop=True)
    missing_df = missing_df[missing_df["missing_pct"] > 0].reset_index(drop=True)
    missing_df["feature_display"] = missing_df["feature"].map(
        lambda x_: MISSING_LABEL_MAP.get(x_, x_)
    )

    if missing_df.empty:
        ax_c.axis("off")
        ax_c.text(0.5, 0.5, "No missing values", ha="center", va="center")
    else:
        y_pos = np.arange(len(missing_df))

        ax_c.barh(
            y_pos,
            missing_df["missing_pct"],
            color=COLOR_MISSING,
            height=0.68
        )

        ax_c.set_yticks(y_pos)
        ax_c.set_yticklabels(missing_df["feature_display"])
        ax_c.invert_yaxis()
        ax_c.set_xlabel("Missing (%)")
        ax_c.set_title(
            "Feature missingness",
            pad=5,
            fontweight="normal"
        )

        max_pct = missing_df["missing_pct"].max()

        for i, row in missing_df.iterrows():
            pct = row["missing_pct"]
            n = int(row["missing_n"])

            ax_c.text(
                pct + max_pct * 0.04,
                i,
                f"{pct:.1f}% ({n})",
                va="center",
                ha="left",
                fontsize=8
            )

        ax_c.set_xlim(0, max_pct * 1.65)

    add_panel_label(ax_c, "c", x=-0.20, y=1.12)
    remove_top_right_spines(ax_c)




    cat_vars = ["Type", "MAT", "Shape", "CT", "TM", "TS"]
    cat_vars = [c for c in cat_vars if c in df_panel.columns]

    CAT_TOPK = 8
    CT_TOPK = 8
    group_col = "CT"

    for ax, col in zip(d_axes_cat, cat_vars):
        plot_categorical_bar(ax, df_panel, col, topk=CAT_TOPK)

    for ax in d_axes_cat[len(cat_vars):]:
        ax.axis("off")

    add_panel_label(d_axes_cat[0], "d", x=-0.18, y=1.20)
    add_group_title(
        fig,
        d_axes_cat,
        "Categorical feature landscape",
        y_offset=0.028,
        fontsize=11
    )




    continuous_vars = []

    if "Size_nm" in df_panel.columns and "CT" in df_panel.columns:
        continuous_vars.append(("Size_nm", "Size by cancer type"))

    if ZETA_COL is not None and "CT" in df_panel.columns:
        continuous_vars.append((ZETA_COL, "Zeta Potential by cancer type"))

    if group_col in df_panel.columns:
        ct_order = clean_cat_series(df_panel[group_col]).value_counts().head(CT_TOPK).index.tolist()
        ct_order = [x_ for x_ in ct_order if x_ != "Other"]
    else:
        ct_order = None

    for ax, (y_col, title_text) in zip(e_axes_cont, continuous_vars):
        plot_box_by_group(
            ax=ax,
            df_obj=df_panel,
            y_col=y_col,
            title_text=title_text,
            group_col=group_col,
            order=ct_order,
            topk=CT_TOPK,
            orientation=box_orientation
        )

    for ax in e_axes_cont[len(continuous_vars):]:
        ax.axis("off")


    if box_orientation == "horizontal" and len(e_axes_cont) > 1:
        for ax in e_axes_cont:
            ax.set_ylabel("Cancer type")
            ax.tick_params(axis="y", left=True, labelleft=True)
            ax.spines["left"].set_visible(True)


    d_bbox = d_axes_cat[0].get_position()
    e_bbox = e_axes_cont[0].get_position()

    e_label_x = d_bbox.x0 + (-0.18) * d_bbox.width
    e_label_y = e_bbox.y0 + 1.20 * e_bbox.height

    fig.text(
        e_label_x,
        e_label_y,
        "(e)",
        fontsize=12,
        fontweight="bold",
        ha="left",
        va="top"
    )

    add_group_title(
        fig,
        e_axes_cont,
        "Numeric feature distributions",
        y_offset=e_title_offset,
        fontsize=11
    )




    source_dir = os.path.join(PANEL_DIR, "source_data")
    os.makedirs(source_dir, exist_ok=True)

    df_plot[[col for col in ["split", TARGET_DE, "log_DE_tumor", "y"] if col in df_plot.columns]].to_csv(
        os.path.join(source_dir, "fig2a_endpoint_distribution_source_data.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    summary.to_csv(
        os.path.join(source_dir, "fig2b_class_fraction_counts_source_data.csv"),
        encoding="utf-8-sig"
    )

    missing_df.to_csv(
        os.path.join(source_dir, "fig2c_missingness_source_data.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    cat_source_rows = []
    for col in cat_vars:
        s = clean_cat_series(df_panel[col])
        s = apply_display_abbreviation(s, col)
        vc = get_top_counts(s, topk=CAT_TOPK)
        pct = vc / vc.sum() * 100

        for level in vc.index:
            cat_source_rows.append({
                "feature": col,
                "display_level": level,
                "count": int(vc.loc[level]),
                "percentage": float(pct.loc[level])
            })

    pd.DataFrame(cat_source_rows).to_csv(
        os.path.join(source_dir, "fig2d_categorical_feature_landscape_source_data.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    cont_source_cols = []

    if "CT" in df_panel.columns:
        cont_source_cols.append("CT")

    for y_col, _ in continuous_vars:
        if y_col in df_panel.columns:
            cont_source_cols.append(y_col)

    cont_source_cols = list(dict.fromkeys(cont_source_cols))

    if len(cont_source_cols) > 0:
        df_panel[cont_source_cols].to_csv(
            os.path.join(source_dir, f"fig2e_continuous_feature_by_cancer_type_{out_suffix}_source_data.csv"),
            index=False,
            encoding="utf-8-sig"
        )




    out_base = os.path.join(
        PANEL_DIR,
        f"Figure2_dataset_landscape_{out_suffix}_Q075_1773_e_aligned"
    )

    fig.savefig(
        out_base + ".png",
        dpi=900,
        bbox_inches="tight",
        facecolor="white"
    )

    fig.savefig(
        out_base + ".pdf",
        bbox_inches="tight",
        facecolor="white"
    )

    fig.savefig(
        out_base + ".svg",
        bbox_inches="tight",
        facecolor="white"
    )

    plt.show()
    plt.close(fig)

    print(f"Figure 2 {box_orientation} version text: ")
    print(out_base + ".png")
    print(out_base + ".pdf")
    print(out_base + ".svg")







make_figure2(box_orientation="horizontal")


















INDIV_DIR = os.path.join(SAVE_DIR, "Figure2_individual_panels")
os.makedirs(INDIV_DIR, exist_ok=True)

EXPORT_DPI = 900



SHOW_PANEL_LABEL = False


def save_individual_figure(fig, base_name):
    """
    text PNG/PDF/SVG
    """
    fig.savefig(
        os.path.join(INDIV_DIR, f"{base_name}.png"),
        dpi=EXPORT_DPI,
        bbox_inches="tight",
        facecolor="white"
    )

    fig.savefig(
        os.path.join(INDIV_DIR, f"{base_name}.pdf"),
        bbox_inches="tight",
        facecolor="white"
    )

    fig.savefig(
        os.path.join(INDIV_DIR, f"{base_name}.svg"),
        bbox_inches="tight",
        facecolor="white"
    )

    plt.show()
    plt.close(fig)





fig, ax = plt.subplots(figsize=(3.6, 2.55))

train_values = df_plot.loc[df_plot["split"] == "train", "log_DE_tumor"].dropna()
test_values = df_plot.loc[df_plot["split"] == "test", "log_DE_tumor"].dropna()

all_values = df_plot["log_DE_tumor"].dropna()
bins = np.linspace(all_values.min(), all_values.max(), 22)

ax.hist(
    train_values,
    bins=bins,
    density=True,
    alpha=0.75,
    color=COLOR_TRAIN,
    edgecolor="none",
    label="train"
)

ax.hist(
    test_values,
    bins=bins,
    density=True,
    alpha=0.70,
    color=COLOR_TEST,
    edgecolor="none",
    label="test"
)

ax.axvline(
    thr_log,
    color="black",
    linestyle="--",
    linewidth=1.2
)

ymax = ax.get_ylim()[1]
ax.text(
    thr_log + 0.05,
    ymax * 0.92,
    r"Q$_{0.75}$ = 1.773",
    rotation=0,
    va="top",
    ha="left",
    fontsize=7.6,
    bbox=dict(facecolor="white", edgecolor="none", alpha=0.65, pad=0.5)
)

ax.set_xlabel(r"log$_{10}$(DE$_{tumor}$ + 0.01)")
ax.set_ylabel("Density")

ax.legend(
    frameon=False,
    loc="upper left",
    bbox_to_anchor=(0.12, 0.98),
    fontsize=8,
    handlelength=1.4,
    borderaxespad=0.0
)

if SHOW_PANEL_LABEL:
    add_panel_label(ax, "a", x=-0.16, y=1.08)

remove_top_right_spines(ax)

save_individual_figure(fig, "fig2a_endpoint_distribution")





fig, ax = plt.subplots(figsize=(2.05, 2.55))

summary = (
    df_plot
    .groupby(["split", "y"])
    .size()
    .unstack(fill_value=0)
    .reindex(["train", "test"])
)

for cls in [0, 1]:
    if cls not in summary.columns:
        summary[cls] = 0

summary = summary[[0, 1]]
fraction = summary.div(summary.sum(axis=1), axis=0)

x = np.arange(len(summary.index))
bar_width = 0.65

ax.bar(
    x,
    fraction[0],
    width=bar_width,
    color=COLOR_NONHIGH,
    edgecolor="white",
    linewidth=0.8,
    label="Non-high"
)

ax.bar(
    x,
    fraction[1],
    width=bar_width,
    bottom=fraction[0],
    color=COLOR_HIGH,
    edgecolor="white",
    linewidth=0.8,
    label="High"
)

for i, split_name in enumerate(summary.index):
    nonhigh_count = int(summary.loc[split_name, 0])
    high_count = int(summary.loc[split_name, 1])

    nonhigh_frac = float(fraction.loc[split_name, 0])
    high_frac = float(fraction.loc[split_name, 1])

    if nonhigh_count > 0:
        ax.text(
            x[i],
            nonhigh_frac / 2,
            str(nonhigh_count),
            ha="center",
            va="center",
            fontsize=8,
            color="#1F1F1F"
        )

    if high_count > 0:
        ax.text(
            x[i],
            nonhigh_frac + high_frac / 2,
            str(high_count),
            ha="center",
            va="center",
            fontsize=8,
            color="white"
        )

ax.set_xticks(x)
ax.set_xticklabels(summary.index)
ax.set_ylabel("Fraction")
ax.set_ylim(0, 1.05)
ax.set_yticks([0, 0.5, 1.0])

handles, labels = ax.get_legend_handles_labels()
label_to_handle = dict(zip(labels, handles))

ax.legend(
    [label_to_handle["High"], label_to_handle["Non-high"]],
    ["High", "Non-high"],
    frameon=False,
    loc="upper center",
    bbox_to_anchor=(0.55, 1.28),
    ncol=2,
    handlelength=1.0,
    columnspacing=0.8
)

if SHOW_PANEL_LABEL:
    add_panel_label(ax, "b", x=-0.42, y=1.12)

remove_top_right_spines(ax)

save_individual_figure(fig, "fig2b_class_fraction")





fig, ax = plt.subplots(figsize=(3.9, 2.55))

missing_df = pd.DataFrame({
    "feature": base_features,
    "missing_n": [int(df_for_all[col].isna().sum()) for col in base_features],
    "missing_pct": [float(df_for_all[col].isna().mean() * 100) for col in base_features]
})

missing_df = missing_df.sort_values("missing_pct", ascending=False).reset_index(drop=True)
missing_df = missing_df[missing_df["missing_pct"] > 0].reset_index(drop=True)
missing_df["feature_display"] = missing_df["feature"].map(
    lambda x_: MISSING_LABEL_MAP.get(x_, x_)
)

if missing_df.empty:
    ax.axis("off")
    ax.text(0.5, 0.5, "No missing values", ha="center", va="center")
else:
    y_pos = np.arange(len(missing_df))

    ax.barh(
        y_pos,
        missing_df["missing_pct"],
        color=COLOR_MISSING,
        height=0.68
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(missing_df["feature_display"])
    ax.invert_yaxis()

    ax.set_xlabel("Missing (%)")
    ax.set_title(
        "Feature missingness",
        pad=5,
        fontweight="normal"
    )

    max_pct = missing_df["missing_pct"].max()

    for i, row in missing_df.iterrows():
        pct = row["missing_pct"]
        n = int(row["missing_n"])

        ax.text(
            pct + max_pct * 0.04,
            i,
            f"{pct:.1f}% ({n})",
            va="center",
            ha="left",
            fontsize=8
        )

    ax.set_xlim(0, max_pct * 1.65)

if SHOW_PANEL_LABEL:
    add_panel_label(ax, "c", x=-0.18, y=1.12)

remove_top_right_spines(ax)

save_individual_figure(fig, "fig2c_feature_missingness")






cat_panel_map = {
    "Type": "fig2d_type",
    "MAT": "fig2d_material",
    "Shape": "fig2d_shape",
    "CT": "fig2d_cancer_type",
    "TM": "fig2d_tumor_model",
    "TS": "fig2d_targeting_strategy"
}

CAT_TOPK = 8

for col, base_name in cat_panel_map.items():
    if col not in df_panel.columns:
        continue


    if col == "MAT":
        figsize = (3.6, 2.75)
    elif col == "CT":
        figsize = (3.6, 2.75)
    else:
        figsize = (3.35, 2.45)

    fig, ax = plt.subplots(figsize=figsize)

    plot_categorical_bar(
        ax=ax,
        df_obj=df_panel,
        col=col,
        topk=CAT_TOPK
    )

    save_individual_figure(fig, base_name)






CT_TOPK = 8
group_col = "CT"

if group_col in df_panel.columns:
    ct_order = clean_cat_series(df_panel[group_col]).value_counts().head(CT_TOPK).index.tolist()
    ct_order = [x_ for x_ in ct_order if x_ != "Other"]
else:
    ct_order = None



if "Size_nm" in df_panel.columns and "CT" in df_panel.columns:
    fig, ax = plt.subplots(figsize=(5.2, 2.75))

    plot_box_by_group(
        ax=ax,
        df_obj=df_panel,
        y_col="Size_nm",
        title_text="Size by cancer type",
        group_col="CT",
        order=ct_order,
        topk=CT_TOPK,
        orientation="horizontal"
    )


    ax.set_ylabel("Cancer type")
    ax.tick_params(axis="y", left=True, labelleft=True)
    ax.spines["left"].set_visible(True)

    save_individual_figure(fig, "fig2e_size_by_cancer_type")



if ZETA_COL is not None and "CT" in df_panel.columns:
    fig, ax = plt.subplots(figsize=(5.2, 2.75))

    plot_box_by_group(
        ax=ax,
        df_obj=df_panel,
        y_col=ZETA_COL,
        title_text="Zeta Potential by cancer type",
        group_col="CT",
        order=ct_order,
        topk=CT_TOPK,
        orientation="horizontal"
    )


    ax.set_ylabel("Cancer type")
    ax.tick_params(axis="y", left=True, labelleft=True)
    ax.spines["left"].set_visible(True)

    save_individual_figure(fig, "fig2e_zeta_potential_by_cancer_type")





source_dir = os.path.join(INDIV_DIR, "source_data")
os.makedirs(source_dir, exist_ok=True)

df_plot[[col for col in ["split", TARGET_DE, "log_DE_tumor", "y"] if col in df_plot.columns]].to_csv(
    os.path.join(source_dir, "fig2a_endpoint_distribution_source_data.csv"),
    index=False,
    encoding="utf-8-sig"
)

summary.to_csv(
    os.path.join(source_dir, "fig2b_class_fraction_counts_source_data.csv"),
    encoding="utf-8-sig"
)

missing_df.to_csv(
    os.path.join(source_dir, "fig2c_missingness_source_data.csv"),
    index=False,
    encoding="utf-8-sig"
)

cat_source_rows = []

for col in cat_panel_map.keys():
    if col not in df_panel.columns:
        continue

    s = clean_cat_series(df_panel[col])
    s = apply_display_abbreviation(s, col)
    vc = get_top_counts(s, topk=CAT_TOPK)
    pct = vc / vc.sum() * 100

    for level in vc.index:
        cat_source_rows.append({
            "feature": col,
            "display_level": level,
            "count": int(vc.loc[level]),
            "percentage": float(pct.loc[level])
        })

pd.DataFrame(cat_source_rows).to_csv(
    os.path.join(source_dir, "fig2d_categorical_feature_composition_source_data.csv"),
    index=False,
    encoding="utf-8-sig"
)

cont_source_cols = []

if "CT" in df_panel.columns:
    cont_source_cols.append("CT")

if "Size_nm" in df_panel.columns:
    cont_source_cols.append("Size_nm")

if ZETA_COL is not None:
    cont_source_cols.append(ZETA_COL)

cont_source_cols = list(dict.fromkeys(cont_source_cols))

if len(cont_source_cols) > 0:
    df_panel[cont_source_cols].to_csv(
        os.path.join(source_dir, "fig2e_numeric_feature_distributions_source_data.csv"),
        index=False,
        encoding="utf-8-sig"
    )

print("Figure 2 individual panels text: ")
print(INDIV_DIR)
print("Source data text: ")
print(source_dir)






from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt




if "SAVE_DIR" not in globals():
    SAVE_DIR = str(RESULTS_DIR)

RUN_DIR = Path(SAVE_DIR).resolve()

FIG3_DIR = RUN_DIR / "output" / "Figure3" / "optimized_single_panels"
SRC_DIR = RUN_DIR / "source_data" / "Figure3" / "optimized_single_panels"

FIG3_DIR.mkdir(parents=True, exist_ok=True)
SRC_DIR.mkdir(parents=True, exist_ok=True)

print("RUN_DIR =", RUN_DIR)
print("Output =", FIG3_DIR)




mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 8,
    "axes.titlesize": 8.8,
    "axes.labelsize": 8.2,
    "xtick.labelsize": 7.4,
    "ytick.labelsize": 7.6,
    "axes.linewidth": 0.75,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "legend.frameon": False,
    "axes.unicode_minus": False,
})




COLOR_PRIMARY = "#3F6FA6"
COLOR_NEUTRAL = "#BEC6CF"
COLOR_ERROR = "#A8B0B8"
COLOR_EDGE = "#66717D"
COLOR_TEXT = "#23272B"




DISPLAY_NAMES = {
    "best_cat": "CatBoost",
    "best_stacking": "Stacking",
    "best_voting": "Voting",
    "best_svm": "SVM-RBF",
    "best_xgb": "XGBoost",
    "best_lgbm": "LightGBM",
    "best_knn": "KNN",
    "best_lr": "LR",
    "best_dt": "DT",
    "best_dnn": "DNN",
}

ALT_NAMES = {
    "catboost": "CatBoost",
    "cat": "CatBoost",
    "stacking": "Stacking",
    "voting": "Voting",
    "svm-rbf": "SVM-RBF",
    "svm rbf": "SVM-RBF",
    "svm_rbf": "SVM-RBF",
    "xgboost": "XGBoost",
    "xgb": "XGBoost",
    "lightgbm": "LightGBM",
    "lgbm": "LightGBM",
    "knn": "KNN",
    "logistic regression": "LR",
    "logistic_regression": "LR",
    "logistic reg.": "LR",
    "logistic reg": "LR",
    "lr": "LR",
    "decision tree": "DT",
    "decision_tree": "DT",
    "dt": "DT",
    "dnn": "DNN",
}

MODEL_ORDER = [
    "CatBoost",
    "Stacking",
    "Voting",
    "SVM-RBF",
    "XGBoost",
    "LightGBM",
    "KNN",
    "LR",
    "DT",
    "DNN",
]

MODEL_ORDER_MAP = {name: i for i, name in enumerate(MODEL_ORDER)}

def display_name(model_name: str) -> str:
    model_name = str(model_name)

    if model_name in DISPLAY_NAMES:
        return DISPLAY_NAMES[model_name]

    key = (
        model_name
        .lower()
        .replace("best_", "")
        .replace("_", " ")
        .strip()
    )

    if key in ALT_NAMES:
        return ALT_NAMES[key]

    return model_name.replace("best_", "").replace("_", " ").strip()

def remove_top_right_spines(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)




cv_path = RUN_DIR / "model_summary" / "model_metrics_summary_cv_raw.csv"

if not cv_path.exists():
    raise FileNotFoundError(
        f"text: {cv_path}\n"
        "text, text model_summary/model_metrics_summary_cv_raw.csv "
    )

df = pd.read_csv(cv_path)

required_cols = ["model_name", "PR-AUC_mean", "PR-AUC_std"]
missing_cols = [c for c in required_cols if c not in df.columns]

if missing_cols:
    raise ValueError(f"model_metrics_summary_cv_raw.csv text: {missing_cols}")


rf_like_names = {
    "best_rf",
    "rf",
    "random_forest",
    "random forest",
    "best_random_forest"
}

df = df[
    ~df["model_name"]
    .astype(str)
    .str.lower()
    .isin(rf_like_names)
].copy()

df["display_name"] = df["model_name"].map(display_name)


df_plot = df[df["display_name"].isin(MODEL_ORDER)].copy()
df_plot["model_order"] = df_plot["display_name"].map(MODEL_ORDER_MAP)

df_plot = (
    df_plot
    .sort_values("model_order", ascending=True)
    .reset_index(drop=True)
)

if df_plot.empty:
    raise ValueError(
        "df_plot text text model_name text DISPLAY_NAMES / ALT_NAMES text "
    )


df_plot.to_csv(
    SRC_DIR / "Figure3a_classifier_landscape_source_data.csv",
    index=False,
    encoding="utf-8-sig"
)




fig, ax = plt.subplots(figsize=(3.25, 3.05), dpi=300)

y = np.arange(len(df_plot))
is_primary = df_plot["display_name"].eq("CatBoost")


ax.errorbar(
    df_plot["PR-AUC_mean"],
    y,
    xerr=df_plot["PR-AUC_std"],
    fmt="none",
    ecolor=COLOR_ERROR,
    elinewidth=0.75,
    capsize=2.0,
    capthick=0.75,
    alpha=0.90,
    zorder=1,
)


ax.scatter(
    df_plot["PR-AUC_mean"],
    y,
    s=np.where(is_primary, 38, 22),
    c=np.where(is_primary, COLOR_PRIMARY, COLOR_NEUTRAL),
    edgecolors=np.where(is_primary, COLOR_PRIMARY, COLOR_EDGE),
    linewidths=0.55,
    zorder=2,
)


ax.set_yticks(y)
ax.set_yticklabels(df_plot["display_name"])
ax.invert_yaxis()
ax.tick_params(axis="y", pad=2)


xmin = float((df_plot["PR-AUC_mean"] - df_plot["PR-AUC_std"]).min())
xmax = float((df_plot["PR-AUC_mean"] + df_plot["PR-AUC_std"]).max())

x_left = max(0.20, xmin - 0.025)
x_right = min(1.00, xmax + 0.025)

ax.set_xlim(x_left, x_right)


ax.set_xticks(np.arange(0.3, 0.75, 0.1))


ax.set_xlabel("Cross-validation PR-AUC")


ax.set_title("")


ax.grid(False)

remove_top_right_spines(ax)


ax.spines["left"].set_color(COLOR_TEXT)
ax.spines["bottom"].set_color(COLOR_TEXT)


plt.tight_layout(pad=0.8)




out_base = FIG3_DIR / "Figure3a_classifier_landscape_no_title"

fig.savefig(
    out_base.with_suffix(".png"),
    dpi=900,
    bbox_inches="tight",
    facecolor="white"
)

fig.savefig(
    out_base.with_suffix(".pdf"),
    bbox_inches="tight",
    facecolor="white"
)

fig.savefig(
    out_base.with_suffix(".svg"),
    bbox_inches="tight",
    facecolor="white"
)

fig.savefig(
    out_base.with_suffix(".tiff"),
    dpi=900,
    bbox_inches="tight",
    facecolor="white"
)

plt.show()
plt.close(fig)

print("Figure 3a exported:")
print(out_base.with_suffix(".png"))
print(out_base.with_suffix(".pdf"))
print(out_base.with_suffix(".svg"))
print(out_base.with_suffix(".tiff"))
print("Source data exported:")
print(SRC_DIR / "Figure3a_classifier_landscape_source_data.csv")








from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap




if "SAVE_DIR" not in globals():
    SAVE_DIR = str(RESULTS_DIR)

RUN_DIR = Path(SAVE_DIR).resolve()

FIG3_DIR = RUN_DIR / "output" / "Figure3" / "optimized_single_panels"
SRC_DIR = RUN_DIR / "source_data" / "Figure3" / "optimized_single_panels"

FIG3_DIR.mkdir(parents=True, exist_ok=True)
SRC_DIR.mkdir(parents=True, exist_ok=True)

print("RUN_DIR =", RUN_DIR)
print("Output =", FIG3_DIR)




mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 7.0,
    "axes.titlesize": 7.0,
    "axes.labelsize": 7.0,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.7,
    "axes.linewidth": 0.55,
    "xtick.major.width": 0.55,
    "ytick.major.width": 0.55,
    "legend.frameon": False,
    "axes.unicode_minus": False,
})




COLOR_TEXT = "#23272B"
COLOR_BORDER = "#FFFFFF"


CMAP_BLUE_SOFT = LinearSegmentedColormap.from_list(
    "soft_blues_v2",
    ["#F7FAFC", "#E4F0F7", "#B9D7EA", "#74ADD1", "#2E78B7"]
)

DISPLAY_NAMES = {
    "best_cat": "CatBoost",
    "best_stacking": "Stacking",
    "best_voting": "Voting",
    "best_svm": "SVM-RBF",
    "best_xgb": "XGBoost",
    "best_lgbm": "LightGBM",
    "best_knn": "KNN",
    "best_lr": "LR",
    "best_dt": "DT",
    "best_dnn": "DNN",
}

ALT_NAMES = {
    "catboost": "CatBoost",
    "cat": "CatBoost",
    "stacking": "Stacking",
    "voting": "Voting",
    "svm-rbf": "SVM-RBF",
    "svm rbf": "SVM-RBF",
    "svm_rbf": "SVM-RBF",
    "xgboost": "XGBoost",
    "xgb": "XGBoost",
    "lightgbm": "LightGBM",
    "lgbm": "LightGBM",
    "knn": "KNN",
    "logistic regression": "LR",
    "logistic_regression": "LR",
    "logistic reg.": "LR",
    "logistic reg": "LR",
    "lr": "LR",
    "decision tree": "DT",
    "decision_tree": "DT",
    "dt": "DT",
    "dnn": "DNN",
}

METRICS = ["PR-AUC", "ROC-AUC", "F1", "Precision", "Recall"]

METRIC_LABELS = {
    "PR-AUC": "PR-AUC",
    "ROC-AUC": "ROC-AUC",
    "F1": "F1",
    "Precision": "Precision",
    "Recall": "Recall",
}

def display_name(model_name: str) -> str:
    model_name = str(model_name)

    if model_name in DISPLAY_NAMES:
        return DISPLAY_NAMES[model_name]

    key = (
        model_name
        .lower()
        .replace("best_", "")
        .replace("_", " ")
        .strip()
    )

    if key in ALT_NAMES:
        return ALT_NAMES[key]

    return model_name.replace("best_", "").replace("_", " ").strip()




cv_path = RUN_DIR / "model_summary" / "model_metrics_summary_cv_raw.csv"

if not cv_path.exists():
    raise FileNotFoundError(
        f"text: {cv_path}\n"
        "text, text model_summary/model_metrics_summary_cv_raw.csv "
    )

df = pd.read_csv(cv_path)

required_cols = ["model_name"] + [f"{m}_mean" for m in METRICS]
missing_cols = [c for c in required_cols if c not in df.columns]

if missing_cols:
    raise ValueError(f"model_metrics_summary_cv_raw.csv text: {missing_cols}")


rf_like_names = {
    "best_rf",
    "rf",
    "random_forest",
    "random forest",
    "best_random_forest",
}

df = df[
    ~df["model_name"]
    .astype(str)
    .str.lower()
    .isin(rf_like_names)
].copy()

df["display_name"] = df["model_name"].map(display_name)


df_plot = (
    df
    .sort_values("PR-AUC_mean", ascending=False)
    .head(6)
    .reset_index(drop=True)
)


source_cols = ["model_name", "display_name"] + [f"{m}_mean" for m in METRICS]
source_path = SRC_DIR / "Figure3a_top_model_cv_metrics_source_data.csv"

df_plot[source_cols].to_csv(
    source_path,
    index=False,
    encoding="utf-8-sig"
)




fig, ax = plt.subplots(figsize=(4.05, 2.05), dpi=300)

mat = df_plot[[f"{m}_mean" for m in METRICS]].to_numpy()


vmin = 0.32
vmax = 0.86

im = ax.imshow(
    mat,
    cmap=CMAP_BLUE_SOFT,
    vmin=vmin,
    vmax=vmax,
    aspect="auto",
    interpolation="nearest"
)

n_rows, n_cols = mat.shape


ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)

ax.grid(
    which="minor",
    color=COLOR_BORDER,
    linewidth=0.45
)

ax.tick_params(
    which="minor",
    bottom=False,
    left=False
)


for i in range(n_rows):
    for j in range(n_cols):
        val = mat[i, j]
        txt_color = "white" if val >= 0.70 else COLOR_TEXT

        ax.text(
            j,
            i,
            f"{val:.2f}",
            ha="center",
            va="center",
            fontsize=6.0,
            color=txt_color
        )


ax.set_xticks(np.arange(n_cols))
ax.set_xticklabels(
    [METRIC_LABELS[m] for m in METRICS],
    rotation=0,
    ha="center"
)

ax.set_yticks(np.arange(n_rows))
ax.set_yticklabels(df_plot["display_name"])


ax.set_xlabel("")
ax.set_ylabel("")


ax.tick_params(axis="both", length=0)


for spine in ax.spines.values():
    spine.set_visible(False)






plt.tight_layout(pad=0.35)




out_base = FIG3_DIR / "Figure3a_top_model_cv_metrics_no_inner_title"

fig.savefig(
    out_base.with_suffix(".png"),
    dpi=900,
    bbox_inches="tight",
    facecolor="white"
)

fig.savefig(
    out_base.with_suffix(".pdf"),
    bbox_inches="tight",
    facecolor="white"
)

fig.savefig(
    out_base.with_suffix(".svg"),
    bbox_inches="tight",
    facecolor="white"
)

fig.savefig(
    out_base.with_suffix(".tiff"),
    dpi=900,
    bbox_inches="tight",
    facecolor="white"
)

plt.show()
plt.close(fig)

print("Figure 3a exported:")
print(out_base.with_suffix(".png"))
print(out_base.with_suffix(".pdf"))
print(out_base.with_suffix(".svg"))
print(out_base.with_suffix(".tiff"))
print("Source data exported:")
print(source_path)






from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt




if "SAVE_DIR" not in globals():
    SAVE_DIR = str(RESULTS_DIR)

RUN_DIR = Path(SAVE_DIR).resolve()

FIG3_DIR = RUN_DIR / "output" / "Figure3" / "optimized_single_panels"
SRC_DIR = RUN_DIR / "source_data" / "Figure3" / "optimized_single_panels"

FIG3_DIR.mkdir(parents=True, exist_ok=True)
SRC_DIR.mkdir(parents=True, exist_ok=True)

print("RUN_DIR =", RUN_DIR)
print("Output =", FIG3_DIR)




mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,

    "font.size": 6.8,
    "axes.titlesize": 6.4,
    "axes.labelsize": 6.5,
    "xtick.labelsize": 6.2,
    "ytick.labelsize": 6.2,

    "axes.linewidth": 0.52,
    "xtick.major.width": 0.52,
    "ytick.major.width": 0.52,
    "legend.frameon": False,
    "axes.unicode_minus": False,
})




COLOR_TEXT = "#23272B"
COLOR_GRID = "#F1F3F6"

COLOR_CAT = "#6F95BD"
COLOR_CAT_LIGHT = "#EAF2FA"

COLOR_OTHER_POINT = "#9AA4AF"
COLOR_OTHER_BOX = "#F2F4F7"
COLOR_OTHER_EDGE = "#7F8A96"

COLOR_WHITE = "#FFFFFF"

DISPLAY_NAMES = {
    "best_cat": "CatBoost",
    "best_stacking": "Stacking",
    "best_voting": "Voting",
    "best_svm": "SVM-RBF",
    "best_xgb": "XGBoost",
    "best_lgbm": "LightGBM",
    "best_knn": "KNN",
    "best_lr": "LR",
    "best_dt": "DT",
    "best_dnn": "DNN",
}

RF_LIKE_NAMES = {
    "best_rf",
    "rf",
    "random_forest",
    "random forest",
    "best_random_forest",
}

TOP_N_MODELS = 5

def display_name(model_name):
    model_name = str(model_name)
    return DISPLAY_NAMES.get(
        model_name,
        model_name.replace("best_", "").replace("_", " ").strip()
    )

def find_metric_col(df_obj):
    candidate_cols = [
        "PR-AUC",
        "AUPRC",
        "pr_auc",
        "PR_AUC",
        "AUC_PR",
        "average_precision",
        "Average precision",
        "average_precision_score",
    ]
    for col in candidate_cols:
        if col in df_obj.columns:
            return col

    raise ValueError(
        "fold metric text PR-AUC/AUPRC/average_precision text \n"
        f"text: {list(df_obj.columns)}"
    )




cv_summary_path = RUN_DIR / "model_summary" / "model_metrics_summary_cv_raw.csv"

if not cv_summary_path.exists():
    raise FileNotFoundError(
        f"text: {cv_summary_path}\n"
        "text, text model_summary/model_metrics_summary_cv_raw.csv "
    )

df_summary = pd.read_csv(cv_summary_path)

required_cols = ["model_name", "PR-AUC_mean"]
missing_cols = [c for c in required_cols if c not in df_summary.columns]

if missing_cols:
    raise ValueError(
        f"model_metrics_summary_cv_raw.csv text: {missing_cols}"
    )


df_summary = df_summary[
    ~df_summary["model_name"]
    .astype(str)
    .str.lower()
    .isin(RF_LIKE_NAMES)
].copy()

df_summary["display_name"] = df_summary["model_name"].map(display_name)

df_top = (
    df_summary
    .sort_values("PR-AUC_mean", ascending=False)
    .head(TOP_N_MODELS)
    .reset_index(drop=True)
)

top_models = df_top["model_name"].astype(str).tolist()
top_display = df_top["display_name"].astype(str).tolist()

print("Top models:", top_display)




fold_frames = []

for model in top_models:
    fold_path = RUN_DIR / f"{model}_cv_fold_metrics.csv"

    if not fold_path.exists():
        raise FileNotFoundError(
            f"text fold-level text: {fold_path}\n"
            f"text {model}_cv_fold_metrics.csv text "
        )

    tmp = pd.read_csv(fold_path)
    metric_col = find_metric_col(tmp)

    tmp = tmp.copy()
    tmp["model_name"] = model
    tmp["display_name"] = display_name(model)
    tmp["fold_pr_auc"] = pd.to_numeric(tmp[metric_col], errors="coerce")

    if "fold" not in tmp.columns:
        tmp["fold"] = np.arange(1, len(tmp) + 1)

    fold_frames.append(
        tmp[["model_name", "display_name", "fold", "fold_pr_auc"]]
    )

df_fold = pd.concat(fold_frames, axis=0, ignore_index=True)
df_fold = df_fold.dropna(subset=["fold_pr_auc"]).copy()

df_fold["display_name"] = pd.Categorical(
    df_fold["display_name"],
    categories=top_display,
    ordered=True
)

df_fold = df_fold.sort_values(["display_name", "fold"]).copy()


source_path = SRC_DIR / "Figure3b_fold_level_pr_auc_stability_source_data.csv"

df_fold.to_csv(
    source_path,
    index=False,
    encoding="utf-8-sig"
)

print("Source data exported:")
print(source_path)




labels = top_display

data = []

for model_label in labels:
    vals = (
        df_fold.loc[df_fold["display_name"].astype(str) == model_label, "fold_pr_auc"]
        .dropna()
        .to_numpy()
    )
    data.append(vals)

positions = np.arange(len(labels))




fig, ax = plt.subplots(figsize=(3.90, 2.15), dpi=300)

box_width = 0.30

bp = ax.boxplot(
    data,
    positions=positions,
    widths=box_width,
    patch_artist=True,
    showfliers=False,
    medianprops=dict(
        color=COLOR_TEXT,
        linewidth=0.70
    ),
    boxprops=dict(
        linewidth=0.60,
        color=COLOR_OTHER_EDGE
    ),
    whiskerprops=dict(
        linewidth=0.60,
        color=COLOR_OTHER_EDGE
    ),
    capprops=dict(
        linewidth=0.60,
        color=COLOR_OTHER_EDGE
    )
)


for i, patch in enumerate(bp["boxes"]):
    if labels[i] == "CatBoost":
        patch.set_facecolor(COLOR_CAT_LIGHT)
        patch.set_edgecolor(COLOR_CAT)
        patch.set_alpha(0.62)
    else:
        patch.set_facecolor(COLOR_OTHER_BOX)
        patch.set_edgecolor(COLOR_OTHER_EDGE)
        patch.set_alpha(0.62)


rng = np.random.default_rng(2026)

for i, (model_label, vals) in enumerate(zip(labels, data)):
    jitter = rng.normal(loc=0.0, scale=0.022, size=len(vals))
    x = positions[i] + jitter

    if model_label == "CatBoost":
        point_color = COLOR_CAT
        point_size = 12
        point_alpha = 0.82
    else:
        point_color = COLOR_OTHER_POINT
        point_size = 11
        point_alpha = 0.80

    ax.scatter(
        x,
        vals,
        s=point_size,
        color=point_color,
        edgecolor=COLOR_WHITE,
        linewidth=0.28,
        alpha=point_alpha,
        zorder=3
    )




ax.set_xticks(positions)
ax.set_xticklabels(
    labels,
    rotation=0,
    ha="center"
)

ax.set_ylabel("Fold PR-AUC", fontsize=6.5)

ax.set_ylim(0.25, 0.75)
ax.set_yticks([0.3, 0.4, 0.5, 0.6, 0.7])

ax.grid(
    True,
    axis="y",
    color=COLOR_GRID,
    linewidth=0.40,
    linestyle="-"
)
ax.grid(False, axis="x")

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_linewidth(0.52)
ax.spines["bottom"].set_linewidth(0.52)

ax.tick_params(
    axis="both",
    length=2.0,
    width=0.52,
    color=COLOR_TEXT,
    labelsize=6.2
)







plt.tight_layout(pad=0.35)




out_base = FIG3_DIR / "Figure3b_fold_level_pr_auc_stability_box_jitter_no_title"

fig.savefig(
    out_base.with_suffix(".png"),
    dpi=900,
    bbox_inches="tight",
    facecolor="white"
)

fig.savefig(
    out_base.with_suffix(".pdf"),
    bbox_inches="tight",
    facecolor="white"
)

fig.savefig(
    out_base.with_suffix(".svg"),
    bbox_inches="tight",
    facecolor="white"
)

fig.savefig(
    out_base.with_suffix(".tiff"),
    dpi=900,
    bbox_inches="tight",
    facecolor="white"
)

plt.show()
plt.close(fig)

print("Figure 3b exported:")
print(out_base.with_suffix(".png"))
print(out_base.with_suffix(".pdf"))
print(out_base.with_suffix(".svg"))
print(out_base.with_suffix(".tiff"))
print("Source data exported:")
print(source_path)






from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt




if "SAVE_DIR" not in globals():
    SAVE_DIR = str(RESULTS_DIR)

RUN_DIR = Path(SAVE_DIR).resolve()

FIG3_DIR = RUN_DIR / "output" / "Figure3" / "optimized_single_panels"
SRC_DIR = RUN_DIR / "source_data" / "Figure3" / "optimized_single_panels"

FIG3_DIR.mkdir(parents=True, exist_ok=True)
SRC_DIR.mkdir(parents=True, exist_ok=True)

print("RUN_DIR =", RUN_DIR)
print("Output =", FIG3_DIR)




mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,

    "font.size": 6.8,
    "axes.titlesize": 6.8,
    "axes.labelsize": 6.8,
    "xtick.labelsize": 6.3,
    "ytick.labelsize": 6.6,

    "axes.linewidth": 0.58,
    "xtick.major.width": 0.52,
    "ytick.major.width": 0.52,
    "xtick.major.size": 2.2,
    "ytick.major.size": 0,

    "legend.fontsize": 6.4,
    "legend.frameon": False,
    "axes.unicode_minus": False,
})




COLOR_OOF = "#7FA6CA"
COLOR_TEST = "#C9866D"
COLOR_LINE = "#A8B0B8"
COLOR_TEXT = "#23272B"

METRICS = ["PR-AUC", "ROC-AUC", "F1", "Precision", "Recall"]

METRIC_LABELS = {
    "PR-AUC": "PR-AUC",
    "ROC-AUC": "ROC-AUC",
    "F1": "F1",
    "Precision": "Precision",
    "Recall": "Recall",
}

def remove_top_right_spines(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)




oof_path = RUN_DIR / "model_summary" / "selected_primary_model_oof.csv"
test_path = RUN_DIR / "model_summary" / "selected_primary_model_test.csv"

if not oof_path.exists():
    raise FileNotFoundError(f"text: {oof_path}")

if not test_path.exists():
    raise FileNotFoundError(f"text: {test_path}")

oof = pd.read_csv(oof_path)
test = pd.read_csv(test_path)

missing_oof = [m for m in METRICS if m not in oof.columns]
missing_test = [m for m in METRICS if m not in test.columns]

if missing_oof:
    raise ValueError(f"selected_primary_model_oof.csv text: {missing_oof}")

if missing_test:
    raise ValueError(f"selected_primary_model_test.csv text: {missing_test}")


oof_values = oof.iloc[0][METRICS].astype(float)
test_values = test.iloc[0][METRICS].astype(float)

source_df = pd.DataFrame({
    "metric": METRICS,
    "metric_label": [METRIC_LABELS[m] for m in METRICS],
    "OOF": [float(oof_values[m]) for m in METRICS],
    "Independent test": [float(test_values[m]) for m in METRICS],
})

source_df.to_csv(
    SRC_DIR / "Figure3c_oof_vs_independent_test_source_data.csv",
    index=False,
    encoding="utf-8-sig"
)





fig, ax = plt.subplots(figsize=(2.80, 1.95), dpi=300)


y = np.arange(len(METRICS))[::-1]

for i, metric in enumerate(METRICS):
    yy = y[i]
    x_oof = float(oof_values[metric])
    x_test = float(test_values[metric])


    ax.plot(
        [x_oof, x_test],
        [yy, yy],
        color=COLOR_LINE,
        linewidth=0.78,
        solid_capstyle="round",
        zorder=1
    )


    ax.scatter(
        x_oof,
        yy,
        s=19,
        color=COLOR_OOF,
        edgecolor="white",
        linewidth=0.32,
        zorder=3,
        label="OOF" if i == 0 else None
    )


    ax.scatter(
        x_test,
        yy,
        s=19,
        color=COLOR_TEST,
        edgecolor="white",
        linewidth=0.32,
        zorder=3,
        label="Independent test" if i == 0 else None
    )




ax.set_yticks(y)
ax.set_yticklabels([METRIC_LABELS[m] for m in METRICS])


all_values = np.concatenate([
    source_df["OOF"].to_numpy(),
    source_df["Independent test"].to_numpy()
])

x_min = max(0.40, all_values.min() - 0.020)
x_max = min(0.93, all_values.max() + 0.020)

ax.set_xlim(x_min, x_max)
ax.set_xticks([0.4, 0.6, 0.8])
ax.set_xlabel("Metric value")


ax.grid(False)

remove_top_right_spines(ax)

ax.spines["left"].set_color(COLOR_TEXT)
ax.spines["bottom"].set_color(COLOR_TEXT)
ax.spines["left"].set_linewidth(0.58)
ax.spines["bottom"].set_linewidth(0.58)

ax.tick_params(
    axis="x",
    length=2.2,
    width=0.52,
    color=COLOR_TEXT,
    labelsize=6.3
)

ax.tick_params(
    axis="y",
    length=0,
    width=0.52,
    color=COLOR_TEXT,
    labelsize=6.6,
    pad=2
)


ax.set_title("")


ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, 1.13),
    ncol=2,
    handlelength=1.0,
    handletextpad=0.35,
    columnspacing=1.0,
    borderaxespad=0.0
)


plt.tight_layout(pad=0.35)




out_base = FIG3_DIR / "Figure3c_oof_vs_independent_test_no_title_no_grid_further_compressed"

fig.savefig(
    out_base.with_suffix(".png"),
    dpi=900,
    bbox_inches="tight",
    facecolor="white"
)

fig.savefig(
    out_base.with_suffix(".pdf"),
    bbox_inches="tight",
    facecolor="white"
)

fig.savefig(
    out_base.with_suffix(".svg"),
    bbox_inches="tight",
    facecolor="white"
)

fig.savefig(
    out_base.with_suffix(".tiff"),
    dpi=900,
    bbox_inches="tight",
    facecolor="white"
)

plt.show()
plt.close(fig)

print("Figure 3c exported:")
print(out_base.with_suffix(".png"))
print(out_base.with_suffix(".pdf"))
print(out_base.with_suffix(".svg"))
print(out_base.with_suffix(".tiff"))
print("Source data exported:")
print(SRC_DIR / "Figure3c_oof_vs_independent_test_source_data.csv")






from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt




if "SAVE_DIR" not in globals():
    SAVE_DIR = str(RESULTS_DIR)

RUN_DIR = Path(SAVE_DIR).resolve()

FIG3_DIR = RUN_DIR / "output" / "Figure3" / "optimized_single_panels"
SRC_DIR = RUN_DIR / "source_data" / "Figure3" / "optimized_single_panels"

FIG3_DIR.mkdir(parents=True, exist_ok=True)
SRC_DIR.mkdir(parents=True, exist_ok=True)

print("RUN_DIR =", RUN_DIR)
print("Output =", FIG3_DIR)




mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,

    "font.size": 6.6,
    "axes.titlesize": 6.2,
    "axes.labelsize": 6.3,
    "xtick.labelsize": 5.9,
    "ytick.labelsize": 6.0,

    "axes.linewidth": 0.50,
    "xtick.major.width": 0.50,
    "ytick.major.width": 0.50,
    "legend.frameon": False,
    "axes.unicode_minus": False,
})




COLOR_TEXT = "#23272B"
COLOR_GRID = "#F4F6F8"


COLOR_LOW = "#A6B0BA"
COLOR_LOW_EDGE = "#87929E"
COLOR_LOW_BOX = "#F4F6F8"


COLOR_HIGH = "#C98C72"
COLOR_HIGH_EDGE = "#B97962"
COLOR_HIGH_BOX = "#FAEEE9"

COLOR_WHITE = "#FFFFFF"





pred_path = RUN_DIR / "best_cat_test_predictions.csv"

fallback_path = (
    RUN_DIR
    / "source_data"
    / "Figure3"
    / "rich_multi_panel_exploration"
    / "panel_E_test_score_distribution.csv"
)

if pred_path.exists():
    pred = pd.read_csv(pred_path).rename(
        columns={
            "y_true_test": "true_label",
            "y_prob_test": "predicted_score",
            "y_pred_test": "predicted_label",
        }
    )
elif fallback_path.exists():
    pred = pd.read_csv(fallback_path)
else:
    raise FileNotFoundError(
        "text best_cat_test_predictions.csv, text "
        "source_data/Figure3/rich_multi_panel_exploration/panel_E_test_score_distribution.csv "
    )

required_cols = ["true_label", "predicted_score"]
missing_cols = [c for c in required_cols if c not in pred.columns]

if missing_cols:
    raise ValueError(
        f"text: {missing_cols}\n"
        f"text: {list(pred.columns)}"
    )

pred = pred.copy()
pred["true_label"] = pred["true_label"].astype(int)
pred["predicted_score"] = pd.to_numeric(pred["predicted_score"], errors="coerce")
pred = pred.dropna(subset=["true_label", "predicted_score"]).copy()

pred["true_class"] = np.where(
    pred["true_label"].eq(1),
    "True high",
    "True non-high"
)


source_path = SRC_DIR / "Figure3e_test_score_separation_source_data.csv"
pred.to_csv(
    source_path,
    index=False,
    encoding="utf-8-sig"
)

print("Source data exported:")
print(source_path)




class_order = ["True non-high", "True high"]
display_labels = ["True\nnon-high", "True\nhigh"]

data = []
for cls in class_order:
    vals = (
        pred.loc[pred["true_class"].eq(cls), "predicted_score"]
        .dropna()
        .to_numpy()
    )
    data.append(vals)

positions = np.arange(len(class_order))




fig, ax = plt.subplots(figsize=(2.35, 2.15), dpi=300)

box_width = 0.38

bp = ax.boxplot(
    data,
    positions=positions,
    widths=box_width,
    patch_artist=True,
    showfliers=False,
    medianprops=dict(
        color=COLOR_TEXT,
        linewidth=0.72
    ),
    boxprops=dict(
        linewidth=0.62
    ),
    whiskerprops=dict(
        linewidth=0.60
    ),
    capprops=dict(
        linewidth=0.60
    )
)


box_fills = [COLOR_LOW_BOX, COLOR_HIGH_BOX]
box_edges = [COLOR_LOW_EDGE, COLOR_HIGH_EDGE]

for i, patch in enumerate(bp["boxes"]):
    patch.set_facecolor(box_fills[i])
    patch.set_edgecolor(box_edges[i])
    patch.set_alpha(0.78)

for i, whisker in enumerate(bp["whiskers"]):
    whisker.set_color(box_edges[i // 2])
    whisker.set_linewidth(0.60)

for i, cap in enumerate(bp["caps"]):
    cap.set_color(box_edges[i // 2])
    cap.set_linewidth(0.60)


rng = np.random.default_rng(2026)

point_colors = [COLOR_LOW, COLOR_HIGH]
point_edges = [COLOR_WHITE, COLOR_WHITE]

for i, vals in enumerate(data):
    jitter = rng.normal(loc=0.0, scale=0.045, size=len(vals))
    x = positions[i] + jitter

    ax.scatter(
        x,
        vals,
        s=10,
        color=point_colors[i],
        edgecolor=point_edges[i],
        linewidth=0.25,
        alpha=0.78,
        zorder=3
    )




ax.set_xticks(positions)
ax.set_xticklabels(display_labels)

ax.set_ylabel("Predicted score", fontsize=6.3)

ax.set_ylim(-0.02, 1.02)
ax.set_yticks([0.0, 0.25, 0.50, 0.75, 1.0])

ax.grid(
    True,
    axis="y",
    color=COLOR_GRID,
    linewidth=0.35,
    linestyle="-"
)
ax.grid(False, axis="x")

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.spines["left"].set_linewidth(0.50)
ax.spines["bottom"].set_linewidth(0.50)

ax.tick_params(
    axis="both",
    length=1.8,
    width=0.50,
    color=COLOR_TEXT,
    labelsize=6.0
)

ax.tick_params(axis="x", pad=3)

ax.set_title(
    "Test score separation",
    loc="center",
    pad=1.5,
    fontsize=6.2,
    fontweight="normal",
    color=COLOR_TEXT
)

plt.tight_layout(pad=0.35)




out_base = FIG3_DIR / "Figure3e_test_score_separation_box_jitter_refined"

fig.savefig(
    out_base.with_suffix(".png"),
    dpi=900,
    bbox_inches="tight",
    facecolor="white"
)

fig.savefig(
    out_base.with_suffix(".pdf"),
    bbox_inches="tight",
    facecolor="white"
)

fig.savefig(
    out_base.with_suffix(".svg"),
    bbox_inches="tight",
    facecolor="white"
)

fig.savefig(
    out_base.with_suffix(".tiff"),
    dpi=900,
    bbox_inches="tight",
    facecolor="white"
)

plt.show()
plt.close(fig)

print("Figure 3e exported:")
print(out_base.with_suffix(".png"))
print(out_base.with_suffix(".pdf"))
print(out_base.with_suffix(".svg"))
print(out_base.with_suffix(".tiff"))






from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns




if "SAVE_DIR" not in globals():
    SAVE_DIR = str(RESULTS_DIR)

RUN_DIR = Path(SAVE_DIR).resolve()

FIG3_DIR = RUN_DIR / "output" / "Figure3" / "optimized_single_panels"
SRC_DIR = RUN_DIR / "source_data" / "Figure3" / "optimized_single_panels"

FIG3_DIR.mkdir(parents=True, exist_ok=True)
SRC_DIR.mkdir(parents=True, exist_ok=True)

print("RUN_DIR =", RUN_DIR)
print("Output =", FIG3_DIR)




mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,

    "font.size": 6.6,
    "axes.titlesize": 6.2,
    "axes.labelsize": 6.3,
    "xtick.labelsize": 6.0,
    "ytick.labelsize": 6.0,

    "axes.linewidth": 0.50,
    "xtick.major.width": 0.50,
    "ytick.major.width": 0.50,
    "legend.frameon": False,
    "axes.unicode_minus": False,
})




COLOR_TEXT = "#23272B"


COLOR_NONHIGH = "#8FA9C8"
COLOR_HIGH = "#C98C72"




pred_path = RUN_DIR / "best_cat_test_predictions.csv"

fallback_path = (
    RUN_DIR
    / "source_data"
    / "Figure3"
    / "rich_multi_panel_exploration"
    / "panel_E_test_score_distribution.csv"
)

if pred_path.exists():
    pred = pd.read_csv(pred_path)
    print("Loaded:", pred_path)
elif fallback_path.exists():
    pred = pd.read_csv(fallback_path)
    print("Loaded fallback:", fallback_path)
else:
    raise FileNotFoundError(
        "text best_cat_test_predictions.csv, text fallback text: \n"
        f"{fallback_path}"
    )

print("Columns:", list(pred.columns))




true_label_candidates = [
    "true_label",
    "y_true_test",
    "y_test",
    "label",
    "y_true",
]

score_candidates = [
    "predicted_score",
    "y_prob_test",
    "y_score_test",
    "test_pred_score",
    "test_pred_proba",
    "y_proba_test",
    "proba_test",
    "pred_test_proba",
    "score",
]

true_col = None
score_col = None

for c in true_label_candidates:
    if c in pred.columns:
        true_col = c
        break

for c in score_candidates:
    if c in pred.columns:
        score_col = c
        break

if true_col is None:
    raise ValueError(
        "text text: \n"
        f"{list(pred.columns)}\n"
        f"text: {true_label_candidates}"
    )

if score_col is None:
    raise ValueError(
        "text/text text: \n"
        f"{list(pred.columns)}\n"
        f"text: {score_candidates}"
    )

print("true label column =", true_col)
print("score column      =", score_col)




df_e = pred[[true_col, score_col]].copy()
df_e = df_e.rename(columns={
    true_col: "true_label",
    score_col: "predicted_score"
})

df_e["true_label"] = pd.to_numeric(df_e["true_label"], errors="coerce")
df_e["predicted_score"] = pd.to_numeric(df_e["predicted_score"], errors="coerce")

df_e = df_e.dropna(subset=["true_label", "predicted_score"]).copy()
df_e["true_label"] = df_e["true_label"].astype(int)


df_e["group"] = np.where(
    df_e["true_label"].eq(1),
    "High-delivery",
    "Non-high"
)


source_path = SRC_DIR / "Figure3e_test_score_separation_kde_full_source_data.csv"
df_e.to_csv(
    source_path,
    index=False,
    encoding="utf-8-sig"
)

print("Source data exported:")
print(source_path)




fig, ax = plt.subplots(figsize=(2.65, 2.15), dpi=300)


sns.kdeplot(
    data=df_e[df_e["group"].eq("Non-high")],
    x="predicted_score",
    fill=True,
    color=COLOR_NONHIGH,
    alpha=0.60,
    linewidth=1.10,
    bw_adjust=0.78,
    cut=3,
    warn_singular=False,
    label="Non-high",
    ax=ax
)


sns.kdeplot(
    data=df_e[df_e["group"].eq("High-delivery")],
    x="predicted_score",
    fill=True,
    color=COLOR_HIGH,
    alpha=0.56,
    linewidth=1.10,
    bw_adjust=0.78,
    cut=3,
    warn_singular=False,
    label="High-delivery",
    ax=ax
)





ax.set_xticks([0.0, 0.4, 0.8, 1.2])
ax.set_yticks([0, 1, 2, 3])

ax.set_xlabel("Predicted score", fontsize=6.3)
ax.set_ylabel("Density", fontsize=6.3)


ax.grid(False)


for gridline in ax.get_xgridlines():
    gridline.set_visible(False)

for gridline in ax.get_ygridlines():
    gridline.set_visible(False)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.spines["left"].set_linewidth(0.50)
ax.spines["bottom"].set_linewidth(0.50)
ax.spines["left"].set_color(COLOR_TEXT)
ax.spines["bottom"].set_color(COLOR_TEXT)

ax.tick_params(
    axis="both",
    length=1.8,
    width=0.50,
    color=COLOR_TEXT,
    labelsize=6.0
)

ax.set_title(
    "Test score separation",
    loc="center",
    pad=1.5,
    fontsize=6.2,
    fontweight="normal",
    color=COLOR_TEXT
)


ax.legend(
    loc="upper right",
    fontsize=5.3,
    handlelength=0.9,
    handletextpad=0.30,
    labelspacing=0.20,
    borderaxespad=0.2,
    frameon=False
)

plt.tight_layout(pad=0.35)




out_base = FIG3_DIR / "Figure3e_test_score_separation_kde_full_display_no_grid_final"

fig.savefig(
    out_base.with_suffix(".png"),
    dpi=900,
    bbox_inches="tight",
    facecolor="white"
)

fig.savefig(
    out_base.with_suffix(".pdf"),
    bbox_inches="tight",
    facecolor="white"
)

fig.savefig(
    out_base.with_suffix(".svg"),
    bbox_inches="tight",
    facecolor="white"
)

fig.savefig(
    out_base.with_suffix(".tiff"),
    dpi=900,
    bbox_inches="tight",
    facecolor="white"
)

plt.show()
plt.close(fig)

print("Figure 3e exported:")
print(out_base.with_suffix(".png"))
print(out_base.with_suffix(".pdf"))
print(out_base.with_suffix(".svg"))
print(out_base.with_suffix(".tiff"))






import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt




plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10
plt.rcParams["axes.linewidth"] = 1.0
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["svg.fonttype"] = "none"




if "SAVE_DIR" not in globals():
    SAVE_DIR = str(RESULTS_DIR)

RUN_DIR = Path(SAVE_DIR).resolve()

FIG3_DIR = RUN_DIR / "output" / "Figure3" / "optimized_single_panels"
SRC_DIR = RUN_DIR / "source_data" / "Figure3" / "optimized_single_panels"

FIG3_DIR.mkdir(parents=True, exist_ok=True)
SRC_DIR.mkdir(parents=True, exist_ok=True)

print("RUN_DIR =", RUN_DIR)
print("Output =", FIG3_DIR)




methods = ["None", "Class\nweight", "ROS"]
x = np.arange(len(methods))

oof_vals = [0.472, 0.507, 0.519]
holdout_vals = [0.815, 0.844, 0.801]




color_oof = "#6f97c4"
color_hold = "#c57b5a"




fig, ax = plt.subplots(figsize=(4.4, 3.8), dpi=600)

ax.plot(
    x, oof_vals,
    color=color_oof,
    marker="o",
    markersize=7,
    linewidth=2.0,
    label="OOF",
    zorder=3
)

ax.plot(
    x, holdout_vals,
    color=color_hold,
    marker="o",
    markersize=7,
    linewidth=2.0,
    label="Hold-out",
    zorder=3
)




ax.set_xticks(x)
ax.set_xticklabels(methods)
ax.set_ylabel("PR-AUC")
ax.set_title("Imbalance sensitivity", pad=8)

ax.set_ylim(0.40, 0.90)
ax.set_yticks(np.arange(0.4, 0.91, 0.1))

ax.grid(False)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.tick_params(axis="x", length=4, width=1.0, pad=6)
ax.tick_params(axis="y", length=4, width=1.0, pad=4)

ax.legend(
    loc="upper left",
    frameon=False,
    fontsize=9,
    handlelength=1.4,
    handletextpad=0.5,
    borderaxespad=0.4
)

plt.tight_layout()




out_base = FIG3_DIR / "Figure3f_imbalance_sensitivity_final"

fig.savefig(
    out_base.with_suffix(".png"),
    dpi=900,
    bbox_inches="tight",
    facecolor="white"
)

fig.savefig(
    out_base.with_suffix(".pdf"),
    bbox_inches="tight",
    facecolor="white"
)

fig.savefig(
    out_base.with_suffix(".svg"),
    bbox_inches="tight",
    facecolor="white"
)

fig.savefig(
    out_base.with_suffix(".tiff"),
    dpi=900,
    bbox_inches="tight",
    facecolor="white"
)

plt.show()
plt.close(fig)




source_df = pd.DataFrame({
    "method": ["None", "Class weight", "ROS"],
    "OOF_PR_AUC": oof_vals,
    "Hold_out_PR_AUC": holdout_vals
})

src_path = SRC_DIR / "Figure3f_imbalance_sensitivity_final_source_data.csv"
source_df.to_csv(src_path, index=False, encoding="utf-8-sig")




print("Figure 3f exported:")
print(out_base.with_suffix(".png"))
print(out_base.with_suffix(".pdf"))
print(out_base.with_suffix(".svg"))
print(out_base.with_suffix(".tiff"))

print("Source data exported:")
print(src_path)






import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt




plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 9
plt.rcParams["axes.titlesize"] = 10
plt.rcParams["axes.labelsize"] = 9
plt.rcParams["xtick.labelsize"] = 8
plt.rcParams["ytick.labelsize"] = 8
plt.rcParams["axes.linewidth"] = 0.9
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["svg.fonttype"] = "none"




RUN_DIR = Path(
    str(RESULTS_DIR)
)

OUT_DIR = RUN_DIR / "output" / "Figure3" / "optimized_single_panels"
SRC_DIR = RUN_DIR / "source_data" / "Figure3" / "optimized_single_panels"

OUT_DIR.mkdir(parents=True, exist_ok=True)
SRC_DIR.mkdir(parents=True, exist_ok=True)

print("RUN_DIR =", RUN_DIR)
print("Output =", OUT_DIR)

PANEL_PREFIX = "Figure3f"





if "summary_df" in globals() and isinstance(summary_df, pd.DataFrame):
    df = summary_df.copy()
    print("Loaded: summary_df from current notebook")
else:
    csv_path = RUN_DIR / "catboost_imbalance_sensitivity_summary.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"text summary_df, text: \n{csv_path}"
        )
    df = pd.read_csv(csv_path)
    print("Loaded:", csv_path)

print("Columns:", list(df.columns))




def normalize_method_name(row):
    text = " ".join([
        str(row.get("imbalance_method_run", "")),
        str(row.get("model_name", ""))
    ]).lower()

    if "none" in text:
        return "None"
    if "ros" in text or "random" in text or "over" in text:
        return "ROS"
    if "class" in text or "weight" in text:
        return "Class weight"

    return text


df["Method"] = df.apply(normalize_method_name, axis=1)

method_order = ["None", "Class weight", "ROS"]

df = df[df["Method"].isin(method_order)].copy()
df = df.drop_duplicates(subset=["Method"]).copy()
df["Method"] = pd.Categorical(df["Method"], categories=method_order, ordered=True)
df = df.sort_values("Method").reset_index(drop=True)

if df.shape[0] != 3:
    raise ValueError(
        f"text 3 text, text: {df['Method'].tolist()}"
    )

print("Detected methods:", df["Method"].tolist())









metric_order = [
    "PR-AUC",
    "ROC-AUC",
    "Precision",
    "Recall",
    "F1",
]

metric_labels_display = [
    "PR-AUC",
    "ROC-AUC",
    "Precision",
    "Recall",
    "F1 score",
]

metric_col_candidates = {
    "PR-AUC": [
        "TEST_PR-AUC", "TEST_AUPRC", "TEST_PR_AUC"
    ],
    "ROC-AUC": [
        "TEST_ROC-AUC", "TEST_AUROC", "TEST_ROC_AUC"
    ],
    "Precision": [
        "TEST_Precision"
    ],
    "Recall": [
        "TEST_Recall"
    ],
    "F1": [
        "TEST_F1", "TEST_F1-score", "TEST_F1_score"
    ],
}


def pick_existing_column(df_, candidates, metric_name):
    for c in candidates:
        if c in df_.columns:
            return c

    raise KeyError(
        f"text {metric_name} text \n"
        f"text: {candidates}"
    )


metric_cols = {}
for metric in metric_order:
    metric_cols[metric] = pick_existing_column(
        df,
        metric_col_candidates[metric],
        metric
    )

print("Detected metric columns:")
for k, v in metric_cols.items():
    print(f"  {k}: {v}")




plot_df = df[["Method"] + [metric_cols[m] for m in metric_order]].copy()
plot_df.columns = ["Method"] + metric_order

src_path = SRC_DIR / f"{PANEL_PREFIX}_radar_single_holdout_source_data.csv"
plot_df.to_csv(src_path, index=False, encoding="utf-8-sig")

print("Source data exported:")
print(src_path)




def values_to_xy(values, angles_deg, rmin=0.4, rmax=1.0):
    values = np.asarray(values, dtype=float)


    r = (values - rmin) / (rmax - rmin)
    r = np.clip(r, 0, 1)

    angles_rad = np.deg2rad(angles_deg)

    x = r * np.cos(angles_rad)
    y = r * np.sin(angles_rad)


    x = np.r_[x, x[0]]
    y = np.r_[y, y[0]]

    return x, y


def grid_polygon(level, angles_deg):
    angles_rad = np.deg2rad(angles_deg)

    x = level * np.cos(angles_rad)
    y = level * np.sin(angles_rad)


    x = np.r_[x, x[0]]
    y = np.r_[y, y[0]]

    return x, y




angles_deg = np.array([90, 18, -54, -126, 162], dtype=float)

rmin = 0.4
rmax = 1.0

tick_values = [0.4, 0.6, 0.8, 1.0]
tick_levels = [(t - rmin) / (rmax - rmin) for t in tick_values]




color_map = {
    "None": "#B8C2CF",
    "Class weight": "#33B5E5",
    "ROS": "#F07F5A",
}

line_width_map = {
    "None": 1.6,
    "Class weight": 2.1,
    "ROS": 1.9,
}

grid_color_outer = "#707070"
grid_color_inner = "#D0D0D0"
spoke_color = "#C6C6C6"




fig, ax = plt.subplots(figsize=(3.35, 3.15), dpi=600)

ax.set_aspect("equal")
ax.axis("off")




for t, level in zip(tick_values, tick_levels):
    gx, gy = grid_polygon(level, angles_deg)

    ax.plot(
        gx,
        gy,
        color=grid_color_outer if t in [0.4, 1.0] else grid_color_inner,
        linewidth=0.85 if t in [0.4, 1.0] else 0.65,
        zorder=1
    )




for a in angles_deg:
    a_rad = np.deg2rad(a)

    ax.plot(
        [0, np.cos(a_rad)],
        [0, np.sin(a_rad)],
        color=spoke_color,
        linewidth=0.65,
        zorder=1
    )




for t, level in zip(tick_values, tick_levels):
    ax.text(
        0.022,
        level,
        f"{t:.1f}",
        ha="left",
        va="center",
        fontsize=7.2,
        color="#4D4D4D"
    )




label_radius = 1.13

for label, a in zip(metric_labels_display, angles_deg):
    a_rad = np.deg2rad(a)

    x_lab = label_radius * np.cos(a_rad)
    y_lab = label_radius * np.sin(a_rad)

    ha = "center"
    va = "center"

    if a == 90:
        ha, va = "center", "bottom"
    elif a == 18:
        ha, va = "left", "center"
    elif a == -54:
        ha, va = "left", "center"
    elif a == -126:
        ha, va = "right", "center"
    elif a == 162:
        ha, va = "right", "center"

    ax.text(
        x_lab,
        y_lab,
        label,
        ha=ha,
        va=va,
        fontsize=8.0,
        color="#222222"
    )




handles = []

for _, row in plot_df.iterrows():
    method = row["Method"]
    values = row[metric_order].astype(float).values

    xs, ys = values_to_xy(
        values,
        angles_deg,
        rmin=rmin,
        rmax=rmax
    )

    line, = ax.plot(
        xs,
        ys,
        color=color_map[method],
        linewidth=line_width_map[method],
        label=method,
        solid_capstyle="round",
        solid_joinstyle="round",
        zorder=3
    )

    handles.append(line)




ax.legend(
    handles=handles,
    labels=method_order,
    loc="lower center",
    bbox_to_anchor=(0.5, 0.015),
    ncol=3,
    frameon=False,
    fontsize=7.2,
    handlelength=1.45,
    handletextpad=0.45,
    columnspacing=0.85,
    borderaxespad=0.0
)


ax.set_xlim(-1.32, 1.32)
ax.set_ylim(-1.28, 1.20)

plt.tight_layout(pad=0.05)




out_base = OUT_DIR / f"{PANEL_PREFIX}_radar_single_holdout_pentagon_JCR_compact"

fig.savefig(
    out_base.with_suffix(".png"),
    dpi=900,
    bbox_inches="tight",
    facecolor="white"
)

fig.savefig(
    out_base.with_suffix(".pdf"),
    bbox_inches="tight",
    facecolor="white"
)

fig.savefig(
    out_base.with_suffix(".svg"),
    bbox_inches="tight",
    facecolor="white"
)

fig.savefig(
    out_base.with_suffix(".tiff"),
    dpi=900,
    bbox_inches="tight",
    facecolor="white"
)

plt.show()
plt.close(fig)

print("JCR compact radar version exported:")
print(out_base.with_suffix(".png"))
print(out_base.with_suffix(".pdf"))
print(out_base.with_suffix(".svg"))
print(out_base.with_suffix(".tiff"))
print("Source data:")
print(src_path)






from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl




mpl.rcParams.update({
    "font.family": "Arial",
    "font.size": 6.6,
    "axes.titlesize": 7.0,
    "axes.labelsize": 6.6,
    "xtick.labelsize": 6.0,
    "ytick.labelsize": 6.0,
    "axes.linewidth": 0.55,
    "xtick.major.width": 0.55,
    "ytick.major.width": 0.55,
    "xtick.major.size": 2.4,
    "ytick.major.size": 2.4,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "axes.unicode_minus": False,
})




if "SAVE_DIR" in globals():
    RUN_DIR = Path(SAVE_DIR).resolve()
else:
    RUN_DIR = Path(
        str(RESULTS_DIR)
    )

OUT_DIR = (
    RUN_DIR
    / "output"
    / "Figure3"
    / "final_single_panels"
    / "jcr_compact_no_grid_blue_line_black_prevalence"
)

SRC_DIR = (
    RUN_DIR
    / "source_data"
    / "Figure3"
    / "final_single_panels"
    / "jcr_compact_no_grid_blue_line_black_prevalence"
)

OUT_DIR.mkdir(parents=True, exist_ok=True)
SRC_DIR.mkdir(parents=True, exist_ok=True)

print("RUN_DIR =", RUN_DIR)
print("OUT_DIR =", OUT_DIR)
print("SRC_DIR =", SRC_DIR)

PANEL_PREFIX = "Figure3_screening"


EXPORT_SUPPLEMENTARY_SCREENING = True




candidate_files = [
    RUN_DIR / "best_cat_test_predictions.csv",
    RUN_DIR / "output" / "best_cat_test_predictions.csv",
    RUN_DIR / "output" / "Figure3" / "best_cat_test_predictions.csv",
    RUN_DIR / "output" / "Figure3" / "final_6panel" / "best_cat_test_predictions.csv",
    Path("./best_cat_test_predictions.csv"),
]

pred_path = None
for p in candidate_files:
    if p.exists():
        pred_path = p
        break

if pred_path is None:
    raise FileNotFoundError(
        "text best_cat_test_predictions.csv \n"
        "text, text candidate_files "
    )

pred = pd.read_csv(pred_path)
print("Loaded:", pred_path)
print("Columns:", list(pred.columns))




def pick_col(df, candidates, label):
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(
        f"text {label} text \n"
        f"text: {candidates}\n"
        f"text: {list(df.columns)}"
    )

true_col = pick_col(
    pred,
    [
        "y_true_test",
        "y_test",
        "true_test",
        "test_true",
        "label_test",
        "y_true",
    ],
    "test-set true label"
)

score_col = pick_col(
    pred,
    [
        "y_prob_test",
        "y_score_test",
        "test_prob",
        "test_score",
        "pred_test_proba",
        "pred_score",
        "probability",
        "score",
    ],
    "test-set predicted score/probability"
)

print("true label column =", true_col)
print("score column =", score_col)




y_true = pred[true_col].to_numpy().astype(int)
y_score = pred[score_col].to_numpy().astype(float)

valid_mask = np.isfinite(y_score)
y_true = y_true[valid_mask]
y_score = y_score[valid_mask]

n = len(y_true)
n_pos = int(np.sum(y_true == 1))

if n == 0:
    raise ValueError("text ")

if n_pos == 0:
    raise ValueError("text, text Precision@k / Recall@k / EF ")

prevalence = n_pos / n

print(f"n_test = {n}")
print(f"n_positive = {n_pos}")
print(f"test prevalence = {prevalence:.4f}")


order = np.argsort(-y_score)
y_true_sorted = y_true[order]
y_score_sorted = y_score[order]




def precision_at_k(y_sorted, k):
    k = min(int(k), len(y_sorted))
    k = max(k, 1)
    return np.sum(y_sorted[:k] == 1) / k

def recall_at_k(y_sorted, k):
    k = min(int(k), len(y_sorted))
    k = max(k, 1)
    return np.sum(y_sorted[:k] == 1) / np.sum(y_sorted == 1)

def ef_at_k(y_sorted, k):
    return precision_at_k(y_sorted, k) / prevalence

def ef_at_percent(y_sorted, percent):
    k = int(np.ceil(len(y_sorted) * percent / 100))
    k = max(k, 1)
    return ef_at_k(y_sorted, k), k

k_max = min(40, n)
k_curve = np.arange(1, k_max + 1)

precision_curve = np.array([precision_at_k(y_true_sorted, k) for k in k_curve])
recall_curve = np.array([recall_at_k(y_true_sorted, k) for k in k_curve])
ef_curve = np.array([ef_at_k(y_true_sorted, k) for k in k_curve])


percent_curve = np.arange(1, 21)
fraction_ef_curve = []
fraction_k_curve = []

for percent in percent_curve:
    ef_value, k_fraction = ef_at_percent(y_true_sorted, percent)
    fraction_ef_curve.append(ef_value)
    fraction_k_curve.append(k_fraction)

fraction_ef_curve = np.array(fraction_ef_curve)
fraction_k_curve = np.array(fraction_k_curve)


highlight_k = [5, 10, 20]
highlight_k = [k for k in highlight_k if k <= k_max]

highlight_precision = np.array([precision_at_k(y_true_sorted, k) for k in highlight_k])
highlight_recall = np.array([recall_at_k(y_true_sorted, k) for k in highlight_k])
highlight_ef = np.array([ef_at_k(y_true_sorted, k) for k in highlight_k])

ef5_percent, k_5_percent = ef_at_percent(y_true_sorted, 5)
ef10_percent, k_10_percent = ef_at_percent(y_true_sorted, 10)

print("\nHighlighted screening metrics:")
for k, p, r, ef in zip(highlight_k, highlight_precision, highlight_recall, highlight_ef):
    print(f"k={k}: Precision={p:.3f}, Recall={r:.3f}, EF={ef:.3f}")

print(f"EF5% = {ef5_percent:.3f}, k = {k_5_percent}")
print(f"EF10% = {ef10_percent:.3f}, k = {k_10_percent}")




curve_df = pd.DataFrame({
    "k": k_curve,
    "Precision_at_k": precision_curve,
    "Recall_at_k": recall_curve,
    "EF_at_k": ef_curve,
})

fraction_df = pd.DataFrame({
    "top_percent": percent_curve,
    "top_k_equivalent": fraction_k_curve,
    "EF_at_top_percent": fraction_ef_curve,
})

summary_rows = []

for k, p, r, ef in zip(highlight_k, highlight_precision, highlight_recall, highlight_ef):
    summary_rows.append({"metric": f"Precision@{k}", "k_or_fraction": k, "value": p})
    summary_rows.append({"metric": f"Recall@{k}", "k_or_fraction": k, "value": r})
    summary_rows.append({"metric": f"EF@{k}", "k_or_fraction": k, "value": ef})

summary_rows.append({"metric": "EF5%", "k_or_fraction": k_5_percent, "value": ef5_percent})
summary_rows.append({"metric": "EF10%", "k_or_fraction": k_10_percent, "value": ef10_percent})

summary_df_export = pd.DataFrame(summary_rows)

curve_src_path = SRC_DIR / f"{PANEL_PREFIX}_curve_source_data.csv"
fraction_src_path = SRC_DIR / f"{PANEL_PREFIX}_fractional_enrichment_source_data.csv"
summary_src_path = SRC_DIR / f"{PANEL_PREFIX}_summary_source_data.csv"

curve_df.to_csv(curve_src_path, index=False, encoding="utf-8-sig")
fraction_df.to_csv(fraction_src_path, index=False, encoding="utf-8-sig")
summary_df_export.to_csv(summary_src_path, index=False, encoding="utf-8-sig")

print("\nSource data exported:")
print(curve_src_path)
print(fraction_src_path)
print(summary_src_path)





main_color = "#2F6FB0"
main_color_dark = "#2F6FB0"


baseline_color = "#000000"
baseline_text_color = "#000000"

text_color = "#23272B"


SINGLE_FIGSIZE = (2.65, 1.90)


DISPLAY_DPI = 180
EXPORT_DPI = 900




def style_axis_jcr_compact(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.spines["left"].set_color(text_color)
    ax.spines["bottom"].set_color(text_color)
    ax.spines["left"].set_linewidth(0.55)
    ax.spines["bottom"].set_linewidth(0.55)


    ax.grid(False)

    ax.tick_params(
        axis="both",
        width=0.55,
        length=2.4,
        color=text_color,
        labelcolor=text_color,
        pad=1.6
    )

def export_current_figure(fig, out_base):
    """
    text png / svg / tiff
    PNG text TIFF text 900 DPI
    SVG text, text DPI
    """
    fig.savefig(
        out_base.with_suffix(".png"),
        dpi=EXPORT_DPI,
        bbox_inches="tight",
        facecolor="white"
    )

    fig.savefig(
        out_base.with_suffix(".svg"),
        bbox_inches="tight",
        facecolor="white"
    )

    fig.savefig(
        out_base.with_suffix(".tiff"),
        dpi=EXPORT_DPI,
        bbox_inches="tight",
        facecolor="white"
    )

    print("Exported:")
    print(out_base.with_suffix(".png"))
    print(out_base.with_suffix(".svg"))
    print(out_base.with_suffix(".tiff"))

def annotate_points_jcr(ax, xs, ys, labels, ylim, mode="default"):
    """
    JCR text:
    - Precision@k text k=5 / k=10 / k=20 text;
    - text;
    - text, text;
    - text
    """
    y_span = ylim[1] - ylim[0]

    for x, y, label in zip(xs, ys, labels):
        x_text = x
        y_text = y + 0.045 * y_span
        va = "bottom"


        if y > ylim[0] + 0.88 * y_span:
            y_text = y - 0.065 * y_span
            va = "top"


        if mode == "precision":
            if "k=5" in label:
                x_text = x - 0.55
                y_text = y + 0.055 * y_span
                va = "bottom"
            elif "k=10" in label:
                x_text = x + 0.55
                y_text = y + 0.055 * y_span
                va = "bottom"
            elif "k=20" in label:
                x_text = x + 0.25
                y_text = y + 0.055 * y_span
                va = "bottom"


        if mode == "ef":
            if "k=5" in label:
                x_text = x - 0.25
            elif "k=10" in label:
                x_text = x + 0.25

            y_text = y - 0.080 * y_span
            va = "top"

            if "k=20" in label:
                y_text = y + 0.050 * y_span
                va = "bottom"


        if mode == "fraction":
            y_text = y + 0.050 * y_span
            va = "bottom"

            if y > ylim[0] + 0.86 * y_span:
                y_text = y - 0.080 * y_span
                va = "top"

        ax.text(
            x_text,
            y_text,
            label,
            ha="center",
            va=va,
            fontsize=4.8,
            color=text_color,
            linespacing=0.84,
            zorder=5,
            bbox=dict(
                facecolor="white",
                edgecolor="none",
                alpha=0.72,
                pad=0.22
            ),
            clip_on=False
        )

def add_baseline_label(ax, x_pos, y_pos, label, ylim):
    """
    baseline text:
    - test prevalence text;
    - text;
    - text
    """
    y_span = ylim[1] - ylim[0]

    ax.text(
        x_pos - 0.45,
        y_pos + y_span * 0.030,
        label,
        ha="right",
        va="bottom",
        fontsize=5.0,
        color=baseline_text_color,
        alpha=1.00,
        zorder=4
    )

def plot_single_curve_panel_jcr(
    x,
    y,
    title,
    xlabel,
    ylabel,
    ylim,
    out_name,
    baseline=None,
    baseline_label=None,
    highlight_x=None,
    highlight_y=None,
    highlight_labels=None,
    xlim=None,
    xticks=None,
    yticks=None,
    annotation_mode="default",
    panel_label=None,
    show_title=True,
):
    fig, ax = plt.subplots(figsize=SINGLE_FIGSIZE, dpi=DISPLAY_DPI)


    ax.plot(
        x,
        y,
        color=main_color,
        linewidth=1.08,
        solid_capstyle="round",
        solid_joinstyle="round",
        zorder=3
    )


    if baseline is not None:
        ax.axhline(
            baseline,
            color=baseline_color,
            linestyle=(0, (3, 2)),
            linewidth=0.70,
            alpha=0.65,
            zorder=1
        )

        if baseline_label is not None:
            x_for_label = xlim[1] if xlim is not None else x[-1]
            add_baseline_label(ax, x_for_label, baseline, baseline_label, ylim)


    if highlight_x is not None and highlight_y is not None:
        ax.scatter(
            highlight_x,
            highlight_y,
            s=12,
            color=main_color_dark,
            edgecolor="white",
            linewidth=0.35,
            zorder=4
        )

        if highlight_labels is not None:
            annotate_points_jcr(
                ax=ax,
                xs=highlight_x,
                ys=highlight_y,
                labels=highlight_labels,
                ylim=ylim,
                mode=annotation_mode
            )


    if show_title and title:
        ax.set_title(
            title,
            loc="left",
            pad=2.2,
            fontweight="semibold"
        )

    ax.set_xlabel(xlabel, labelpad=1.8)
    ax.set_ylabel(ylabel, labelpad=1.8)
    ax.set_ylim(*ylim)

    if xlim is not None:
        ax.set_xlim(*xlim)

    if xticks is not None:
        ax.set_xticks(xticks)

    if yticks is not None:
        ax.set_yticks(yticks)

    style_axis_jcr_compact(ax)


    if panel_label is not None:
        ax.text(
            -0.18,
            1.08,
            panel_label,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=6.8,
            fontweight="bold",
            color=text_color
        )

    fig.tight_layout(pad=0.22)

    out_base = OUT_DIR / out_name
    export_current_figure(fig, out_base)

    plt.show()
    plt.close(fig)






precision_ylim = (0.00, 1.10)

plot_single_curve_panel_jcr(
    x=k_curve,
    y=precision_curve,
    title="Precision@k",
    xlabel="Top-k ranked samples",
    ylabel="Precision@k",
    ylim=precision_ylim,
    out_name="Figure3d_precision_at_k_blue_line_black_prevalence",
    baseline=prevalence,
    baseline_label="test prevalence",
    highlight_x=highlight_k,
    highlight_y=highlight_precision,
    highlight_labels=[f"k={k}\n{v:.2f}" for k, v in zip(highlight_k, highlight_precision)],
    xlim=(1, k_max),
    xticks=[5, 10, 20, 30, 40] if k_max >= 40 else [5, 10, 20, k_max],
    yticks=np.arange(0.0, 1.01, 0.2),
    annotation_mode="precision",
    panel_label=None,
    show_title=False,
)






recall_ylim = (0.00, 1.05)

plot_single_curve_panel_jcr(
    x=k_curve,
    y=recall_curve,
    title="Recall@k",
    xlabel="Top-k ranked samples",
    ylabel="Recall@k",
    ylim=recall_ylim,
    out_name="Figure3e_recall_at_k_blue_line",
    baseline=None,
    baseline_label=None,
    highlight_x=highlight_k,
    highlight_y=highlight_recall,
    highlight_labels=[f"k={k}\n{v:.2f}" for k, v in zip(highlight_k, highlight_recall)],
    xlim=(1, k_max),
    xticks=[5, 10, 20, 30, 40] if k_max >= 40 else [5, 10, 20, k_max],
    yticks=np.arange(0.0, 1.01, 0.2),
    annotation_mode="default",
    panel_label=None,
    show_title=False,
)





if EXPORT_SUPPLEMENTARY_SCREENING:
    ef_ymax = max(4.05, ef_curve.max() * 1.08)
    ef_ylim = (0.00, ef_ymax)

    plot_single_curve_panel_jcr(
        x=k_curve,
        y=ef_curve,
        title="EF@k",
        xlabel="Top-k ranked samples",
        ylabel="Enrichment factor",
        ylim=ef_ylim,
        out_name="Supplementary_EF_at_k_blue_line_black_random",
        baseline=1.0,
        baseline_label="random",
        highlight_x=highlight_k,
        highlight_y=highlight_ef,
        highlight_labels=[f"k={k}\n{v:.2f}" for k, v in zip(highlight_k, highlight_ef)],
        xlim=(1, k_max),
        xticks=[5, 10, 20, 30, 40] if k_max >= 40 else [5, 10, 20, k_max],
        yticks=np.arange(0, np.ceil(ef_ymax) + 0.1, 1.0),
        annotation_mode="ef",
        panel_label=None,
        show_title=True,
    )





if EXPORT_SUPPLEMENTARY_SCREENING:
    fraction_ymax = max(4.05, fraction_ef_curve.max() * 1.08)
    fraction_ylim = (0.00, fraction_ymax)

    highlight_percent = [5, 10]
    highlight_fraction_ef = [ef5_percent, ef10_percent]

    plot_single_curve_panel_jcr(
        x=percent_curve,
        y=fraction_ef_curve,
        title="Fractional enrichment",
        xlabel="Top-ranked fraction (%)",
        ylabel="Enrichment factor",
        ylim=fraction_ylim,
        out_name="Supplementary_fractional_enrichment_blue_line_black_random",
        baseline=1.0,
        baseline_label="random",
        highlight_x=highlight_percent,
        highlight_y=highlight_fraction_ef,
        highlight_labels=[f"{p}%\n{v:.2f}" for p, v in zip(highlight_percent, highlight_fraction_ef)],
        xlim=(1, 20),
        xticks=[1, 5, 10, 15, 20],
        yticks=np.arange(0, np.ceil(fraction_ymax) + 0.1, 1.0),
        annotation_mode="fraction",
        panel_label=None,
        show_title=True,
    )

print("\nDone. JCR compact no-grid single screening panels exported.")
print("Main Figure 3 panels without internal titles:")
print(OUT_DIR / "Figure3d_precision_at_k_blue_line_black_prevalence.png")
print(OUT_DIR / "Figure3e_recall_at_k_blue_line.png")
print("Supplementary panels:")
print(OUT_DIR / "Supplementary_EF_at_k_blue_line_black_random.png")
print(OUT_DIR / "Supplementary_fractional_enrichment_blue_line_black_random.png")






import os
import joblib
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap
from IPython.display import display

from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)




if "SAVE_DIR" not in globals():
    SAVE_DIR = str(RESULTS_DIR)

RUN_DIR = Path(SAVE_DIR).resolve()

FIG4_OUT_DIR = RUN_DIR / "output" / "Figure4"
FIG4_SRC_DIR = RUN_DIR / "source_data" / "Figure4"


FIG4_PANEL_DIR = FIG4_OUT_DIR / "individual_panels_no_labels_PR_AUC_count_only_summary_title"

FIG4_OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG4_SRC_DIR.mkdir(parents=True, exist_ok=True)
FIG4_PANEL_DIR.mkdir(parents=True, exist_ok=True)

print("RUN_DIR        =", RUN_DIR)
print("FIG4_OUT_DIR   =", FIG4_OUT_DIR)
print("FIG4_SRC_DIR   =", FIG4_SRC_DIR)
print("FIG4_PANEL_DIR =", FIG4_PANEL_DIR)





def configure_figure4_style():
    mpl.rcParams.update({
        "font.family": "Arial",
        "font.size": 7.8,
        "axes.titlesize": 8.6,
        "axes.labelsize": 7.6,
        "xtick.labelsize": 6.9,
        "ytick.labelsize": 6.9,
        "legend.fontsize": 6.8,
        "axes.linewidth": 0.70,
        "xtick.major.width": 0.60,
        "ytick.major.width": 0.60,
        "xtick.major.size": 2.8,
        "ytick.major.size": 2.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "figure.dpi": 300,
        "savefig.dpi": 600,
    })

configure_figure4_style()

COLORS = {
    "blue": "#2B6CA3",
    "blue_dark": "#123E66",
    "blue_mid": "#6F9FC2",
    "blue_light": "#DFECF4",
    "teal": "#5C9E9A",
    "teal_dark": "#3F7F7B",
    "salmon": "#C76F5B",
    "gray": "#6E7781",
    "light_gray": "#DCE2E8",
    "grid": "#E7EBF0",
    "axis": "#222222",
    "text": "#222222",
    "muted": "#8B96A3",
}

CMAP_CM = LinearSegmentedColormap.from_list(
    "detumor_blues",
    ["#F5F8FB", "#D9E8F2", "#9EC8DD", "#4F95BE", "#0E4A86"]
)

CLASS_TICK_LABELS = ["Non-high", "High"]


def clean_axes(ax, grid=False):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLORS["axis"])
    ax.spines["bottom"].set_color(COLORS["axis"])
    ax.tick_params(colors=COLORS["axis"], labelcolor=COLORS["axis"])

    if grid:
        ax.grid(True, axis="y", color=COLORS["grid"], lw=0.45, alpha=0.90)
        ax.set_axisbelow(True)





def load_primary_payload():
    candidates = [
        RUN_DIR / "best_cat.joblib",
        RUN_DIR / "best_catboost.joblib",
        RUN_DIR / "model" / "best_cat.joblib",
        RUN_DIR / "model" / "best_catboost.joblib",
        RUN_DIR / "publication_style_best_model" / "best_cat.joblib",
        RUN_DIR / "publication_style_best_model" / "best_catboost.joblib",
    ]

    for p in candidates:
        if p.exists():
            print("[payload loaded]", p)
            return joblib.load(p), p

    print("[warning] primary payload joblib not found.")
    return None, None


def read_best_cat_test_predictions():
    candidates = [
        RUN_DIR / "best_cat_test_predictions.csv",
        RUN_DIR / "best_catboost_test_predictions.csv",
        RUN_DIR / "selected_primary_model_test.csv",
        RUN_DIR / "model_summary" / "best_cat_test_predictions.csv",
        RUN_DIR / "model_summary" / "selected_primary_model_test.csv",
        RUN_DIR / "publication_style_best_model" / "best_cat_test_predictions.csv",
        RUN_DIR / "publication_style_best_model" / "selected_primary_model_test.csv",
    ]

    for p in candidates:
        if p.exists():
            pred = pd.read_csv(p)
            print("[test predictions loaded]", p)
            return pred, p

    raise FileNotFoundError(
        "Cannot find hold-out test prediction CSV. "
        "Please check best_cat_test_predictions.csv or selected_primary_model_test.csv."
    )


def read_best_cat_oof_predictions(payload=None):
    candidates = [
        RUN_DIR / "best_cat_oof_predictions.csv",
        RUN_DIR / "best_catboost_oof_predictions.csv",
        RUN_DIR / "selected_primary_model_oof.csv",
        RUN_DIR / "model_summary" / "best_cat_oof_predictions.csv",
        RUN_DIR / "model_summary" / "selected_primary_model_oof.csv",
        RUN_DIR / "publication_style_best_model" / "best_cat_oof_predictions.csv",
        RUN_DIR / "publication_style_best_model" / "selected_primary_model_oof.csv",
    ]

    for p in candidates:
        if p.exists():
            oof = pd.read_csv(p)
            print("[OOF predictions loaded]", p)
            return oof, p

    if payload is not None and isinstance(payload, dict) and ("oof_prob_train" in payload):
        if "y_train" not in globals():
            raise FileNotFoundError(
                "Payload contains oof_prob_train, but y_train is not available "
                "in the current environment."
            )

        oof = pd.DataFrame({
            "y_true_train": np.asarray(y_train).astype(int),
            "y_prob_oof_train": np.asarray(payload["oof_prob_train"], dtype=float),
        })

        if "best_threshold" not in payload:
            raise ValueError(
                "Payload contains oof_prob_train but does not contain best_threshold. "
                "Cannot generate OOF predicted labels safely."
            )

        thr = float(payload["best_threshold"])
        oof["y_pred_oof_train"] = (oof["y_prob_oof_train"] >= thr).astype(int)

        print("[OOF predictions reconstructed from payload]")
        return oof, "payload:oof_prob_train"

    raise FileNotFoundError(
        "Cannot find OOF prediction CSV and cannot reconstruct from payload."
    )


payload, payload_path = load_primary_payload()
test_pred_raw, test_pred_path = read_best_cat_test_predictions()
oof_pred_raw, oof_pred_path = read_best_cat_oof_predictions(payload=payload)

if payload is not None and isinstance(payload, dict) and "best_threshold" in payload:
    BEST_THR = float(payload["best_threshold"])
elif "BEST_THR" in globals():
    BEST_THR = float(globals()["BEST_THR"])
elif "final_thr" in globals():
    BEST_THR = float(globals()["final_thr"])
else:
    BEST_THR = None

print("BEST_THR =", BEST_THR)





def standardize_prediction_table(df, table_name="prediction"):
    df = df.copy()

    rename_map = {
        "y_true_test": "y_true",
        "y_true_train": "y_true",
        "true_label": "y_true",
        "label": "y_true",
        "target": "y_true",
        "y": "y_true",

        "y_prob_test": "y_prob",
        "y_prob_oof_train": "y_prob",
        "predicted_score": "y_prob",
        "predicted_probability": "y_prob",
        "probability": "y_prob",
        "score": "y_prob",
        "y_score": "y_prob",
        "prob_high": "y_prob",
        "high_probability": "y_prob",

        "y_pred_test": "y_pred",
        "y_pred_oof_train": "y_pred",
        "predicted_label": "y_pred",
        "prediction": "y_pred",
        "pred": "y_pred",

        "cv_fold": "fold",
        "fold_id": "fold",
        "Fold": "fold",
    }

    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    required = ["y_true", "y_prob"]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            f"{table_name} table missing columns: {missing}; "
            f"current columns: {df.columns.tolist()}"
        )

    df["y_true"] = pd.to_numeric(df["y_true"], errors="coerce")
    df["y_prob"] = pd.to_numeric(df["y_prob"], errors="coerce")
    df = df.dropna(subset=["y_true", "y_prob"]).copy()

    df["y_true"] = df["y_true"].astype(int)

    if "y_pred" not in df.columns:
        if BEST_THR is None:
            raise ValueError(
                f"{table_name} does not contain y_pred, and no real BEST_THR was found. "
                "Please provide y_pred in the prediction file or load the payload with best_threshold."
            )

        df["y_pred"] = (df["y_prob"] >= BEST_THR).astype(int)
        print(f"[{table_name}] y_pred generated using BEST_THR = {BEST_THR:.6f}")
    else:
        df["y_pred"] = pd.to_numeric(df["y_pred"], errors="coerce")
        df["y_pred"] = df["y_pred"].fillna(
            (df["y_prob"] >= BEST_THR).astype(int)
            if BEST_THR is not None else np.nan
        )

        if df["y_pred"].isna().any():
            raise ValueError(
                f"{table_name} contains missing y_pred values and no BEST_THR is available."
            )

        df["y_pred"] = df["y_pred"].astype(int)

    keep_cols = ["y_true", "y_prob", "y_pred"]

    if "fold" in df.columns:
        df["fold"] = pd.to_numeric(df["fold"], errors="coerce")
        keep_cols.append("fold")

    return df[keep_cols].copy()


test_pred = standardize_prediction_table(test_pred_raw, table_name="test")
oof_pred = standardize_prediction_table(oof_pred_raw, table_name="OOF")


if "fold" in oof_pred.columns and oof_pred["fold"].notna().any():
    min_fold = int(oof_pred["fold"].dropna().min())
    max_fold = int(oof_pred["fold"].dropna().max())

    if min_fold == 0 and max_fold <= 4:
        oof_pred["fold"] = oof_pred["fold"] + 1

test_pred.to_csv(
    FIG4_SRC_DIR / "holdout_test_predictions_standardized.csv",
    index=False,
)

oof_pred.to_csv(
    FIG4_SRC_DIR / "oof_predictions_standardized.csv",
    index=False,
)

print("test_pred shape:", test_pred.shape)
print("oof_pred shape :", oof_pred.shape)





def get_fold_assignment_for_train():
    if "y_train" not in globals() or "group_id_train" not in globals():
        raise RuntimeError(
            "OOF prediction table has no fold column, and y_train / group_id_train "
            "are not available. Cannot generate Fold 1-5 confusion matrices."
        )

    y_arr = np.asarray(y_train).astype(int)
    groups_arr = np.asarray(group_id_train)

    if "get_cv_splitter" in globals():
        splitter = get_cv_splitter()
    elif "build_cv_splitter" in globals():
        splitter = build_cv_splitter(
            y=y_arr,
            groups=groups_arr,
            n_splits=int(globals().get("N_SPLITS", 5)),
            random_state=int(globals().get("RANDOM_STATE", 42)),
        )
    elif "cv_splitter" in globals():
        splitter = cv_splitter
    else:
        raise RuntimeError(
            "Cannot find get_cv_splitter / build_cv_splitter / cv_splitter. "
            "Please add a fold column to the OOF prediction CSV."
        )

    fold_assignment = np.full(len(y_arr), fill_value=-1, dtype=int)

    for fold_id, (_, va_idx) in enumerate(
        splitter.split(np.zeros(len(y_arr)), y_arr, groups_arr),
        start=1,
    ):
        fold_assignment[va_idx] = fold_id

    if np.any(fold_assignment < 0):
        raise RuntimeError("Some training samples did not receive a fold assignment.")

    return fold_assignment


if "fold" in oof_pred.columns and oof_pred["fold"].notna().any():
    print("[fold assignment] using fold column from OOF prediction table.")
else:
    fold_assignment = get_fold_assignment_for_train()

    if len(fold_assignment) != len(oof_pred):
        raise RuntimeError(
            f"Fold assignment length mismatch: "
            f"fold_assignment={len(fold_assignment)}, oof_pred={len(oof_pred)}."
        )

    oof_pred["fold"] = fold_assignment
    print("[fold assignment] reconstructed from CV splitter.")

oof_pred["fold"] = oof_pred["fold"].astype(int)

oof_pred.to_csv(
    FIG4_SRC_DIR / "oof_predictions_with_fold.csv",
    index=False,
)





def compute_cm_df(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    counts_df = pd.DataFrame(
        cm,
        index=["True non-high", "True high"],
        columns=["Pred non-high", "Pred high"],
    )

    norm_df = counts_df.div(counts_df.sum(axis=1).replace(0, np.nan), axis=0)

    return counts_df, norm_df


fold_cm_counts = {}
fold_cm_norm = {}

fold_ids = sorted(oof_pred["fold"].dropna().astype(int).unique().tolist())

for fold in fold_ids:
    fold_df = oof_pred[oof_pred["fold"] == fold].copy()

    cm_counts, cm_norm = compute_cm_df(
        fold_df["y_true"],
        fold_df["y_pred"],
    )

    fold_cm_counts[fold] = cm_counts
    fold_cm_norm[fold] = cm_norm

    cm_counts.to_csv(
        FIG4_SRC_DIR / f"panel_fold{fold}_confusion_matrix_counts.csv"
    )
    cm_norm.to_csv(
        FIG4_SRC_DIR / f"panel_fold{fold}_confusion_matrix_row_normalized.csv"
    )

oof_cm_counts, oof_cm_norm = compute_cm_df(
    oof_pred["y_true"],
    oof_pred["y_pred"],
)

oof_cm_counts.to_csv(
    FIG4_SRC_DIR / "panel_overall_oof_confusion_matrix_counts.csv"
)
oof_cm_norm.to_csv(
    FIG4_SRC_DIR / "panel_overall_oof_confusion_matrix_row_normalized.csv"
)

y_test_true = test_pred["y_true"].to_numpy().astype(int)
y_test_prob = test_pred["y_prob"].to_numpy().astype(float)
y_test_pred = test_pred["y_pred"].to_numpy().astype(int)

fpr, tpr, roc_thr = roc_curve(y_test_true, y_test_prob)
roc_auc_val = roc_auc_score(y_test_true, y_test_prob)

precision_arr, recall_arr, pr_thr = precision_recall_curve(y_test_true, y_test_prob)
pr_auc_val = average_precision_score(y_test_true, y_test_prob)
test_prevalence = float(np.mean(y_test_true))

pd.DataFrame({
    "fpr": fpr,
    "tpr": tpr,
    "threshold": roc_thr,
}).to_csv(
    FIG4_SRC_DIR / "panel_roc_curve_holdout_test.csv",
    index=False,
)

pd.DataFrame({
    "recall": recall_arr,
    "precision": precision_arr,
    "threshold": np.r_[pr_thr, np.nan],
}).to_csv(
    FIG4_SRC_DIR / "panel_precision_recall_curve_holdout_test.csv",
    index=False,
)

holdout_metrics = {
    "Accuracy": accuracy_score(y_test_true, y_test_pred),
    "Precision": precision_score(y_test_true, y_test_pred, zero_division=0),
    "Recall": recall_score(y_test_true, y_test_pred, zero_division=0),
    "F1": f1_score(y_test_true, y_test_pred, zero_division=0),
    "ROC-AUC": roc_auc_val,
    "PR-AUC": pr_auc_val,
}

pd.DataFrame(
    [{"metric": k, "value": v} for k, v in holdout_metrics.items()]
).to_csv(
    FIG4_SRC_DIR / "panel_holdout_test_metrics_PR_AUC.csv",
    index=False,
)

print("\n[Figure 4 real data summary]")
print("Test prediction file:", test_pred_path)
print("OOF prediction file :", oof_pred_path)
print("Fold IDs:", fold_ids)
print(f"ROC-AUC = {roc_auc_val:.3f}")
print(f"PR-AUC  = {pr_auc_val:.3f}")
print(f"Prevalence = {test_prevalence:.3f}")





def plot_confusion_matrix_panel(
    ax,
    counts_df,
    norm_df=None,
    title="",
    show_xlabel=True,
    show_ylabel=True,
    show_xticklabels=True,
    show_yticklabels=True,
    vmax=None,
    show_summary=True,
):
    values = counts_df.to_numpy(dtype=float)

    if vmax is None:
        vmax = np.nanmax(values) if np.nanmax(values) > 0 else 1




    tn = int(values[0, 0])
    fp = int(values[0, 1])
    fn = int(values[1, 0])
    tp = int(values[1, 1])

    ax.imshow(
        values,
        cmap=CMAP_CM,
        vmin=0,
        vmax=vmax * 1.08,
        interpolation="nearest",
    )

    ax.text(
        0.5,
        1.105,
        title,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=7.6,
        fontweight="bold",
        color="#222222",
    )

    if show_summary:
        summary_text = f"(TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp})"
        ax.text(
            0.5,
            1.035,
            summary_text,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=5.8,
            fontweight="normal",
            color="#222222",
        )

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    if show_xticklabels:
        ax.set_xticklabels(CLASS_TICK_LABELS, fontsize=6.5)
    else:
        ax.set_xticklabels([])

    if show_yticklabels:
        ax.set_yticklabels(CLASS_TICK_LABELS, fontsize=6.5)
    else:
        ax.set_yticklabels([])

    if show_xlabel:
        ax.set_xlabel("Predicted label", fontsize=6.7, labelpad=1.6)
    else:
        ax.set_xlabel("")

    if show_ylabel:
        ax.set_ylabel("True label", fontsize=6.7, labelpad=1.8)
    else:
        ax.set_ylabel("")

    for i in range(2):
        for j in range(2):
            count = int(values[i, j])
            color = "white" if values[i, j] > 0.55 * vmax else "#222222"

            ax.text(
                j,
                i,
                f"{count}",
                ha="center",
                va="center",
                fontsize=8.6,
                fontweight="bold" if values[i, j] > 0.55 * vmax else "normal",
                color=color,
            )

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.tick_params(length=0, pad=1.8)


def panel_fold_cm(
    ax,
    fold_id,
    show_xlabel=True,
    show_ylabel=True,
    show_xticklabels=True,
    show_yticklabels=True,
):
    if fold_id not in fold_cm_counts:
        raise ValueError(
            f"Fold {fold_id} not found. Existing folds: {list(fold_cm_counts.keys())}"
        )

    max_fold_value = max(
        fold_cm_counts[k].to_numpy().max()
        for k in fold_cm_counts
    )

    plot_confusion_matrix_panel(
        ax,
        fold_cm_counts[fold_id],
        norm_df=fold_cm_norm[fold_id],
        title=f"Fold {fold_id}",
        show_xlabel=show_xlabel,
        show_ylabel=show_ylabel,
        show_xticklabels=show_xticklabels,
        show_yticklabels=show_yticklabels,
        vmax=max_fold_value,
        show_summary=True,
    )


def panel_overall_oof_cm(
    ax,
    show_xlabel=True,
    show_ylabel=True,
    show_xticklabels=True,
    show_yticklabels=True,
):
    plot_confusion_matrix_panel(
        ax,
        oof_cm_counts,
        norm_df=oof_cm_norm,
        title="Overall OOF",
        show_xlabel=show_xlabel,
        show_ylabel=show_ylabel,
        show_xticklabels=show_xticklabels,
        show_yticklabels=show_yticklabels,
        vmax=oof_cm_counts.to_numpy().max(),
        show_summary=True,
    )


def panel_holdout_radar(ax):
    labels = list(holdout_metrics.keys())
    values = np.array([holdout_metrics[k] for k in labels], dtype=float)

    n_metrics = len(labels)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False)

    angles_closed = np.r_[angles, angles[0]]
    values_closed = np.r_[values, values[0]]

    ax.set_theta_offset(0.0)
    ax.set_theta_direction(1)

    ax.plot(
        angles_closed,
        values_closed,
        color=COLORS["blue"],
        lw=1.25,
        marker="o",
        ms=2.7,
        zorder=3,
    )

    ax.fill(
        angles_closed,
        values_closed,
        color=COLORS["blue"],
        alpha=0.11,
        zorder=2,
    )

    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=6.7)

    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.4", "0.6", "0.8", "1.0"], fontsize=5.8)

    ax.grid(color=COLORS["grid"], lw=0.45)
    ax.spines["polar"].set_color("#AEB8C2")
    ax.spines["polar"].set_linewidth(0.62)

    ax.set_title(
        "Hold-out test metrics",
        fontsize=8.1,
        fontweight="bold",
        pad=7,
    )

    radial_offset = {
        "Accuracy": 0.145,
        "Precision": 0.105,
        "Recall": 0.125,
        "F1": 0.110,
        "ROC-AUC": 0.135,
        "PR-AUC": 0.125,
    }

    angular_shift = {
        "Accuracy": 0.000,
        "Precision": -0.030,
        "Recall": 0.030,
        "F1": 0.000,
        "ROC-AUC": -0.030,
        "PR-AUC": 0.030,
    }

    for angle, lab, val in zip(angles, labels, values):
        r_txt = val - radial_offset.get(lab, 0.11)
        r_txt = np.clip(r_txt, 0.08, 0.90)
        theta_txt = angle + angular_shift.get(lab, 0.0)

        ax.text(
            theta_txt,
            r_txt,
            f"{val:.2f}",
            ha="center",
            va="center",
            fontsize=5.8,
            color=COLORS["text"],
            zorder=6,
        )


def panel_roc(ax):
    ax.plot(
        fpr,
        tpr,
        color=COLORS["blue"],
        lw=1.55,
        label=f"ROC-AUC = {roc_auc_val:.3f}",
    )

    ax.plot(
        [0, 1],
        [0, 1],
        color=COLORS["gray"],
        lw=0.95,
        ls="--",
        label="Chance",
    )

    ax.set_title("ROC curve", fontsize=8.5, fontweight="bold", pad=3.5)
    ax.set_xlabel("False positive rate", labelpad=2.0)
    ax.set_ylabel("True positive rate", labelpad=2.0)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)

    ax.legend(
        frameon=False,
        loc="lower right",
        handlelength=2.0,
        borderaxespad=0.2,
    )

    clean_axes(ax, grid=True)


def panel_pr(ax):
    ax.plot(
        recall_arr,
        precision_arr,
        color=COLORS["teal"],
        lw=1.55,
        label=f"PR-AUC = {pr_auc_val:.3f}",
    )

    ax.axhline(
        test_prevalence,
        color=COLORS["salmon"],
        lw=0.95,
        ls="--",
        label=f"Prevalence = {test_prevalence:.2f}",
    )

    ax.set_title("Precision-recall curve", fontsize=8.5, fontweight="bold", pad=3.5)
    ax.set_xlabel("Recall", labelpad=2.0)
    ax.set_ylabel("Precision", labelpad=2.0)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)

    ax.legend(
        frameon=False,
        loc="lower left",
        handlelength=2.0,
        borderaxespad=0.2,
    )

    clean_axes(ax, grid=True)









def save_single_panel(draw_func, filename, figsize=(3.0, 2.6), projection=None, show=True):
    if projection == "polar":
        fig_p, ax_p = plt.subplots(
            figsize=figsize,
            subplot_kw={"projection": "polar"}
        )
    else:
        fig_p, ax_p = plt.subplots(figsize=figsize)

    draw_func(ax_p)


    out_png = FIG4_PANEL_DIR / f"{filename}.png"
    fig_p.savefig(
        out_png,
        dpi=600,
        bbox_inches="tight",
        facecolor="white"
    )
    print("[saved panel]", out_png)


    out_pdf = FIG4_PANEL_DIR / f"{filename}.pdf"
    fig_p.savefig(
        out_pdf,
        format="pdf",
        bbox_inches="tight",
        facecolor="white"
    )
    print("[saved panel]", out_pdf)


    out_svg = FIG4_PANEL_DIR / f"{filename}.svg"
    fig_p.savefig(
        out_svg,
        format="svg",
        bbox_inches="tight",
        facecolor="white"
    )
    print("[saved panel]", out_svg)


    out_tiff = FIG4_PANEL_DIR / f"{filename}.tiff"

    try:
        fig_p.savefig(
            out_tiff,
            dpi=900,
            format="tiff",
            bbox_inches="tight",
            facecolor="white",
            pil_kwargs={"compression": "tiff_lzw"}
        )
    except TypeError:
        fig_p.savefig(
            out_tiff,
            dpi=900,
            format="tiff",
            bbox_inches="tight",
            facecolor="white"
        )

    print("[saved panel]", out_tiff)

    if show:
        display(fig_p)

    plt.close(fig_p)



for fold_id in range(1, 6):
    save_single_panel(
        lambda ax, fid=fold_id: panel_fold_cm(
            ax,
            fid,
            show_xlabel=True,
            show_ylabel=True,
            show_xticklabels=True,
            show_yticklabels=True,
        ),
        f"Figure4_panel_{fold_id}_fold{fold_id}_confusion_matrix_count_only_summary_title",
        figsize=(2.60, 2.42),
        show=True,
    )


save_single_panel(
    lambda ax: panel_overall_oof_cm(
        ax,
        show_xlabel=True,
        show_ylabel=True,
        show_xticklabels=True,
        show_yticklabels=True,
    ),
    "Figure4_panel_6_overall_oof_confusion_matrix_count_only_summary_title",
    figsize=(2.85, 2.60),
    show=True,
)


save_single_panel(
    panel_holdout_radar,
    "Figure4_panel_7_holdout_test_metrics_radar_PR_AUC",
    figsize=(2.95, 2.95),
    projection="polar",
    show=True,
)


save_single_panel(
    panel_roc,
    "Figure4_panel_8_holdout_test_roc_curve",
    figsize=(3.20, 2.55),
    show=True,
)


save_single_panel(
    panel_pr,
    "Figure4_panel_9_holdout_test_precision_recall_curve_PR_AUC",
    figsize=(3.20, 2.55),
    show=True,
)

print("\nFigure 4 individual panels exported and displayed.")
print("Panel dir :", FIG4_PANEL_DIR)
print("Source dir:", FIG4_SRC_DIR)
























import os
import json
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from pathlib import Path
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap





if "SAVE_DIR" not in globals():
    SAVE_DIR = str(RESULTS_DIR)

RUN_DIR = Path(SAVE_DIR).resolve()

FIG5_OUT_DIR = RUN_DIR / "output" / "Figure5_revised_v4"
FIG5_SRC_DIR = RUN_DIR / "source_data" / "Figure5_revised_v4"
FIG5_PANEL_DIR = FIG5_OUT_DIR / "individual_panels_no_labels"

FIG5_OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG5_SRC_DIR.mkdir(parents=True, exist_ok=True)
FIG5_PANEL_DIR.mkdir(parents=True, exist_ok=True)

FIG5_BASENAME = "Figure5_candidate_prioritization_recommended_space_v4"

print("RUN_DIR        =", RUN_DIR)
print("FIG5_OUT_DIR   =", FIG5_OUT_DIR)
print("FIG5_SRC_DIR   =", FIG5_SRC_DIR)
print("FIG5_PANEL_DIR =", FIG5_PANEL_DIR)





def configure_figure5_style():
    mpl.rcParams.update({
        "font.family": "Arial",
        "font.size": 7.6,
        "axes.titlesize": 8.4,
        "axes.titleweight": "bold",
        "axes.labelsize": 7.5,
        "xtick.labelsize": 6.8,
        "ytick.labelsize": 6.8,
        "legend.fontsize": 6.3,
        "axes.linewidth": 0.70,
        "xtick.major.width": 0.60,
        "ytick.major.width": 0.60,
        "xtick.major.size": 2.6,
        "ytick.major.size": 2.6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "figure.dpi": 300,
        "savefig.dpi": 600,
    })

configure_figure5_style()

COLORS5 = {
    "blue": "#2B6CA3",
    "blue_dark": "#123E66",
    "blue_mid": "#6F9FC2",
    "blue_light": "#DDEBF4",
    "teal": "#5C9E9A",
    "orange": "#D68A63",
    "orange_light": "#E8B79A",
    "grey": "#AEB8C2",
    "grey_mid": "#7E8B97",
    "grey_light": "#E7EBF0",
    "text": "#222222",
    "axis": "#222222",
    "white": "#FFFFFF",
}

CMAP5_BLUE = LinearSegmentedColormap.from_list(
    "fig5_blue",
    ["#F6FAFC", "#DDEBF4", "#8BB8D6", "#2B6CA3"]
)

CMAP5_SCORE = LinearSegmentedColormap.from_list(
    "fig5_score",
    ["#E9EEF3", "#BFD8E8", "#6F9FC2", "#123E66"]
)


def style_axis_fig5(ax, grid=False):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.spines["left"].set_linewidth(0.70)
    ax.spines["bottom"].set_linewidth(0.70)
    ax.spines["left"].set_color(COLORS5["axis"])
    ax.spines["bottom"].set_color(COLORS5["axis"])

    ax.tick_params(
        axis="both",
        width=0.60,
        length=2.6,
        colors=COLORS5["axis"],
        labelcolor=COLORS5["axis"],
        pad=2.0
    )

    if grid:
        ax.grid(True, axis="y", color=COLORS5["grey_light"], lw=0.45, alpha=0.70)
        ax.grid(False, axis="x")
    else:
        ax.grid(False)

    ax.set_axisbelow(True)


def add_panel_label(ax, label):
    ax.text(
        -0.12, 1.08, label,
        transform=ax.transAxes,
        fontsize=11.0,
        fontweight="bold",
        va="top",
        ha="left",
        color=COLORS5["text"]
    )


def soften_legend(leg):
    if leg is not None:
        leg.set_frame_on(False)
        for txt in leg.get_texts():
            txt.set_fontsize(6.2)


def save_figure_all_formats(fig, out_dir, basename):
    paths = {}
    for ext in ["png", "pdf", "svg", "tiff"]:
        out = Path(out_dir) / f"{basename}.{ext}"
        fig.savefig(out, bbox_inches="tight", facecolor="white")
        paths[ext] = str(out)
        print("[saved]", out)
    return paths


def save_panel_all_formats(fig, basename):
    paths = {}
    for ext in ["png", "pdf", "svg"]:
        out = FIG5_PANEL_DIR / f"{basename}.{ext}"
        fig.savefig(out, bbox_inches="tight", facecolor="white")
        paths[ext] = str(out)
        print("[saved panel]", out)
    return paths





def resolve_fig5_candidates():
    if "formula_candidates_scored" in globals() and globals()["formula_candidates_scored"] is not None:
        df0 = globals()["formula_candidates_scored"].copy()
        return df0, "globals.formula_candidates_scored"

    candidate_paths = [
        RUN_DIR / "formula_generation" / "generated_candidates_scored_with_novelty.csv",
        RUN_DIR / "formula_generation" / "generated_candidates_scored.csv",
    ]

    for p in candidate_paths:
        if p.exists():
            df0 = pd.read_csv(p, encoding="utf-8-sig")
            return df0, str(p)

    raise FileNotFoundError(
        "text text, text formula_candidates_scored "
        "text generated_candidates_scored.csv "
    )


def resolve_score_column(df):
    candidate_cols = [
        "pred_proba_model",
        "Model probability",
        "model_probability",
        "pred_prob",
        "probability",
        "score",
        "predicted_score",
    ]

    for c in candidate_cols:
        if c in df.columns:
            return c

    raise ValueError(
        "text text: "
        "pred_proba_model / Model probability / pred_prob / probability / score "
    )


def resolve_priority_cutoff(df, score_col):
    candidate_global_names = [
        "final_thr",
        "BEST_THR",
        "best_thr",
        "candidate_score_threshold",
        "decision_threshold",
    ]

    for name in candidate_global_names:
        if name in globals() and globals()[name] is not None:
            try:
                val = float(globals()[name])
                if np.isfinite(val):
                    return val, f"globals.{name}"
            except Exception:
                pass

    for obj_name in ["formula_payload", "payload"]:
        obj = globals().get(obj_name, None)
        if isinstance(obj, dict):
            for key in [
                "best_threshold",
                "threshold",
                "best_thr",
                "optimal_threshold",
                "decision_threshold",
            ]:
                if key in obj and obj[key] is not None:
                    try:
                        val = float(obj[key])
                        if np.isfinite(val):
                            return val, f"{obj_name}.{key}"
                    except Exception:
                        pass

    val = float(pd.to_numeric(df[score_col], errors="coerce").quantile(0.75))
    return val, "candidate Q0.75 cutoff"


candidates_fig5, candidate_source = resolve_fig5_candidates()
score_col = resolve_score_column(candidates_fig5)

candidates_fig5[score_col] = pd.to_numeric(candidates_fig5[score_col], errors="coerce")
candidates_fig5 = candidates_fig5.replace([np.inf, -np.inf], np.nan)
candidates_fig5 = candidates_fig5.dropna(subset=[score_col]).copy()
candidates_fig5 = candidates_fig5.sort_values(score_col, ascending=False).reset_index(drop=True)
candidates_fig5["Candidate rank"] = np.arange(1, len(candidates_fig5) + 1)

priority_cutoff, priority_cutoff_source = resolve_priority_cutoff(candidates_fig5, score_col)

print("\n[Figure 5 candidate table]")
print("source       =", candidate_source)
print("shape        =", candidates_fig5.shape)
print("score column =", score_col)
print("cutoff       =", priority_cutoff, "| source:", priority_cutoff_source)






PREFERRED_PROFILE_NUM_COLS = ["Size", "Zeta Potential", "Admin"]


PREFERRED_RANGE_NUM_COLS = ["Size", "Zeta Potential"]

PREFERRED_PROFILE_CAT_COLS = ["Type", "MAT", "TS", "CT"]
PREFERRED_FALLBACK_CAT_COLS = ["Shape"]

FIG5_PROFILE_NUM_COLS = [c for c in PREFERRED_PROFILE_NUM_COLS if c in candidates_fig5.columns]
FIG5_RANGE_NUM_COLS = [c for c in PREFERRED_RANGE_NUM_COLS if c in candidates_fig5.columns]

FIG5_PROFILE_CAT_COLS = [c for c in PREFERRED_PROFILE_CAT_COLS if c in candidates_fig5.columns]
for c in PREFERRED_FALLBACK_CAT_COLS:
    if len(FIG5_PROFILE_CAT_COLS) < 4 and c in candidates_fig5.columns:
        FIG5_PROFILE_CAT_COLS.append(c)

print("profile numeric cols   =", FIG5_PROFILE_NUM_COLS)
print("range numeric cols     =", FIG5_RANGE_NUM_COLS)
print("profile category cols  =", FIG5_PROFILE_CAT_COLS)





def _to_num(s):
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _short_text(x, max_len=12):
    if pd.isna(x):
        return "NA"
    s = str(x)
    return s if len(s) <= max_len else s[:max_len - 1] + "…"


def build_retention_summary(df, score_col):
    scores = _to_num(df[score_col]).dropna()
    thresholds = [0.95, 0.90, 0.80, 0.70, 0.60, 0.50]

    rows = []
    n_total = len(scores)

    for thr in thresholds:
        retained = int((scores >= thr).sum())
        rows.append({
            "score_threshold": thr,
            "retained_n": retained,
            "retained_fraction": retained / n_total if n_total > 0 else np.nan,
            "retained_percent": 100 * retained / n_total if n_total > 0 else np.nan,
            "n_total": n_total,
        })

    return pd.DataFrame(rows)


def build_top_candidate_profile(df, score_col, num_cols, cat_cols, topn=10):
    top = df.head(min(topn, len(df))).copy()

    out = pd.DataFrame()
    out["Rank"] = top["Candidate rank"].astype(int)
    out["Score"] = _to_num(top[score_col]).round(3)

    for c in num_cols:
        if c in top.columns:
            out[c] = _to_num(top[c])

    for c in cat_cols:
        if c in top.columns:
            out[c] = top[c].map(lambda x: _short_text(x, 12))

    return out


def build_working_range_summary(df, score_col, num_cols, topn=50):
    top = df.head(min(topn, len(df))).copy()
    rows = []

    for c in num_cols:
        if c not in df.columns:
            continue

        s_all = _to_num(df[c]).dropna()
        s_top = _to_num(top[c]).dropna()

        if len(s_all) < 3 or len(s_top) < 3:
            continue

        for group_name, s in [
            ("All candidates", s_all),
            (f"Top {len(top)} candidates", s_top),
        ]:
            rows.append({
                "feature": c,
                "group": group_name,
                "n": int(len(s)),
                "q10": float(s.quantile(0.10)),
                "median": float(s.median()),
                "q90": float(s.quantile(0.90)),
                "min": float(s.min()),
                "max": float(s.max()),
            })

    return pd.DataFrame(rows)


retention_fig5 = build_retention_summary(candidates_fig5, score_col)

profile_fig5 = build_top_candidate_profile(
    candidates_fig5,
    score_col=score_col,
    num_cols=FIG5_PROFILE_NUM_COLS,
    cat_cols=FIG5_PROFILE_CAT_COLS,
    topn=10
)

working_range_fig5 = build_working_range_summary(
    candidates_fig5,
    score_col=score_col,
    num_cols=FIG5_RANGE_NUM_COLS,
    topn=50
)





candidates_fig5.to_csv(
    FIG5_SRC_DIR / "Figure5_all_generated_candidates_scored.csv",
    index=False,
    encoding="utf-8-sig"
)

retention_fig5.to_csv(
    FIG5_SRC_DIR / "panel_b_candidate_retention_thresholds.csv",
    index=False,
    encoding="utf-8-sig"
)

profile_fig5.to_csv(
    FIG5_SRC_DIR / "panel_c_top10_candidate_formulation_profile.csv",
    index=False,
    encoding="utf-8-sig"
)

working_range_fig5.to_csv(
    FIG5_SRC_DIR / "panel_d_recommended_working_range_real_units.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n[Figure 5 source data saved]")
print("-", FIG5_SRC_DIR / "panel_b_candidate_retention_thresholds.csv")
print("-", FIG5_SRC_DIR / "panel_c_top10_candidate_formulation_profile.csv")
print("-", FIG5_SRC_DIR / "panel_d_recommended_working_range_real_units.csv")





def draw_ranked_score_panel(ax, df, score_col):
    scores = _to_num(df[score_col]).to_numpy(dtype=float)
    ranks = np.arange(1, len(scores) + 1)
    percent = 100 * (ranks - 1) / max(1, len(scores) - 1)

    ax.plot(
        percent,
        scores,
        color=COLORS5["blue"],
        lw=1.45,
        solid_capstyle="round"
    )


    ax.axvspan(0, 1, color=COLORS5["blue_light"], alpha=0.60, lw=0)
    ax.axvspan(1, 5, color=COLORS5["blue_light"], alpha=0.28, lw=0)


    for thr, color in [
        (0.90, COLORS5["orange"]),
        (0.80, COLORS5["grey_mid"]),
    ]:
        if np.nanmin(scores) <= thr <= np.nanmax(scores):
            ax.axhline(
                thr,
                color=color,
                lw=0.85,
                ls="--",
                alpha=0.90
            )
            ax.text(
                98.5,
                thr + 0.01,
                f"{thr:.2f}",
                ha="right",
                va="bottom",
                fontsize=6.3,
                color=color
            )

    ax.set_xlim(0, 100)
    ax.set_ylim(
        max(0.0, np.nanmin(scores) - 0.03),
        min(1.03, np.nanmax(scores) + 0.04)
    )

    ax.set_title("Score-ranked candidate library", pad=3)
    ax.set_xlabel("Candidate percentile (%)")
    ax.set_ylabel("Predicted high-delivery score")

    style_axis_fig5(ax, grid=True)


def draw_retention_panel(ax, retention_df):
    plot_df = retention_df.sort_values("score_threshold", ascending=True).copy()

    y = np.arange(len(plot_df))
    retained_n = plot_df["retained_n"].to_numpy(dtype=float)
    retained_percent = plot_df["retained_percent"].to_numpy(dtype=float)
    thresholds = plot_df["score_threshold"].to_numpy(dtype=float)

    ax.barh(
        y,
        retained_percent,
        height=0.62,
        color=COLORS5["blue_light"],
        edgecolor=COLORS5["blue_mid"],
        lw=0.8
    )

    for yi, n, pct in zip(y, retained_n, retained_percent):
        ax.text(
            pct + max(retained_percent.max() * 0.018, 0.15),
            yi,
            f"{pct:.2f}% (n = {int(n):,})",
            va="center",
            ha="left",
            fontsize=6.5,
            color=COLORS5["text"]
        )

    ax.set_yticks(y)
    ax.set_yticklabels([f"Score >= {t:.2f}" for t in thresholds])

    ax.set_xlabel("Retained candidates (%)")
    ax.set_title("Candidate retention across score thresholds", pad=3)

    max_pct = max(retained_percent.max(), 1)
    ax.set_xlim(0, max_pct * 1.32)

    style_axis_fig5(ax, grid=True)


def _normalize_value(value, values_ref):
    values_ref = pd.Series(values_ref).dropna().astype(float)

    if pd.isna(value) or len(values_ref) < 3:
        return 0.0

    lo = float(values_ref.quantile(0.05))
    hi = float(values_ref.quantile(0.95))

    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(values_ref.min())
        hi = float(values_ref.max())

    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return 0.0

    return float(np.clip((float(value) - lo) / (hi - lo), 0, 1))


def draw_candidate_profile_table(ax, profile_df, df_all, score_col):
    ax.axis("off")

    if profile_df.empty:
        ax.text(0.5, 0.5, "No candidate profile", ha="center", va="center")
        return

    display_df = profile_df.copy()

    rename_map = {
        "Zeta Potential": "Zeta",
        "Admin": "Admin",
        "Score": "Score",
        "Rank": "Rank",
        "Size": "Size",
    }

    display_df = display_df.rename(columns=rename_map)

    preferred_order = [
        "Rank", "Score", "Size", "Zeta", "Admin",
        "Type", "MAT", "TS", "CT", "Shape"
    ]

    cols = [c for c in preferred_order if c in display_df.columns]

    if len(cols) > 9:
        cols = cols[:9]

    display_df = display_df[cols]

    n_rows = len(display_df)

    width_map = {
        "Rank": 0.075,
        "Score": 0.090,
        "Size": 0.090,
        "Zeta": 0.095,
        "Admin": 0.095,
        "Type": 0.105,
        "MAT": 0.115,
        "TS": 0.105,
        "CT": 0.115,
        "Shape": 0.115,
    }

    widths = np.array([width_map.get(c, 0.10) for c in cols], dtype=float)
    widths = widths / widths.sum()

    x_edges = np.concatenate([[0], np.cumsum(widths)])
    row_h = 1 / (n_rows + 1.35)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.set_title("Top-ranked candidate formulation profile", pad=3)

    y_top = 1 - row_h


    for j, c in enumerate(cols):
        x0 = x_edges[j]
        w = widths[j]

        ax.add_patch(Rectangle(
            (x0, y_top),
            w,
            row_h,
            facecolor=COLORS5["blue_dark"],
            edgecolor="white",
            lw=0.5
        ))

        ax.text(
            x0 + w / 2,
            y_top + row_h / 2,
            c,
            ha="center",
            va="center",
            fontsize=5.7,
            fontweight="bold",
            color="white"
        )


    for i in range(n_rows):
        y0 = y_top - (i + 1) * row_h

        for j, c in enumerate(cols):
            x0 = x_edges[j]
            w = widths[j]
            val = display_df.iloc[i][c]

            if c == "Score":
                norm_val = _normalize_value(val, df_all[score_col])
                face = CMAP5_SCORE(norm_val)
                text_color = "white" if norm_val > 0.55 else COLORS5["text"]

            elif c in ["Size", "Zeta", "Admin"]:
                original_col = {
                    "Size": "Size",
                    "Zeta": "Zeta Potential",
                    "Admin": "Admin",
                }.get(c, c)

                if original_col in df_all.columns:
                    norm_val = _normalize_value(val, df_all[original_col])
                    face = CMAP5_BLUE(norm_val)
                    text_color = "white" if norm_val > 0.65 else COLORS5["text"]
                else:
                    face = "#F4F7FA"
                    text_color = COLORS5["text"]

            elif c == "Rank":
                face = "#FFFFFF"
                text_color = COLORS5["text"]

            else:
                face = "#F4F7FA" if i % 2 == 0 else "#FFFFFF"
                text_color = COLORS5["text"]

            ax.add_patch(Rectangle(
                (x0, y0),
                w,
                row_h,
                facecolor=face,
                edgecolor="white",
                lw=0.45
            ))

            if c == "Score":
                txt = f"{float(val):.2f}" if pd.notna(val) else "NA"
            elif c in ["Size", "Zeta", "Admin"]:
                if pd.isna(val):
                    txt = "NA"
                else:
                    v = float(val)
                    if abs(v) >= 100:
                        txt = f"{v:.0f}"
                    elif abs(v) >= 10:
                        txt = f"{v:.1f}"
                    else:
                        txt = f"{v:.2g}"
            else:
                txt = str(val)

            ax.text(
                x0 + w / 2,
                y0 + row_h / 2,
                txt,
                ha="center",
                va="center",
                fontsize=5.45,
                color=text_color
            )


def _feature_label(feat):
    if feat == "Size":
        return "Size (nm)"
    if feat == "Zeta Potential":
        return "Zeta potential (mV)"
    return feat


def _format_range_value(x):
    if not np.isfinite(x):
        return "NA"
    if abs(x) >= 100:
        return f"{x:.0f}"
    if abs(x) >= 10:
        return f"{x:.1f}"
    return f"{x:.2g}"


def draw_working_range_real_units_panel(ax, range_df, num_cols):
    """
    Draw real-unit q10-median-q90 working ranges.
    Only physicochemical features are included.
    Grey = all generated candidates; blue = top-ranked candidates.
    This mapping should be explained in the figure caption.
    """
    ax.axis("off")

    if range_df.empty or len(num_cols) == 0:
        ax.text(0.5, 0.5, "No numeric working range", ha="center", va="center")
        return

    features = [c for c in num_cols if c in range_df["feature"].unique()]
    features = features[:2]

    ax.set_title("Recommended physicochemical working range", pad=3)

    top_start = 0.76
    panel_h = 0.24
    gap = 0.16

    for i, feat in enumerate(features):
        bottom = top_start - i * (panel_h + gap) - panel_h

        subax = ax.inset_axes([0.17, bottom, 0.74, panel_h])

        rows_feat = range_df[range_df["feature"] == feat].copy()
        row_all = rows_feat[rows_feat["group"] == "All candidates"]
        row_top = rows_feat[rows_feat["group"].str.startswith("Top")]

        if row_all.empty or row_top.empty:
            subax.axis("off")
            continue

        r_all = row_all.iloc[0]
        r_top = row_top.iloc[0]

        x_min = min(r_all["q10"], r_top["q10"], r_all["min"], r_top["min"])
        x_max = max(r_all["q90"], r_top["q90"], r_all["max"], r_top["max"])

        if not np.isfinite(x_min) or not np.isfinite(x_max) or x_max <= x_min:
            x_min, x_max = 0, 1

        pad = 0.08 * (x_max - x_min)
        subax.set_xlim(x_min - pad, x_max + pad)


        subax.hlines(
            0.66,
            r_all["q10"],
            r_all["q90"],
            color=COLORS5["grey"],
            lw=1.55,
            alpha=0.88
        )
        subax.scatter(
            r_all["median"],
            0.66,
            s=15,
            color=COLORS5["grey_mid"],
            zorder=3
        )


        subax.hlines(
            0.34,
            r_top["q10"],
            r_top["q90"],
            color=COLORS5["blue"],
            lw=2.0,
            alpha=0.95
        )
        subax.scatter(
            r_top["median"],
            0.34,
            s=20,
            color=COLORS5["blue_dark"],
            zorder=4
        )


        subax.text(
            -0.07,
            0.50,
            _feature_label(feat),
            transform=subax.transAxes,
            ha="right",
            va="center",
            fontsize=7.0,
            color=COLORS5["text"]
        )


        range_txt = (
            f"Top: {_format_range_value(r_top['median'])} "
            f"[{_format_range_value(r_top['q10'])}-{_format_range_value(r_top['q90'])}]"
        )
        subax.text(
            0.98,
            0.88,
            range_txt,
            transform=subax.transAxes,
            ha="right",
            va="top",
            fontsize=5.9,
            color=COLORS5["blue_dark"]
        )

        subax.set_yticks([])
        subax.grid(False)

        subax.spines["top"].set_visible(False)
        subax.spines["right"].set_visible(False)
        subax.spines["left"].set_visible(False)
        subax.spines["bottom"].set_linewidth(0.65)
        subax.spines["bottom"].set_color(COLORS5["axis"])

        subax.tick_params(axis="x", labelsize=6.2, width=0.55, length=2.3, pad=1.5)

        if i < len(features) - 1:
            subax.set_xlabel("")
        else:
            subax.set_xlabel("Feature value", fontsize=7.0, labelpad=1.8)





fig5 = plt.figure(figsize=(9.3, 6.2))

gs = fig5.add_gridspec(
    nrows=2,
    ncols=2,
    left=0.065,
    right=0.985,
    bottom=0.075,
    top=0.92,
    wspace=0.34,
    hspace=0.42
)

ax_a = fig5.add_subplot(gs[0, 0])
ax_b = fig5.add_subplot(gs[0, 1])
ax_c = fig5.add_subplot(gs[1, 0])
ax_d = fig5.add_subplot(gs[1, 1])

draw_ranked_score_panel(ax_a, candidates_fig5, score_col)
draw_retention_panel(ax_b, retention_fig5)
draw_candidate_profile_table(ax_c, profile_fig5, candidates_fig5, score_col)
draw_working_range_real_units_panel(ax_d, working_range_fig5, FIG5_RANGE_NUM_COLS)

for ax, lab in zip(
    [ax_a, ax_b, ax_c, ax_d],
    ["(a)", "(b)", "(c)", "(d)"]
):
    add_panel_label(ax, lab)

fig5.suptitle(
    "Model-guided candidate prioritization and recommended formulation space",
    fontsize=11.0,
    fontweight="bold",
    x=0.065,
    ha="left",
    y=0.98
)

fig5_paths = save_figure_all_formats(fig5, FIG5_OUT_DIR, FIG5_BASENAME)

plt.show()
plt.close(fig5)





def export_single_fig5_panel(draw_func, basename, figsize=(3.45, 2.55), adjust=None):
    fig, ax = plt.subplots(figsize=figsize)
    draw_func(ax)

    if adjust is None:
        fig.subplots_adjust(left=0.18, right=0.96, bottom=0.18, top=0.88)
    else:
        fig.subplots_adjust(**adjust)

    save_panel_all_formats(fig, basename)
    plt.show()
    plt.close(fig)


export_single_fig5_panel(
    lambda ax: draw_ranked_score_panel(ax, candidates_fig5, score_col),
    "Figure5_panel_a_score_ranked_candidate_library",
    figsize=(3.50, 2.55)
)

export_single_fig5_panel(
    lambda ax: draw_retention_panel(ax, retention_fig5),
    "Figure5_panel_b_candidate_retention_thresholds",
    figsize=(3.60, 2.55)
)

export_single_fig5_panel(
    lambda ax: draw_candidate_profile_table(ax, profile_fig5, candidates_fig5, score_col),
    "Figure5_panel_c_top10_candidate_formulation_profile",
    figsize=(4.55, 2.85),
    adjust={"left": 0.03, "right": 0.98, "bottom": 0.05, "top": 0.88}
)

export_single_fig5_panel(
    lambda ax: draw_working_range_real_units_panel(ax, working_range_fig5, FIG5_RANGE_NUM_COLS),
    "Figure5_panel_d_recommended_physicochemical_working_range",
    figsize=(4.10, 2.85),
    adjust={"left": 0.08, "right": 0.96, "bottom": 0.08, "top": 0.86}
)





figure5_meta = {
    "figure": "Figure 5",
    "title": "Model-guided candidate prioritization and recommended formulation space",
    "candidate_source": candidate_source,
    "score_column": score_col,
    "priority_cutoff": float(priority_cutoff),
    "priority_cutoff_source": priority_cutoff_source,
    "n_candidates": int(len(candidates_fig5)),
    "profile_numeric_columns": FIG5_PROFILE_NUM_COLS,
    "range_numeric_columns": FIG5_RANGE_NUM_COLS,
    "profile_categorical_columns": FIG5_PROFILE_CAT_COLS,
    "outputs": fig5_paths,
    "source_data_dir": str(FIG5_SRC_DIR),
    "panel_dir": str(FIG5_PANEL_DIR),
    "notes": {
        "panel_a": "Generated candidates are ranked by predicted high-delivery score and shown as candidate percentile.",
        "panel_b": "Candidate retention is summarized as retained percentage across fixed high-score thresholds, with retained n shown in labels.",
        "panel_c": "Top 10 candidate formulation profile combines score, physicochemical variables, administration-related numeric value, and formulation/experimental categories.",
        "panel_d": "Recommended physicochemical working range shows q10-median-q90 in real units for Size (nm) and Zeta potential (mV). Grey intervals indicate all generated candidates; blue intervals indicate top-ranked candidates; points denote medians."
    }
}

meta_path = FIG5_SRC_DIR / "Figure5_candidate_prioritization_revised_v4_metadata.json"

with open(meta_path, "w", encoding="utf-8") as f:
    json.dump(figure5_meta, f, ensure_ascii=False, indent=2)

print("\n[Figure 5 revised v4 completed]")
print("Full figure:", FIG5_OUT_DIR)
print("Individual panels:", FIG5_PANEL_DIR)
print("Source data:", FIG5_SRC_DIR)
print("Metadata:", meta_path)
