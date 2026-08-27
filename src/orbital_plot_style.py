"""
===============================================================================
Project:       VLEO Mission Analysis and Platform Analysis for an EP-driven,
               EO Satellite
Collaboration: In collaboration with ArianeGroup
Institution:   Cranfield University
Author:        Arnau Pascual
Year:          2026
===============================================================================
File:          orbital_plot_style.py
Description:   Standardized, publication-grade styling, palettes, and figure formatting utilities.
===============================================================================
Usage:
    import orbital_plot_style as ops

    fig, axes = ops.make_figure("2x1", shared_x=True)
    axes[0].plot(t, alt, **ops.plot_kwargs("nrlmsise00"))

    ops.add_zero_line(axes[1])

    ops.save_figure(fig, "altitude_error", output_dir="results/figures")

Design principles
-----------------
* White background, no decorative gradients or shadows.
* STIX / DejaVu Serif typeface with STIX mathtext for LaTeX-compatible
  equations, Greek symbols, subscripts, superscripts, and units.
* Colorblind-accessible and grayscale-compatible palette (combined with
  linestyle and marker differentiation).
* Inward-facing ticks, thin major grid, no minor grid by default.
* constrained_layout for all figures -- never tight_layout().
* Minimum 300 dpi for all raster exports; PDF and SVG also supported.
* All style decisions are explicit and centralized here.
Scientific integrity
--------------------
This module only controls visual presentation.  It never smooths, filters,
resamples, clips, or normalises any scientific data.
"""

from __future__ import annotations
import os
from typing import Optional, Sequence, Tuple, Union
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# =============================================================================
# 1.  rcPARAMS CONFIGURATION

# =============================================================================
def apply_orbital_style() -> None:
    """
    Configure Matplotlib rcParams for publication-quality orbital figures.

    Called automatically when this module is imported.  Safe to call again

    if rcParams need to be reset after external code has modified them.
    """

    mpl.rcParams.update({
        # -- Typography -------------------------------------------------------
        "font.family":                    "serif",
        "font.serif":                     [
            "STIX Two Text", "STIXGeneral", "DejaVu Serif",
            "Times New Roman", "Georgia", "serif",
        ],
        "font.size":                      9.0,
        "axes.titlesize":                 9.5,
        "axes.labelsize":                 9.0,
        "xtick.labelsize":                8.0,
        "ytick.labelsize":                8.0,
        "legend.fontsize":                7.5,
        "figure.titlesize":               10.0,
        "figure.titleweight":             "normal",
        # -- Mathtext (equations, Greek, subscripts, superscripts) -----------
        "mathtext.fontset":               "stix",
        "mathtext.default":               "regular",
        # -- Figure ----------------------------------------------------------
        "figure.dpi":                     120,
        "figure.facecolor":               "white",
        "figure.edgecolor":               "white",
        "figure.constrained_layout.use":  True,
        "figure.constrained_layout.hspace": 0.08,
        "figure.constrained_layout.wspace": 0.06,
        # -- Axes ------------------------------------------------------------
        "axes.facecolor":                 "white",
        "axes.edgecolor":                 "#444444",
        "axes.linewidth":                 0.75,
        "axes.grid":                      True,
        "axes.grid.which":                "major",
        "axes.spines.top":                True,
        "axes.spines.right":              True,
        "axes.labelweight":               "normal",
        "axes.labelpad":                  4.0,
        "axes.titlepad":                  5.0,
        "axes.axisbelow":                 True,
        # -- Grid ------------------------------------------------------------
        "grid.color":                     "#d0d0d0",
        "grid.linewidth":                 0.5,
        "grid.alpha":                     1.0,
        "grid.linestyle":                 "-",
        # -- Ticks -----------------------------------------------------------
        "xtick.direction":                "in",
        "ytick.direction":                "in",
        "xtick.major.size":               4.0,
        "ytick.major.size":               4.0,
        "xtick.minor.size":               2.0,
        "ytick.minor.size":               2.0,
        "xtick.major.width":              0.75,
        "ytick.major.width":              0.75,
        "xtick.minor.width":              0.5,
        "ytick.minor.width":              0.5,
        "xtick.top":                      True,
        "ytick.right":                    True,
        "xtick.minor.visible":            False,
        "ytick.minor.visible":            False,
        # -- Lines -----------------------------------------------------------
        "lines.linewidth":                1.5,
        "lines.markersize":               4.0,
        "lines.markeredgewidth":          0.6,
        "patch.linewidth":                0.75,
        # -- Legend ----------------------------------------------------------
        "legend.frameon":                 True,
        "legend.framealpha":              0.88,
        "legend.edgecolor":               "#c8c8c8",
        "legend.fancybox":                False,
        "legend.shadow":                  False,
        "legend.labelspacing":            0.30,
        "legend.handlelength":            1.80,
        "legend.handletextpad":           0.45,
        "legend.columnspacing":           1.20,
        "legend.borderpad":               0.40,
        # -- Export ----------------------------------------------------------
        "savefig.dpi":                    300,
        "savefig.bbox":                   "tight",
        "savefig.facecolor":              "white",
        "savefig.edgecolor":              "none",
        "pdf.fonttype":                   42,
        "ps.fonttype":                    42,
    })

# Apply the style automatically on import.

apply_orbital_style()

# =============================================================================
# 2.  FIGURE SIZES  (width, height) in inches

# =============================================================================
FIGURE_SIZES = {
    # -- Journal column widths -----------------------------------------------
    "single_col":    (3.50, 2.60),
    "double_col":    (7.00, 3.80),
    # -- Time-series / wide --------------------------------------------------
    "wide":          (9.50, 3.40),
    "wide_2panel":   (9.50, 5.50),
    # -- Square comparison ---------------------------------------------------
    "square":        (4.50, 4.50),
    # -- Multi-panel stacked -------------------------------------------------
    "2panel":        (9.50, 6.50),
    "3panel":        (9.50, 9.00),
    "4panel":        (9.50, 11.50),
    # -- General purpose -----------------------------------------------------
    "standard":      (9.50, 7.00),
    "standard_tall": (9.50, 9.50),
    # -- Presentation (16:9) -------------------------------------------------
    "presentation":  (10.00, 5.50),
    "pres_2panel":   (10.00, 8.00),
}

# =============================================================================
# 3.  SEMANTIC COLOR PALETTE

# =============================================================================
COLORS = {
    # Data series roles

    "primary":         "#1a6faf",
    "reference":       "#c85200",
    "validated":       "#1b7a3e",
    "secondary":       "#7b2d8b",
    # Ground truth

    "truth":           "#111111",
    # Threshold / tolerance

    "threshold_lower": "#c0392b",
    "threshold_upper": "#27ae60",
    "goal_line":       "#8b0000",
    # Reference lines

    "zero_line":       "#444444",
    "annotation":      "#c0392b",
    # Uncertainty bands

    "band_fill":       "#aec7e8",
    # Grayscale

    "gray_dark":       "#444444",
    "gray_mid":        "#888888",
    "gray_light":      "#cccccc",
}

# =============================================================================
# 4.  MODEL STYLE DICTIONARY

# =============================================================================
MODEL_STYLES = {
    # -- Atmospheric density models (primary lines) --------------------------
    "nrlmsise00": {
        "label":     "NRLMSISE-00",
        "color":     "#1a6faf",
        "linestyle": "-",
        "linewidth": 1.5,
    },
    "jb2008": {
        "label":     "JB2008",
        "color":     "#c85200",
        "linestyle": "--",
        "linewidth": 1.5,
    },
    "dtm2000": {
        "label":     "DTM2000",
        "color":     "#1b7a3e",
        "linestyle": "-.",
        "linewidth": 1.5,
    },
    "harrispriester": {
        "label":     "Harris-Priester",
        "color":     "#7b2d8b",
        "linestyle": ":",
        "linewidth": 1.8,
    },
    # -- Cd-bound envelope variants (lo = Cd_min, hi = Cd_max) --------------
    "nrlmsise00_lo": {
        "label":     "NRLMSISE-00 (Cd min)",
        "color":     "#1a6faf",
        "linestyle": "-",
        "linewidth": 1.2,
        "alpha":     0.85,
    },
    "nrlmsise00_hi": {
        "label":     "NRLMSISE-00 (Cd max)",
        "color":     "#0d3d6b",
        "linestyle": "-",
        "linewidth": 1.2,
        "alpha":     0.85,
    },
    "jb2008_lo": {
        "label":     "JB2008 (Cd min)",
        "color":     "#c85200",
        "linestyle": "--",
        "linewidth": 1.2,
        "alpha":     0.85,
    },
    "jb2008_hi": {
        "label":     "JB2008 (Cd max)",
        "color":     "#7a3200",
        "linestyle": "--",
        "linewidth": 1.2,
        "alpha":     0.85,
    },
    "dtm2000_lo": {
        "label":     "DTM2000 (Cd min)",
        "color":     "#1b7a3e",
        "linestyle": "-.",
        "linewidth": 1.2,
        "alpha":     0.85,
    },
    "dtm2000_hi": {
        "label":     "DTM2000 (Cd max)",
        "color":     "#0e4422",
        "linestyle": "-.",
        "linewidth": 1.2,
        "alpha":     0.85,
    },
    "harrispriester_lo": {
        "label":     "Harris-Priester (Cd min)",
        "color":     "#7b2d8b",
        "linestyle": ":",
        "linewidth": 1.4,
        "alpha":     0.85,
    },
    "harrispriester_hi": {
        "label":     "Harris-Priester (Cd max)",
        "color":     "#46185c",
        "linestyle": ":",
        "linewidth": 1.4,
        "alpha":     0.85,
    },
    # -- TLE ground truth / reference data -----------------------------------
    "tle_truth": {
        "label":     "TLE Ground Truth",
        "color":     "#111111",
        "linestyle": "--",
        "linewidth": 1.0,
        "marker":    "o",
        "markersize": 3.0,
    },
    # -- Model with active drag compensation ---------------------------------
    "model_thrust": {
        "label":     "Model (with thrust)",
        "color":     "#1a6faf",
        "linestyle": "-",
        "linewidth": 1.5,
    },
    # -- Mission profile / target references ---------------------------------
    "target_profile": {
        "label":     "Target Altitude Profile",
        "color":     "#c0392b",
        "linestyle": "--",
        "linewidth": 1.5,
    },
    # -- Propulsion / thrust -------------------------------------------------
    "thrust": {
        "label":     "Thrust",
        "color":     "#1b7a3e",
        "linestyle": "-",
        "linewidth": 1.2,
    },
    "propellant": {
        "label":     "Propellant consumed",
        "color":     "#1b7a3e",
        "linestyle": "-",
        "linewidth": 1.5,
    },
    # -- Power / solar -------------------------------------------------------
    "solar_flux": {
        "label":     "F10.7 Solar Flux",
        "color":     "#c85200",
        "linestyle": "-",
        "linewidth": 1.5,
    },
    "power_generated": {
        "label":     "Power generated",
        "color":     "#b8860b",
        "linestyle": "-",
        "linewidth": 1.3,
    },
    "battery_soc": {
        "label":     "Battery SoC",
        "color":     "#1b7a3e",
        "linestyle": "-",
        "linewidth": 1.5,
    },
    "illumination": {
        "label":     "Illumination fraction",
        "color":     "#c85200",
        "linestyle": "-",
        "linewidth": 1.0,
    },
}

# Band fill opacity for model envelopes (fill_between)

BAND_ALPHA = 0.18
LINE_WIDTHS = {
    "main":      1.5,
    "secondary": 1.2,
    "reference": 1.0,
    "thin":      0.75,
    "zero":      0.85,
    "threshold": 1.2,
    "grid":      0.5,
}

MARKER_SIZES = {
    "default": 4.0,
    "small":   2.5,
    "large":   6.0,
}

# =============================================================================
# 5.  MODEL STYLE HELPERS

# =============================================================================
def plot_kwargs(style_key, label=None):
    """
    Return a dict of keyword arguments for ax.plot() from a MODEL_STYLES entry.

    Parameters
    ----------
    style_key : key into MODEL_STYLES (e.g. "nrlmsise00", "tle_truth").
    label     : override the default label string.
    Returns
    -------
    dict with keys valid for plt.plot().  None values are excluded.
    """

    valid = {
        "color", "linewidth", "linestyle", "marker", "markersize",
        "markevery", "alpha", "label", "zorder",
    }

    s = MODEL_STYLES.get(style_key, {}).copy()

    if label is not None:
        s["label"] = label

    return {k: v for k, v in s.items() if k in valid and v is not None}

def band_color(style_key):
    """Return the primary color for a model, used in fill_between calls."""

    return MODEL_STYLES.get(style_key, {}).get("color", COLORS["primary"])

def model_label(style_key):
    """Return the publication display name for a model."""

    return MODEL_STYLES.get(style_key, {}).get("label", style_key.upper())

# =============================================================================
# 6.  FIGURE CREATION

# =============================================================================
def make_figure(layout="1x1", shared_x=False, shared_y=False, figsize=None):
    """
    Create a figure with a standard layout using constrained_layout.

    Parameters
    ----------
    layout   : "rows x cols" string, e.g. "1x1", "2x1", "3x1".
    shared_x : share the x-axis across rows.
    shared_y : share the y-axis across columns.
    figsize  : explicit (width, height) in inches; overrides automatic choice.
    Returns
    -------
    (fig, axes) where axes is a single Axes for 1x1, or an ndarray otherwise.
    """

    rows, cols = (int(x) for x in layout.split("x"))

    if figsize is None:
        if rows == 1 and cols == 1:
            figsize = FIGURE_SIZES["double_col"]
        elif rows == 2 and cols == 1:
            figsize = FIGURE_SIZES["wide_2panel"]
        elif rows == 3 and cols == 1:
            figsize = FIGURE_SIZES["3panel"]
        elif rows == 4 and cols == 1:
            figsize = FIGURE_SIZES["4panel"]

        else:
            figsize = FIGURE_SIZES["standard"]
    fig, axes = plt.subplots(
        rows, cols,
        figsize=figsize,
        sharex=shared_x,
        sharey=shared_y,
        constrained_layout=True,
    )

    return fig, axes

# =============================================================================
# 7.  AXIS FORMATTING HELPERS

# =============================================================================
def format_time_axis(ax, unit="days", epoch_label=None):
    """
    Apply a descriptive, unit-consistent x-axis label for time-domain plots.

    Parameters
    ----------
    ax          : target Axes.
    unit        : "s", "min", "h", "days", or "datetime".
    epoch_label : optional epoch string appended to the label.
    """

    unit_map = {
        "s":        "Time since epoch [s]",
        "min":      "Time since epoch [min]",
        "h":        "Elapsed time [h]",
        "days":     "Elapsed time [days]",
        "datetime": "Date [UTC]",
    }

    label = unit_map.get(unit, f"Time [{unit}]")
    if epoch_label and unit != "datetime":
        label = f"{label}   (epoch: {epoch_label})"

    ax.set_xlabel(label)

def apply_panel_label(ax, letter, x=-0.11, y=1.02):
    """
    Add a bold panel identifier (a), (b), (c) ... in the top-left corner.
    """

    ax.text(
        x, y, f"({letter})",
        transform=ax.transAxes,
        fontsize=9.0,
        fontweight="bold",
        va="bottom",
        ha="right",
        clip_on=False,
    )

def set_sci_formatter(ax, axis="y"):
    """Enable offset scientific notation on the specified axis."""

    from matplotlib.ticker import ScalarFormatter

    fmt = ScalarFormatter(useMathText=True)

    fmt.set_scientific(True)

    fmt.set_powerlimits((-3, 4))

    if axis == "y":
        ax.yaxis.set_major_formatter(fmt)

    else:
        ax.xaxis.set_major_formatter(fmt)

# =============================================================================
# 8.  ERROR / RESIDUAL HELPERS

# =============================================================================
def add_zero_line(ax, **kwargs):
    """Draw a subtle horizontal zero-reference line."""

    defaults = dict(
        color=COLORS["zero_line"],
        linewidth=LINE_WIDTHS["zero"],
        linestyle="--",
        zorder=1,
        label="_nolegend_",
    )

    defaults.update(kwargs)

    ax.axhline(0.0, **defaults)

def add_threshold_lines(ax, lower=None, upper=None,
                        lower_label=None, upper_label=None):
    """
    Draw horizontal tolerance / threshold lines on ax.

    Parameters
    ----------
    ax           : target Axes.
    lower        : y-value for the lower threshold.
    upper        : y-value for the upper threshold.
    lower_label  : legend label for the lower threshold line.
    upper_label  : legend label for the upper threshold line.
    """

    if lower is not None:
        lbl = lower_label or f"Lower threshold ({lower:.0f})"
        ax.axhline(lower, color=COLORS["threshold_lower"],
                   linewidth=LINE_WIDTHS["threshold"],
                   linestyle="--", zorder=2, label=lbl)

    if upper is not None:
        lbl = upper_label or f"Upper threshold ({upper:.0f})"
        ax.axhline(upper, color=COLORS["threshold_upper"],
                   linewidth=LINE_WIDTHS["threshold"],
                   linestyle="--", zorder=2, label=lbl)

def plot_error_band(ax, x, y_low, y_high, color=None, alpha=BAND_ALPHA, **kwargs):
    """
    Draw a semi-transparent uncertainty band between y_low and y_high.

    Parameters
    ----------
    ax     : target Axes.
    x      : x-axis data.
    y_low  : lower bound of the band.
    y_high : upper bound of the band.
    color  : fill color; defaults to COLORS["band_fill"].
    alpha  : transparency level.
    """

    if color is None:
        color = COLORS["band_fill"]
    ax.fill_between(x, y_low, y_high, color=color, alpha=alpha, **kwargs)

def add_error_stats(ax, errors, loc="upper right", prefix=""):
    """
    Annotate an error panel with MAE, RMS, and peak absolute error.

    Parameters
    ----------
    ax     : target Axes.
    errors : sequence of signed error values (NaN ignored).
    loc    : "upper right", "upper left", "lower right", or "lower left".
    prefix : optional string prepended to each stat line.
    """

    arr = np.asarray(
        [e for e in errors if not (isinstance(e, float) and np.isnan(e))],
        dtype=float,
    )

    if arr.size == 0:
        return

    mae  = float(np.mean(np.abs(arr)))
    rms  = float(np.sqrt(np.mean(arr ** 2)))
    peak = float(np.max(np.abs(arr)))
    text = f"{prefix}MAE  = {mae:.4g}\nRMS  = {rms:.4g}\nPeak = {peak:.4g}"
    loc_coords = {
        "upper right": (0.98, 0.97),
        "upper left":  (0.02, 0.97),
        "lower right": (0.98, 0.03),
        "lower left":  (0.02, 0.03),
    }

    xy = loc_coords.get(loc, (0.98, 0.97))
    ha = "right" if "right" in loc else "left"
    va = "top"   if "upper" in loc else "bottom"

    ax.text(
        *xy, text,
        transform=ax.transAxes,
        fontsize=7.0,
        ha=ha, va=va,
        family="monospace",
        bbox=dict(
            boxstyle="round,pad=0.30",
            facecolor="white",
            edgecolor=COLORS["gray_light"],
            alpha=0.92,
            linewidth=0.6,
        ),
        zorder=5,
    )

# =============================================================================
# 9.  LEGEND HELPERS

# =============================================================================
def tidy_legend(ax, loc="best", outside=False, ncol=1):
    """
    Place a clean, publication-ready legend on ax.

    Parameters
    ----------
    ax      : target Axes.
    loc     : legend location string (ignored when outside=True).
    outside : if True, places the legend to the right of the axes area.
    ncol    : number of legend columns.
    """

    kw = dict(
        fontsize=7.5,
        framealpha=0.88,
        edgecolor=COLORS["gray_light"],
        ncol=ncol,
        handlelength=1.6,
        handletextpad=0.4,
        borderpad=0.4,
    )

    if outside:
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0),
                  borderaxespad=0.0, **kw)

    else:
        ax.legend(loc=loc, **kw)

def deduplicate_legend(ax, **legend_kwargs):
    """
    Remove duplicate legend entries (same label) and draw the legend.
    """

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    defaults = dict(fontsize=7.5, edgecolor=COLORS["gray_light"],
                    framealpha=0.88, handlelength=1.6)

    defaults.update(legend_kwargs)

    ax.legend(by_label.values(), by_label.keys(), **defaults)

# =============================================================================
# 10.  FIGURE EXPORT

# =============================================================================
def save_figure(fig, filename, output_dir=None, formats=("pdf", "png"),
                dpi=300, close=False):
    """
    Export a figure to one or more file formats.

    Parameters
    ----------
    fig        : Matplotlib Figure to save.
    filename   : base filename without extension.
    output_dir : destination directory; created automatically if absent.
    formats    : iterable of format strings: "pdf", "png", "svg".
    dpi        : raster resolution (ignored for PDF and SVG).
    close      : if True, close the figure after saving.
    Returns
    -------
    List of absolute file paths that were written.
    """

    if output_dir is None:
        output_dir = os.getcwd()
    os.makedirs(output_dir, exist_ok=True)
    saved = []

    for fmt in formats:
        path = os.path.join(output_dir, f"{filename}.{fmt}")
        save_dpi = dpi if fmt not in ("pdf", "svg") else None

        fig.savefig(
            path,
            format=fmt,
            dpi=save_dpi,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )

        saved.append(os.path.abspath(path))

    if close:
        plt.close(fig)

    return saved

# =============================================================================
# 11.  SELF-TEST

# =============================================================================
if __name__ == "__main__":
    print("orbital_plot_style -- self-test\n")
    print(f"  Font family   : {mpl.rcParams['font.family']}")
    print(f"  Mathtext set  : {mpl.rcParams['mathtext.fontset']}")
    print(f"  Grid color    : {mpl.rcParams['grid.color']}")
    print(f"  Tick direction: {mpl.rcParams['xtick.direction']}")
    print(f"  Export dpi    : {mpl.rcParams['savefig.dpi']}")
    print(f"  Models        : {', '.join(list(MODEL_STYLES.keys())[:4])} ...")
    print(f"  Figure sizes  : {', '.join(list(FIGURE_SIZES.keys()))}")
    print()
    rng = np.random.default_rng(42)
    t   = np.linspace(0, 30, 600)
    err_nrl = 2.5 * np.exp(-0.04 * t) * np.sin(0.9 * t) + 0.4 * rng.standard_normal(600)
    err_jb  = 1.8 * np.exp(-0.05 * t) * np.cos(0.7 * t) + 0.3 * rng.standard_normal(600)
    fig, axes = make_figure("2x1", shared_x=True, figsize=FIGURE_SIZES["wide_2panel"])
    fig.suptitle("orbital_plot_style -- self-test figure")
    ax0, ax1 = axes
    ax0.plot(t, err_nrl, **plot_kwargs("nrlmsise00"))
    plot_error_band(ax0, t, err_nrl - 0.6, err_nrl + 0.6, color=band_color("nrlmsise00"))
    add_zero_line(ax0)
    add_threshold_lines(ax0, lower=-2.0, upper=2.0,
                        lower_label="-2 km tolerance", upper_label="+2 km tolerance")

    ax0.set_ylabel("Altitude error [km]")
    add_error_stats(ax0, err_nrl)
    tidy_legend(ax0)
    apply_panel_label(ax0, "a")
    ax1.plot(t, err_jb, **plot_kwargs("jb2008"))
    add_zero_line(ax1)
    ax1.set_ylabel(r"$\Delta i$ [deg]")
    format_time_axis(ax1, unit="days")
    add_error_stats(ax1, err_jb)
    tidy_legend(ax1)
    apply_panel_label(ax1, "b")

    out = save_figure(fig, "orbital_style_selftest", output_dir=".", formats=("png",))

    print(f"  Demo figure saved to: {out[0]}")
    plt.show()
