"""Figures for each pipeline stage. Only plotting here, no analysis."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

if TYPE_CHECKING:
    from roofsolar.pipeline import TerrainResult

DPI = 130


def _extent(shape: tuple[int, int], pixel: tuple[float, float]) -> tuple[float, ...]:
    """Image extent in local metres from the AOI's top-left corner (north up)."""
    rows, cols = shape
    return (0, cols * pixel[0], rows * pixel[1], 0)


def _style(ax, title: str) -> None:
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("east (m)")
    ax.set_ylabel("south (m)")
    ax.annotate(
        "N",
        xy=(0.95, 0.95),
        xytext=(0.95, 0.83),
        xycoords="axes fraction",
        ha="center",
        fontsize=10,
        fontweight="bold",
        arrowprops=dict(arrowstyle="-|>", color="k"),
    )


def _aoi_outline(ax, aoi: np.ndarray, ext) -> None:
    ax.contour(
        aoi.astype(float),
        levels=[0.5],
        colors="yellow",
        linewidths=1.2,
        extent=(ext[0], ext[1], ext[3], ext[2]),
        origin="upper",
    )


def _compass_wheel(fig, rect, cmap) -> None:
    """Small colour wheel showing which hue means which facing direction."""
    ax = fig.add_axes(rect, projection="polar")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    theta = np.linspace(0, 2 * np.pi, 361)
    r = np.linspace(0.5, 1, 2)
    t, _ = np.meshgrid(theta, r)
    ax.pcolormesh(theta, r, np.degrees(t), cmap=cmap, vmin=0, vmax=360, shading="auto")
    ax.set_yticks([])
    ax.set_xticks(np.radians([0, 90, 180, 270]), ["N", "E", "S", "W"], fontsize=8)
    ax.spines["polar"].set_visible(False)


def _panel_elevation(ax, res, hs, ext):
    ax.imshow(np.where(res.aoi, hs, np.nan), cmap="gray", extent=ext)
    im = ax.imshow(res.elevation.data, cmap="terrain", alpha=0.6, extent=ext)
    _style(ax, "Elevation (m a.s.l.) over hillshade")
    return im


def _panel_slope(ax, res, ext):
    im = ax.imshow(res.slope.data, cmap="magma", vmin=0, vmax=90, extent=ext)
    _style(ax, "Slope (deg)")
    return im


def _panel_aspect(ax, res, ext, flat_deg):
    ax.imshow(
        np.where(res.aoi & np.isnan(res.aspect.data), 0.85, np.nan),
        cmap="gray",
        vmin=0,
        vmax=1,
        extent=ext,
    )
    im = ax.imshow(
        res.aspect.data, cmap="hsv", vmin=0, vmax=360, extent=ext, interpolation="nearest"
    )
    _style(ax, f"Aspect (deg from true N); grey = flat (< {flat_deg:g} deg)")
    return im


def _panel_ndsm(ax, res, ext):
    im = ax.imshow(
        res.ndsm.data,
        cmap="viridis",
        vmin=0,
        vmax=np.nanpercentile(res.ndsm.data, 99.5),
        extent=ext,
    )
    _style(ax, "Height above ground (m)")
    return im


def plot_terrain(
    res: TerrainResult, hs: np.ndarray, roof_px: np.ndarray, ground_px: np.ndarray, out: Path
) -> None:
    """Write all step-1 figures."""
    pixel = res.slope.pixel_size
    ext = _extent(res.aoi.shape, pixel)
    flat_deg = res.stats["flat_slope_threshold_deg"]

    # 01 inputs
    fig, ax = plt.subplots(1, 2, figsize=(15, 6.5))
    ax[0].imshow(res.rgb, extent=ext)
    _aoi_outline(ax[0], res.aoi, ext)
    _style(ax[0], "RGB with area of interest (yellow)")
    fig.colorbar(_panel_elevation(ax[1], res, hs, ext), ax=ax[1], shrink=0.8)
    fig.tight_layout()
    fig.savefig(out / "01_inputs_elevation.png", dpi=DPI)
    plt.close(fig)

    # 02 slope
    fig, ax = plt.subplots(figsize=(9, 7))
    fig.colorbar(_panel_slope(ax, res, ext), ax=ax, shrink=0.8, label="deg")
    fig.tight_layout()
    fig.savefig(out / "02_slope.png", dpi=DPI)
    plt.close(fig)

    # 03 aspect
    fig, ax = plt.subplots(figsize=(9, 7))
    _panel_aspect(ax, res, ext, flat_deg)
    fig.tight_layout()
    _compass_wheel(fig, [0.78, 0.12, 0.17, 0.17], "hsv")
    fig.savefig(out / "03_aspect.png", dpi=DPI)
    plt.close(fig)

    # 04 height above ground
    fig, ax = plt.subplots(figsize=(9, 7))
    fig.colorbar(_panel_ndsm(ax, res, ext), ax=ax, shrink=0.8, label="m")
    fig.tight_layout()
    fig.savefig(out / "04_height_above_ground.png", dpi=DPI)
    plt.close(fig)

    # 05 roof diagnostics: aspect rose, slope histogram, ground slope noise
    fig = plt.figure(figsize=(16, 5))
    axr = fig.add_subplot(1, 3, 1, projection="polar")
    asp = res.aspect.data[roof_px]
    asp = asp[~np.isnan(asp)]
    counts, edges = np.histogram(asp, bins=72, range=(0, 360))
    axr.bar(
        np.radians(edges[:-1] + 2.5),
        counts,
        width=np.radians(5),
        color=plt.cm.hsv(edges[:-1] / 360),
        edgecolor="k",
        linewidth=0.3,
    )
    axr.set_theta_zero_location("N")
    axr.set_theta_direction(-1)
    axr.set_yticklabels([])
    for p in res.stats["roof_candidates"]["aspect_peaks_deg"]:
        axr.annotate(
            f"{p['azimuth_deg']:.0f}°",
            xy=(np.radians(p["azimuth_deg"]), counts.max()),
            ha="center",
            fontsize=9,
            fontweight="bold",
        )
    axr.set_title("Roof-candidate aspect (true N)", pad=18)

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.hist(res.slope.data[roof_px], bins=60, range=(0, 90), color="tab:orange")
    ax2.axvline(
        res.stats["roof_candidates"]["slope_deg"]["median"],
        color="k",
        ls="--",
        label=f"median {res.stats['roof_candidates']['slope_deg']['median']:.1f}°",
    )
    ax2.set(xlabel="slope (deg)", ylabel="pixels", title="Roof-candidate slope")
    ax2.legend()

    ax3 = fig.add_subplot(1, 3, 3)
    gs = res.slope.data[ground_px]
    ax3.hist(gs[~np.isnan(gs)], bins=60, range=(0, 15), color="tab:green")
    ax3.axvline(flat_deg, color="r", ls="--", label=f"flat threshold {flat_deg:g}°")
    ax3.axvline(
        res.stats["flat_ground_slope_deg"]["p95"],
        color="k",
        ls=":",
        label=f"ground p95 {res.stats['flat_ground_slope_deg']['p95']:.1f}°",
    )
    ax3.set(xlabel="slope (deg)", ylabel="pixels", title="Slope on flat ground (noise)")
    ax3.legend()
    fig.tight_layout()
    fig.savefig(out / "05_roof_diagnostics.png", dpi=DPI)
    plt.close(fig)

    # 00 summary (for README / video)
    fig, ax = plt.subplots(2, 3, figsize=(20, 12))
    ax[0, 0].imshow(res.rgb, extent=ext)
    _aoi_outline(ax[0, 0], res.aoi, ext)
    _style(ax[0, 0], "RGB + AOI")
    fig.colorbar(_panel_elevation(ax[0, 1], res, hs, ext), ax=ax[0, 1], shrink=0.7)
    fig.colorbar(_panel_ndsm(ax[0, 2], res, ext), ax=ax[0, 2], shrink=0.7)
    fig.colorbar(_panel_slope(ax[1, 0], res, ext), ax=ax[1, 0], shrink=0.7)
    _panel_aspect(ax[1, 1], res, ext, flat_deg)
    ax[1, 2].imshow(np.where(res.aoi, hs, np.nan), cmap="gray", extent=ext)
    ax[1, 2].imshow(np.where(roof_px, 1.0, np.nan), cmap="autumn", alpha=0.5, extent=ext)
    _style(ax[1, 2], "Roof candidates (preview, refined in step 3)")
    fig.tight_layout()
    fig.savefig(out / "00_summary.png", dpi=100)
    plt.close(fig)


def plot_provenance(res: TerrainResult, off_grid: np.ndarray, out: Path) -> None:
    """Show where the provider re-processed the DSM (heights off the 1 cm grid)."""
    ext = _extent(res.aoi.shape, res.slope.pixel_size)
    fig, ax = plt.subplots(1, 2, figsize=(15, 6.5))
    ax[0].imshow(res.rgb, extent=ext)
    ax[0].imshow(np.where(off_grid, 1.0, np.nan), cmap="autumn", alpha=0.45, extent=ext)
    _aoi_outline(ax[0], res.aoi, ext)
    _style(ax[0], "Re-processed DSM regions (heights not on 1 cm grid)")
    frac = (res.elevation.data * 100) % 1
    ax[1].imshow(
        np.where(res.aoi, np.minimum(frac, 1 - frac), np.nan),
        cmap="magma",
        vmin=0,
        vmax=0.5,
        extent=ext,
        interpolation="nearest",
    )
    _style(ax[1], "Distance of each height from the 1 cm grid (cm)")
    fig.tight_layout()
    fig.savefig(out / "06_data_provenance.png", dpi=DPI)
    plt.close(fig)
