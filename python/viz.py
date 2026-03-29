"""Visualization module for coffee extraction simulation results."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation

from .sim import SimulationResult


def plot_timeseries(result: SimulationResult, save_path: str | None = None):
    """Plot TDS, EY, flow rate, and temperature vs. time."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Coffee Extraction Simulation", fontsize=14)

    t = result.time

    # TDS
    ax = axes[0, 0]
    ax.plot(t, result.tds, "b-", linewidth=1.5, label="Instantaneous TDS")
    if hasattr(result, "cup_tds") and result.cup_tds:
        ax.plot(t, result.cup_tds, "b--", linewidth=1.2, alpha=0.7, label="Cup TDS")
        ax.legend(fontsize=8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("TDS (%)")
    ax.set_title("Total Dissolved Solids")
    ax.grid(True, alpha=0.3)

    # Extraction Yield
    ax = axes[0, 1]
    ax.plot(t, result.extraction_yield, "r-", linewidth=1.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("EY (%)")
    ax.set_title("Extraction Yield")
    ax.grid(True, alpha=0.3)

    # Flow Rate
    ax = axes[1, 0]
    ax.plot(t, result.flow_rate_ml_s, "g-", linewidth=1.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Flow Rate (mL/s)")
    ax.set_title("Flow Rate")
    ax.grid(True, alpha=0.3)

    # Temperature
    ax = axes[1, 1]
    ax.plot(t, result.avg_temperature_c, "orange", linewidth=1.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title("Average Bed Temperature")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close()


def plot_cross_section(
    field: np.ndarray,
    title: str = "Cross Section",
    slice_axis: int = 1,
    slice_index: int | None = None,
    cmap: str = "viridis",
    save_path: str | None = None,
):
    """Plot a 2D cross-section slice of a 3D field.

    Args:
        field: 3D numpy array.
        title: Plot title.
        slice_axis: Axis to slice along (0=x, 1=y, 2=z).
        slice_index: Index along the slice axis. Defaults to middle.
        cmap: Colormap name.
        save_path: Optional path to save figure.
    """
    if slice_index is None:
        slice_index = field.shape[slice_axis] // 2

    if slice_axis == 0:
        data = field[slice_index, :, :]
        xlabel, ylabel = "Y", "Z"
    elif slice_axis == 1:
        data = field[:, slice_index, :]
        xlabel, ylabel = "X", "Z"
    else:
        data = field[:, :, slice_index]
        xlabel, ylabel = "X", "Y"

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(data.T, origin="lower", cmap=cmap, aspect="auto")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.colorbar(im, ax=ax)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close()


def plot_extraction_uniformity(result: SimulationResult, save_path: str | None = None):
    """Top-down heatmap of extraction yield uniformity."""
    if result.extraction_yield_field is None:
        return

    ey = result.extraction_yield_field
    # Average along z-axis for top-down view
    ey_topdown = np.mean(ey, axis=2)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(ey_topdown.T, origin="lower", cmap="RdYlGn", aspect="equal")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Extraction Uniformity (Top-Down View)")
    plt.colorbar(im, ax=ax, label="Extraction Yield")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close()


def plot_brewing_chart(
    tds: float,
    ey: float,
    save_path: str | None = None,
):
    """Plot the final brew on an SCA-style brewing control chart.

    Args:
        tds: Final TDS percentage.
        ey: Final extraction yield percentage.
        save_path: Optional path to save figure.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # SCA ideal zone: EY 18-22%, TDS 1.15-1.45% (pourover)
    ideal = Rectangle((18, 1.15), 4, 0.30, alpha=0.2, color="green", label="SCA Ideal Zone")
    ax.add_patch(ideal)

    # Regions
    ax.axvline(18, color="gray", linestyle="--", alpha=0.3)
    ax.axvline(22, color="gray", linestyle="--", alpha=0.3)
    ax.axhline(1.15, color="gray", linestyle="--", alpha=0.3)
    ax.axhline(1.45, color="gray", linestyle="--", alpha=0.3)

    # Labels for regions
    ax.text(15.5, 1.55, "Under-extracted\nStrong", ha="center", va="center", fontsize=8, alpha=0.5)
    ax.text(20, 1.55, "Strong", ha="center", va="center", fontsize=8, alpha=0.5)
    ax.text(24.5, 1.55, "Over-extracted\nStrong", ha="center", va="center", fontsize=8, alpha=0.5)
    ax.text(15.5, 1.05, "Under-extracted\nWeak", ha="center", va="center", fontsize=8, alpha=0.5)
    ax.text(20, 1.05, "Weak", ha="center", va="center", fontsize=8, alpha=0.5)
    ax.text(24.5, 1.05, "Over-extracted\nWeak", ha="center", va="center", fontsize=8, alpha=0.5)

    # Plot the brew point
    ax.plot(ey, tds, "ro", markersize=12, label=f"Brew (TDS={tds:.2f}%, EY={ey:.1f}%)")

    ax.set_xlabel("Extraction Yield (%)")
    ax.set_ylabel("TDS (%)")
    ax.set_title("SCA Brewing Control Chart")
    ax.set_xlim(14, 26)
    ax.set_ylim(0.8, 1.7)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close()


def plot_3d_volume(
    field: np.ndarray,
    title: str = "3D Volume",
    inside_bed: np.ndarray | None = None,
    cmap: str = "hot",
    save_path: str | None = None,
):
    """Render a 3D volume visualization of a scalar field using PyVista.

    Args:
        field: 3D numpy array to visualize.
        title: Plot title.
        inside_bed: Optional boolean mask. If provided, only voxels where
            inside_bed is True are shown.
        cmap: Colormap name.
        save_path: Optional path to save a screenshot.
    """
    try:
        import pyvista as pv
    except ImportError:
        print("pyvista is required for 3D volume rendering. "
              "Install it with: pip install pyvista")
        return

    data = field.copy()
    if inside_bed is not None:
        data[~inside_bed] = np.nan

    grid = pv.ImageData(dimensions=np.array(data.shape) + 1)
    grid.cell_data["values"] = data.flatten(order="F")

    plotter = pv.Plotter()
    plotter.add_volume(
        grid,
        scalars="values",
        cmap=cmap,
        opacity="sigmoid",
    )
    plotter.add_title(title)

    if save_path:
        plotter.screenshot(save_path)
    else:
        plotter.show()


def animate_cross_section(
    snapshots: list[np.ndarray],
    title: str = "Concentration Evolution",
    slice_axis: int = 1,
    cmap: str = "hot",
    save_path: str = "animation.gif",
):
    """Animate a list of 3D snapshots as 2D cross-section slices.

    Args:
        snapshots: List of 3D numpy arrays (one per frame).
        title: Animation title.
        slice_axis: Axis to slice along (0=x, 1=y, 2=z).
        cmap: Colormap name.
        save_path: Path to save the animation GIF.
    """
    if not snapshots:
        print("No snapshots to animate.")
        return

    # Extract 2D slices
    def _get_slice(volume):
        idx = volume.shape[slice_axis] // 2
        if slice_axis == 0:
            return volume[idx, :, :]
        elif slice_axis == 1:
            return volume[:, idx, :]
        else:
            return volume[:, :, idx]

    slices = [_get_slice(s) for s in snapshots]

    # Compute global colorbar range
    vmin = min(s.min() for s in slices)
    vmax = max(s.max() for s in slices)
    if vmax == vmin:
        vmax = vmin + 1e-12

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(slices[0].T, origin="lower", cmap=cmap, aspect="auto",
                   vmin=vmin, vmax=vmax)
    ax.set_title(f"{title} (frame 0/{len(slices) - 1})")
    plt.colorbar(im, ax=ax)

    def _update(frame):
        im.set_data(slices[frame].T)
        ax.set_title(f"{title} (frame {frame}/{len(slices) - 1})")
        return [im]

    anim = FuncAnimation(fig, _update, frames=len(slices), interval=200, blit=True)
    anim.save(save_path, writer="pillow")
    plt.close()
    print(f"Animation saved to {save_path}")
