from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dataloader.schema import Sample


def _safe_title_channel(channels: list[str], idx: int) -> str:
    if 0 <= idx < len(channels):
        return channels[idx]
    return f"channel_{idx}"


def _feature_colorbar_label(channel_name: str) -> str:
    key = str(channel_name).strip().lower()
    mapping = {
        "frp": "Fire radiative power (MW)",
        "count": "Detection count per grid cell",
        "noaa_flood": "Flood indicator / event intensity",
        "t2m": "2 m temperature",
        "d2m": "2 m dew point temperature",
        "u10": "10 m zonal wind",
        "v10": "10 m meridional wind",
        "sp": "Surface pressure",
        "swvl1": "Volumetric soil water (layer 1)",
        "landfire_fuel": "Fuel model code",
    }
    return mapping.get(key, f"{channel_name} intensity")


def _sample_bbox(sample: Sample) -> Optional[tuple[float, float, float, float]]:
    cfg = sample.meta.get("config", {})
    grid_cfg = cfg.get("grid", {}) if isinstance(cfg, dict) else {}
    bbox = grid_cfg.get("bbox") if isinstance(grid_cfg, dict) else None
    if bbox is None or len(bbox) != 4:
        return None
    return (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))


def _time_label(sample: Sample, time_index: int) -> str:
    cfg = sample.meta.get("config", {})
    time_cfg = cfg.get("time", {}) if isinstance(cfg, dict) else {}
    start = time_cfg.get("start_date") if isinstance(time_cfg, dict) else None
    cadence = time_cfg.get("frequency") if isinstance(time_cfg, dict) else None
    if start is None:
        return f"t={time_index}"
    try:
        ts = pd.date_range(start=start, periods=time_index + 1, freq=cadence)[time_index]
        return str(ts)
    except Exception:
        return f"t={time_index} | start={start} | cadence={cadence}"


def infer_observed_and_synthetic_indices(
    sample: Sample,
    channel_index: int = 0,
    feature_name: Optional[str] = None,
) -> tuple[int, int]:
    """Return one observed and one synthetic time index for a selected channel."""
    x = sample.x
    channels = list(sample.meta.get("channels", []))
    x_mask = sample.meta.get("x_synthetic_mask")

    if x.ndim != 4 or x.shape[1] == 0:
        raise ValueError("Sample does not contain visualizable feature channels.")
    if feature_name is not None and feature_name in channels:
        channel_index = channels.index(feature_name)
    if not isinstance(x_mask, np.ndarray) or x_mask.size == 0:
        raise ValueError("x_synthetic_mask is required.")

    channel_mask = x_mask[:, channel_index]
    observed_candidates = np.where(~channel_mask)[0]
    synthetic_candidates = np.where(channel_mask)[0]
    if observed_candidates.size == 0:
        raise ValueError("No observed time slice available for the selected channel.")
    if synthetic_candidates.size == 0:
        raise ValueError("No synthetic time slice available for the selected channel.")
    return int(observed_candidates[0]), int(synthetic_candidates[0])


def infer_observed_and_synthetic_pairs(
    sample: Sample,
    channel_index: int = 0,
    feature_name: Optional[str] = None,
    max_pairs: int = 3,
) -> list[tuple[int, int, int]]:
    """Return multiple (previous_observed_idx, synthetic_idx, next_observed_idx) tuples.

    Each synthetic index is matched to the nearest previous and next observed time indices.
    """
    x = sample.x
    channels = list(sample.meta.get("channels", []))
    x_mask = sample.meta.get("x_synthetic_mask")

    if x.ndim != 4 or x.shape[1] == 0:
        raise ValueError("Sample does not contain visualizable feature channels.")
    if feature_name is not None and feature_name in channels:
        channel_index = channels.index(feature_name)
    if not isinstance(x_mask, np.ndarray) or x_mask.size == 0:
        raise ValueError("x_synthetic_mask is required.")

    channel_mask = x_mask[:, channel_index]
    observed = np.where(~channel_mask)[0]
    synthetic = np.where(channel_mask)[0]
    if observed.size == 0:
        raise ValueError("No observed time slice available for the selected channel.")
    if synthetic.size == 0:
        raise ValueError("No synthetic time slice available for the selected channel.")

    pairs: list[tuple[int, int, int]] = []
    for syn_idx in synthetic:
        previous_observed = observed[observed < syn_idx]
        next_observed = observed[observed > syn_idx]
        if previous_observed.size == 0 or next_observed.size == 0:
            continue
        prev_obs = int(previous_observed[-1])
        next_obs = int(next_observed[0])
        pairs.append((prev_obs, int(syn_idx), next_obs))
        if len(pairs) >= max_pairs:
            break
    if not pairs:
        raise ValueError("No synthetic slice has both previous and next observed references.")
    return pairs


def _set_geo_axes(ax, bbox: Optional[tuple[float, float, float, float]]) -> None:
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    if bbox is not None:
        ax.set_xlim(bbox[0], bbox[2])
        ax.set_ylim(bbox[1], bbox[3])


def _image_extent(bbox: Optional[tuple[float, float, float, float]]) -> Optional[list[float]]:
    if bbox is None:
        return None
    return [bbox[0], bbox[2], bbox[1], bbox[3]]


def save_sample_overview_png(
    sample: Sample,
    output_path: str,
    time_index: int = 0,
    channel_index: int = 0,
    feature_name: Optional[str] = None,
) -> str:
    """Save a compact visualization of one sample time slice.

    Layout:
    - selected feature map from x
    - label grid y
    - x synthetic mask for selected channel
    - y synthetic mask
    """
    out = Path(output_path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    x = sample.x
    y = sample.y
    channels = list(sample.meta.get("channels", []))
    x_mask = sample.meta.get("x_synthetic_mask")
    y_mask = sample.meta.get("y_synthetic_mask")

    if x.ndim != 4:
        raise ValueError(f"Expected x to have shape (T, C, H, W), got {x.shape}")
    if y.ndim != 3:
        raise ValueError(f"Expected y to have shape (T, H, W), got {y.shape}")
    if not (0 <= time_index < x.shape[0]):
        raise IndexError(f"time_index out of range: {time_index}")

    if feature_name is not None and feature_name in channels:
        channel_index = channels.index(feature_name)
    if x.shape[1] == 0:
        channel_index = 0
    elif not (0 <= channel_index < x.shape[1]):
        raise IndexError(f"channel_index out of range: {channel_index}")

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    bbox = _sample_bbox(sample)
    extent = _image_extent(bbox)
    fig.suptitle(f"Sample overview | {_time_label(sample, time_index)}", fontsize=14)

    if x.shape[1] > 0:
        feature = x[time_index, channel_index]
        im0 = axes[0, 0].imshow(feature, cmap="viridis", extent=extent, origin="lower", aspect="auto")
        fig.colorbar(
            im0,
            ax=axes[0, 0],
            fraction=0.046,
            pad=0.04,
            label=_feature_colorbar_label(_safe_title_channel(channels, channel_index)),
        )
        axes[0, 0].set_title(f"Feature map: {_safe_title_channel(channels, channel_index)}")
    else:
        axes[0, 0].text(0.5, 0.5, "No feature channels", ha="center", va="center")
        axes[0, 0].set_title("x feature")
    _set_geo_axes(axes[0, 0], bbox)

    im1 = axes[0, 1].imshow(
        y[time_index],
        cmap="magma",
        vmin=np.min(y),
        vmax=np.max(y) if np.max(y) > 0 else 1,
        extent=extent,
        origin="lower",
        aspect="auto",
    )
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04, label="Label code")
    axes[0, 1].set_title("Label grid")
    _set_geo_axes(axes[0, 1], bbox)

    if isinstance(x_mask, np.ndarray) and x_mask.size > 0 and x.shape[1] > 0:
        x_syn = np.full((y.shape[1], y.shape[2]), float(x_mask[time_index, channel_index]), dtype=float)
        im2 = axes[1, 0].imshow(x_syn, cmap="gray", vmin=0, vmax=1, extent=extent, origin="lower", aspect="auto")
        fig.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04, label="0=observed, 1=synthetic")
        axes[1, 0].set_title("x synthetic mask")
    else:
        axes[1, 0].text(0.5, 0.5, "No x synthetic mask", ha="center", va="center")
        axes[1, 0].set_title("x synthetic mask")
    _set_geo_axes(axes[1, 0], bbox)

    if isinstance(y_mask, np.ndarray) and y_mask.size > 0:
        im3 = axes[1, 1].imshow(
            y_mask[time_index].astype(float), cmap="gray", vmin=0, vmax=1, extent=extent, origin="lower", aspect="auto"
        )
        fig.colorbar(im3, ax=axes[1, 1], fraction=0.046, pad=0.04, label="0=observed, 1=synthetic")
        axes[1, 1].set_title("y synthetic mask")
    else:
        axes[1, 1].text(0.5, 0.5, "No y synthetic mask", ha="center", va="center")
        axes[1, 1].set_title("y synthetic mask")
    _set_geo_axes(axes[1, 1], bbox)

    fig.text(
        0.5,
        0.01,
        "Top row shows the selected feature and label grid. Bottom row shows whether values are observed or synthetic.",
        ha="center",
        fontsize=9,
    )
    plt.tight_layout(rect=[0, 0.04, 1, 0.97])
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return str(out)


def save_feature_label_png(
    sample: Sample,
    output_path: str,
    time_index: int = 0,
    feature_name: Optional[str] = None,
    channel_index: int = 0,
) -> str:
    """Save a two-panel visualization: one feature map and one label grid."""
    out = Path(output_path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    x = sample.x
    y = sample.y
    channels = list(sample.meta.get("channels", []))
    if feature_name is not None and feature_name in channels:
        channel_index = channels.index(feature_name)
    if x.shape[1] == 0:
        raise ValueError("No feature channels available for visualization.")

    bbox = _sample_bbox(sample)
    extent = _image_extent(bbox)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))

    im0 = axes[0].imshow(x[time_index, channel_index], cmap="viridis", extent=extent, origin="lower", aspect="auto")
    fig.colorbar(
        im0,
        ax=axes[0],
        fraction=0.046,
        pad=0.04,
        label=_feature_colorbar_label(_safe_title_channel(channels, channel_index)),
    )
    axes[0].set_title(f"{_safe_title_channel(channels, channel_index)} | {_time_label(sample, time_index)}")

    im1 = axes[1].imshow(
        y[time_index],
        cmap="magma",
        vmin=np.min(y),
        vmax=np.max(y) if np.max(y) > 0 else 1,
        extent=extent,
        origin="lower",
        aspect="auto",
    )
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label="Label code")
    axes[1].set_title(f"Label grid | {_time_label(sample, time_index)}")

    for ax in axes:
        _set_geo_axes(ax, bbox)

    fig.text(
        0.5,
        0.01,
        "Left: selected feature intensity. Right: label ids over the same area and time step.",
        ha="center",
        fontsize=9,
    )
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return str(out)


def save_feature_only_png(
    sample: Sample,
    output_path: str,
    time_index: int = 0,
    feature_name: Optional[str] = None,
    channel_index: int = 0,
    title_prefix: str = "Feature map",
) -> str:
    """Save a single feature map with geographic axes and a colorbar."""
    out = Path(output_path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    x = sample.x
    channels = list(sample.meta.get("channels", []))
    if x.ndim != 4 or x.shape[1] == 0:
        raise ValueError("No feature channels available for visualization.")
    if feature_name is not None and feature_name in channels:
        channel_index = channels.index(feature_name)
    if not (0 <= channel_index < x.shape[1]):
        raise IndexError(f"channel_index out of range: {channel_index}")

    bbox = _sample_bbox(sample)
    extent = _image_extent(bbox)
    name = _safe_title_channel(channels, channel_index)

    fig, ax = plt.subplots(figsize=(7.5, 5.4))
    im = ax.imshow(x[time_index, channel_index], cmap="viridis", extent=extent, origin="lower", aspect="auto")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=_feature_colorbar_label(name))
    ax.set_title(f"{title_prefix}: {name} | {_time_label(sample, time_index)}")
    _set_geo_axes(ax, bbox)
    fig.text(0.5, 0.01, "Single-channel geographic visualization for one time step.", ha="center", fontsize=9)
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return str(out)


def save_real_vs_synthetic_png(
    sample: Sample,
    output_path: str,
    channel_index: int = 0,
    feature_name: Optional[str] = None,
    observed_time_index: Optional[int] = None,
    synthetic_time_index: Optional[int] = None,
) -> str:
    """Save a side-by-side comparison between an observed and synthetic time slice."""
    out = Path(output_path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    x = sample.x
    channels = list(sample.meta.get("channels", []))
    x_mask = sample.meta.get("x_synthetic_mask")

    if x.ndim != 4 or x.shape[1] == 0:
        raise ValueError("Sample does not contain visualizable feature channels.")
    if feature_name is not None and feature_name in channels:
        channel_index = channels.index(feature_name)

    if not isinstance(x_mask, np.ndarray) or x_mask.size == 0:
        raise ValueError("x_synthetic_mask is required for real-vs-synthetic visualization.")

    if observed_time_index is None:
        observed_time_index, inferred_synth = infer_observed_and_synthetic_indices(
            sample, channel_index=channel_index
        )
        if synthetic_time_index is None:
            synthetic_time_index = inferred_synth
    elif synthetic_time_index is None:
        _, synthetic_time_index = infer_observed_and_synthetic_indices(
            sample, channel_index=channel_index
        )

    bbox = _sample_bbox(sample)
    extent = _image_extent(bbox)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    name = _safe_title_channel(channels, channel_index)

    im0 = axes[0].imshow(x[observed_time_index, channel_index], cmap="viridis", extent=extent, origin="lower", aspect="auto")
    axes[0].set_title(f"Observed | {name} | {_time_label(sample, observed_time_index)}")
    fig.colorbar(
        im0,
        ax=axes[0],
        fraction=0.046,
        pad=0.04,
        label=_feature_colorbar_label(name),
    )

    im1 = axes[1].imshow(x[synthetic_time_index, channel_index], cmap="viridis", extent=extent, origin="lower", aspect="auto")
    axes[1].set_title(f"Synthetic | {name} | {_time_label(sample, synthetic_time_index)}")
    fig.colorbar(
        im1,
        ax=axes[1],
        fraction=0.046,
        pad=0.04,
        label=_feature_colorbar_label(name),
    )

    diff = x[synthetic_time_index, channel_index] - x[observed_time_index, channel_index]
    vmax = np.max(np.abs(diff)) if np.max(np.abs(diff)) > 0 else 1.0
    im2 = axes[2].imshow(diff, cmap="coolwarm", vmin=-vmax, vmax=vmax, extent=extent, origin="lower", aspect="auto")
    axes[2].set_title("Difference | synthetic - observed")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04, label="Change relative to observed field")

    for ax in axes:
        _set_geo_axes(ax, bbox)

    fig.text(
        0.5,
        0.01,
        "Synthetic slices may appear empty when no event signal is present and the resampling rule yields zero-valued cells.",
        ha="center",
        fontsize=9,
    )
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return str(out)


def save_real_vs_synthetic_pairs_png(
    sample: Sample,
    output_path: str,
    channel_index: int = 0,
    feature_name: Optional[str] = None,
    max_pairs: int = 3,
) -> str:
    """Save multiple observed-vs-synthetic comparison pairs in one figure."""
    out = Path(output_path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    x = sample.x
    channels = list(sample.meta.get("channels", []))
    if x.ndim != 4 or x.shape[1] == 0:
        raise ValueError("Sample does not contain visualizable feature channels.")
    if feature_name is not None and feature_name in channels:
        channel_index = channels.index(feature_name)

    pairs = infer_observed_and_synthetic_pairs(
        sample,
        channel_index=channel_index,
        feature_name=feature_name,
        max_pairs=max_pairs,
    )
    bbox = _sample_bbox(sample)
    extent = _image_extent(bbox)
    name = _safe_title_channel(channels, channel_index)

    nrows = len(pairs)
    fig, axes = plt.subplots(nrows, 4, figsize=(18, 4.6 * nrows))
    if nrows == 1:
        axes = np.array([axes])

    for row_idx, (prev_observed_idx, synthetic_time_index, next_observed_idx) in enumerate(pairs):
        row_axes = axes[row_idx]

        im0 = row_axes[0].imshow(
            x[prev_observed_idx, channel_index], cmap="viridis", extent=extent, origin="lower", aspect="auto"
        )
        row_axes[0].set_title(f"Previous observed | {name} | {_time_label(sample, prev_observed_idx)}")
        fig.colorbar(
            im0,
            ax=row_axes[0],
            fraction=0.046,
            pad=0.04,
            label=_feature_colorbar_label(name),
        )

        im1 = row_axes[1].imshow(
            x[synthetic_time_index, channel_index], cmap="viridis", extent=extent, origin="lower", aspect="auto"
        )
        row_axes[1].set_title(f"Synthetic | {name} | {_time_label(sample, synthetic_time_index)}")
        fig.colorbar(
            im1,
            ax=row_axes[1],
            fraction=0.046,
            pad=0.04,
            label=_feature_colorbar_label(name),
        )

        im2 = row_axes[2].imshow(
            x[next_observed_idx, channel_index], cmap="viridis", extent=extent, origin="lower", aspect="auto"
        )
        row_axes[2].set_title(f"Next observed | {name} | {_time_label(sample, next_observed_idx)}")
        fig.colorbar(
            im2,
            ax=row_axes[2],
            fraction=0.046,
            pad=0.04,
            label=_feature_colorbar_label(name),
        )

        diff = x[synthetic_time_index, channel_index] - x[prev_observed_idx, channel_index]
        vmax = np.max(np.abs(diff)) if np.max(np.abs(diff)) > 0 else 1.0
        im3 = row_axes[3].imshow(
            diff, cmap="coolwarm", vmin=-vmax, vmax=vmax, extent=extent, origin="lower", aspect="auto"
        )
        row_axes[3].set_title("Difference | synthetic - previous")
        fig.colorbar(im3, ax=row_axes[3], fraction=0.046, pad=0.04, label="Change relative to previous observed field")

        for ax in row_axes:
            _set_geo_axes(ax, bbox)

    fig.text(
        0.5,
        0.01,
        "Each row shows previous observed, synthetic target, next observed, and the difference between synthetic and previous observed. Empty synthetic maps can be valid when the imputed event field is zero.",
        ha="center",
        fontsize=9,
    )
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return str(out)


def save_points_png(
    points: pd.DataFrame,
    output_path: str,
    bbox: Optional[Sequence[float]] = None,
    color_column: str = "frp",
    title: str = "Point observations",
) -> str:
    """Save a scatter plot for point-based geospatial observations."""
    out = Path(output_path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    if points.empty:
        raise ValueError("Point DataFrame is empty.")
    if "longitude" not in points.columns or "latitude" not in points.columns:
        raise ValueError("Point DataFrame must contain longitude and latitude columns.")

    fig, ax = plt.subplots(figsize=(8, 6.2))
    if color_column in points.columns:
        sc = ax.scatter(
            points["longitude"],
            points["latitude"],
            c=points[color_column],
            s=18,
            cmap="inferno",
            alpha=0.75,
            edgecolors="none",
        )
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label=color_column)
    else:
        ax.scatter(points["longitude"], points["latitude"], s=18, alpha=0.75, edgecolors="none")

    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    if bbox is not None and len(bbox) == 4:
        ax.set_xlim(float(bbox[0]), float(bbox[2]))
        ax.set_ylim(float(bbox[1]), float(bbox[3]))

    fig.text(
        0.5,
        0.01,
        "Each point is one observed detection. Color represents point intensity or magnitude.",
        ha="center",
        fontsize=9,
    )
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return str(out)
