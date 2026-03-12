from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any
import json

import numpy as np

from dataloader.schema import Sample


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _label_legend(sample: Sample) -> list[str]:
    cfg = sample.meta.get("config", {})
    label_cfg = cfg.get("label", {}) if isinstance(cfg, dict) else {}
    mapping = label_cfg.get("label_mapping", {}) if isinstance(label_cfg, dict) else {}
    selected = label_cfg.get("label_hazards", None) if isinstance(label_cfg, dict) else None
    default_id = label_cfg.get("label_default_value", 0) if isinstance(label_cfg, dict) else 0

    legend = [f"{default_id}=default/background"]
    selected_set = {str(h).strip().lower() for h in selected or []}
    y_codes = {int(v) for v in np.unique(sample.y).tolist()} if sample.y.size else set()
    for hazard, code in mapping.items():
        hz = str(hazard).strip().lower()
        cid = int(code)
        if hz in selected_set or cid in y_codes:
            legend.append(f"{cid}={hz}")

    seen = set()
    return [item for item in legend if not (item in seen or seen.add(item))]


def inspect_sample(sample: Sample) -> dict[str, Any]:
    """Return a structured summary for one loaded sample."""
    x = sample.x
    y = sample.y
    meta = sample.meta
    channels = list(meta.get("channels", []))
    cfg = meta.get("config", {})
    label_cfg = cfg.get("label", {}) if isinstance(cfg, dict) else {}
    time_cfg = cfg.get("time", {}) if isinstance(cfg, dict) else {}
    grid_cfg = cfg.get("grid", {}) if isinstance(cfg, dict) else {}

    x_mask = meta.get("x_synthetic_mask")
    y_mask = meta.get("y_synthetic_mask")

    per_channel_synth: dict[str, float | None] = {}
    if isinstance(x_mask, np.ndarray) and x_mask.ndim == 2 and x_mask.shape[1] == len(channels):
        for i, name in enumerate(channels):
            per_channel_synth[name] = float(x_mask[:, i].mean())
    else:
        for name in channels:
            per_channel_synth[name] = None

    issues: list[str] = []
    if x.ndim == 4 and x.shape[1] == 0:
        issues.append("No usable feature channels were loaded.")
    if x.size == 0:
        issues.append("Feature tensor is empty.")
    if y.size == 0:
        issues.append("Label tensor is empty.")
    if isinstance(x_mask, np.ndarray) and x_mask.size and float(x_mask.mean()) > 0.8:
        issues.append("Synthetic coverage is high in x (>80%).")
    if isinstance(y_mask, np.ndarray) and y_mask.size and float(y_mask.mean()) > 0.8:
        issues.append("Synthetic coverage is high in y (>80%).")

    summary = {
        "x_shape": tuple(int(v) for v in x.shape),
        "y_shape": tuple(int(v) for v in y.shape),
        "channels": channels,
        "data_sources": cfg.get("sources", []),
        "label_source": label_cfg.get("label_source"),
        "target_hazards": label_cfg.get("label_hazards"),
        "label_legend": _label_legend(sample),
        "time_range": {
            "start": time_cfg.get("start_date"),
            "end": time_cfg.get("end_date"),
            "cadence": time_cfg.get("frequency"),
        },
        "bbox": grid_cfg.get("bbox"),
        "spatial_resolution_deg": grid_cfg.get("resolution_deg"),
        "x_synthetic_ratio": float(x_mask.mean()) if isinstance(x_mask, np.ndarray) and x_mask.size else None,
        "y_synthetic_ratio": float(y_mask.mean()) if isinstance(y_mask, np.ndarray) and y_mask.size else None,
        "per_channel_synthetic_ratio": per_channel_synth,
        "y_unique_values": [int(v) for v in np.unique(y).tolist()] if y.size else [],
        "issues": issues,
    }
    return summary


def print_sample_summary(sample: Sample) -> None:
    """Print a compact human-readable summary."""
    s = inspect_sample(sample)
    print("sample summary")
    print("sources:", s["data_sources"])
    print("x shape:", s["x_shape"])
    print("y shape:", s["y_shape"])
    print("channels:", s["channels"])
    print("label source:", s["label_source"])
    print("target hazards:", s["target_hazards"])
    print("label legend:", s["label_legend"])
    print("time range:", s["time_range"])
    print("bbox:", s["bbox"])
    print("spatial resolution (deg):", s["spatial_resolution_deg"])
    print("x synthetic ratio:", s["x_synthetic_ratio"])
    print("y synthetic ratio:", s["y_synthetic_ratio"])
    print("per-channel synthetic ratio:", s["per_channel_synthetic_ratio"])
    print("y unique values:", s["y_unique_values"])
    if s["issues"]:
        print("issues:")
        for item in s["issues"]:
            print("-", item)
    else:
        print("issues: none")


def save_sample_report_html(sample: Sample, output_path: str) -> str:
    """Save a one-page HTML report for a sample."""
    out = Path(output_path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    s = inspect_sample(sample)
    summary_json = json.dumps(_jsonable(s), indent=2)

    def li(items: list[str]) -> str:
        return "".join(f"<li>{escape(str(item))}</li>" for item in items)

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Sample Report</title>
  <style>
    :root {{
      --bg: #f5f1e8;
      --panel: #fffdf8;
      --ink: #1e2a2f;
      --muted: #5c6b70;
      --line: #d8d1c2;
      --accent: #146356;
      --warn: #8a4b08;
    }}
    body {{
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      background: linear-gradient(180deg, #efe7d8 0%, var(--bg) 100%);
      color: var(--ink);
    }}
    .wrap {{
      max-width: 1080px;
      margin: 24px auto;
      padding: 0 16px 32px;
    }}
    .hero {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 20px 22px;
      box-shadow: 0 10px 30px rgba(0,0,0,0.05);
    }}
    h1, h2 {{
      margin: 0 0 10px;
      font-weight: 700;
    }}
    .muted {{
      color: var(--muted);
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 16px;
      margin-top: 16px;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 16px 18px;
    }}
    .kpi {{
      font-size: 28px;
      font-weight: 700;
      color: var(--accent);
    }}
    ul {{
      margin: 8px 0 0 18px;
      padding: 0;
    }}
    pre {{
      white-space: pre-wrap;
      background: #f4f7f7;
      border-radius: 10px;
      padding: 12px;
      border: 1px solid var(--line);
      overflow: auto;
    }}
    .issues {{
      color: var(--warn);
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="hero">
      <h1>Sample Report</h1>
      <div class="muted">Structured summary of one dataloader output.</div>
    </div>
    <div class="grid">
      <div class="card">
        <h2>Shapes</h2>
        <div class="kpi">x = {escape(str(s["x_shape"]))}</div>
        <div class="kpi">y = {escape(str(s["y_shape"]))}</div>
      </div>
      <div class="card">
        <h2>Sources</h2>
        <ul>{li([str(v) for v in s["data_sources"]])}</ul>
      </div>
      <div class="card">
        <h2>Channels</h2>
        <ul>{li([str(v) for v in s["channels"]])}</ul>
      </div>
      <div class="card">
        <h2>Labels</h2>
        <div>label source: <strong>{escape(str(s["label_source"]))}</strong></div>
        <div>target hazards: <strong>{escape(str(s["target_hazards"]))}</strong></div>
        <ul>{li([str(v) for v in s["label_legend"]])}</ul>
      </div>
      <div class="card">
        <h2>Coverage</h2>
        <div>x synthetic ratio: <strong>{escape(str(s["x_synthetic_ratio"]))}</strong></div>
        <div>y synthetic ratio: <strong>{escape(str(s["y_synthetic_ratio"]))}</strong></div>
        <div class="muted">Per-channel ratios:</div>
        <pre>{escape(json.dumps(s["per_channel_synthetic_ratio"], indent=2))}</pre>
      </div>
      <div class="card">
        <h2>Query</h2>
        <div>time: <strong>{escape(str(s["time_range"]))}</strong></div>
        <div>bbox: <strong>{escape(str(s["bbox"]))}</strong></div>
        <div>resolution: <strong>{escape(str(s["spatial_resolution_deg"]))} deg</strong></div>
      </div>
      <div class="card">
        <h2>Label Values</h2>
        <div>Unique y values:</div>
        <pre>{escape(str(s["y_unique_values"]))}</pre>
      </div>
      <div class="card issues">
        <h2>Issues</h2>
        <ul>{li([str(v) for v in s["issues"]] or ["none"])}</ul>
      </div>
    </div>
    <div class="card" style="margin-top:16px;">
      <h2>Raw Summary JSON</h2>
      <pre>{escape(summary_json)}</pre>
    </div>
  </div>
</body>
</html>
"""
    out.write_text(html, encoding="utf-8")
    return str(out)
