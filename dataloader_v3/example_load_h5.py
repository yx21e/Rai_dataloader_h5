#!/usr/bin/env python3

from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataloader_v3 import load_sample_h5, to_torch_batch


def main() -> None:
    path = "/home/yangshuang/output/sample_v3.h5"
    sample = load_sample_h5(path)
    x_t, y_t, _ = to_torch_batch(sample)

    print("loaded h5:", path)
    print("x shape:", sample.x.shape, "| torch:", tuple(x_t.shape))
    print("y shape:", sample.y.shape, "| torch:", tuple(y_t.shape))
    print("channels:", sample.meta.get("channels", []))


if __name__ == "__main__":
    main()
