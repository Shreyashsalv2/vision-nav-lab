"""Generate the synthetic recorded sequence.   python scripts/make_synthetic.py --out data/synthetic [--frames 300]"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vnav import synthetic  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    out = synthetic.generate(a.out, synthetic.SyntheticConfig(frames=a.frames, seed=a.seed))
    print(f"wrote {a.frames} frames, ground truth and camera.json to {out}")


if __name__ == "__main__":
    main()
