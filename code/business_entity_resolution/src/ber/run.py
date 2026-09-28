"""Pipeline entry point.

    python -m ber.run prepare --split train      # L0+L1: load + normalize -> parquet
"""
import argparse
import time

from . import config, io
from .normalize import normalize_frame


def prepare(split: str) -> None:
    for src in config.SOURCES:
        out = config.artifact("norm", f"{split}_s{src}.parquet")
        if out.exists():
            print(f"[prepare] {out.name} cached")
            continue
        t = time.time()
        df = io.load_source(split, src)
        norm = normalize_frame(df)
        norm.to_parquet(out, compression="zstd", index=False)
        print(f"[prepare] {split} s{src}: {len(norm):,} rows in {time.time() - t:.0f}s -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prepare"])
    ap.add_argument("--split", choices=config.SPLITS, required=True)
    args = ap.parse_args()
    if args.stage == "prepare":
        prepare(args.split)


if __name__ == "__main__":
    main()
