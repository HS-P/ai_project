"""
Output/full_runs/ 결과를 집계하고, best 모델을 Output/best/ 에 복사.

Usage:
  python aggregate_results.py              # 표 출력 + best 자동 선택
  python aggregate_results.py --metric val # val_acc 기준 (기본)
  python aggregate_results.py --metric train # train_acc 기준
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
FULL_DIR = _ROOT / "Output" / "research" / "full_runs"
BEST_DIR = _ROOT / "Output" / "research" / "best"


def load_all_results():
    results = []
    for sub in FULL_DIR.iterdir():
        if not sub.is_dir():
            continue
        rj = sub / "result.json"
        if rj.is_file():
            with open(rj) as f:
                results.append(json.load(f))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", choices=("val", "train", "balanced"), default="val")
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    results = load_all_results()
    if not results:
        print("No results yet in Output/full_runs/. Run train experiments first.")
        return

    if args.metric == "val":
        key = lambda r: -r["val_acc"]
        title = "by Val acc"
    elif args.metric == "train":
        key = lambda r: -r["train_acc"]
        title = "by Train acc"
    else:
        key = lambda r: -(0.4 * r["train_acc"] + 0.6 * r["val_acc"])
        title = "by balanced (0.4*train + 0.6*val)"

    results.sort(key=key)

    print(f"\n{'='*100}")
    print(f"FULL DATA RESULTS ({len(results)} runs, sorted {title})")
    print(f"{'='*100}")
    print(f"{'tag':<26} {'Train%':>8} {'Val%':>8} {'WTrain':>7} {'WVal':>5} {'SVs':>6} {'Time(s)':>8}")
    print("-" * 100)
    for r in results[:args.top]:
        print(f"{r['tag']:<26} {r['train_acc']*100:>7.4f}% {r['val_acc']*100:>7.4f}% "
              f"{r['n_wrong_train']:>7} {r['n_wrong_val']:>5} {r['n_support_vectors']:>6} "
              f"{r['train_time_sec']:>7.1f}s")

    best = results[0]
    print(f"\n>>> BEST ({args.metric}): {best['tag']}")
    print(f"    Train: {best['train_acc']*100:.4f}% | Val: {best['val_acc']*100:.4f}%")
    print(f"    Config: {best['config']}")

    # Copy best to Output/best/
    best_src = FULL_DIR / best["tag"]
    BEST_DIR.mkdir(parents=True, exist_ok=True)
    # clear best dir first
    for f in BEST_DIR.iterdir():
        if f.is_file():
            f.unlink()
    for f in best_src.iterdir():
        if f.is_file():
            shutil.copy2(f, BEST_DIR / f.name)
    # Also save aggregated table
    with open(BEST_DIR / "all_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[best] Copied {best['tag']} -> {BEST_DIR}/")
    print(f"[best] Aggregated table: {BEST_DIR / 'all_results.json'}")


if __name__ == "__main__":
    main()
