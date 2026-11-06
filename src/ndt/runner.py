"""Command-line entry point: ``python -m ndt`` / ``ndt``."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

from .experiments import EXPERIMENTS, Config
from .network import resolve_device
from .parallel import default_jobs, describe_environment

__all__ = ["main", "write_csv"]


def write_csv(rows, path: str) -> None:
    if not rows:
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ndt",
        description="Monotone spline networks for neural distribution transformation.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("env", help="print the compute environment (CPU, GPU, versions)")

    run = sub.add_parser("run", help="run one or more experiments")
    run.add_argument("--experiments", default="all",
                     help="'all' or a comma-separated list: " + ", ".join(sorted(EXPERIMENTS)))
    run.add_argument("--out", default="results", help="output directory (default: results)")
    run.add_argument("--quick", action="store_true", help="small, fast versions of every experiment")
    run.add_argument("--jobs", type=int, default=0, help="parallel workers (default: all cores - 1)")
    run.add_argument("--device", default="cpu", help="cpu, cuda or auto (default: cpu)")
    run.add_argument("--seed", type=int, default=20261015)
    run.add_argument("--figures", action="store_true", help="also render the figure set")
    return parser


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)

    if args.command == "env":
        print(json.dumps(describe_environment(), indent=2))
        return 0

    names = sorted(EXPERIMENTS) if args.experiments == "all" \
        else [n.strip() for n in args.experiments.split(",") if n.strip()]
    unknown = [n for n in names if n not in EXPERIMENTS]
    if unknown:
        print("unknown experiment(s): %s" % ", ".join(unknown), file=sys.stderr)
        print("available: %s" % ", ".join(sorted(EXPERIMENTS)), file=sys.stderr)
        return 2

    jobs = args.jobs or default_jobs()
    device = resolve_device(args.device)
    os.makedirs(args.out, exist_ok=True)

    environment = describe_environment()
    environment["resolved_device"] = device
    environment["jobs"] = jobs
    with open(os.path.join(args.out, "environment.json"), "w", encoding="utf-8") as fh:
        json.dump(environment, fh, indent=2)

    cfg = Config(quick=args.quick, jobs=jobs, device=device, seed=args.seed,
                 meta={"device_used": device, "jobs": jobs, "quick": args.quick})

    print("ndt run | device=%s jobs=%d quick=%s" % (device, jobs, args.quick))
    summary = {}
    for name in names:
        start = time.perf_counter()
        rows = EXPERIMENTS[name](cfg)
        elapsed = time.perf_counter() - start
        write_csv(rows, os.path.join(args.out, name + ".csv"))
        summary[name] = {"rows": len(rows), "seconds": round(elapsed, 2)}
        print("  %-12s %4d rows  %6.1f s" % (name, len(rows), elapsed))

    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump({"environment": environment, "runs": summary}, fh, indent=2)

    if args.figures:
        from .plotting import make_all

        figures = make_all(args.out)
        print("figures: %s" % ", ".join(os.path.basename(p) for p in figures))

    print("results written to %s" % os.path.abspath(args.out))
    return 0
