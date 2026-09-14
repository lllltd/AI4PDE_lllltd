"""Command-line interface: generate data, train, evaluate, or run everything."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pipeline import (
    evaluate_stage,
    generate_stage,
    load_config,
    prepare_run_dir,
    run_all,
    train_stage,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cornell-style nonlinear Darcy DeepONet")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "generate", "train", "evaluate"):
        command = subparsers.add_parser(name)
        command.add_argument("--config", type=Path, required=True)
        command.add_argument("--output-dir", type=Path)
        if name == "evaluate":
            command.add_argument("--checkpoint", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    run_dir = prepare_run_dir(config, args.output_dir)
    if args.command == "run":
        metrics = run_all(config, run_dir)
        print(json.dumps(metrics, indent=2))
    elif args.command == "generate":
        generate_stage(config, run_dir)
        print(f"Saved dataset to {run_dir / 'data.npz'}")
    elif args.command == "train":
        train_stage(config, run_dir)
        print(f"Saved checkpoints and training history to {run_dir}")
    else:
        metrics = evaluate_stage(config, run_dir, args.checkpoint)
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
