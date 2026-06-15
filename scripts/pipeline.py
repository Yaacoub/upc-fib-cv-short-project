#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pitchvision.config import load_config
from pitchvision.frames import extract_all_frames, extract_evenly_spaced_frames, list_videos
from pitchvision.pipeline import run_checkpoint, run_final_pipeline


def cmd_extract_frames(args: argparse.Namespace) -> None:
    videos = list_videos(args.videos)
    if args.all:
        saved = extract_all_frames(videos, args.output)
        print(f"Extracted all {len(saved)} frames into {args.output}")
    else:
        saved = extract_evenly_spaced_frames(videos, args.output, args.count)
        print(f"Extracted {len(saved)} frames into {args.output}")


def cmd_run(args: argparse.Namespace) -> None:
    config = load_config(ROOT / "config" / "pipeline.yaml")
    results_path = run_final_pipeline(args.images, args.output, config, args.manual_labels)
    print(f"Wrote final PitchVision results to {results_path}")


def cmd_run_checkpoint(args: argparse.Namespace) -> None:
    config = load_config(ROOT / "config" / "pipeline.yaml")
    results_path = run_checkpoint(args.images, args.output, config, args.manual_labels)
    print(f"Wrote checkpoint results to {results_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PitchVision broadcast-to-minimap pipeline")
    subparsers = parser.add_subparsers(required=True)

    extract = subparsers.add_parser(
        "extract-frames",
        help="Extract frames from videos (an evenly-spaced sample by default, or every frame with --all)",
    )
    extract.add_argument("--videos", type=Path, required=True, help="Video file or directory")
    extract.add_argument("--output", type=Path, required=True, help="Output image directory")
    extract.add_argument("--count", type=int, default=50, help="Number of frames to extract (ignored if --all is set)")
    extract.add_argument(
        "--all",
        action="store_true",
        help=(
            "Extract every frame from the source videos instead of an evenly-spaced "
            "sample. Use this to feed `run` so the final stitched video covers the "
            "whole match."
        ),
    )
    extract.set_defaults(func=cmd_extract_frames)

    run = subparsers.add_parser("run", help="Run the complete detection, tracking, projection, and mini-map pipeline")
    run.add_argument("--images", type=Path, required=True, help="Input image directory")
    run.add_argument("--output", type=Path, required=True, help="Output directory")
    run.add_argument("--manual-labels", type=Path, default=None, help="Optional CSV with manual team labels")
    run.set_defaults(func=cmd_run)

    checkpoint = subparsers.add_parser("run-checkpoint", help="Run only the 50%% checkpoint stages")
    checkpoint.add_argument("--images", type=Path, required=True, help="Input image directory")
    checkpoint.add_argument("--output", type=Path, required=True, help="Output directory")
    checkpoint.add_argument("--manual-labels", type=Path, default=None, help="Optional CSV with manual team labels")
    checkpoint.set_defaults(func=cmd_run_checkpoint)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
