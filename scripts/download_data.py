#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
load_dotenv(dotenv_path=ROOT / ".env")


def _password() -> str:
    password = os.environ.get("SOCCERNET_PASSWORD")
    if not password:
        raise RuntimeError("Set SOCCERNET_PASSWORD in your environment or .env file.")
    return password


def _downloader(local_dir: Path):
    from SoccerNet.Downloader import SoccerNetDownloader

    return SoccerNetDownloader(LocalDirectory=str(local_dir))


def cmd_download_videos(args: argparse.Namespace) -> None:
    from SoccerNet.utils import getListGames

    downloader = _downloader(args.local_dir)
    downloader.password = _password()

    files = [f"1_{args.resolution}.mkv"]
    games: list[tuple[str, str]] = []
    for split in ["train"]:
        for game in getListGames(split, task="spotting"):
            games.append((split, game))

    games = games[:1]

    if not games:
        raise RuntimeError("No SoccerNet games matched the requested split/task.")

    for split, game in games:
        print(f"Downloading {files} for {split}: {game}")
        downloader.downloadGame(game=game, files=files, spl=split, verbose=True)

    print(f"Downloaded {len(games)} SoccerNet game(s) to {args.local_dir}")


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--local-dir",
        type=Path,
        default=Path("data/soccernet"),
        help="Local SoccerNet root directory",
    )

    parser = argparse.ArgumentParser(
        description="Download the SoccerNet videos used by the PitchVision checkpoint",
        parents=[common],
    )

    subparsers = parser.add_subparsers(required=True)

    videos = subparsers.add_parser(
        "download-videos",
        help="Download a small number of full broadcast videos after NDA approval",
        parents=[common],
    )
    videos.add_argument(
        "--resolution",
        choices=["224p", "480p", "720p", "1080p"],
        default="720p",
        help="Video resolution suffix to download (e.g. 224p, 720p)",
    )
    videos.set_defaults(func=cmd_download_videos)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
