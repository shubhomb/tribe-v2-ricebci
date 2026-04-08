#!/usr/bin/env python3
"""
Batch-process all DEVO2 videos through TribeModel and generate brain plots.

For each video:
  1. Runs TribeModel.get_events_dataframe + predict
  2. Generates a PlotBrain timestep figure
  3. Saves the plot as a PNG

After processing, reorganizes the DEVO2 folder so each video gets its own
subdirectory containing the .mp4, any artifacts (.wav, .tsv), and the plot.
"""

import os
import re
import shutil
import sys
from pathlib import Path

import torch

# ── Configuration ──────────────────────────────────────────────────────────
DEVO2_DIR = Path("sample_video/DEVO2")
CACHE_FOLDER = Path("./cache")
GPU_DEVICE = 0  # change if GPU 0 is busy
N_TIMESTEPS = 15
PLOT_KWARGS = dict(cmap="fire", norm_percentile=99, vmin=0.6, alpha_cmap=(0, 0.2), show_stimuli=True)


def stem_from_video(video_path: Path) -> str:
    """Extract a clean directory name from a video filename."""
    return video_path.stem  # e.g. "311-Baking Cookies Timelapse-ED"


def process_videos():
    torch.cuda.set_device(GPU_DEVICE)

    from tribev2.demo_utils import TribeModel
    from tribev2.plotting import PlotBrain

    model = TribeModel.from_pretrained("facebook/tribev2", cache_folder=str(CACHE_FOLDER))
    plotter = PlotBrain(mesh="fsaverage5")

    videos = sorted(DEVO2_DIR.glob("*.mp4"))
    print(f"Found {len(videos)} videos in {DEVO2_DIR}")

    for i, video_path in enumerate(videos, 1):
        stem = stem_from_video(video_path)
        plot_path = DEVO2_DIR / f"{stem}.png"

        if plot_path.exists():
            print(f"[{i}/{len(videos)}] Skipping {stem} (plot already exists)")
            continue

        print(f"[{i}/{len(videos)}] Processing: {stem}")
        try:
            df = model.get_events_dataframe(video_path=str(video_path))
            preds, segments = model.predict(events=df)
            print(f"  Predictions shape: {preds.shape}")

            n = min(N_TIMESTEPS, preds.shape[0])
            fig = plotter.plot_timesteps(preds[:n], segments=segments[:n], **PLOT_KWARGS)
            fig.savefig(str(plot_path), dpi=150, bbox_inches="tight")
            import matplotlib.pyplot as plt
            plt.close(fig)
            print(f"  Saved plot: {plot_path}")
        except Exception as e:
            print(f"  ERROR processing {stem}: {e}", file=sys.stderr)
            continue

    print("\nAll videos processed.")


def reorganize():
    """
    Reorganize DEVO2 so each video gets its own subdirectory.
    Files that don't belong to a specific video stay at the top level.

    Directory structure:
        DEVO2/
            311-Baking Cookies Timelapse-ED/
                311-Baking Cookies Timelapse-ED.mp4
                311-Baking Cookies Timelapse-ED.wav
                311-Baking Cookies Timelapse-ED.tsv
                311-Baking Cookies Timelapse-ED.png
            ...
            DEVO2_Participant_Results.json
            DEVO-2 Supplementary Table 1-...xlsx
    """
    videos = sorted(DEVO2_DIR.glob("*.mp4"))
    video_stems = {stem_from_video(v) for v in videos}

    for stem in sorted(video_stems):
        subdir = DEVO2_DIR / stem
        subdir.mkdir(exist_ok=True)

        # Move all files whose name starts with this stem
        for f in DEVO2_DIR.iterdir():
            if f.is_dir():
                continue
            if f.stem == stem or f.name.startswith(stem + "."):
                dest = subdir / f.name
                print(f"  Moving {f.name} -> {subdir.name}/")
                shutil.move(str(f), str(dest))

    print("\nReorganization complete.")
    # Show final structure
    for subdir in sorted(DEVO2_DIR.iterdir()):
        if subdir.is_dir() and not subdir.name.startswith("."):
            files = [f.name for f in sorted(subdir.iterdir())]
            print(f"  {subdir.name}/  ({len(files)} files: {', '.join(files)})")
        elif subdir.is_file():
            print(f"  {subdir.name}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process DEVO2 videos and reorganize folder")
    parser.add_argument("--process-only", action="store_true", help="Only generate plots, don't reorganize")
    parser.add_argument("--reorganize-only", action="store_true", help="Only reorganize, don't process videos")
    parser.add_argument("--gpu", type=int, default=GPU_DEVICE, help=f"GPU device index (default: {GPU_DEVICE})")
    parser.add_argument("--n-timesteps", type=int, default=N_TIMESTEPS, help=f"Max timesteps to plot (default: {N_TIMESTEPS})")
    args = parser.parse_args()

    GPU_DEVICE = args.gpu
    N_TIMESTEPS = args.n_timesteps

    if args.reorganize_only:
        reorganize()
    elif args.process_only:
        process_videos()
    else:
        process_videos()
        reorganize()
