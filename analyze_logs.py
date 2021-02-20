#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from collections import defaultdict
from pathlib import Path
import traceback
import warnings

import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm

from libyana.exputils import argutils
from libyana.logutils import logio
from meshreg import analyze

parser = argparse.ArgumentParser()
parser.add_argument("--save_roots", default=["tmp"], nargs="+")
parser.add_argument("--sort_loss", default="loss_v2d_person")
parser.add_argument("--destination", default="results/tables")
parser.add_argument("--no_gifs", action="store_true")
parser.add_argument("--no_videos", action="store_true")
parser.add_argument("--video_resize", default=0.5, type=float)
parser.add_argument("--monitor_metrics",
                    nargs="+",
                    default=[
                        "objverts3d_chamfer", "objverts3d_mepe_trans",
                        "joints3d_epe_mean", "joints3d_cent_epe_mean",
                        "recov_joint3d", "pose_reg", "total_loss"
                    ])
parser.add_argument("--min", action="store_true")
parser.add_argument("--show_extremes", default=-2, type=int)
parser.add_argument("--subfolders", action="store_true")
parser.add_argument("--no_html", action="store_true")
parser.add_argument("--max_samples", type=int)

args = parser.parse_args()
argutils.print_args(args)

destination = Path(args.destination)
destination.mkdir(exist_ok=True, parents=True)

all_folders = []
explored_folders = []
for save_root in args.save_roots:
    save_root = Path(save_root)
    df_data = []
    if args.subfolders:
        folders = list(save_root.iterdir())
    else:
        folders = [save_root]
    for folder in folders:
        explored_folders.append(folder)
        res_path = folder / "plotly.html"
        if res_path.exists():
            all_folders.append(folder)
print(f"Got {len(all_folders)} folders")

plots = defaultdict(list)
if len(all_folders) == 0:
    raise ValueError(f"No results in folders {explored_folders} !")
chosen_folders = sorted(all_folders)
if args.max_samples is not None:
    chosen_folders = chosen_folders[:args.max_samples]
    print(chosen_folders[-1])
for folder_idx, folder in enumerate(tqdm(chosen_folders)):
    try:
        res_data, plots = analyze.parse_logs(folder,
                                             args.monitor_metrics,
                                             compact=args.min,
                                             gifs=not args.no_gifs,
                                             plots=plots)
        df_data.append(res_data)
    except Exception:
        warnings.warn(f"Ran into error for {res_path}")
        traceback.print_exc()
# Add before/after optimization results
df = pd.DataFrame(df_data)
print(df[[f"{metric}_train" for metric in args.monitor_metrics]])
print(df[[f"{metric}_val" for metric in args.monitor_metrics]])
if not args.no_html:
    analyze.make_exp_html(df,
                          plots,
                          args.monitor_metrics,
                          destination,
                          compact=not args.min)
