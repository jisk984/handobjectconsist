import os
import pickle

import numpy as np
import pandas as pd
from libyana.logutils import log2html, logio
from meshreg import logutils


def parse_logs(folder,
               monitor_metrics=None,
               compact=True,
               plots=None,
               gifs=True):
    res_dict = {}
    train_path = os.path.join(folder, "train.txt")
    val_path = os.path.join(folder, "val.txt")
    opt_path = os.path.join(folder, "opt.pkl")
    train_df = pd.DataFrame(logio.get_logs(train_path))
    val_df = pd.DataFrame(logio.get_logs(val_path))
    with open(opt_path, "rb") as p_f:
        opts = pickle.load(p_f)
    for monitor_metric in monitor_metrics:
        res_dict[f"{monitor_metric}_train"] = train_df[monitor_metric].tolist(
        )[-1]
        res_dict[f"{monitor_metric}_val"] = val_df[monitor_metric].tolist()[-1]
        plots[f"{monitor_metric}_train"].append(
            train_df[monitor_metric].tolist())
        plots[f"{monitor_metric}_val"].append(val_df[monitor_metric].tolist())
    for key, val in opts.items():
        if isinstance(val, list):
            res_dict[key] = tuple(val)
        else:
            res_dict[key] = val

    # res_dict.update(res["args"])
    last_train_name = [
        img_path
        for img_path in sorted(os.listdir(os.path.join(folder, "images")))
        if "train" in img_path
    ][-1]
    last_val_name = [
        img_path
        for img_path in sorted(os.listdir(os.path.join(folder, "images")))
        if "val" in img_path and "batch" in img_path
    ][-1]
    res_dict["last_train_img_path"] = os.path.join(folder, "images",
                                                   last_train_name)
    res_dict["last_val_img_path"] = os.path.join(folder, "images",
                                                 last_val_name)
    return res_dict, plots


def make_exp_html(df_data, plots, metric_names, destination, compact=False):
    if compact:
        print(df_data)
    main_plot_str = logutils.make_compare_plots(plots,
                                                local_folder=destination)
    df_html = log2html.df2html(df_data,
                               local_folder=str(destination),
                               collapsible=not compact)
    with (destination / "raw.html").open("w") as h_f:
        h_f.write(df_html)

    with open("htmlassets/index.html", "rt") as t_f:
        html_str = t_f.read()
    with open(destination / "raw.html", "rt") as t_f:
        table_str = t_f.read()
    full_html_str = (html_str.replace("TABLEPLACEHOLDER", table_str))
    with open(destination / "index.html", "wt") as h_f:
        h_f.write(full_html_str)
