#!/usr/bin/env python
# -*- coding: utf-8 -*-
from itertools import cycle
from bokeh import plotting as plt
from bokeh import embed, layouts, palettes
from bokeh.models import HoverTool

# Creat unique ids for each new html div
HTML_IDX = [0]

COLORS = (palettes.Colorblind[max(palettes.Colorblind)] +
          palettes.Bokeh[max(palettes.Bokeh)])
DASH_PATTERNS = ("solid", "dashed")


def make_compare_plots(plots,
                       local_folder,
                       tools="pan,wheel_zoom,box_zoom,reset,save"):
    bokeh_figs = []
    for metric, metric_vals in plots.items():
        cycle_colors = cycle(COLORS)
        cycle_dash_patterns = cycle(DASH_PATTERNS)
        bokeh_p = plt.figure(
            tools=tools,
            x_axis_label="iter_steps",
            title=metric,
        )
        bokeh_p.add_tools(HoverTool())
        for run_idx, vals in enumerate(metric_vals):
            color = next(cycle_colors)
            dash_pattern = next(cycle_dash_patterns)
            bokeh_p.line(
                list(range(len(vals))),
                vals,
                legend_label=f"{run_idx:03d}",
                color=color,
                line_dash=dash_pattern,
            )
        bokeh_figs.append(bokeh_p)
    bokeh_grid = layouts.gridplot([bokeh_figs])
    js_str, html_str = embed.components(bokeh_grid)
    with (local_folder / "add_js.txt").open("at") as js_f:
        js_f.write(js_str)
    return html_str


def plotvals2bokeh(plot_vals,
                   local_folder,
                   tools="pan,wheel_zoom,box_zoom,reset,save"):
    bokeh_p = plt.figure(tools=tools, height=100, width=150)
    bokeh_p.line(list(range(len(plot_vals))), plot_vals)
    bokeh_p.add_tools(HoverTool())
    js_str, html_str = embed.components(bokeh_p)
    with (local_folder / "add_js.txt").open("at") as js_f:
        js_f.write(js_str)

    return html_str
