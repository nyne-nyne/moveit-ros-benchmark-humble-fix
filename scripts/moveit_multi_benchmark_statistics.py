#!/usr/bin/env python3
import argparse
import io
import itertools
import os
import sys
from pathlib import Path

import IPython
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import colormaps
from matplotlib.backends.backend_pdf import PdfPages

# TODO: See https://matplotlib.org/stable/users/explain/customizing.html#the-matplotlibrc-file
#       for more details, perhaps a better way of doing this?
mpl.rcParams['axes.linewidth'] = 0.5
mpl.rcParams['patch.linewidth'] = 0.5
mpl.rcParams['legend.fontsize'] = 6
mpl.rcParams['legend.fancybox'] = False
mpl.rcParams['legend.edgecolor'] = (0.0, 0.0, 0.0)

def skip_line(f: io.TextIOWrapper):
    f.readline()

def skip_lines(n: int, f: io.TextIOWrapper):
    for _ in range(n):
        skip_line(f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""Generate a PDF file containing the plots of the same experiment run with different parameters, such as the planning time and attempts.
                                                    The metrics are assumed to be the same across all experiments. Works with the result files produced by the MoveIt2 benchmark package in ROS2-Humble.""",
                                     epilog="Get the benchmark package for ROS2-Humble + MoveIt2 from https://github.com/valantano/moveit-ros-benchmark-humble-fix or https://github.com/nyne-nyne/moveit-ros-benchmark-humble-fix.")
    parser.add_argument("results", type=Path, nargs='+',
                        help="Path to one or more directories containing the benchmark experiment logs.")
    parser.add_argument("--ishell", default=False, action="store_true",
                        help="Run an interactive (IPython) shell at the end of the script. Defaults to 'False'")
    parser.add_argument("--colours", default="Set1",
                        help="Colourmap to use for the plots. Defaults to 'Set1'. Refer to matplotlib's colormap documentation for more details.")
    parser.add_argument("--dir", type=Path, required=False, default=os.getcwd(),
                        help="Colourmap to use for the plots. Defaults to 'Set1.'")
    parser.add_argument("--blacklist", nargs='+',required=False, default=None,
                        help="List of metrics to not plot. Default behaviour is to plot all metrics.")

    args = parser.parse_args()

    if not os.path.isdir(args.dir):
        print(f"Creating new directory {args.dir}...")
        os.mkdir(args.dir)
    
    # Pair up all the corresponding experiments, across the benchmarks
    dirs = []
    for benchmark_result in args.results:
            if os.path.isdir(benchmark_result):
                dirs.append(sorted([os.path.join(benchmark_result, f) for f in os.listdir(benchmark_result) if f.endswith(".log")]))
            else:
                print(f"{benchmark_result} is not a directory. Skipping...", file=sys.stderr)                    

    for experiment_group in zip(*dirs):
        # TODO: Turn stuff like this into a class, and just use self for all the class-level stuff?
        # TODO: Is this slower than the single-benchmark script?
        metrics = dict()
        benchmarks = dict()
        runs = -1
        file_name = "" # experiment_name + scene_name + extension (.pdf)
        exp_name = ""
        scene_name = ""

        for result_file in experiment_group:
            bm_name = ""
            with open(result_file, "r") as f:
                print("Processing result " + str(result_file), file=sys.stderr)
                planner_results = dict()

                skip_line(f)
                exp_name = f.readline().split()[-1]
                skip_lines(6, f)

                attempts = f.readline().split()[-1]
                planning_time = f.readline().split()[-1]
                bm_name = f"{planning_time}s{attempts}a" # TODO: Add the robot name as well? If you've got multiple robots, raise your hand!
                skip_line(f)
                scene_name = f.readline().split()[-1]
                file_name = f"{exp_name}_{scene_name}"
                skip_lines(5, f)

                runs = max(int(f.readline().split()[0]), runs)
                skip_lines(2, f)

                no_of_planners = int(f.readline().split()[0])

                for p in range(no_of_planners):
                    # planner_name = f.readline().strip()
                    planner_name = f.readline().replace("(ompl)", '').strip()
                    skip_line(f)

                    # Parse the metrics on the first time, skip these lines on the next
                    # Assumes the first planner has the full metrics, and didn't error out
                    no_of_metrics = int(f.readline().split()[0])
                    if not metrics:
                        for metric in range(no_of_metrics):
                            metric_name, metric_type = f.readline().split()
                            metrics[metric_name] = metric_type
                    else:
                        skip_lines(no_of_metrics, f)
                    skip_line(f)

                    # Get the run data, for our planner
                    data = io.StringIO("".join(itertools.islice(f, runs)))

                    # In the current file format, any NaN's are the result of a failed solve.
                    try:
                        df = pd.read_table(data, sep=";\s?", header=None).drop(columns=12) # type: ignore
                    except KeyError as e: # This is for when the planner is unable to solve 100% of the time. Less than total metrics are given.
                        df = pd.DataFrame({row_id: np.repeat(np.NaN, runs) for row_id in range(len(metrics))})

                    planner_results[planner_name] = df;
                    data.close()
                    skip_line(f)

                benchmarks[bm_name] = xr.DataArray(data=list(planner_results.values()), # TODO: Add some attributes as well, from the benchmark file!
                                                   dims=["planner", "run", "metric"],
                                                   coords={"planner": list(planner_results.keys()),
                                                           "metric": list(metrics.keys()),
                                                           "run": range(runs)}, name=bm_name)

        # All the data across the same experiment into a dataset
        # This is fine :) since it's references. Just using it for the dimension alignment
        # when there are more/less planners involved. And, allows for playing around with
        # it easier if using the interactive shell argument
        # TODO: put this stuff in a m e g a dataset? With allllll the data?
        collected = xr.Dataset(benchmarks,
                                coords={"metric": list(metrics.keys()),
                                        "run": range(runs),
                                        "benchmark": list(benchmarks.keys())})

        # It's plotting time.
        with PdfPages(os.path.join(args.dir, file_name + ".pdf")) as pdf:
            try: 
                colours = colormaps[args.colours].colors
            except KeyError as e:
                print(f"Unknown colourmap name {args.map}. See https://matplotlib.org/stable/users/explain/colors/colormaps.html#choosing-colormaps-in-matplotlib. Defaulting to 'Set1'", file=sys.stderr)
                colours = colormaps["Set1"].colors

            if args.blacklist:
                print(f"Skipping plotting of the following metric(s): {', '.join(args.blacklist[:-1])}", file=sys.stderr)
            for metric, data_type in metrics.items():
                if metric in args.blacklist:
                    continue

                if data_type == "BOOLEAN": # Plots the data in a bar-graph
                    # Yes xarray has methods for this, but it doesn't have anything for
                    # box-plots (as of commenting time). This was done in order to keep stuff consistent,
                    # and for more control over the plotting
                    fig, ax = plt.subplots(tight_layout=True)
                    ax.grid(linestyle="--", markerfacecolor="xkcd:light blue grey", alpha=0.4, linewidth=0.6)
                    x = np.arange(len(collected.planner))
                    m = (1.0 - len(collected.benchmark))/2.0
                    width = 1.0/(len(collected.benchmark) + 0.25)

                    for colour_i, benchmark in enumerate(collected.values()):
                        results = []
                        offset = width * m
                        cfailed_planners = [False]*collected.planner.size # This is for indicated that a planner failed; for disambiguation from data that's just 0

                        for planner in collected.planner: # Multiple groupby selection is not available yet, Ubuntu 22.04 :(
                            results.append(benchmark.sel(planner=planner, metric=metric).mean(dim='run', skipna=True) * 100)

                        ax.bar(x + offset, results, width, color=colours[colour_i % len(colours)], label=benchmark.name)
                        m += 1.0

                    ax.set_ylim(0, 100)
                    ax.legend(bbox_to_anchor=(0.0, 1.0), loc='lower left', ncol=collected.benchmark.size)
                    ax.tick_params(axis='y', labelsize=6)
                    ax.tick_params(axis='x', labelsize=6)
                    ax.set_xticks(x, collected.planner.data, rotation=80)
                    ax.set_ylabel(f"{metric.replace('_', ' ').title()} (%)", fontsize=8)
                    #ax.set_xlabel("Motion planning algorithm", fontsize=12)
                    pdf.savefig()
                    plt.close()
                else: # Uses a box-plot for all other data
                    # This methodology was inspired from the answers here https://stackoverflow.com/questions/16592222/how-to-create-grouped-boxplot
                    fig, ax = plt.subplots(tight_layout=True)
                    ax.grid(linestyle="--", markerfacecolor="xkcd:light blue grey", alpha=0.4, linewidth=0.6)
                    x = np.arange(len(collected.planner))
                    m = (1.0 - len(collected.benchmark))/2.0
                    width = 1/(len(collected.benchmark) + 0.25)

                    for colour_i, benchmark in enumerate(collected.values()):
                        offset = width * m
                        results = []

                        for planner in collected.planner:
                            results.append(benchmark.sel(metric=metric, planner=planner))

                        boxplot = ax.boxplot(results, positions=x + offset, widths=width * 0.85, sym='+', vert=True,
                                             flierprops=dict(markeredgewidth=0.5, markeredgecolor=colours[colour_i % len(colours)]), patch_artist=True, labels=['']*collected.planner.size)
                        # Colouring the rest of the boxplot
                        for element in boxplot.keys():
                            plt.setp(boxplot[element], color=colours[colour_i % len(colours)])
                            plt.setp(boxplot[element], linewidth=0.6)
                        #plt.setp(boxplot["whiskers"], linestyle="--")
                        plt.setp(boxplot["boxes"], facecolor="white")

                        # Enable the legend, using a bar plot for a more consistent legend look
                        # TODO: Another way to do them with handles? https://matplotlib.org/stable/users/explain/axes/legend_guide.html
                        ax.bar(x + offset, [np.nan], 0, color=colours[colour_i], label=benchmark.name)
                        m += 1.0

                    ax.legend(bbox_to_anchor=(0.0, 1.0), loc='lower left', ncol=collected.benchmark.size)
                    ax.set_xticks(x, collected.planner.data, rotation=80)
                    ax.tick_params(axis='y', labelsize=6)
                    ax.tick_params(axis='x', labelsize=6)
                    ax.set_ylabel(f"{metric.replace('_', ' ').title()}", fontsize=8)
                    pdf.savefig()
                    plt.close()

        if args.ishell:
            print(f"Starting shell for data collect for experiment {exp_name} using scene {scene_name}")
            IPython.embed()
