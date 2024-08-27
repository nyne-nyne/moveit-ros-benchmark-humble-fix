#!/usr/bin/env python3
#TODO: temp testing, remove later, or add as an option
import IPython
# TODO: Save the results of each benchmark into a PDF file, instead of just displaying them
import argparse
import sys
import io
import itertools
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import pandas as pd

def skip_line(f: io.TextIOWrapper):
    f.readline()

def skip_lines(n: int, f: io.TextIOWrapper):
    for _ in range(n):
        skip_line(f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Assumes same experiment is being compared across the files, with the same properties for each
    # run. The only thing that differs is the attempts and/or the planning time. Sometimes the planner(s)
    parser.add_argument("experiments", type=Path, nargs='+',
                        help="Path to file(s) containing benchmark logs, of the same experiment.")
    #parser.add_argument("benchmark_dirs", type=str, nargs='+',
    #                    help="Path to one or more directories that contain benchmark experiment log files.")
    # TODO, : list the files? and then look match 'em up, by sorting them?
    #         This would just mean, the user has to give the benchmark folder paths
    args = parser.parse_args()

    # TODO: Turn stuff like this into a class, and just use self for all the class-level stuff?
    metrics = dict()
    benchmarks = dict()
    runs = -1

    # TODO: parse the file for experiment name as well as scene name; for the file to save to
    for result_file in args.experiments:
        bm_name = ""
        with open(result_file, "r") as f:
            planner_results = dict()

            print("Processing result " + str(result_file), end='', file=sys.stderr)
            skip_lines(8, f)

            attempts = f.readline().split()[-1]
            planning_time = f.readline().split()[-1]
            bm_name = f"{planning_time}s{attempts}a" # TODO: Add the robot name as well? If you've got multiple robots, raise your hand!
            skip_lines(7, f)

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

                # TODO: Best values to fill in for Nan's? The mean? 0's? Would it affect the box-plots?
                #       Typically, boxplots will not plot if I use a np.array with a some np.nan's.
                #       But, they when the type is float, but it seems to work if the type is object, which is
                #       which is how xarray is storing the values...so it seems to work.
                #       https://docs.xarray.dev/en/latest/user-guide/plotting.html#missing-values
                try:
                    df = pd.read_table(data, sep=";\s?", header=None).drop(columns=12) # type: ignore
                except KeyError as e: # this happens for planner failure, less than total metrics are given
                    df = pd.DataFrame({row_id: np.repeat(np.NaN, runs) for row_id in range(len(metrics))})
                
                planner_results[planner_name] = df;
                data.close()
                skip_line(f)

            benchmarks[bm_name] = xr.DataArray(data=list(planner_results.values()),
                                               dims=["planner", "run", "metric"],
                                               coords={"planner": list(planner_results.keys()),
                                                       "metric": list(metrics.keys()),
                                                       "run": range(runs)}, name=bm_name)

    # All the data across the same experiment into a dataset
    # This is fine :) since it's references. Just using it for the dimension alignment
    # when there are more/less planners involved.
    collected = xr.Dataset(benchmarks,
                            coords={"metric": list(metrics.keys()),
                                    "run": range(runs),
                                    "benchmark": list(benchmarks.keys())})

    # It's plotting time.
    for metric, data_type in metrics.items():
        if data_type == "BOOLEAN": # bar-graph
            fig, ax = plt.subplots()
            x = np.arange(len(collected.planner))
            m = (1.0 - len(collected.benchmark))/2.0
            width = 0.4
            ax.set_xticks(x, collected.planner.to_numpy(), rotation=90)

            for benchmark in collected.values():
                offset = width * m
                results = []
                for i, planner in enumerate(collected.planner): # multiple groupby selection is not available yet :(
                    result = benchmark.sel(planner=planner, metric=metric).mean(dim='run') * 100
                    if result.isnull():
                        plt.setp(ax.get_xticklabels()[i], color='red') # This indicates that the one or more benchmark failed; the one that didn't will have a plot...
                    results.append(result)

                ax.bar(x + offset, results, width, label=benchmark.name)
                m += 1.0

            ax.set_ylim(0.0, 100)
            ax.legend()
            ax.tick_params(axis='y', labelsize=10)
            ax.tick_params(axis='x', labelsize=10)
            ax.set_ylabel(f"{metric.replace('_', ' ').title()} (%)", fontsize=12)
            #ax.set_xlabel("Motion planning algorithm", fontsize=12)
            plt.show()
        else: # box-plot
            # Heavy inspiration, picked from answers here https://stackoverflow.com/questions/16592222/how-to-create-grouped-boxplotsa
            fig, ax = plt.subplots()
            x = np.arange(len(collected.planner)) * 1.5 # TODO: Get the multiplier from the length
            box_spacing = 0.1
            width = 0.5
            m = (1.0 - len(collected.benchmark))/2.0
            colours = ["red", "blue", "green", "cyan", "teal", "orange"] # TODO: Autogenerate these? Colour names? Or hex-values?

            mark_locations = []
            for colour_i, benchmark in enumerate(collected.values()):
                offset = width * m
                results = []
                for i, planner in enumerate(collected.planner.to_numpy()):
                    result = benchmark.sel(metric=metric, planner=planner)
                    if int(result.count()) != runs:
                        mark_locations.append(i)
                    results.append(result)

                boxplot = ax.boxplot(results, positions=x + offset, widths=width*(1.0 - box_spacing), sym='+', vert=True,
                                     patch_artist=True, labels=['']*len(collected.planner.to_numpy()))

                # Colouring
                # TODO: How to get the outliers the same colour, from the same palette?
                for element in ["boxes", "whiskers", "fliers", "means", "medians", "caps"]:
                    plt.setp(boxplot[element], color=colours[colour_i])
                plt.setp(boxplot['boxes'], facecolor="white")

                # Enable the legend, using a bar plot for a more consistent legend look
                ax.bar(x + offset, [np.nan], 0, color=colours[colour_i], label=benchmark.name)
                m += 1.0

            ax.legend()
            ax.set_xticks(x, collected.planner.to_numpy(), rotation=90)
            for i in mark_locations:
                plt.setp(ax.get_xticklabels()[i], color='red')
            ax.tick_params(axis='y', labelsize=10)
            ax.tick_params(axis='x', labelsize=10)
            ax.set_ylabel(f"{metric.replace('_', ' ').title()}", fontsize=12)
            plt.show()


    IPython.embed()

