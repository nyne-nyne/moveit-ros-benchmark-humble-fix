# License

This repository is based on the [moveit2/humble/moveit_ros/benchmarks](https://github.com/ros-planning/moveit2/tree/humble/moveit_ros/benchmarks) repository. However, the authors and maintainers of the original repository are not affiliated with this repository in any way.

The code in this repository is licensed under the BSD 3-Clause License. See the LICENSE file for more information.


# Benchmark Fix for MoveIt2 Humble

This repository contains a fix for using the `moveit_ros_benchmarks` package in ROS 2 with MoveIt2 Humble. The package allows you to compare different planners from different planner families against each other.

## Complications Encountered

While following the tutorial [here](https://moveit.picknik.ai/humble/doc/examples/benchmarking/benchmarking_tutorial.html), I encountered the following complications:

1. **`warehouse_mongo` Dependency:** The `warehouse_mongo` package is no longer available for ROS 2, causing an issue with the benchmarking process.
2. **Incorrect YAML Parsing:** The `demo1.yaml` file is not correctly parsed by the `BenchmarkOptions.cpp`, resulting in a crash with the following error:
    
    ```bash
    [moveit_run_benchmark-1] terminate called after throwing an instance of 'rclcpp::exceptions::InvalidParameterTypeException'
    [moveit_run_benchmark-1]   what():  parameter 'benchmark_config.planning_pipelines.pipeline1.name' has invalid type: expected [string_array] got [string]
    
    ```
    
3. **Planning Pipeline Initialization:** The `BenchmarkExecutor.cpp` does not correctly initialize the planning pipeline, leading to two possible issues:
    - It either always uses the CHOMP planner.
    - It uses the OMPL planner but fails to find the configurations and falls back to the default planner configured inside the MoveIt Config Package. This issue is accompanied by the following warning:
    
    ```bash
    [moveit_run_benchmark-1] [WARN] [moveit.ompl_planning.planning_context_manager]: Cannot find planning configuration for group 'arm' using planner 'TRRTkConfigDefault'. Will use defaults instead.
    [moveit_run_benchmark-1] [INFO] [moveit.ompl_planning.model_based_planning_context]: Planner configuration 'arm' will use planner 'geometric::RRTConnect'. Additional configuration parameters will be set when the planner is constructed.
    
    ```
    

## Fixes Implemented

I have addressed the above-mentioned issues to the best of my ability.

The Boost progress bar is currently not functioning, but it does not affect the core functionality.

## Usage

To use this repository, follow the instructions below:

1. **MoveIt2 Config:** Ensure that you have the MoveIt2 Config installed.
2. **Create Scenarios:** Utilize the `benchmark_pre_step.launch.py` file to create scenarios with states, queries, and scene objects, as described in the tutorial. Save these scenarios using SQLite.
3. **Configure the Benchmark:** Use the `/config/ompl_benchmark_kairosAB.yaml` file to configure your benchmark according to the instructions provided in the tutorial.
4. **Start the Benchmark:** Launch the benchmark using the `/launch/benchmark.launch.py` file.

Feel free to explore this repository and adapt it to your needs. If you encounter any issues, please don't hesitate to report them.


# Old README.md MoveIt ROS Benchmarks

This package provides methods to benchmark motion planning algorithms and aggregate/plot statistics. Results can be viewed in [Planner Arena](http://plannerarena.org/).

For more information and usage example please see [moveit tutorials](https://ros-planning.github.io/moveit_tutorials/doc/benchmarking/benchmarking_tutorial.html).
