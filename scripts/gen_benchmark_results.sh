#!/bin/bash
# Usage: gen_benchmark_results.sh /path/to/benchmark/directory /path/to/store/results/directory
# Generates the results in the form of a pdf, from the benchmark file(s) created by moveit. Each result file will take on the name of the experiment.
# Depends on https://github.com/nyne-nyne/moveit-ros-benchmark-humble-fix. The package is renamed to moveit_ros_benchmarks_fix to avoid name collisions.
# Ensure that the workspace containing the ros2 distribution and the benchmark are installed and built
if [ "$#" -ne 2 ]
then
  echo "Usage: gen_benchmark_results.sh /path/to/benchmark/directory/ /path/to/store/experiment/results/directory/"
  exit 1
fi
benchmark_dir=$1
result_dir=$2

[ ! -d $benchmark_dir ] && echo "Benchmark directory doesn't exist :(" && exit 1
mkdir -p $result_dir


find $benchmark_dir -type f -print0 | while IFS= read -r -d $'\0' file;
do
    filename=$(grep "scene_name" "$file" | awk '{print $2}')_$(grep "Experiment" "$file" | awk '{print $2}')
    echo "Saving result of experiment to $result_dir/$filename.pdf"
    
    ros2 run moveit_ros_benchmarks_fix moveit_benchmark_statistics.py --plot="$result_dir/$filename.pdf" $file
done
