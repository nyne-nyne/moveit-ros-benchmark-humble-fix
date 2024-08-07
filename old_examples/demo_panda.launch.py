from launch import LaunchDescription
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder
from launch_param_builder import ParameterBuilder
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():

    moveit_ros_benchmarks_config = (
        ParameterBuilder("moveit_ros_benchmarks_fix")
        .yaml(
            parameter_namespace="benchmark_config",
            file_path="demo1.yaml",
        )
        .to_dict()
    )

    moveit_configs = MoveItConfigsBuilder("moveit_resources_panda").to_dict()

    sqlite_database = (
        get_package_share_directory("moveit_benchmark_resources")
        + "/databases/panda_test_db.sqlite"
    )

    warehouse_ros_config = {
        # For warehouse_ros_sqlite
        "warehouse_plugin": "warehouse_ros_sqlite::DatabaseConnection",
        "benchmark_config": {
            "warehouse": {
                "warehouse_plugin": "warehouse_ros_sqlite::DatabaseConnection",
                "host": sqlite_database,
                "port": 33828,
                "scene_name": "",
            },
        },

    }
    
    # moveit_ros_benchmark demo executable
    moveit_ros_benchmarks_node = Node(
        name="moveit_run_benchmark",
        package="moveit_ros_benchmarks_fix",
        # prefix='xterm -e gdb --ex=run --args',
        executable="moveit_run_benchmark",
        output="screen",
        parameters=[
            moveit_ros_benchmarks_config,
            moveit_configs,
            warehouse_ros_config,
        ],
    )

    return LaunchDescription([moveit_ros_benchmarks_node])
