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

    warehouse_ros_config = {
        # For warehouse_ros_sqlite
        "warehouse_plugin": "warehouse_ros_sqlite::DatabaseConnection",
        "warehouse_host": "/tmp/demo1_db.sqlite",
        "warehouse_port": 33828,
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
