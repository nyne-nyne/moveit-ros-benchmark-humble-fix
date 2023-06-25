from launch import LaunchDescription
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder
from launch_param_builder import ParameterBuilder


def generate_launch_description():

    moveit_ros_benchmarks_config = (
        ParameterBuilder("moveit_ros_benchmarks")
        .yaml(
            parameter_namespace="benchmark_config",
            file_path="ompl_benchmark_kairosAB.yaml",
        )
        .to_dict()
    )

    moveit_configs = MoveItConfigsBuilder("kairosab_ur10_egh").to_dict()

    # moveit_configs = (
    #     MoveItConfigsBuilder("kairosab_ur10_egh")
    #     .trajectory_execution(file_path="config/moveit_controllers.yaml")
    #     .robot_description_kinematics(file_path="config/kinematics.yaml")
    #     .robot_description(file_path="config/rbkairos.urdf.xacro")
    #     .planning_pipelines()
    #     .to_dict()
    # )

    warehouse_ros_config = {
        "warehouse_plugin": "warehouse_ros_sqlite::DatabaseConnection", # needed to use sqlite instead of mongodb
        # "warehouse_host": "localhost"  # not needed because host and port are set in benchmark_config
    }

    ompl_planning_pipeline_config = {
        "planning_plugin": "ompl_interface/OMPLPlanner",    # currently a workaround for the Error: Multiple planning plugins available. You should specify the '~planning_plugin' parameter. Using 'chomp_interface/CHOMPPlanner' for now.
    }

    # moveit_ros_benchmark demo executable
    moveit_ros_benchmarks_node = Node(
        name="moveit_run_benchmark",
        package="moveit_ros_benchmarks",
        # prefix='xterm -e gdb --ex=run --args',
        executable="moveit_run_benchmark",
        output="screen",
        parameters=[
            moveit_ros_benchmarks_config,
            moveit_configs,
            warehouse_ros_config,
            # ompl_planning_pipeline_config,
        ],
    )

    # Warehouse mongodb server
    # mongodb_server_node = Node(
    #     package="warehouse_ros_sqlite",
    #     executable="mongo_wrapper_ros.py",
    #     parameters=[
    #         {"warehouse_port": 33829},
    #         {"warehouse_host": "localhost"},
    #         {"warehouse_plugin": "warehouse_ros_sqlit::DatabaseConnection"},
    #     ],
    #     output="screen",
    # )

    return LaunchDescription([moveit_ros_benchmarks_node]) #, mongodb_server_node])
