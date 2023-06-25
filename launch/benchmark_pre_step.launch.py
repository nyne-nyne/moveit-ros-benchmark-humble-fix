import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess
from ament_index_python.packages import get_package_share_directory
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    moveit_config = (
        MoveItConfigsBuilder("kairosab_ur10_egh")
        .robot_description(file_path="config/rbkairos.urdf.xacro")
        .trajectory_execution(file_path="config/moveit_controllers.yaml")
        .to_moveit_configs()
    )
    
    warehouse_ros_config = {
        "warehouse_plugin": "warehouse_ros_sqlite::DatabaseConnection",
        "warehouse_host": "localhost"
    }

    # MoveGroup Node | remapping needed 
    run_move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        remappings=[('/joint_states', '/kairosAB/arm/joint_states')],
        parameters=[moveit_config.to_dict(), warehouse_ros_config,],
    )

    # RViz
    rviz_config_file = (
        get_package_share_directory("kairos_simple_mover_group")
        + "/config/moveit_group_tutorial.rviz"
    )
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2_moveit2",
        output="log",
        arguments=["-d", rviz_config_file],
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            warehouse_ros_config,
        ],
    )

    # Static TF
    static_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="static_transform_publisher",
        output="log",
        arguments=["0.0", "0.0", "0.0", "0.0", "0.0",
                   "0.0", "world", "kairosAB_base_footprint"],
    )

    return LaunchDescription(
        [
            static_tf,
            rviz_node,
            run_move_group_node,
            # motion_planning_node,
            # robot_state_publisher,
            # ros2_control_node,
        ]
    )
