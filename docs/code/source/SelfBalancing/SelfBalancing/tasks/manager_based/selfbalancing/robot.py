"""Two-wheeled self-balancing robot (TWIP), loaded from its URDF."""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import DelayedPDActuatorCfg
from isaaclab.assets import ArticulationCfg

# Project root = 6 folders up from this file (.../tasks/manager_based/selfbalancing/robot.py), so the
# path works wherever the project is.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), *([os.pardir] * 6)))
ROBOT_URDF_PATH = os.path.join(_PROJECT_ROOT, "assets", "RobotTwoWheel", "urdf", "SelfBalancingRobot_simplified.urdf")

TwoWheel_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        # converted from the URDF when the simulation starts, so it always matches the file
        asset_path=ROBOT_URDF_PATH,
        fix_base=False,
        root_link_name="base_link",
        # upper plate, battery and motors are merged into base_link -> 3 bodies: base_link, wheel1, wheel2
        merge_fixed_joints=True,
        self_collision=False,
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            drive_type="force",
            target_type="none",  # no position/velocity target: the actuator below applies torque
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=None, damping=None),
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=100.0,
            max_angular_velocity=100.0,
            max_depenetration_velocity=100.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # wheel bottom is 0.068 m below base_link (see the URDF) -> spawn 2 mm higher
        pos=(0.0, 0.0, 0.070),
        joint_pos={"wheel1_motor1_joint": 0.0, "wheel2_motor2_joint": 0.0},
    ),
    actuators={
        # Torque-controlled DC motors (stiffness 0) whose commands arrive 2-8 physics steps (10-40 ms)
        # late, resampled every episode -- the real motor driver reacts with a similar lag.
        "wheels": DelayedPDActuatorCfg(
            joint_names_expr=["wheel1_motor1_joint", "wheel2_motor2_joint"],
            effort_limit=0.49,  # Nm
            stiffness=0.0,
            damping=0.002,
            min_delay=2,
            max_delay=8,
        ),
    },
)
