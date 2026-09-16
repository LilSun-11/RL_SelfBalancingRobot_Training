# robot.py
import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets import ArticulationCfg

# Repo root, resolved relative to this file so the asset path works regardless of where the repo
# is cloned (not hardcoded to one machine's home directory).
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), *([os.pardir] * 6)))
ROBOT_TWO_WHEEL_URDF_PATH = os.path.join(_REPO_ROOT, "assets", "RobotTwoWheel", "urdf", "RobotTwoWheel.urdf")

##
# TWIP robot configuration (RobotTwoWheel -- a real URDF exported from SolidWorks with STL meshes,
# unlike the old TwoWheel.urdf which used primitive geometry)
##

TwoWheel_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        # Spawn straight from the source URDF (not a prebaked USD) so it never drifts from URDF
        # edits. Keeps the original ROS package layout (RobotTwoWheel/urdf/*.urdf +
        # RobotTwoWheel/meshes/*.STL) so the Isaac Sim URDF importer can resolve
        # "package://RobotTwoWheel/meshes/..." on its own.
        asset_path=ROBOT_TWO_WHEEL_URDF_PATH,
        fix_base=False,
        root_link_name="base_link",
        # Only 3 links (base_link, link_L, link_R) joined by 2 continuous joints -- no fixed joints
        # to merge (unlike the old TwoWheel.urdf, which had several).
        merge_fixed_joints=False,
        self_collision=False,
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            drive_type="force",
            # No PD drive baked into the USD; the "wheels" actuator sets gains at runtime.
            target_type="none",
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
        # joint_L/joint_R origin is z=-0.034 from base_link; wheel radius (link_L/R.STL bounding
        # box) ~0.033 -> wheel bottom = -0.067 from base_link -> spawn base_link at 0.069 (with a
        # small margin) so the wheels just touch the ground.
        pos=(0.0, 0.0, 0.069),
        joint_pos={
            "joint_L": 0.0,
            "joint_R": 0.0,
        },
    ),
    actuators={
        "wheels": IdealPDActuatorCfg(
            joint_names_expr=["joint_L", "joint_R"],
            # Inherited from the old robot (same JGB37-520 motor assumption) -- UNCONFIRMED for this
            # chassis, which is much heavier (~0.967 kg vs ~0.184 kg base_link), so real torque needs
            # are likely higher. Check the motor datasheet.
            effort_limit=0.49,
            stiffness=0.0,
            damping=0.002,
        ),
    },
)
