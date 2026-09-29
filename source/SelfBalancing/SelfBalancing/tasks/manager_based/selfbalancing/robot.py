# robot.py
import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import DelayedPDActuatorCfg
from isaaclab.assets import ArticulationCfg

# Repo root, resolved relative to this file so the asset path works regardless of where the repo
# is cloned (not hardcoded to one machine's home directory).
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), *([os.pardir] * 6)))
ROBOT_TWO_WHEEL_URDF_PATH = os.path.join(
    _REPO_ROOT, "assets", "RobotTwoWheel", "urdf", "SelfBalancingRobot_simplified.urdf"
)

##
# TWIP robot configuration (SelfBalancingRobot_simplified -- box/cylinder primitives with
# analytically computed inertias per part, instead of the SolidWorks STL meshes of RobotTwoWheel.urdf)
##

TwoWheel_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        # Spawn straight from the source URDF (not a prebaked USD) so it never drifts from URDF edits.
        asset_path=ROBOT_TWO_WHEEL_URDF_PATH,
        fix_base=False,
        root_link_name="base_link",
        # Merges the fixed-joint links (upper_base_link, battery_link, motor1, motor2) into base_link,
        # leaving 3 rigid bodies: base_link (~1.09 kg combined), wheel1, wheel2.
        merge_fixed_joints=True,
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
        # base_link->motor z=-0.0235, motor->wheel z=-0.0105, wheel radius=0.034 -> wheel bottom =
        # -0.068 from base_link -> spawn base_link at 0.070 (small margin) so the wheels just touch.
        pos=(0.0, 0.0, 0.070),
        joint_pos={
            "wheel1_motor1_joint": 0.0,
            "wheel2_motor2_joint": 0.0,
        },
    ),
    actuators={
        # DelayedPDActuatorCfg (not the plain IdealPDActuatorCfg) delays the commanded effort by a
        # random number of PHYSICS steps, resampled every episode reset -- models the real motor
        # driver/H-bridge's response lag when reversing direction (dead-time to avoid shoot-through,
        # electrical/mechanical response), which the sim's actuator otherwise applies instantly. On
        # real hardware, action reversals were observed to lag noticeably; a plain instant-torque
        # actuator gives the policy no reason to ever compensate for that.
        # min/max_delay are in physics steps (sim.dt=1/200s -> 5ms/step, see selfbalancing_env_cfg.py)
        # -- 2-8 steps (10-40ms) is a placeholder guess, NOT measured from the real motor. Replace with
        # the actual observed reversal lag once measured on hardware.
        "wheels": DelayedPDActuatorCfg(
            joint_names_expr=["wheel1_motor1_joint", "wheel2_motor2_joint"],
            # Inherited from the old robot (same JGB37-520 motor assumption) -- UNCONFIRMED for this
            # chassis (~1.16 kg total), so real torque needs may be higher. Check the motor datasheet.
            effort_limit=0.49,
            stiffness=0.0,
            damping=0.002,
            min_delay=2,
            max_delay=8,
        ),
    },
)
