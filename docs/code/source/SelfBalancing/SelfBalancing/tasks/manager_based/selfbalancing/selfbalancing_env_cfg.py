"""Self-balancing robot task: balance, follow a forward velocity and a heading."""

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import GaussianNoiseCfg as Gnoise

from . import mdp
from .robot import TwoWheel_CFG

WHEEL_JOINT_NAMES = ["wheel1_motor1_joint", "wheel2_motor2_joint"]  # left (+Y), right (-Y)
MOTOR_TORQUE_MAX = 0.49  # Nm, same as effort_limit in robot.py

VELOCITY_RANGE = (-0.15, 0.15)  # m/s
YAW_RANGE = (-math.pi, math.pi)  # rad, relative to the spawn heading
YAW_MAX_STEP = math.radians(60.0)  # rad, max change of the target heading at each resample
RESAMPLING_TIME = (5.0, 7.0)  # s
YAW_CURRICULUM_ITERATIONS = 2000  # PPO iterations with the target heading held at 0
STEPS_PER_ITERATION = 32  # = num_steps_per_env in agents/rsl_rl_ppo_cfg.py


##
# Scene
##


@configclass
class SelfbalancingSceneCfg(InteractiveSceneCfg):
    """Ground plane, light and the robot."""

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(
            size=(100.0, 100.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.5,
                dynamic_friction=1.2,
                restitution=0.0,
                friction_combine_mode="max",
                restitution_combine_mode="min",
            ),
        ),
    )
    robot: ArticulationCfg = TwoWheel_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )


##
# MDP
##


@configclass
class ActionsCfg:
    """One torque per wheel: action in [-1, 1] x MOTOR_TORQUE_MAX."""

    wheel_effort = mdp.JointEffortActionCfg(asset_name="robot", joint_names=WHEEL_JOINT_NAMES, scale=MOTOR_TORQUE_MAX)


@configclass
class CommandsCfg:
    """Forward velocity and heading targets."""

    target_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        ranges=VELOCITY_RANGE,
        resampling_time_range=RESAMPLING_TIME,
        rel_standing_envs=0.15,  # 15% of the time: stand still
        debug_vis=True,  # green = target, blue = actual velocity
    )
    target_yaw = mdp.UniformYawCommandCfg(
        asset_name="robot",
        ranges=YAW_RANGE,
        resampling_time_range=RESAMPLING_TIME,
        max_step=YAW_MAX_STEP,
        standing_command_name="target_velocity",
        debug_vis=True,  # orange = target heading
    )


@configclass
class ObservationsCfg:
    """Only what the real robot can measure (IMU, encoders, its own last command) plus the commands."""

    @configclass
    class PolicyCfg(ObsGroup):
        pitch_angle = ObsTerm(func=mdp.imu_pitch_angle, noise=Gnoise(mean=0.0, std=0.02))
        pitch_rate = ObsTerm(func=mdp.imu_pitch_rate, noise=Gnoise(mean=0.0, std=0.04))
        yaw_angle = ObsTerm(
            func=mdp.relative_yaw, params={"command_name": "target_yaw"}, noise=Gnoise(mean=0.0, std=0.02)
        )
        yaw_rate = ObsTerm(func=mdp.imu_yaw_rate, noise=Gnoise(mean=0.0, std=0.04))
        wheel1_vel = ObsTerm(
            func=mdp.joint_vel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[WHEEL_JOINT_NAMES[0]])},
            noise=Gnoise(mean=0.0, std=0.05),
        )
        wheel2_vel = ObsTerm(
            func=mdp.joint_vel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[WHEEL_JOINT_NAMES[1]])},
            noise=Gnoise(mean=0.0, std=0.05),
        )
        last_action1 = ObsTerm(func=mdp.last_action_index, params={"index": 0})
        last_action2 = ObsTerm(func=mdp.last_action_index, params={"index": 1})
        velocity_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "target_velocity"})
        yaw_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "target_yaw"})

        def __post_init__(self) -> None:
            self.enable_corruption = True  # add the noise (training only)
            self.concatenate_terms = True  # one vector, in the order above

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Domain randomization and disturbances."""

    # -- startup: fixed per robot for the whole run
    wheel_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["wheel1", "wheel2"]),
            "static_friction_range": (1.5, 1.5),
            "dynamic_friction_range": (1.2, 1.2),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 1,
        },
    )
    randomize_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["base_link"]),
            "com_range": {"x": (-0.005, 0.005), "y": (-0.0001, 0.0001), "z": (-0.005, 0.005)},
        },
    )
    randomize_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["base_link"]),
            "mass_distribution_params": (0.8, 1.2),
            "operation": "scale",
            "recompute_inertia": True,
        },
    )
    randomize_motor_friction = EventTerm(
        func=mdp.randomize_wheel_motor_friction,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=WHEEL_JOINT_NAMES),
            "friction_range": (0.010, 0.014),  # Nm
            "asymmetry": 0.1,  # left and right differ by up to +-10%
        },
    )

    # -- reset: every new episode
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pose_range": {"pitch": (-0.52, 0.52)},  # start tilted up to +-30 deg
            "velocity_range": {"pitch": (-0.5, 0.5)},
        },
    )
    reset_wheels = EventTerm(
        func=mdp.reset_joints_by_offset_symmetric,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=WHEEL_JOINT_NAMES),
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        },
    )

    # -- interval: a short forward/backward kick every 4-6 s (0 N = off; try (-200.0, 200.0))
    push_robot = EventTerm(
        func=mdp.push_by_external_force_local_x,
        mode="interval",
        interval_range_s=(4.0, 6.0),
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["base_link"]),
            "force_range": (0.0, 0.0),
            "body_offset_z": 0.10,
        },
    )


@configclass
class RewardsCfg:
    """Stay upright, follow the commands, move smoothly."""

    # -- survive
    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    terminating = RewTerm(func=mdp.is_terminated, weight=-50000.0)

    # -- stay upright
    upright = RewTerm(func=mdp.base_upright_penalty, weight=-1000.0, params={"asset_cfg": SceneEntityCfg("robot")})
    upright_bonus = RewTerm(
        func=mdp.base_upright_reward, weight=0.5, params={"asset_cfg": SceneEntityCfg("robot"), "std": 0.1}
    )
    pitch_rate = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.1, params={"asset_cfg": SceneEntityCfg("robot")})
    wheel_speed = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.00005,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=WHEEL_JOINT_NAMES)},
    )

    # -- follow the velocity command
    velocity_tracking = RewTerm(
        func=mdp.velocity_command_error_l2,
        weight=-5.0,
        params={"command_name": "target_velocity", "asset_cfg": SceneEntityCfg("robot"), "norm_scale": 0.2},
    )
    velocity_tracking_bonus = RewTerm(
        func=mdp.velocity_command_tracking_bonus,
        weight=7.5,
        params={"command_name": "target_velocity", "asset_cfg": SceneEntityCfg("robot"), "std": 0.05},
    )

    # -- follow the heading command
    yaw_tracking = RewTerm(
        func=mdp.yaw_command_error_l2, weight=-0.5, params={"command_name": "target_yaw", "norm_scale": 1.0}
    )
    yaw_tracking_bonus = RewTerm(
        func=mdp.yaw_command_tracking_bonus, weight=2.0, params={"command_name": "target_yaw", "std": 0.2}
    )
    yaw_rate = RewTerm(func=mdp.ang_vel_z_l2, weight=-0.5, params={"asset_cfg": SceneEntityCfg("robot")})

    # -- smooth motor commands
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.1)


@configclass
class TerminationsCfg:
    """End an episode on timeout or when the robot falls."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    fell_over = DoneTerm(func=mdp.bad_orientation, params={"asset_cfg": SceneEntityCfg("robot"), "limit_angle": 0.7})


@configclass
class CurriculumsCfg:
    """Keep the target heading at 0 for the first YAW_CURRICULUM_ITERATIONS iterations."""

    yaw_range = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "commands.target_yaw.ranges",
            "modify_fn": mdp.step_value_curriculum,
            "modify_params": {
                "start_value": (0.0, 0.0),
                "end_value": YAW_RANGE,
                "num_steps": YAW_CURRICULUM_ITERATIONS * STEPS_PER_ITERATION,
            },
        },
    )


##
# Environment
##


@configclass
class SelfbalancingEnvCfg(ManagerBasedRLEnvCfg):
    scene: SelfbalancingSceneCfg = SelfbalancingSceneCfg(num_envs=8196, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumsCfg = CurriculumsCfg()

    def __post_init__(self) -> None:
        self.decimation = 2  # policy every 2 physics steps
        self.episode_length_s = 20.0
        self.viewer.eye = (0.8, 0.8, 0.5)
        self.sim.dt = 1 / 200  # 200 Hz physics -> 100 Hz control, same as the firmware
        self.sim.render_interval = self.decimation


@configclass
class SelfbalancingEnvCfg_PLAY(SelfbalancingEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 360
        self.scene.env_spacing = 2.0
        self.observations.policy.enable_corruption = False  # no sensor noise
        self.curriculum.yaw_range = None  # full heading range right away
