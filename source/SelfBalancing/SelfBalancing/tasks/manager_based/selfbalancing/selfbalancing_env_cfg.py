# selfbalancing_env_cfg.py
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
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

WHEEL_JOINT_NAMES = ["joint_L", "joint_R"]  # RobotTwoWheel.urdf -- unlike the old wheel1/2_motor
MOTOR_TORQUE_MAX = 0.49  # Nm — matches the "wheels" actuator's effort_limit in robot.py (inherited
# from the old robot, UNCONFIRMED for this chassis -- base_link is ~5x heavier, motor may need revisiting)

WHEEL_RADIUS = 0.033  # m — measured from the link_L/link_R.STL mesh bounding box

##
# Scene definition
##


@configclass
class TwoWheelSceneCfg(InteractiveSceneCfg):
    """Ground plane + TWIP robot."""

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(
            size=(100.0, 100.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.5,
                dynamic_friction=1.2,
                restitution=0.0,
                # "max": always use the higher friction of the two contacting materials.
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

    # No IMU sensor: pitch_angle/pitch_rate read robot.data.projected_gravity_b/root_ang_vel_b
    # directly instead (see mdp.imu_pitch_angle/imu_pitch_rate). mdp.imu_lin_acc/imu_ang_vel are
    # still available if a noisy raw-accelerometer/gyro model is needed later.


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action: 2 independent motor commands in [-1, 1] -> torque [-0.49, 0.49] Nm per wheel.

    Uses the built-in JointEffortActionCfg (one action per wheel) instead of a symmetric single-
    action term: with randomize_wheel_motor_friction_symmetric enabled, the two wheels can still
    accumulate a position mismatch over time even under identical torque (PhysX's Coulomb friction
    model isn't perfectly smooth), and a symmetric action gives the policy no way to correct for
    that. With independent actions the policy can learn to bias torque to compensate -- the
    trade-off is it also gains the ability to yaw on purpose, so wheel_vel_diff_l2/wheel_pos_diff_l2
    in RewardsCfg exist to keep it motivated to stay in sync. Changing action_dim (1 -> 2) means old
    checkpoints can't be resumed.
    """

    wheel_effort = mdp.JointEffortActionCfg(
        asset_name="robot",
        joint_names=WHEEL_JOINT_NAMES,
        scale=MOTOR_TORQUE_MAX,
    )


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    # Target linear velocity (m/s) along body X, sampled once per episode (resampling_time_range =
    # episode_length_s). The robot must learn to hold a constant forward/backward speed rather than
    # travel to and hold a fixed position (see UniformVelocityCommand in mdp/commands.py).
    target_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        ranges=(-0.4, 0.4),
        resampling_time_range=(20.0, 20.0),
        debug_vis=True,
    )


@configclass
class ObservationsCfg:
    """Observation: tilt angle/rate (IMU-equivalent) + per-wheel encoder distance/velocity + target
    velocity."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # Gaussian noise mimics a real IMU (accelerometer/gyro never read perfectly).
        pitch_angle = ObsTerm(func=mdp.imu_pitch_angle, noise=Gnoise(mean=0.0, std=0.01))
        pitch_rate = ObsTerm(func=mdp.imu_pitch_rate, noise=Gnoise(mean=0.0, std=0.02))
        # Distance traveled (m) per wheel since episode reset, from the wheel encoder angle -- kept
        # per-wheel (not averaged) so the policy can see both encoders independently.
        wheel1_distance = ObsTerm(
            func=mdp.wheel_distance,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=[WHEEL_JOINT_NAMES[0]]),
                "wheel_radius": WHEEL_RADIUS,
            },
        )
        wheel2_distance = ObsTerm(
            func=mdp.wheel_distance,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=[WHEEL_JOINT_NAMES[1]]),
                "wheel_radius": WHEEL_RADIUS,
            },
        )
        # Per-wheel angular velocity (rad/s) -- split into two 1D terms (like wheel1/2_distance
        # above) instead of one 2D joint_vel term, for a consistent declaration style.
        wheel1_vel = ObsTerm(
            func=mdp.joint_vel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[WHEEL_JOINT_NAMES[0]])},
        )
        wheel2_vel = ObsTerm(
            func=mdp.joint_vel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[WHEEL_JOINT_NAMES[1]])},
        )
        # Target linear velocity (m/s) the policy needs to track.
        velocity_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "target_velocity"})

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # Wheel-ground friction: fixed at startup (degenerate range = constant), matching the ground's
    # physics material (static=1.5, dynamic=1.2) so wheels grip instead of slipping.
    wheel_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["link_L", "link_R"]),
            "static_friction_range": (1.5, 1.5),
            "dynamic_friction_range": (1.2, 1.2),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 1,
        },
    )

    # Randomize chassis CoM (base_link already merges upper_base/battery/bolts/motors via
    # merge_fixed_joints) so the policy doesn't overfit to one ideal CoM.
    randomize_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["base_link"]),
            "com_range": {"x": (-0.001, 0.001), "y": (-0.0001, 0.0001), "z": (-0.001, 0.001)},
        },
    )

    # Mechanical joint friction (motor/gearbox), distinct from wheel_friction (ground contact) and
    # from the actuator's own fixed damping (IdealPDActuatorCfg, damping=0.002) -- this is applied by
    # PhysX on top of actuator effort. Uses a custom function (not the built-in
    # randomize_joint_parameters) so both wheels get the same sampled value per env, instead of
    # independently-sampled friction that could itself become a source of drift. mode="startup":
    # fixed per env for its whole lifetime (a hardware trait, not something that varies per episode).
    # Isaac Sim >=5.0 treats this as an effort unit (Nm), not a unitless coefficient -- keep the range
    # small relative to effort_limit=0.49 Nm.
    randomize_wheel_friction_motor = EventTerm(
        func=mdp.randomize_wheel_motor_friction_symmetric,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=WHEEL_JOINT_NAMES),
            "friction_range": (0.010, 0.012),
        },
    )

    # Wheel axis (joint_L/joint_R, axis="0 1 0") = Y_robot = "pitch" in standard roll/pitch/yaw
    # convention (unlike the old TWIP robot, whose wheel axis = X = "roll"), so the initial tilt must
    # randomize "pitch" here.
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pose_range": {"pitch": (-0.52, 0.52)},
            "velocity_range": {"pitch": (-0.5, 0.5)},
        },
    )

    reset_wheels = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=WHEEL_JOINT_NAMES),
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        },
    )

    # Force-based push (not a velocity set) along the robot's own body X axis (not world X, since a
    # yaw-drifted robot's world X no longer points along its actual heading), applied above
    # base_link's true CoM (body_offset_z) to simulate a hit on the upper body rather than at the
    # CoM/wheel axis. Uses instantaneous_wrench_composer (auto-clears every physics step), so it's a
    # brief kick, not a sustained force. Magnitude ~ m*dv/dt for a ~1.5 m/s kick over one physics step
    # (dt=1/200s, mass ~1.15 kg) -> ~345 N; force_range is set below that, room to increase if a
    # stronger kick is wanted.
    push_robot = EventTerm(
        func=mdp.push_by_external_force_local_x,
        mode="interval",
        interval_range_s=(5.0, 7.0),
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["base_link"]),
            "force_range": (-200.0, 200.0),
            "body_offset_z": 0.10,
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP.

    All terms use an ~O(1) per-step scale (see norm_scale) so weights stay comparable to each other
    and to "upright" -- no curriculum, weights are fixed from the start.

    Two goals: stay upright (upright/upright_bonus + pitch_rate, plus yaw_rate to suppress spinning
    now that the two wheels are driven independently), and track target_velocity (velocity_tracking,
    ground-truth body velocity). Plus action_rate for smoothness.
    """

    # -- Baseline --
    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    terminating = RewTerm(func=mdp.is_terminated, weight=-100000.0)

    # -- Stay upright --
    upright = RewTerm(
        func=mdp.base_upright_penalty,
        weight=-50.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    upright_bonus = RewTerm(
        func=mdp.base_upright_reward,
        weight=2.0,
        params={"asset_cfg": SceneEntityCfg("robot"), "std": 0.1},
    )
    pitch_rate = RewTerm(
        func=mdp.ang_vel_xy_l2,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    # Named "wheel_speed" (not "wheel_vel") to avoid clashing if a "wheel_vel"-named term is ever
    # added elsewhere in this class.
    wheel_speed = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.00005,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=WHEEL_JOINT_NAMES)},
    )
    # With independent per-wheel torque, the robot has an actual mechanism to spin in place, and
    # nothing else stops it from doing so (observed in practice) -- this penalizes yaw rate directly.
    yaw_rate = RewTerm(
        func=mdp.ang_vel_z_l2,
        weight=-0.10,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # -- Track target velocity --
    # Penalize squared error between the body's TRUE linear velocity (root_lin_vel_b, ground-truth)
    # and target_velocity. No per-wheel split needed (unlike the old encoder-based distance terms):
    # root_lin_vel_b is already the true body velocity, so wheel-spin-averaging can't fake it -- two
    # wheels spinning against each other shows up as yaw (ang_vel_z), already penalized above.
    velocity_tracking = RewTerm(
        func=mdp.velocity_command_error_l2,
        weight=-0.25,
        params={
            "command_name": "target_velocity",
            "asset_cfg": SceneEntityCfg("robot"),
            "norm_scale": 0.20,
        },
    )
    # Bounded exponential bonus (like upright_bonus) for tracking target_velocity closely --
    # complements velocity_tracking's unbounded penalty with a clear positive signal.
    velocity_tracking_bonus = RewTerm(
        func=mdp.velocity_command_tracking_bonus,
        weight=0.5,
        params={
            "command_name": "target_velocity",
            "asset_cfg": SceneEntityCfg("robot"),
            "std": 0.1,
        },
    )
    # lin_vel_x_normalized_l2 would penalize velocity magnitude directly -- that's the right shape
    # for a "hold position" task, but directly contradicts a nonzero velocity_tracking target, so it
    # stays out of this reward set.

    # Penalizes the two wheels' instantaneous angular velocity differing (unlike wheel_speed, which
    # penalizes magnitude regardless of whether the two wheels match) -- catches wheel-speed mismatch
    # the moment it starts, complementing wheel_pos_diff_l2 (accumulated, see below) which only
    # catches it once the position gap is large enough.
    wheel_vel_diff = RewTerm(
        func=mdp.wheel_vel_diff_l2,
        weight=-0.005,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=WHEEL_JOINT_NAMES)},
    )
    # wheel_pos_diff = RewTerm(
    #     func=mdp.wheel_pos_diff_l2,
    #     weight=-0.01,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=WHEEL_JOINT_NAMES)},
    # )

    # Smooth actions: suppresses high-frequency wheel jitter (observed during play -- causes violent
    # pitch oscillation and feeds back into IMU noise).
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.005)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # (2) Fell past the allowed tilt angle (~40°, from projected_gravity_b)
    fell_over = DoneTerm(
        func=mdp.bad_orientation,
        params={"asset_cfg": SceneEntityCfg("robot"), "limit_angle": 0.7},
    )
    # (3) out_of_range (encoder-based) stays disabled — only the ground-truth version below is
    # active, to match a ground-truth reward term rather than one that trusts wheel encoders.
    # out_of_range = DoneTerm(
    #     func=mdp.distance_command_error_exceeded,
    #     params={
    #         "command_name": "target_distance",
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=WHEEL_JOINT_NAMES),
    #         "threshold": 0.20,
    #         "wheel_radius": WHEEL_RADIUS,
    #     },
    # )
    # (4) Ground-truth safety net: with target_velocity, unbounded drift is the intended behavior (a
    # velocity command has no destination), so this stays disabled unless a position-holding command
    # comes back.
    # out_of_range_true_pos = DoneTerm(
    #     func=mdp.base_position_error_exceeded,
    #     params={"asset_cfg": SceneEntityCfg("robot"), "threshold": 0.20},
    # )


##
# Environment configuration
##


@configclass
class SelfBalancingEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: TwoWheelSceneCfg = TwoWheelSceneCfg(num_envs=8196, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 200.0
        # viewer settings
        self.viewer.eye = (0.8, 0.8, 0.5)
        # simulation settings
        # dt * decimation = 0.005 * 2 = 0.01s -> 100 Hz control loop
        self.sim.dt = 1 / 200
        self.sim.render_interval = self.decimation


class SelfBalancingEnvCfg_PLAY(SelfBalancingEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # smaller scene for play/observation
        self.scene.num_envs = 36
        self.scene.env_spacing = 2.5
        # disable observation noise during play
        self.observations.policy.enable_corruption = False
        # keep push_robot active during play to see how the policy handles disturbances
        # Shortened for play: the training config's episode_length_s=200.0 is too long to watch a
        # respawn happen (time_out still works either way -- it's just a long wait at 200s).
        self.episode_length_s = 20.0
