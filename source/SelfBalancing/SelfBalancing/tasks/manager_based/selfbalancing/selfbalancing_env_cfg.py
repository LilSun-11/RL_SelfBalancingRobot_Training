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
from isaaclab.sensors import ImuCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import GaussianNoiseCfg as Gnoise

from . import mdp
from .twip import TwoWheel_CFG

WHEEL_JOINT_NAMES = ["joint_L", "joint_R"]  # RobotTwoWheel.urdf -- khác wheel1/2_motor cũ
MOTOR_TORQUE_MAX = 0.49  # Nm — khớp effort_limit của actuator "wheels" trong twip.py (CHƯA XÁC NHẬN
# cho RobotTwoWheel, kế thừa từ robot cũ -- base_link nặng hơn ~5x, cần kiểm tra lại động cơ thật)

# Theo dõi vị trí của cả 2 bánh xe (trung bình), dùng để train tiến/lùi theo 1 trục X.
# joint_pos của khớp bánh xe là góc quay liên tục, không wrap [-π, π] (quay vô hạn vòng, -∞ đến +∞).
WHEEL_RADIUS = 0.033  # m -- đo từ bounding box mesh link_L.STL/link_R.STL (±0.033 cả x lẫn z)

##
# Scene definition
##


@configclass
class TwoWheelSceneCfg(InteractiveSceneCfg):
    """Sân chơi + robot TWIP."""

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(
            size=(100.0, 100.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.5,
                dynamic_friction=1.2,
                restitution=0.0,
                # "max": luôn lấy ma sát lớn nhất giữa 2 vật liệu tiếp xúc (bánh xe - sàn),
                # tránh bị kéo xuống do combine mode mặc định "average" nếu 1 bên có ma sát thấp hơn.
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

    # IMU gắn trên thân xe, mô phỏng sensor vật lý (MPU6050, ICM42688...)
    # lin_acc_b = R^T * (a_body + gravity_bias) — đúng như accelerometer thực đo
    # khác với projected_gravity = R^T * g (đo hoàn hảo, không có nhiễu từ gia tốc thân xe)
    imu: ImuCfg = ImuCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        offset=ImuCfg.OffsetCfg(
            # pos: ước lượng gắn giữa thân (mesh base_link.STL cao tới z~0.130 so với gốc, xem
            # bounding box) -- CHƯA có vị trí PCB/mount thật, cần chỉnh lại nếu có thiết kế cụ thể.
            pos=(0.0, 0.0, 0.08),
            # KHÔNG cần xoay 90° như robot cũ: RobotTwoWheel.urdf có trục bánh xe = Y_robot (chuẩn
            # "pitch" thông thường) nên sensor-frame mặc định đã khớp trục pitch của robot, không cần
            # hoán trục X<->Y như robot TWIP cũ (trục bánh xe = X, phải xoay để dồn về sensor-Y).
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        gravity_bias=(0.0, 0.0, 9.81),
        update_period=0.0,
    )


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action: 1 xung PWM động cơ duy nhất trong [-1, 1] -> torque [-0.49, 0.49] Nm, áp CÙNG 1 giá
    trị cho cả 2 bánh (đảm bảo 2 bánh luôn đồng bộ tuyệt đối -- xem SymmetricWheelEffortAction trong
    mdp/actions.py). Tham khảo từ balancing-robot-pai: action_dim=1 khớp firmware thật (chỉ gửi 1
    lệnh chung cho cả 2 bánh), nên export ONNX cũng chỉ có 1 output, robot không thể tự rẽ/lệch hướng."""

    wheel_effort = mdp.SymmetricWheelEffortActionCfg(
        asset_name="robot",
        joint_names=WHEEL_JOINT_NAMES,
        scale=MOTOR_TORQUE_MAX,
    )


@configclass
class CommandsCfg:
    """Command terms cho MDP."""

    # quãng đường tiến/lùi mục tiêu (m), lấy mẫu 1 lần/episode (resampling_time_range = episode_length_s)
    # tạm thời ranges=(0.0, 0.0): ép policy học giữ nguyên vị trí bánh xe = 0 trước, chưa cho di chuyển
    # tự do -- khi policy giữ vị trí tốt rồi mới mở dần ranges ra (-0.5, 0.5) để học tiến/lùi thật sự.
    target_distance = mdp.UniformDistanceCommandCfg(
        asset_cfg=SceneEntityCfg("robot", joint_names=WHEEL_JOINT_NAMES),
        wheel_radius=WHEEL_RADIUS,
        ranges=(0.0, 0.0),
        resampling_time_range=(10.0, 10.0),
        debug_vis=True,
    )


@configclass
class ObservationsCfg:
    """Observation: góc/tốc độ nghiêng (IMU) + quãng đường/vận tốc từng bánh (encoder) + vị trí mục
    tiêu (distance_command)."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # nhiễu Gaussian mô phỏng IMU thật (accelerometer/gyro không bao giờ đo chính xác tuyệt đối)
        pitch_angle = ObsTerm(func=mdp.imu_pitch_angle, noise=Gnoise(mean=0.0, std=0.01))
        pitch_rate = ObsTerm(func=mdp.imu_pitch_rate, noise=Gnoise(mean=0.0, std=0.02))
        # quãng đường đã đi (m) của riêng từng bánh, tính từ góc quay khớp encoder (mốc 0 = lúc reset
        # episode) -- không gộp trung bình 2 bánh như trước, để policy thấy được cả 2 encoder riêng.
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
        # vận tốc góc (rad/s) của từng bánh -- tách riêng 2 term 1 chiều (giống wheel1/2_distance ở
        # trên) thay vì 1 term joint_vel 2 chiều, cho đồng nhất cách khai báo.
        wheel1_vel = ObsTerm(
            func=mdp.joint_vel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[WHEEL_JOINT_NAMES[0]])},
        )
        wheel2_vel = ObsTerm(
            func=mdp.joint_vel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[WHEEL_JOINT_NAMES[1]])},
        )
        # quãng đường tiến/lùi mục tiêu (m) mà policy cần đạt được
        # distance_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "target_distance"})

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # ma sát bánh xe - mặt sàn: gán 1 lần lúc khởi tạo (range hẹp = giá trị cố định, không random mỗi episode),
    # khớp với physics_material của ground (static=1.5, dynamic=1.2) để bánh xe bám sàn, không bị trượt.
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

    # randomize vị trí trọng tâm của thân xe (base_link, đã gộp upper_base/battery/bolts/motors qua
    # merge_fixed_joints) để policy không overfit vào 1 vị trí CoM lý tưởng, robust hơn khi CoM thực tế lệch.
    randomize_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["base_link"]),
            "com_range": {"x": (-0.001, 0.001), "y": (-0.0001, 0.0001), "z": (-0.001, 0.001)},
        },
    )

    # Randomize ma sát NGHỈ (static/Coulomb) và ma sát NHỚT (viscous) của khớp động cơ bánh xe -- khác
    # với wheel_friction ở trên (ma sát bánh xe - MẶT SÀN, rigid body material) và với damping trong
    # IdealPDActuatorCfg (twip.py, cố định = 0.002, mô hình hoá viscous damping NGAY TRONG công thức
    # PD của actuator) -- đây là ma sát CƠ KHÍ của khớp (chổi than/hộp số...), do vật lý engine (PhysX)
    # áp thêm, độc lập với effort do actuator tính ra.
    # Dùng hàm riêng (mdp.randomize_wheel_motor_friction_symmetric, KHÔNG phải built-in
    # mdp.randomize_joint_parameters) để cả 2 bánh nhận CÙNG 1 giá trị friction trong 1 env -- built-in
    # sample độc lập theo từng khớp, có thể khiến 2 bánh lệch friction nhau, phá vỡ tính đối xứng mà
    # SymmetricWheelEffortAction (áp cùng 1 torque cho cả 2 bánh) đang đảm bảo.
    # mode="startup": mỗi env có 1 giá trị cố định suốt vòng đời (đặc tính phần cứng, coi như không
    # đổi giữa các episode) -- robust hơn với sai lệch ma sát cơ khí thực tế giữa các robot.
    # LƯU Ý: từ Isaac Sim 5.0 trở lên, giá trị này là ĐƠN VỊ EFFORT (Nm cho khớp revolute), không còn
    # là hệ số không đơn vị như bản cũ -- range chọn nhỏ so với effort_limit=0.49 Nm, cần tinh chỉnh
    # thực nghiệm.
    # TẠM TẮT (2026-09-03): dù đã xác nhận cả process_actions lẫn write_joint_friction_coefficient_to_sim
    # tính/ghi giá trị GIỐNG HỆT NHAU cho joint_L/joint_R (kiểm chứng bằng runtime test thật), bật term
    # này vẫn gây robot lệch trái/phải rõ rệt -- tắt đi thì hết hẳn hiện tượng. Nhiều khả năng do bản
    # chất ma sát Coulomb (static/stick-slip) là mô hình KHÔNG TRƠN/rời rạc (bánh "bám" hay "trượt" phụ
    # thuộc ngưỡng tức thời) -- dù input đối xứng tuyệt đối, sai số dấu-phẩy-động cực nhỏ giữa 2 body có
    # thể khiến thời điểm "bứt" ma sát của mỗi bánh lệch nhau, và hệ cân bằng ngược vốn rất nhạy sẽ
    # khuếch đại chênh lệch đó theo thời gian -- không phải bug trong code, mà là đặc tính bất ổn của mô
    # hình Coulomb friction ở gần vận tốc 0 trong PhysX. Mở lại nếu muốn domain-randomize ma sát cơ khí
    # khớp, nhưng cân nhắc range nhỏ hơn hoặc theo dõi kỹ hành vi yaw khi bật.
    randomize_wheel_friction_motor = EventTerm(
        func=mdp.randomize_wheel_motor_friction_symmetric,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=WHEEL_JOINT_NAMES),
            "friction_range": (0.010, 0.012),
        },
    )

    # trục bánh xe thật (joint_L/joint_R, axis="0 1 0") = Y_robot = "pitch" trong quy ước
    # roll/pitch/yaw chuẩn -- KHÁC robot TWIP cũ (trục bánh xe = X = "roll"), nên random góc nghiêng
    # ban đầu phải dùng key "pitch" ở đây.
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

    # Đẩy bằng LỰC (thay vì set thẳng vận tốc) -- chỉ theo trục X của env (world, cố định hướng dù xe
    # đang nghiêng), đặt tại điểm lệch +Z so với base_link (body_offset_z, mô phỏng va chạm vào THÂN
    # TRÊN robot thay vì ngay khối tâm/trục bánh xe -- xem mdp/events.py). Lực chỉ tồn tại đúng 1
    # physics step (instantaneous_wrench_composer tự reset mỗi step) nên là 1 cú hích ngắn, không kéo
    # dài suốt cả interval_range_s.
    # Trục X (không phải Y như robot TWIP cũ) vì trục bánh xe (joint_L/joint_R) của RobotTwoWheel là
    # Y_robot -- hướng LĂN (hướng có thể sửa bằng torque bánh) là X, nên đẩy theo X mới tạo mất thăng
    # bằng CÓ Ý NGHĨA huấn luyện (đẩy theo Y sẽ lật ngang, robot không có cách nào tự chống lại).
    # body_offset_z=0.10: ước lượng theo bounding box mesh base_link.STL (cao tới z~0.130 so với gốc)
    # -- CHƯA khớp vị trí "thân trên" thật, chỉ là điểm giữa-cao hợp lý, có thể chỉnh lại.
    # F=m*Δv/dt với xung ~1.5 m/s trong 1 physics step (dt=1/200s), khối lượng xe ~1.15 kg:
    # F ~ 1.15*1.5/0.005 ~ 345 N -- range (-250,250) đang nhỏ hơn ước lượng này, có thể tăng nếu muốn
    # cú hích mạnh hơn.
    push_robot = EventTerm(
        func=mdp.push_by_external_force_x,
        mode="interval",
        interval_range_s=(5.0, 7.0),
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["base_link"]),
            "force_range": (-250.0, 250.0),
            "body_offset_z": 0.10,
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP -- thiết kế đơn giản hoá lại từ đầu.

    Nguyên tắc: mọi term đều dùng thang đo ~O(1) mỗi step (chuẩn hoá sai số/vận tốc nhỏ trước khi
    bình phương, xem norm_scale), để weight có thể so sánh trực tiếp với nhau và với "upright" --
    không dùng curriculum (weight cố định ngay từ đầu) để đơn giản, dễ resume, dễ debug.

    2 nhóm mục tiêu:
      - Đứng thẳng: upright/upright_bonus (vị trí góc) + pitch_rate (vận tốc góc, chống dao động).
      - Đứng yên tại target_distance: true_position_tracking (vị trí THẬT, ground-truth, không
        dùng encoder) + lin_vel_x (vận tốc, tín hiệu phanh nhanh hơn vì không phải đợi sai số vị
        trí tích luỹ).
    Cộng thêm action_rate (chống giật/rung tần số cao).
    """

    # -- Nền tảng --
    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    terminating = RewTerm(func=mdp.is_terminated, weight=-50000.0)

    # -- Đứng thẳng --
    upright = RewTerm(
        func=mdp.base_upright_penalty,
        weight=-10000.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    upright_bonus = RewTerm(
        func=mdp.base_upright_reward,
        weight=5.0,
        params={"asset_cfg": SceneEntityCfg("robot"), "std": 0.1},
    )
    pitch_rate = RewTerm(
        func=mdp.ang_vel_xy_l2,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    # Phạt trực tiếp tốc độ quay bánh xe (rad/s) -- khác với action_rate (chỉ phạt Δaction giữa 2
    # step, tức độ "giật", không phạt độ lớn): term này phạt thẳng bình phương vận tốc góc, nên 1
    # action lớn nhưng giữ ỔN ĐỊNH (không đổi) vẫn bị phạt nếu khiến bánh quay nhanh, ép robot quay
    # bánh chậm/vừa đủ để cân bằng thay vì quay nhanh liên tục.
    # Đặt tên "wheel_speed" (không phải "wheel_vel") để tránh trùng thuộc tính với "wheel_vel" trong
    # khối reward vị trí đang comment tắt bên dưới -- nếu bỏ comment khối đó, "wheel_vel" ở dưới sẽ
    # ghi đè âm thầm lên term cùng tên định nghĩa trước đó trong class.
    # wheel_speed = RewTerm(
    #     func=mdp.joint_vel_l2,
    #     weight=-0.0005,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=WHEEL_JOINT_NAMES)},
    # )
    # yaw_rate/yaw_angle đã bỏ (2026-08-24): từ khi ActionsCfg.wheel_effort đổi sang
    # SymmetricWheelEffortActionCfg (1 action, cùng torque cho cả 2 bánh), robot không còn cơ chế
    # chủ động rẽ/xoay quanh Z nữa (không thể chỉnh lệch torque 2 bánh), nên 2 reward chống xoay này
    # không còn cần thiết.

    # -- Đứng yên tại vị trí mục tiêu (target_distance) --
    # Chỉ phạt theo vị trí THẬT (root_pos_w, ground-truth), KHÔNG dùng encoder (wheel_distance) cho
    # reward -- encoder vẫn còn trong observation để policy "thấy" được, nhưng không dùng làm cơ sở
    # tính reward vì có thể bị "lừa" khi bánh xe trượt trên sàn (encoder báo sai số ~0 trong khi vị
    # trí thật đã lệch xa -- từng thấy 90.3% episode huỷ do out_of_range_true_pos trong khi encoder
    # báo gần như hoàn hảo).
    # true_position_tracking = RewTerm(
    #     func=mdp.true_position_error_l2,
    #     weight=-100.0,
    #     params={
    #         "command_name": "target_distance",
    #         "asset_cfg": SceneEntityCfg("robot"),
    #         "norm_scale": 0.20,
    #     },
    # )
    # Thử nghiệm: phạt độ lớn vị trí đo qua ENCODER (wheel_distance, khác với true_position_tracking
    # dùng ground-truth ở trên) -- xem có đủ để giữ vị trí mà không gây mất ổn định như
    # true_position_tracking hay không. Vẫn có rủi ro bị "lừa" nếu bánh xe trượt (encoder báo sai số
    # thấp hơn thực tế), nên chỉ coi đây là thử nghiệm, không thay thế lưới an toàn ground-truth.
    # encoder_distance_tracking = RewTerm(
    #     func=mdp.distance_command_error_l2,
    #     weight=-400.0,
    #     params={
    #         "command_name": "target_distance",
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=WHEEL_JOINT_NAMES),
    #         "wheel_radius": WHEEL_RADIUS,
    #         "norm_scale": 0.2,
    #     },
    # )
    # Thưởng dương (exponential, chặn trên = 1.0) khi quãng đường đo qua encoder gần đúng vị trí mục
    # tiêu (target_distance) -- bổ sung cho encoder_distance_tracking (phạt, không chặn) ở trên, tạo
    # "hố thưởng" nhọn quanh đúng vị trí 0 giống cặp base_upright_penalty/base_upright_reward.
    # encoder_distance_tracking_bonus = RewTerm(
    #     func=mdp.distance_command_tracking_bonus,
    #     weight=10.0,
    #     params={
    #         "command_name": "target_distance",
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=WHEEL_JOINT_NAMES),
    #         "std": 0.1,
    #         "wheel_radius": WHEEL_RADIUS,
    #     },
    # )
    # norm_scale=0.5 (m/s) khác với norm_scale=0.20 (m) của các term vị trí ở trên -- vận tốc bị bình
    # phương nên norm_scale nhỏ như vị trí sẽ biến 1 cú di chuyển sửa lỗi ngắn (vd để bắt kịp cú ngã)
    # thành phạt khổng lồ, khiến robot thà đứng yên chịu ngã còn hơn di chuyển để tự cứu.
    lin_vel_x = RewTerm(
        func=mdp.lin_vel_x_normalized_l2,
        weight=-30.0,
        params={"asset_cfg": SceneEntityCfg("robot"), "norm_scale": 0.5},
    )
    # Phạt tốc độ quay bánh xe (rad/s) -- bổ sung cho lin_vel_x (đo vận tốc thân xe): tránh quay bánh
    # vô ích (vd 2 bánh quay ngược nhau/rung tại chỗ mà không tạo chuyển động thân xe thật).
    # wheel_vel = RewTerm(
    #     func=mdp.joint_vel_l2,
    #     weight=-0.0005,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=WHEEL_JOINT_NAMES)},
    # )
    # Phạt góc quay (tích luỹ) 2 bánh lệch nhau -- vd bánh này quay nhiều/ít hơn bánh kia theo thời
    # gian nghĩa là xe đang rẽ/lệch hướng chứ không đi thẳng, dù vận tốc tức thời có thể trông ổn.
    # wheel_pos_diff = RewTerm(
    #     func=mdp.wheel_pos_diff_l2,
    #     weight=-5.0,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=WHEEL_JOINT_NAMES)},
    # )

    # -- Mượt hành động: chặn hành vi giật/rung bánh xe tần số cao (quan sát được khi play, gây pitch
    # dao động dữ dội và tự làm nhiễu IMU) -- weight tăng mạnh so với bản trước (-0.1 -> -1.0).
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.050)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # (2) Ngã quá góc cho phép (~40°, tính từ projected_gravity_b)
    fell_over = DoneTerm(
        func=mdp.bad_orientation,
        params={"asset_cfg": SceneEntityCfg("robot"), "limit_angle": 0.7},
    )
    # (3) out_of_range (encoder-based) TẮT -- chỉ dùng out_of_range_true_pos (ground-truth), khớp với
    # true_position_tracking (ground-truth) ở RewardsCfg, không dùng bản encoder vì có thể bị "lừa"
    # khi bánh xe trượt (xem giải thích ở true_position_tracking).
    # out_of_range = DoneTerm(
    #     func=mdp.distance_command_error_exceeded,
    #     params={
    #         "command_name": "target_distance",
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=WHEEL_JOINT_NAMES),
    #         "threshold": 0.20,
    #         "wheel_radius": WHEEL_RADIUS,
    #     },
    # )
    # (4) lưới an toàn ground-truth: huỷ episode nếu vị trí thật lệch quá xa target_distance, tránh
    # robot "bỏ chạy" vô hạn khỏi vị trí mục tiêu mà không bị phạt tương xứng.
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
        self.episode_length_s = 20.0
        # viewer settings
        self.viewer.eye = (0.8, 0.8, 0.5)
        # simulation settings
        # dt * decimation = 0.005 * 2 = 0.01s -> chu kỳ lấy mẫu/điều khiển của hệ là 10ms
        self.sim.dt = 1 / 200
        self.sim.render_interval = self.decimation


class SelfBalancingEnvCfg_PLAY(SelfBalancingEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # scene nhỏ hơn để play/quan sát
        self.scene.num_envs = 36
        self.scene.env_spacing = 2.5
        # tắt nhiễu observation khi play
        self.observations.policy.enable_corruption = False
        # giữ nguyên lực đẩy ngẫu nhiên (push_robot) khi play để xem policy phản ứng thế nào với
        # nhiễu/va chạm.
