# twip.py
import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets import ArticulationCfg

# Repo root, resolved relative to this file so the asset path works regardless of where the repo
# is cloned (not hardcoded to one machine's home directory).
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), *([os.pardir] * 6)))
ROBOT_TWO_WHEEL_URDF_PATH = os.path.join(_REPO_ROOT, "assets", "RobotTwoWheel", "urdf", "RobotTwoWheel.urdf")

##
# Configuration cho Robot TWIP (RobotTwoWheel -- URDF thật xuất từ SolidWorks, mesh STL, khác với
# TwoWheel.urdf cũ dùng hình học primitive)
##

TwoWheel_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        # Spawn thẳng từ URDF nguồn (không dùng USD dựng sẵn) để tránh lệch cấu hình mỗi khi URDF
        # thay đổi. Giữ nguyên cấu trúc thư mục package ROS gốc (RobotTwoWheel/urdf/*.urdf +
        # RobotTwoWheel/meshes/*.STL) để Isaac Sim URDF importer tự resolve được các
        # "package://RobotTwoWheel/meshes/..." trong file URDF (tìm ngược lên thư mục cùng tên
        # package chứa urdf/ -- không cần sửa lại path trong URDF).
        asset_path=ROBOT_TWO_WHEEL_URDF_PATH,
        fix_base=False,
        root_link_name="base_link",
        # URDF này chỉ có đúng 3 link (base_link, link_L, link_R) nối bằng 2 joint continuous, không
        # có joint fixed nào để gộp (khác TwoWheel.urdf cũ có nhiều link phụ nối fixed).
        merge_fixed_joints=False,
        self_collision=False,
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            drive_type="force",
            # không bake PD drive vào USD, actuator "wheels" bên dưới sẽ set gains lúc runtime
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
        # Đo trực tiếp từ URDF + bounding box mesh STL (không có trong URDF, phải đọc mesh):
        # joint_L/joint_R origin z = -0.034 so với base_link; bán kính bánh xe (link_L/link_R.STL)
        # ~0.033 (đo bounding box mesh, cả x lẫn z đều ~±0.033 -> bánh gần như hình trụ tròn đều).
        # đáy bánh xe = -0.034 - 0.033 = -0.067 so với base_link
        # → spawn base_link ở z=0.067 + margin (0.002) để bánh vừa chạm đất
        pos=(0.0, 0.0, 0.069),
        joint_pos={
            "joint_L": 0.0,
            "joint_R": 0.0,
        },
    ),
    actuators={
        "wheels": IdealPDActuatorCfg(
            joint_names_expr=["joint_L", "joint_R"],
            # KẾ THỪA effort_limit=0.49 Nm từ robot cũ (giả định cùng loại động cơ JGB37-520) --
            # CHƯA XÁC NHẬN cho RobotTwoWheel này, base_link nặng hơn (~0.967 kg so với ~0.184 kg cũ)
            # nên rất có thể cần torque lớn hơn thật -- cần kiểm tra lại datasheet động cơ thực tế.
            effort_limit=0.49,
            stiffness=0.0,
            damping=0.002,
        ),
    },
)
