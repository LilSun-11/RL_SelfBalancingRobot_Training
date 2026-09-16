from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply, sample_uniform

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def push_by_external_force_x(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    force_range: tuple[float, float],
    body_offset_z: float = 0.079,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["base_link"]),
) -> None:
    """Đẩy robot bằng LỰC (thay vì set thẳng vận tốc như push_by_setting_velocity), CHỈ theo trục X
    của môi trường (world X) -- is_global=True nên hướng đẩy luôn cố định theo X bất kể robot đang
    nghiêng thế nào (khác lực theo trục X CỦA THÂN XE, sẽ xoay theo robot).

    Dùng trục X (không phải Y như bản gốc cho robot TWIP cũ) vì RobotTwoWheel.urdf có trục bánh xe
    (joint_L/joint_R, axis="0 1 0") = Y_robot -- nghĩa là hướng LĂN của bánh xe (hướng robot có thể
    tự sửa bằng cách tăng/giảm torque bánh) là X. Đẩy theo X mới đúng là "đẩy theo hướng lăn", buộc
    policy phải phản ứng bằng torque bánh xe để giữ thăng bằng -- đẩy theo Y (dọc trục bánh xe) sẽ
    tạo lật NGANG mà 2 bánh nối tiếp không có cách nào chống lại được, không có ý nghĩa huấn luyện.

    Đặt lực tại điểm lệch +Z so với KHỐI TÂM THẬT của base_link (body_offset_z) để mô phỏng va chạm
    vào THÂN TRÊN robot: cùng độ lớn lực, đặt càng cao thì moment lật quanh trục bánh xe (pitch) càng
    lớn, giống va chạm thật hơn so với đẩy ngay tại khối tâm.

    QUAN TRỌNG: dùng body_com_pos_w (khối tâm THẬT trong world frame, đã cộng cả offset trong
    <inertial> của URDF lẫn dịch chuyển do randomize_com lúc runtime) làm gốc, KHÔNG dùng root_pos_w
    (gốc toạ độ kinematic/link frame -- chỉ là vị trí gán trong URDF, không nhất thiết trùng khối
    tâm). Nếu dùng root_pos_w mà khối tâm thật lệch trục Y (dù chỉ vài mm, do CoM lệch tâm hoặc do
    randomize_com), điểm đặt lực sẽ lệch khỏi khối tâm theo Y -> lực đẩy theo X sinh thêm 1 moment
    quay quanh Z (yaw) ngoài ý muốn (τ_z = -Δy × F_x). Dùng khối tâm thật cho X/Y đảm bảo điểm đặt lực
    LUÔN nằm trên mặt phẳng X-Z đi qua khối tâm (Δy = 0 tuyệt đối, không phụ thuộc CoM bị lệch bao
    nhiêu) -> triệt tiêu hoàn toàn moment yaw ký sinh, chỉ còn đúng moment pitch chủ đích.
    Hướng "lên" của body_offset_z vẫn lấy theo root_quat_w (hướng link/mesh thật) chứ không lấy theo
    body_com_quat_w (hướng trục quán tính chính, có thể hơi lệch so với mesh do các thành phần chéo
    ixy/ixz/iyz khác 0 trong URDF) -- muốn "cao hơn so với thân xe thật", không phải "cao hơn theo
    trục quán tính".

    Khi is_global=True, "positions" phải là toạ độ THẾ GIỚI tuyệt đối (không phải offset local).

    Dùng instantaneous_wrench_composer (KHÔNG phải permanent_wrench_composer): composer này tự
    reset về 0 sau mỗi physics step, nên lực chỉ tồn tại đúng 1 cú hích ngắn giống
    push_by_setting_velocity, không bị "kẹt" liên tục suốt cả interval_range_s tới lần đẩy tiếp theo.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    body_ids = asset_cfg.body_ids
    num_bodies = len(body_ids) if isinstance(body_ids, list) else 1

    forces = torch.zeros(len(env_ids), num_bodies, 3, device=asset.device)
    forces[:, :, 0] = sample_uniform(*force_range, (len(env_ids), num_bodies), asset.device)

    com_pos = asset.data.body_com_pos_w[env_ids][:, body_ids, :]
    root_quat = asset.data.root_quat_w[env_ids]
    offset_local = torch.zeros(len(env_ids), 3, device=asset.device)
    offset_local[:, 2] = body_offset_z
    z_offset_w = quat_apply(root_quat, offset_local).unsqueeze(1).expand(-1, num_bodies, -1)
    positions = com_pos + z_offset_w

    asset.instantaneous_wrench_composer.set_forces_and_torques(
        forces=forces,
        positions=positions,
        body_ids=body_ids,
        env_ids=env_ids,
        is_global=True,
    )


def randomize_wheel_motor_friction_symmetric(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    friction_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Randomize ma sát khớp động cơ (static/dynamic/viscous) -- CÙNG 1 giá trị cho MỌI khớp trong
    asset_cfg.joint_ids (vd cả 2 bánh xe) trong 1 env, khác với built-in mdp.randomize_joint_parameters
    (sample ĐỘC LẬP theo từng khớp, khiến 2 bánh có thể lệch friction nhau).

    SymmetricWheelEffortAction áp CÙNG 1 torque cho cả 2 bánh (đảm bảo đồng bộ) -- nếu để friction cơ
    khí lệch nhau giữa 2 bánh sẽ phá vỡ chính tính đối xứng đó (cùng lệnh torque nhưng phản ứng khác
    nhau), nên ở đây chỉ sample 1 giá trị/loại friction mỗi env, dùng broadcast (shape (E,1) ghi vào
    ô (E, num_joints)) để áp CHUNG cho tất cả khớp được chọn thay vì sample riêng từng khớp.

    static/dynamic/viscous vẫn sample ĐỘC LẬP với nhau (không dùng chung 1 giá trị cho cả 3 loại,
    giống hành vi built-in), dynamic được clamp <= static cho đúng vật lý (ma sát động luôn nhỏ hơn
    hoặc bằng ma sát nghỉ).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids
    # mode="startup" gọi apply() không truyền env_ids -> None nghĩa là áp cho MỌI env (giống cách
    # built-in mdp.randomize_joint_parameters tự resolve trong __call__).
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    static = sample_uniform(*friction_range, (len(env_ids), 1), asset.device)
    dynamic = torch.minimum(sample_uniform(*friction_range, (len(env_ids), 1), asset.device), static)
    viscous = sample_uniform(*friction_range, (len(env_ids), 1), asset.device)

    asset.write_joint_friction_coefficient_to_sim(
        joint_friction_coeff=static,
        joint_dynamic_friction_coeff=dynamic,
        joint_viscous_friction_coeff=viscous,
        joint_ids=joint_ids,
        env_ids=env_ids,
    )
