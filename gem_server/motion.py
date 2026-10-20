"""Export authoritative SOMA joint transforms without a renderer or retargeter."""
import json
import warnings
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation


def to_motion(params, soma, fps, metadata):
    import torch
    from soma.geometry.rig_utils import apply_joint_orient_local, joint_local_to_world
    from soma.geometry.transforms import SE3_from_Rt
    from gem.utils.rotation_conversions import axis_angle_to_matrix

    device = soma.device
    identity = params["identity_coeffs"].mean(dim=0, keepdim=True).to(device).float()
    scale = params["scale_params"].mean(dim=0, keepdim=True).to(device).float()
    soma.prepare_identity(identity, scale[:, 1:], repose_to_bind_pose=False, global_scale=scale[:, :1])
    sk = soma.batched_skinning
    sk.rebind(soma._cached_bind_transforms_world, soma._cached_rest_shape)
    body = params["body_pose"].to(device).float().reshape(-1, 76, 3)
    orient = params["global_orient"].to(device).float().reshape(-1, 1, 3)
    aa = torch.cat([orient, body], dim=1)
    local_rot = soma._pad_poses(axis_angle_to_matrix(aa))
    length, count = local_rot.shape[:2]

    def world(rot, hips):
        trans = sk.local_translations.reshape(-1, count, 3)[:1].expand(rot.shape[0], -1, -1).clone()
        trans[:, 1] = hips
        rot = apply_joint_orient_local(rot, sk.joint_orient, sk._orient_parent_T)
        return joint_local_to_world(SE3_from_Rt(rot, trans), soma.joint_parent_ids)

    rest_rot = torch.eye(3, device=device).repeat(1, count, 1, 1)
    rest = world(rest_rot, torch.zeros(1, 3, device=device))
    posed = world(local_rot, params["transl"].to(device).float())
    names_data = soma.rig_data.get("joint_names")
    if names_data is None:
        raise RuntimeError("SOMA rig 缺少 joint_names，无法可靠导出骨架")
    names = [n.decode() if isinstance(n, bytes) else str(n) for n in names_data]
    if len(names) != count:
        raise RuntimeError(f"SOMA names/transform mismatch: {len(names)} / {count}")
    parents = soma.joint_parent_ids.cpu().numpy().astype(int)
    parents[0] = -1
    rest = rest[0].detach().cpu().numpy()
    posed = posed.detach().cpu().numpy()
    if not np.isfinite(posed).all():
        raise RuntimeError("GEM 输出包含 NaN/Inf，拒绝导出损坏动作")
    return {"schema": "gem-soma-motion-v1", "fps": float(fps), "frames": length,
            "unit": "meters", "up_axis": "Y", "names": names, "parents": parents.tolist(),
            "rest_world": np.round(rest, 7).tolist(),
            "pose_world": np.round(posed, 7).tolist(), "metadata": metadata}


def write_motion(motion, folder):
    folder = Path(folder)
    (folder / "motion.json").write_text(json.dumps(motion, ensure_ascii=False, separators=(",", ":")), "utf-8")
    write_bvh(motion, folder / "motion.bvh")


def write_bvh(motion, path):
    rest, posed = np.asarray(motion["rest_world"]), np.asarray(motion["pose_world"])
    parents, names = motion["parents"], motion["names"]
    count = len(names)
    children = [[] for _ in names]
    for j, p in enumerate(parents):
        if p >= 0:
            children[p].append(j)
    local = posed.copy()
    local_rest = rest.copy()
    for j, p in enumerate(parents):
        if p >= 0:
            local[:, j] = np.linalg.inv(posed[:, p]) @ posed[:, j]
            local_rest[j] = np.linalg.inv(rest[p]) @ rest[j]
    offsets = local_rest[:, :3, 3]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        euler = Rotation.from_matrix(local[:, :, :3, :3].reshape(-1, 3, 3).copy()).as_euler("ZYX", degrees=True).reshape(-1, count, 3)
    order = []
    lines = ["HIERARCHY"]

    def joint(j, depth):
        ind = "  " * depth
        name = "".join(c if c.isalnum() or c in "_.-" else "_" for c in names[j])
        order.append(j)
        lines.extend([f"{ind}{'ROOT' if depth == 0 else 'JOINT'} {name}", f"{ind}{{",
                      f"{ind}  OFFSET " + " ".join(f"{x:.7f}" for x in offsets[j]),
                      f"{ind}  CHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation"])
        for child in children[j]:
            joint(child, depth + 1)
        if not children[j]:
            lines.extend([f"{ind}  End Site", f"{ind}  {{", f"{ind}    OFFSET 0 0.01 0", f"{ind}  }}"])
        lines.append(f"{ind}}}")
    joint(0, 0)
    lines.extend(["MOTION", f"Frames: {len(posed)}", f"Frame Time: {1 / motion['fps']:.9f}"])
    for t in range(len(posed)):
        vals = []
        for j in order:
            vals.extend(local[t, j, :3, 3] - offsets[j])
            vals.extend(euler[t, j])
        lines.append(" ".join(f"{x:.7f}" for x in vals))
    Path(path).write_text("\n".join(lines) + "\n", "utf-8")
