"""固定棋盘格 + ``solvePnP`` 的双相机**世界系**外参标定。

与 ``stereo.py`` 的分工
--------------------------------------------------------------------------
* ``stereo.py``：``cv2.stereoCalibrate``，直接解相机 0 → 相机 1 的相对位姿，
  不涉及世界系。适合"只想要一个基线"的场合，也留作交叉校验。
* 本模块（推荐路线，与 EasyMocap 的固定棋盘格思路一致）：把棋盘格当作**世界系
  基准**，对每台相机各自 ``solvePnP`` 求 ``X_cam = R @ X_world + t``，
  再由两个位姿合成相对位姿。好处是：
    - 每台相机的位姿是独立可解释、可单独检查的（相机在哪、朝哪）；
    - 后期 LED 三维动捕直接拿到世界系；
    - 不要求两台相机同时曝光（静止标定板本来就不需要）。

共面点的处理
--------------------------------------------------------------------------
88 个角点全部在 Z=0 平面上，属于平面 PnP。``cv2.solvePnPGeneric`` 配
``SOLVEPNP_IPPE`` 会给出**两个候选解**，本模块用重投影误差挑选，再用
``cv2.solvePnPRefineLM`` 精化（LM 迭代，不是自写优化）。

角点标号反转
--------------------------------------------------------------------------
两台相机从不同方向拍同一块板，检测器给出的角点 0 **未必是同一个物理角点**。
图案本身是中心对称的，180° 反转在图案上无法区分，只能靠：

1. 几何一致性检查（本模块自动）：若只翻了一台，两台相机的射线对应到不同物理
   角点，三角化出来的点会明显偏离已知的世界坐标（厘米级）；
2. 人工确认（界面）：把解出的世界系 X/Y 轴投影回原始图像，看是否与实物 12 格
   长边一致。整体 180°（两台都翻或都不翻）在几何上等价，必须人来定。

因此本模块**不会**替你猜"整体 180°"，只会修掉"两台之间不一致"的那一半。
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from calibration.board import BoardSpec, to_jsonable
from calibration.intrinsic import D_from, K_from
from calibration import stereo as st

MIN_FRAMES = 3            # 单台相机至少几张图才能判外参稳定性
RECOMMENDED_FRAMES = 8    # 建议张数（同一块固定板，多拍几张看抖动）


# ---------------------------------------------------------------------------
# 单相机：世界系位姿
# ---------------------------------------------------------------------------
def _ippe_candidates(obj: np.ndarray, img: np.ndarray, K: np.ndarray,
                     D: np.ndarray) -> tuple[list, list, list]:
    """``solvePnPGeneric`` + IPPE：平面目标会返回两个候选解。"""
    out = cv2.solvePnPGeneric(obj.astype(np.float32), img.astype(np.float32),
                              K.astype(np.float64), D.astype(np.float64),
                              flags=cv2.SOLVEPNP_IPPE)
    # OpenCV 4.10 的 Python 绑定返回 (retval, rvecs, tvecs, reprojErrs)
    rvecs = list(out[1]) if len(out) > 1 else []
    tvecs = list(out[2]) if len(out) > 2 else []
    errs = []
    if len(out) > 3 and out[3] is not None:
        arr = np.asarray(out[3]).reshape(-1)
        errs = [float(v) for v in arr]
    while len(errs) < len(rvecs):
        errs.append(float("inf"))
    return rvecs, tvecs, errs


def _reproj_stats(obj, img, rvec, tvec, K, D) -> dict:
    proj, _ = cv2.projectPoints(obj, np.asarray(rvec, np.float64),
                                np.asarray(tvec, np.float64).reshape(3, 1), K, D)
    err = np.linalg.norm(proj.reshape(-1, 2) - img.reshape(-1, 2), axis=1)
    worst = int(np.argmax(err))
    return {
        "rms_px": round(float(np.sqrt(np.mean(err ** 2))), 4),
        "mean_px": round(float(err.mean()), 4),
        "median_px": round(float(np.median(err)), 4),
        "max_px": round(float(err.max()), 4),
        "worst_corner": {"index": worst, "px": round(float(err[worst]), 4)},
    }


def _apply_correspondence(corners: np.ndarray, spec: BoardSpec,
                          cls: str) -> np.ndarray:
    """把检测到的角点按某一种格点对称重排，使其与世界点编号对应。

    矩形角点阵列的对称群是 D2，共 4 个元素：恒等、绕板法向 180°、沿行镜像、
    沿列镜像。**这 4 种在纯几何上完全不可区分**（图案是矩形格点，任何一种重排
    都能被某个合法位姿精确解释），所以不能靠残差判断，只能靠物理先验
    （相机必须在图案正面一侧）筛掉一半，剩下的一半（相差 180°）交给人工确认。
    """
    g = np.asarray(corners, np.float32).reshape(spec.rows, spec.cols, 2)
    if cls == "row":
        g = g[::-1, :]
    elif cls == "col":
        g = g[:, ::-1]
    elif cls == "both":
        g = g[::-1, ::-1]
    elif cls != "none":
        raise ValueError("未知的格点对应类型：%s" % cls)
    return g.reshape(-1, 2).copy()


def _compose_180(cls: str) -> str:
    """在当前对应上再叠加一次 180°（绕板法向），用于人工翻转。"""
    return {"none": "both", "both": "none", "row": "col", "col": "row"}[cls]


def _solve_once(obj, img, K, D) -> dict:
    """单次 PnP：IPPE 两候选取相机世界 Z 最大者，再 LM 精化。"""
    rvecs, tvecs, errs = _ippe_candidates(obj, img, K, D)
    if not rvecs:
        ok, rvec, tvec = cv2.solvePnP(obj, img, K, D, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            raise ValueError("solvePnP 失败（角点退化或数量不足）")
        rvecs, tvecs, errs = [rvec], [tvec], [float("inf")]

    centres = []
    for rv, tv in zip(rvecs, tvecs):
        Rc = cv2.Rodrigues(np.asarray(rv, np.float64))[0]
        centres.append(float((-Rc.T @ np.asarray(tv, np.float64).reshape(3))[2]))
    best = int(np.argmax(centres))
    rvec = np.asarray(rvecs[best], np.float64).reshape(3, 1)
    tvec = np.asarray(tvecs[best], np.float64).reshape(3, 1)
    candidates = [{"reproj_px": round(float(e), 4) if math.isfinite(e) else None,
                   "camera_z_m": round(z, 4)} for e, z in zip(errs, centres)]

    refined = False
    try:
        rvec, tvec = cv2.solvePnPRefineLM(obj, img, K, D, rvec, tvec)
        refined = True
    except cv2.error:
        refined = False

    R = cv2.Rodrigues(rvec)[0]
    t = tvec.reshape(3)
    rep = _reproj_stats(obj, img, rvec, tvec, K, D)
    centre = -R.T @ t
    z_cam = (R @ obj.T).T + t
    return {
        "R": R, "t": t, "rvec": np.asarray(rvec).reshape(3),
        "tvec": np.asarray(tvec).reshape(3),
        "reprojection": rep, "ippe_candidates": candidates, "refined": refined,
        "centre": centre, "camera_z": float(centre[2]),
        "points_in_front": float(np.mean(z_cam[:, 2] > 0)),
    }


def solve_pose(corners, spec: BoardSpec, K, D, *, correspondence: str = "auto",
               flip180: bool = False, refine: bool = True) -> dict:
    """单台相机对固定棋盘格求世界系位姿。

    返回 ``R``、``t`` 满足 ``X_cam = R @ X_world + t``（米）。

    ``correspondence``：
      * ``"auto"``（默认）—— 在 D2 的 4 种格点对应里，只保留"解出的相机位于图案
        正面一侧（世界 Z > 0）"的那些，再取重投影最小者。这一步解决的是"检测器
        的行列方向与世界 +Y 可能相反"这个纯坐标约定问题。
      * 也可以显式指定 ``"none" / "row" / "col" / "both"``。
    ``flip180``：在选定对应上再叠加一次绕板法向的 180°。这是**留给人工确认**的
    自由度（见模块 docstring），不影响重投影误差，只改变世界系 X/Y 的朝向。
    """
    obj = spec.world_points().astype(np.float64)
    raw = np.asarray(corners, np.float32).reshape(-1, 2)
    if len(raw) != spec.count:
        raise ValueError("角点数量 %d 与世界模型 %d 不符" % (len(raw), spec.count))
    K = np.asarray(K, np.float64)
    D = np.asarray(D, np.float64).reshape(-1)

    if correspondence == "auto":
        viable = []
        for cls in ("none", "row", "col", "both"):
            img = _apply_correspondence(raw, spec, cls)
            try:
                s = _solve_once(obj, img, K, D)
            except cv2.error:
                continue
            if s["camera_z"] > 0 and s["points_in_front"] >= 0.99:
                viable.append((cls, s))
        if not viable:
            # 实在没有正 Z 解：说明棋盘格朝向与假设不符，如实报错而不是硬塞一个
            raise ValueError(
                "找不到“相机位于图案正面一侧”的位姿：检查棋盘格是否正面向下、"
                "内角点行列数是否填反（12×9 方格 → 11×8 内角点）")
        viable.sort(key=lambda t: t[1]["reprojection"]["rms_px"])
        cls, S = viable[0]
        alternatives = [c for c, _ in viable[1:]]
    else:
        cls = correspondence
        S = _solve_once(obj, _apply_correspondence(raw, spec, cls), K, D)
        alternatives = []

    used = cls
    if flip180:
        used = _compose_180(used)
        S = _solve_once(obj, _apply_correspondence(raw, spec, used), K, D)

    R, t = S["R"], S["t"]
    centre = S["centre"]
    dist_centre = float(np.linalg.norm(centre))
    axis = R[2, :].copy()
    height_pattern = float(centre[2])
    height_floor = height_pattern + float(spec.surface_height_m)

    warnings = []
    if S["reprojection"]["rms_px"] > 1.0:
        warnings.append("重投影 RMS %.3fpx 偏大，检查角点检测或内参"
                        % S["reprojection"]["rms_px"])
    if S["points_in_front"] < 1.0:
        warnings.append("有角点落在相机后方（%.0f%% 在前方），位姿不可信"
                        % (100 * S["points_in_front"]))
    if dist_centre < 0.15:
        warnings.append("相机距板中心只有 %.3f m，距离过近" % dist_centre)

    return {
        "correspondence": used,
        "correspondence_alternatives": alternatives,
        "flip180": bool(flip180),
        "R": R.tolist(),
        "t": t.tolist(),
        "rvec": S["rvec"].tolist(),
        "tvec": S["tvec"].tolist(),
        "reprojection": S["reprojection"],
        "ippe_candidates": S["ippe_candidates"],
        "candidate_selection": "取相机世界 Z 最大者（图案正面朝上，相机必在其上方）",
        "refined": S["refined"],
        "camera_centre_world": centre.tolist(),
        "height_above_pattern_m": round(height_pattern, 6),
        "height_above_floor_m": round(height_floor, 6),
        "distance_to_pattern_centre_m": round(dist_centre, 6),
        "optical_axis_world": axis.tolist(),
        "points_in_front": round(S["points_in_front"], 4),
        "rotation": _rotation_diag(R),
        "warnings": warnings,
    }


def _rotation_diag(R: np.ndarray) -> dict:
    orth = float(np.max(np.abs(R @ R.T - np.eye(3))))
    det = float(np.linalg.det(R))
    return {"orthogonality_error": round(orth, 9), "determinant": round(det, 9),
            "valid": bool(orth < 1e-6 and abs(det - 1.0) < 1e-6)}


def cam_to_world(pose: dict) -> dict:
    """``X_world = R_cw @ X_cam + t_cw``（相机系 → 世界系）。"""
    R = np.asarray(pose["R"], np.float64)
    t = np.asarray(pose["t"], np.float64).reshape(3)
    R_cw = R.T
    t_cw = -R_cw @ t
    return {"R_wc": R_cw.tolist(), "t_wc": t_cw.tolist(),
            "convention": "X_world = R_wc @ X_cam + t_wc"}


# ---------------------------------------------------------------------------
# 双相机：相对位姿 + 一致性
# ---------------------------------------------------------------------------
def relative_pose(pose0: dict, pose1: dict) -> dict:
    """由两个世界系位姿合成相机 0 → 相机 1 的相对位姿。

    ``X_cam1 = R_10 @ X_cam0 + T_10``
    """
    R0 = np.asarray(pose0["R"], np.float64)
    t0 = np.asarray(pose0["t"], np.float64).reshape(3)
    R1 = np.asarray(pose1["R"], np.float64)
    t1 = np.asarray(pose1["t"], np.float64).reshape(3)

    R10 = R1 @ R0.T
    T10 = t1 - R10 @ t0
    c0 = -R0.T @ t0
    c1 = -R1.T @ t1

    tx = np.array([[0, -T10[2], T10[1]], [T10[2], 0, -T10[0]], [-T10[1], T10[0], 0]])
    E = tx @ R10

    ang = math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(R10) - 1) / 2.0))))
    return {
        "R_10": R10.tolist(),
        "T_10": T10.tolist(),
        "E": E.tolist(),
        "convention": "X_cam1 = R_10 @ X_cam0 + T_10",
        "T_unit": "m",
        "baseline_m": round(float(np.linalg.norm(c1 - c0)), 6),
        "baseline_vector_c0_to_c1_m": (c1 - c0).tolist(),
        "rotation_angle_deg": round(ang, 4),
        "camera0_centre_m": c0.tolist(),
        "camera1_centre_m": c1.tolist(),
        "rotation": _rotation_diag(R10),
    }


def pair_consistency(pose0: dict, pose1: dict, spec: BoardSpec,
                     K0, D0, K1, D1, corners0, corners1,
                     cls0: str, cls1: str) -> dict:
    """用两个位姿三角化棋盘格角点，与**已知世界坐标**比对。

    能发现"标定板在两次采集之间被碰动""某一路角点残缺/误检"这类问题；
    **不能**发现整体 180° 或格点镜像（那些是图案自身的对称，残差恒为 0）。
    """
    rel = relative_pose(pose0, pose1)
    R10 = np.asarray(rel["R_10"], np.float64)
    T10 = np.asarray(rel["T_10"], np.float64).reshape(3)

    img0 = _apply_correspondence(corners0, spec, cls0)
    img1 = _apply_correspondence(corners1, spec, cls1)

    X0 = st.triangulate(img0, img1, K0, D0, K1, D1, R10, T10)   # 相机 0 系
    R0 = np.asarray(pose0["R"], np.float64)
    t0 = np.asarray(pose0["t"], np.float64).reshape(3)
    Xw = (X0 - t0) @ R0

    truth = spec.world_points().astype(np.float64)
    err = np.linalg.norm(Xw - truth, axis=1)
    worst = int(np.argmax(err))
    return {
        "rms_m": round(float(np.sqrt(np.mean(err ** 2))), 6),
        "mean_m": round(float(err.mean()), 6),
        "max_m": round(float(err.max()), 6),
        "worst_corner": {"index": worst, "m": round(float(err[worst]), 6)},
        "blunder": bool(err.max() > 0.02),
        "relative": rel,
    }


# ---------------------------------------------------------------------------
# 外参总入口
# ---------------------------------------------------------------------------
def solve_rig(frames: dict, spec: BoardSpec, K0, D0, K1, D1, *,
              flip180: bool = False) -> dict:
    """用两台相机各自的多张固定板图像求解世界系外参。

    ``frames``：``{0: [corners, ...], 1: [corners, ...]}``，每项是同一台相机对
    **同一块固定棋盘格**的多次检测结果（同一物理姿态，多张图用于抖动评估）。

    ``flip180``：**全局**绕板法向转 180°。必须两台一起翻，不能只翻一台——单侧翻
    会让两台相机的世界系相差 180°，相对位姿和三角化全错（实测基线会从 2032mm
    跳到 1635mm）。

    为什么必须有人工开关：矩形角点阵列的对称群是 D2，4 种格点对应在纯几何上
    完全等价（重投影误差逐位相同）。本模块只能用"相机必须在图案正面一侧"
    筛掉两个镜像解，剩下的一对相差 180°，只能靠人看世界系坐标轴是否沿 12 格
    长边来定。
    """
    if 0 not in frames or 1 not in frames:
        raise ValueError("需要同时提供相机 0 和相机 1 的图像")
    n0, n1 = len(frames[0]), len(frames[1])
    if n0 < MIN_FRAMES or n1 < MIN_FRAMES:
        raise ValueError("每台相机至少需要 %d 张固定板图像（当前 %d / %d）"
                         % (MIN_FRAMES, n0, n1))

    # 每台相机：所有帧各解一次；代表位姿取重投影最好的一帧
    poses = []
    for cam, (K, D) in ((0, (K0, D0)), (1, (K1, D1))):
        per_frame = [solve_pose(c, spec, K, D, flip180=flip180) for c in frames[cam]]
        per_frame.sort(key=lambda p: p["reprojection"]["rms_px"])
        poses.append(per_frame)

    P0, P1 = poses[0][0], poses[1][0]
    cons = pair_consistency(P0, P1, spec, K0, D0, K1, D1,
                            frames[0][0], frames[1][0],
                            P0["correspondence"], P1["correspondence"])
    rel = cons["relative"]

    warn = []
    if cons["blunder"]:
        warn.append("两台相机的三角化残差 %.1f mm 偏大：检查角点是否都检测完整、"
                    "标定板在两次采集之间是否被碰动" % (1000 * cons["max_m"]))
    if rel["baseline_m"] < 0.05:
        warn.append("基线只有 %.1f mm，三角化精度会很差" % (1000 * rel["baseline_m"]))
    for cam, p in ((0, P0), (1, P1)):
        if p["height_above_floor_m"] <= 0:
            warn.append("相机 %d 的世界 Z 为负（%.3f m）：图案正面朝上时相机应在其上方，"
                        "请检查该相机的角点是否需要翻转 180°"
                        % (cam, p["height_above_floor_m"]))
        if p["points_in_front"] < 1.0:
            warn.append("相机 %d 有角点落在后方，位姿不可信" % cam)

    # 帧间抖动：同一个静止位姿，多张图的差异反映检测/内参的随机性
    jitter = {}
    for cam, plist in enumerate(poses):
        centres = np.array([p["camera_centre_world"] for p in plist])
        jitter[str(cam)] = {
            "frames": len(plist),
            "centre_std_m": round(float(np.linalg.norm(centres.std(axis=0))), 6),
            "centre_range_m": round(float(np.max(np.linalg.norm(
                centres - centres.mean(axis=0), axis=1))), 6),
            "reproj_rms_px": [round(p["reprojection"]["rms_px"], 4) for p in plist],
        }

    K0a, D0a = np.asarray(K0, np.float64), np.asarray(D0, np.float64).reshape(-1)
    K1a, D1a = np.asarray(K1, np.float64), np.asarray(D1, np.float64).reshape(-1)
    _ = (D0a, D1a)
    F = np.linalg.inv(K1a).T @ np.asarray(rel["E"]) @ np.linalg.inv(K0a)
    rel["F"] = F.tolist()

    return {
        "method": "solvePnP (IPPE candidates + LM refine) per camera against one fixed board",
        "board": spec.to_dict(),
        "flip180": bool(flip180),
        "cameras": {
            "0": dict(P0, cam_to_world=cam_to_world(P0)),
            "1": dict(P1, cam_to_world=cam_to_world(P1)),
        },
        "relative": rel,
        "convention": rel["convention"],
        "T_unit": "m",
        "baseline_m": rel["baseline_m"],
        "consistency": {"rms_m": cons["rms_m"], "mean_m": cons["mean_m"],
                        "max_m": cons["max_m"], "blunder": cons["blunder"],
                        "worst_corner": cons["worst_corner"],
                        "definition": "三角化角点与已知世界坐标的三维偏差（棋盘格为基准）"},
        "stability": jitter,
        "warnings": warn,
        "corner_correspondence": {
            "flip180_global": bool(flip180),
            "auto_class_0": P0["correspondence"],
            "auto_class_1": P1["correspondence"],
            "meaning": ("auto_class 是自动选定的格点对应（保证相机在图案正面一侧）；"
                        "flip180_global=True 表示在该对应上两台一起再转 180°"),
            "global_only": "180° 必须两台一起翻；单侧翻会让相对位姿与三角化全错",
            "how_to_verify": ("把世界系 X/Y 轴投影回图像（GET /api/calib/extrinsic/overlay），"
                              "确认 X 轴与实物 12 格长边一致、0 号角点在预期的那一角"),
            "geometric_note": ("矩形角点阵列有 D2 四重对称，纯几何无法区分；"
                               "本模块只用“相机在板面上方”筛掉镜像解，"
                               "剩下的 180° 必须人工确认"),
        },
        "caveat": ("三角化自洽性只能发现「板动了 / 检测错」这类问题，无法判断整体 180°；"
                   "三维精度请用 validate_rig 的独立数据（多个位置/高度）检验"),
    }


# ---------------------------------------------------------------------------
# 独立验证：多个位置 / 高度
# ---------------------------------------------------------------------------
def validate_rig(placements: list, spec: BoardSpec, rig: dict, K0, D0, K1, D1,
                 expected_centres=None) -> dict:
    """用独立采集（板挪到不同位置/高度）检验三角化精度。

    ``placements``：每项 ``{"id": str, "corners0": ..., "corners1": ...,
    "dt_ms": float|None}``。**不复用**外参标定的那几张图。
    """
    if not placements:
        raise ValueError("没有验证数据")
    K0 = np.asarray(K0, np.float64)
    D0 = np.asarray(D0, np.float64).reshape(-1)
    K1 = np.asarray(K1, np.float64)
    D1 = np.asarray(D1, np.float64).reshape(-1)
    rel = rig["relative"]
    R10 = np.asarray(rel["R_10"], np.float64)
    T10 = np.asarray(rel["T_10"], np.float64).reshape(3)
    F = np.asarray(rel["F"], np.float64)
    R0 = np.asarray(rig["cameras"]["0"]["R"], np.float64)
    t0 = np.asarray(rig["cameras"]["0"]["t"], np.float64).reshape(3)
    cls0 = rig["cameras"]["0"].get("correspondence", "none")
    cls1 = rig["cameras"]["1"].get("correspondence", "none")

    spacing = spec.square_m
    truth = spec.world_points().astype(np.float64)
    per, all_sp, all_rep, all_epi, centres = [], [], [], [], []

    for pl in placements:
        c0 = np.asarray(pl["corners0"], np.float32).reshape(-1, 2)
        c1 = np.asarray(pl["corners1"], np.float32).reshape(-1, 2)
        if len(c0) != spec.count or len(c1) != spec.count:
            continue
        i0 = _apply_correspondence(c0, spec, cls0)
        i1 = _apply_correspondence(c1, spec, cls1)

        X0 = st.triangulate(i0, i1, K0, D0, K1, D1, R10, T10)
        Xw = (X0 - t0) @ R0

        # 重投影：把三角化点投回两台相机
        e0 = st.reproject(X0, i0, K0, D0)
        X1 = X0 @ R10.T + T10
        e1 = st.reproject(X1, i1, K1, D1)
        rep = float(np.sqrt(np.mean(np.concatenate([e0, e1]) ** 2)))

        epi = float(np.median(st.epipolar_errors(i0, i1, F)))

        # 已知间距：横向相邻（30mm）与纵向相邻（30mm），网格是正方形所以都该是 30mm
        pairs = []
        for r in range(spec.rows):
            for c in range(spec.cols):
                k = r * spec.cols + c
                if c + 1 < spec.cols:
                    pairs.append((k, k + 1))
                if r + 1 < spec.rows:
                    pairs.append((k, k + spec.cols))
        d = np.array([np.linalg.norm(Xw[i] - Xw[j]) for i, j in pairs])
        sp_err = d - spacing
        all_sp.extend(np.abs(sp_err).tolist())

        # 平面度：对 Z 做最小二乘（世界系里板面是 Z=0 平面，但验证时板被挪动了，
        # 所以这里改为检查"重建点是否共面"——用 SVD 拟合平面）
        cen = Xw.mean(axis=0)
        u, s, vt = np.linalg.svd(Xw - cen)
        normal = vt[2]
        planarity = float(np.sqrt(np.mean(((Xw - cen) @ normal) ** 2)))

        all_rep.append(rep)
        all_epi.append(epi)
        centres.append(cen)

        per.append({
            "id": pl.get("id"),
            "dt_ms": pl.get("dt_ms"),
            "reprojection_rms_px": round(rep, 4),
            "epipolar_median_px": round(epi, 4),
            "spacing_mean_mm": round(float(d.mean() * 1000), 4),
            "spacing_mean_abs_error_mm": round(float(np.abs(sp_err).mean() * 1000), 4),
            "spacing_max_abs_error_mm": round(float(np.abs(sp_err).max() * 1000), 4),
            "spacing_rel_error_pct": round(float(100 * np.abs(sp_err).mean() / spacing), 4),
            "planarity_rms_mm": round(planarity * 1000, 4),
            "centre_world_m": [round(float(v), 6) for v in cen],
            "board_size_check_mm": {
                "x": round(float(np.linalg.norm(Xw[spec.cols - 1] - Xw[0]) * 1000), 3),
                "y": round(float(np.linalg.norm(
                    Xw[(spec.rows - 1) * spec.cols] - Xw[0]) * 1000), 3),
                "expected_x": round((spec.cols - 1) * spec.square_mm, 3),
                "expected_y": round((spec.rows - 1) * spec.square_mm, 3),
            },
        })

    if not per:
        raise ValueError("没有可用的验证帧对（角点数不符？）")

    centres_arr = np.array(centres)
    agg = {
        "placements": len(per),
        "reprojection_rms_px": {
            "mean": round(float(np.mean(all_rep)), 4),
            "max": round(float(np.max(all_rep)), 4),
        },
        "epipolar_median_px": {
            "mean": round(float(np.mean(all_epi)), 4),
            "max": round(float(np.max(all_epi)), 4),
        },
        "spacing_abs_error_mm": {
            "mean": round(float(np.mean(all_sp)), 4),
            "p95": round(float(np.percentile(all_sp, 95)), 4),
            "max": round(float(np.max(all_sp)), 4),
            "expected_spacing_mm": round(spec.square_mm, 3),
        },
        "centre_z_range_m": [
            round(float(centres_arr[:, 2].min()), 4),
            round(float(centres_arr[:, 2].max()), 4),
        ],
        "centre_xy_extent_m": [
            round(float(np.ptp(centres_arr[:, 0])), 4),
            round(float(np.ptp(centres_arr[:, 1])), 4),
        ],
    }

    cross = []
    if expected_centres:
        for chk in expected_centres:
            try:
                a, b = int(chk["i"]), int(chk["j"])
                measured = float(np.linalg.norm(centres_arr[a] - centres_arr[b]))
                expect = float(chk["mm"]) / 1000.0
                cross.append({
                    "i": a, "j": b,
                    "expected_mm": round(expect * 1000, 3),
                    "measured_mm": round(measured * 1000, 3),
                    "error_mm": round((measured - expect) * 1000, 3),
                    "rel_error_pct": round(100 * abs(measured - expect) / expect, 4),
                })
            except (IndexError, KeyError, ValueError):
                continue

    notes = [
        "验证数据必须是**独立采集**（板挪到别的位置/高度），不能复用外参标定那几张图",
        "spacing 误差用棋盘格 30mm 网格的已知间距检验米制尺度，不依赖任何外部真值",
        "planarity 检查重建点是否共面，能暴露角点误匹配与标号错误",
        "重投影误差小 ≠ 三维精度高；以 spacing 与 planarity 为准",
    ]
    if agg["placements"] < 3:
        notes.append("验证位置只有 %d 个，建议至少 3 个不同位置/高度" % agg["placements"])

    return {"aggregate": agg, "per_placement": per,
            "cross_checks": cross, "notes": notes}


__all__ = ["solve_pose", "cam_to_world", "relative_pose", "pair_consistency",
           "solve_rig", "validate_rig", "to_jsonable", "MIN_FRAMES",
           "RECOMMENDED_FRAMES"]
