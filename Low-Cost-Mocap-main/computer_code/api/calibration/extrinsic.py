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
from calibration.intrinsic import D_from, K_from, undistort_points
from calibration import stereo as st

MIN_FRAMES = 1            # 固定板 + 固定相机：一张图就足以 solvePnP
RECOMMENDED_FRAMES = 3    # 可选多采几组观察位姿抖动，不强制


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
    """单次 PnP：先剔除物理上不可能的 IPPE 候选，再在合理候选里比重投影，最后 LM 精化。

    平面目标的 IPPE 两个解互为"穿过板面的镜像"，重投影误差可能都非常小；因此
    **不能只看误差**，要先用物理条件筛：相机必须在图案正面一侧（世界 Z > 0），
    且所有角点都在相机前方。筛完仍然并列时，标记 ``ambiguous`` 交给上层提示。
    """
    rvecs, tvecs, errs = _ippe_candidates(obj, img, K, D)
    if not rvecs:
        ok, rvec, tvec = cv2.solvePnP(obj, img, K, D, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            raise ValueError("solvePnP 失败（角点退化或数量不足）")
        rvecs, tvecs, errs = [rvec], [tvec], [float("inf")]

    cands = []
    for rv, tv, e in zip(rvecs, tvecs, errs):
        Rc = cv2.Rodrigues(np.asarray(rv, np.float64))[0]
        tc = np.asarray(tv, np.float64).reshape(3)
        z_cam = (Rc @ obj.T).T + tc
        cands.append({
            "rvec": rv, "tvec": tv, "R": Rc, "t": tc,
            "camera_z": float((-Rc.T @ tc)[2]),
            "points_in_front": float(np.mean(z_cam[:, 2] > 0)),
            "rms": float(e) if math.isfinite(e) else None,
        })

    def _rms(c):
        return c["rms"] if c["rms"] is not None else 1e9

    plausible = [c for c in cands if c["camera_z"] > 0 and c["points_in_front"] >= 0.99]
    no_plausible = not plausible
    if plausible:
        plausible.sort(key=_rms)
        chosen = plausible[0]
        ambiguous = (len(plausible) > 1
                     and abs(_rms(plausible[0]) - _rms(plausible[1])) < 0.05)
    else:
        # 没有一个候选能把相机放到板面上方：仍给一个（Z 最大），但上层会报错/告警
        chosen = max(cands, key=lambda c: c["camera_z"])
        ambiguous = False

    rvec = np.asarray(chosen["rvec"], np.float64).reshape(3, 1)
    tvec = np.asarray(chosen["tvec"], np.float64).reshape(3, 1)
    candidates = [{"reproj_px": (round(c["rms"], 4) if c["rms"] is not None else None),
                   "camera_z_m": round(c["camera_z"], 4),
                   "points_in_front": round(c["points_in_front"], 4),
                   "plausible": bool(c["camera_z"] > 0 and c["points_in_front"] >= 0.99)}
                  for c in cands]

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
        "ambiguous": bool(ambiguous), "no_plausible": bool(no_plausible),
    }


def solve_pose(corners, spec: BoardSpec, K, D, *, correspondence: str = "auto",
               flip180: bool = False, refine: bool = True) -> dict:
    """单台相机对固定棋盘格求世界系位姿。

    返回 ``R``、``t`` 满足 ``X_cam = R @ X_world + t``（米）。

    ``correspondence``：
      * ``"auto"``（默认）—— 在 D2 的 4 种格点对应里，只保留"解出的相机位于图案
        正面一侧（世界 Z > 0）"的候选。**刻意不按重投影误差排序**：剩下的两种对应
        相差 180°，误差逐位相同，按误差选等于瞎猜。这里按固定顺序取第一个，
        并把是否仍有歧义告诉上层（``orientation_ambiguous``）。
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

    ambiguous_candidates = False
    if correspondence == "auto":
        viable = []
        for cls in ("none", "row", "col", "both"):
            try:
                s = _solve_once(obj, _apply_correspondence(raw, spec, cls), K, D)
            except cv2.error:
                continue
            if s["camera_z"] > 0 and s["points_in_front"] >= 0.99:
                viable.append((cls, s))
        if not viable:
            raise ValueError(
                "找不到“相机位于图案正面一侧”的位姿：检查棋盘格是否正面向下、"
                "内角点行列数是否填反（12×9 方格 → 11×8 内角点）")
        # 固定顺序取第一个，不按误差挑（见 docstring）
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

    # 两种候选对应相差 180°，几何完全等价；只要还有备选就说明方向未经确认
    orientation_ambiguous = bool(alternatives) or bool(S.get("ambiguous"))
    ambiguous_candidates = bool(S.get("ambiguous"))

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
    if ambiguous_candidates:
        warnings.append("PnP 候选位姿的重投影误差几乎相同，无法自动区分，需人工确认")
    if S.get("no_plausible"):
        warnings.append("IPPE 候选里没有“相机在板面上方”的解，位姿不可信")

    return {
        "correspondence": used,
        "correspondence_alternatives": alternatives,
        "orientation_ambiguous": orientation_ambiguous,
        "flip180": bool(flip180),
        "R": R.tolist(),
        "t": t.tolist(),
        "rvec": S["rvec"].tolist(),
        "tvec": S["tvec"].tolist(),
        "reprojection": S["reprojection"],
        "ippe_candidates": S["ippe_candidates"],
        "candidate_selection": "先按“相机 Z>0 且角点全在前方”筛，再比重投影",
        "refined": S["refined"],
        "camera_centre_world": centre.tolist(),
        "height_above_pattern_m": round(height_pattern, 6),
        "height_above_floor_m": round(height_floor, 6),
        "distance_to_pattern_centre_m": round(dist_centre, 6),
        "optical_axis_world": axis.tolist(),
        "points_in_front": round(S["points_in_front"], 4),
        "rotation": st.rotation_diagnostics(R),
        "warnings": warnings,
    }


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
        "rotation": st.rotation_diagnostics(R10),
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
def _flip_of(cam: int, flip180: bool, flip_cam) -> bool:
    """取该相机的 180° 开关：显式给了就用自己的，否则跟随全局。"""
    if not flip_cam:
        return bool(flip180)
    v = flip_cam.get(str(cam), flip_cam.get(cam))
    return bool(flip180) if v is None else bool(v)


def _multi_view_consistency(chosen: dict, frames: dict, spec: BoardSpec,
                            intrinsics: dict) -> dict:
    """各相机最优帧观测同一块固定板，联合三角化全部角点并比对已知世界坐标。"""
    truth = spec.world_points().astype(np.float64)
    cams = sorted(chosen)
    obs = []
    for c in cams:
        idx, pose = chosen[c]
        K, D = intrinsics[c]
        corners = _apply_correspondence(frames[c][idx], spec,
                                        pose["correspondence"])
        norm = undistort_points(corners, K, D)
        P = np.hstack([np.asarray(pose["R"], np.float64),
                       np.asarray(pose["t"], np.float64).reshape(3, 1)])
        obs.append((c, corners, norm, P))

    X = []
    reproj = {c: [] for c in cams}
    for k in range(spec.count):
        pts = [norm[k] for _c, _raw, norm, _P in obs]
        Ps = [P for _c, _raw, _norm, P in obs]
        Xk = st.triangulate_dlt(pts, Ps)
        if Xk is None:
            Xk = np.full(3, np.nan)
        X.append(Xk)
        for c, raw, _norm, _P in obs:
            K, D = intrinsics[c]
            pose = chosen[c][1]
            proj = st.project_world(Xk.reshape(1, 3), K, D,
                                    pose["R"], pose["t"])
            reproj[c].append(float(np.linalg.norm(
                proj.reshape(2) - np.asarray(raw[k], np.float64).reshape(2))))
    X = np.asarray(X, np.float64)
    ok = np.isfinite(X).all(axis=1)
    err = np.linalg.norm(X[ok] - truth[ok], axis=1) if ok.any() else np.array([])
    worst = int(np.argmax(err)) if err.size else None
    return {
        "cameras": cams,
        "rms_m": round(float(np.sqrt(np.mean(err ** 2))), 6) if err.size else None,
        "mean_m": round(float(err.mean()), 6) if err.size else None,
        "max_m": round(float(err.max()), 6) if err.size else None,
        "worst_corner": ({"index": worst, "m": round(float(err[worst]), 6)}
                         if worst is not None else None),
        "reprojection_rms_px": {
            str(c): (round(float(np.sqrt(np.mean(np.square(reproj[c])))), 4)
                     if reproj[c] else None) for c in cams},
        "definition": ("各相机最优帧联合 DLT 三角化棋盘格角点，与已知世界坐标的"
                       "三维偏差（米）；2 台相机时与 pair_consistency 等价"),
    }


def solve_rig_multi(frames: dict, spec: BoardSpec, intrinsics: dict, *,
                    flip180: bool = False, flip_cam=None,
                    orientation_confirmed: bool = False) -> dict:
    """用 2..N 台相机各自的多张固定板图像求解世界系外参。

    ``frames``：``{逻辑相机: [corners, ...]}``，每台相机对**同一块固定棋盘格**
    的多次检测结果（同一物理姿态，多张图用于抖动评估）。
    ``intrinsics``：``{逻辑相机: (K, D)}``。

    ``flip180``：所有相机一起绕板法向转 180°；``flip_cam``：``{相机: bool}``
    只翻某台。``orientation_confirmed`` 只能由人工确认置真。
    """
    cams = sorted(int(c) for c in frames)
    if len(cams) < 2:
        raise ValueError("至少需要两台相机的固定板图像")
    for c in cams:
        n = len(frames[c])
        if n < MIN_FRAMES:
            raise ValueError("相机 %d 至少需要 %d 张固定板图像（当前 %d）"
                             % (c, MIN_FRAMES, n))
        if c not in intrinsics:
            raise ValueError("相机 %d 缺少内参" % c)

    flips = {c: _flip_of(c, flip180, flip_cam) for c in cams}
    poses = {}
    for c in cams:
        K, D = intrinsics[c]
        per_frame = [(i, solve_pose(corners, spec, K, D, flip180=flips[c]))
                     for i, corners in enumerate(frames[c])]
        per_frame.sort(key=lambda t: t[1]["reprojection"]["rms_px"])
        poses[c] = per_frame
    chosen = {c: poses[c][0] for c in cams}

    ca, cb = cams[0], cams[1]
    Ka, Da = intrinsics[ca]
    Kb, Db = intrinsics[cb]
    ia, Pa = chosen[ca]
    ib, Pb = chosen[cb]
    cons = pair_consistency(Pa, Pb, spec, Ka, Da, Kb, Db,
                            frames[ca][ia], frames[cb][ib],
                            Pa["correspondence"], Pb["correspondence"])
    rel = cons["relative"]
    multi = _multi_view_consistency(chosen, frames, spec, intrinsics)

    warn = []
    if cons["blunder"]:
        warn.append("相机 %d/%d 的三角化残差 %.1f mm 偏大：检查角点是否都检测"
                    "完整、标定板在两次采集之间是否被碰动"
                    % (ca, cb, 1000 * cons["max_m"]))
    if rel["baseline_m"] < 0.05:
        warn.append("基线只有 %.1f mm，三角化精度会很差"
                    % (1000 * rel["baseline_m"]))
    if not orientation_confirmed:
        warn.append("棋盘格朝向（180°）尚未人工确认：几何上无法自动判定，"
                    "错 180° 会让基线与相机位置整体偏掉。请用世界轴叠加确认"
                    "每台相机的 X 轴都沿 12 格长边")
    for c in cams:
        p = chosen[c][1]
        if p["height_above_floor_m"] <= 0:
            warn.append("相机 %d 的世界 Z 为负（%.3f m）：图案正面朝上时相机应"
                        "在其上方，请检查该相机的角点方向"
                        % (c, p["height_above_floor_m"]))
        if p["points_in_front"] < 1.0:
            warn.append("相机 %d 有角点落在后方，位姿不可信" % c)
    if len(set(flips.values())) > 1:
        warn.append("各相机的 180° 设置不同（%s）：只有在各路检测顺序确实相差"
                    " 180° 时才应这样设；请用世界轴叠加逐台核对"
                    % ", ".join("cam%d=%s" % (c, flips[c]) for c in cams))

    jitter = {}
    for c in cams:
        plist = poses[c]
        centres = np.array([p["camera_centre_world"] for _, p in plist])
        entry = {
            "frames": len(plist),
            "used_frame_index": plist[0][0],
            "reproj_rms_px": [round(p["reprojection"]["rms_px"], 4)
                              for _, p in plist],
        }
        if len(plist) >= 2:
            entry["centre_std_m"] = round(
                float(np.linalg.norm(centres.std(axis=0))), 6)
            entry["centre_range_m"] = round(float(np.max(np.linalg.norm(
                centres - centres.mean(axis=0), axis=1))), 6)
        else:
            entry["centre_std_m"] = None
            entry["centre_range_m"] = None
            entry["note"] = "只有 1 帧，不计算帧间抖动；重复采集可观察稳定性"
        jitter[str(c)] = entry

    F = np.linalg.inv(np.asarray(Kb, np.float64)).T @ np.asarray(rel["E"]) \
        @ np.linalg.inv(np.asarray(Ka, np.float64))
    rel["F"] = F.tolist()
    orientation_ok = (bool(orientation_confirmed)
                      and len(set(flips.values())) == 1)

    cam_entries = {str(c): dict(chosen[c][1], cam_to_world=cam_to_world(chosen[c][1]))
                   for c in cams}
    correspondence = {str(c): chosen[c][1]["correspondence"] for c in cams}
    return {
        "method": ("solvePnP (IPPE candidates + LM refine) per camera against "
                   "one fixed board"),
        "board": spec.to_dict(),
        "flip180": bool(flip180),
        "flip_cam": {str(c): flips[c] for c in cams},
        "orientation_confirmed": bool(orientation_confirmed),
        "orientation_ok": orientation_ok,
        "camera_pair": [ca, cb],
        "cameras": cam_entries,
        "relative": rel,
        "convention": rel["convention"],
        "T_unit": "m",
        "baseline_m": rel["baseline_m"],
        "consistency": {"rms_m": cons["rms_m"], "mean_m": cons["mean_m"],
                        "max_m": cons["max_m"], "blunder": cons["blunder"],
                        "worst_corner": cons["worst_corner"],
                        "definition": ("三角化角点与已知世界坐标的三维偏差"
                                       "（棋盘格为基准）")},
        "multi_view": multi,
        "stability": jitter,
        "warnings": warn,
        "corner_correspondence": {
            "flip_cam": {str(c): flips[c] for c in cams},
            "classes": correspondence,
            "class_0": correspondence.get(str(ca)),
            "class_1": correspondence.get(str(cb)),
            "confirmed": bool(orientation_confirmed),
            "meaning": ("class 是自动选定的格点对应（保证相机在图案正面一侧）；"
                        "flip_cam 是在其之上再绕板法向 180°"),
            "how_to_verify": ("把世界系 X/Y 轴投影回图像（GET /api/calib/"
                              "extrinsic/overlay?camera=N&flip180=0/1），确认"
                              "每台相机的 X 轴都沿 12 格长边"),
            "geometric_note": ("矩形角点阵列有 D2 四重对称：4 种对应重投影误差"
                               "逐位相同，三角化一致性残差也都约 0.6mm，几何上"
                               "不可区分；只能人工确认"),
        },
        "caveat": ("三角化自洽性只能发现「板动了 / 检测错」，无法判断 180°；"
                   "三维精度请用 validate_rig 的独立数据（多个位置/高度）检验"),
    }


def solve_rig(frames: dict, spec: BoardSpec, K0, D0, K1, D1, *,
              flip180: bool = False, flip_cam=None,
              orientation_confirmed: bool = False) -> dict:
    """双相机兼容入口：内部走统一的 ``solve_rig_multi``。"""
    return solve_rig_multi(frames, spec, {0: (K0, D0), 1: (K1, D1)},
                           flip180=flip180, flip_cam=flip_cam,
                           orientation_confirmed=orientation_confirmed)


# ---------------------------------------------------------------------------
# 独立验证：多个位置 / 高度
# ---------------------------------------------------------------------------
def validate_rig_multi(placements: list, spec: BoardSpec, rig: dict,
                       intrinsics: dict, expected_centres=None) -> dict:
    """用独立采集（板挪到不同位置/高度）检验多相机三角化精度。

    ``placements``：每项 ``{"id", "corners": {相机: corners}, "dt_ms"}``。
    ``intrinsics``：``{相机: (K, D)}``。每个位置用**当时有效的那几台相机**
    联合 DLT 三角化；某台相机没检测到不影响其他相机。
    """
    if not placements:
        raise ValueError("没有验证数据")
    cams_all = sorted(int(c) for c in rig.get("cameras", {}))
    if not cams_all:
        raise ValueError("外参结果里没有相机")
    KDs = {c: (np.asarray(intrinsics[c][0], np.float64),
               np.asarray(intrinsics[c][1], np.float64).reshape(-1))
           for c in cams_all if c in intrinsics}
    classes = {c: rig["cameras"][str(c)].get("correspondence", "none")
               for c in cams_all}
    rel = rig.get("relative") or {}
    F = np.asarray(rel.get("F"), np.float64) if rel.get("F") else None
    first_pair = [c for c in cams_all if c in KDs][:2]

    spacing = spec.square_m
    truth = spec.world_points().astype(np.float64)
    per, all_sp, all_rep, all_epi, centres = [], [], [], [], []

    for pl in placements:
        corners = {int(c): np.asarray(v, np.float32).reshape(-1, 2)
                   for c, v in (pl.get("corners") or {}).items()
                   if v is not None}
        use = [c for c in cams_all if c in corners
               and len(corners[c]) == spec.count and c in KDs]
        if len(use) < 2:
            continue
        raw, norm, proj = {}, {}, {}
        for c in use:
            K, D = KDs[c]
            raw[c] = _apply_correspondence(corners[c], spec, classes[c])
            norm[c] = undistort_points(raw[c], K, D)
            pose = rig["cameras"][str(c)]
            proj[c] = np.hstack([np.asarray(pose["R"], np.float64),
                                 np.asarray(pose["t"], np.float64).reshape(3, 1)])

        Xw = []
        reproj = {c: [] for c in use}
        for k in range(spec.count):
            Xk = st.triangulate_dlt([norm[c][k] for c in use],
                                    [proj[c] for c in use])
            if Xk is None:
                Xk = np.full(3, np.nan)
            Xw.append(Xk)
            for c in use:
                K, D = KDs[c]
                pose = rig["cameras"][str(c)]
                p = st.project_world(Xk.reshape(1, 3), K, D,
                                     pose["R"], pose["t"])
                reproj[c].append(float(np.linalg.norm(
                    p.reshape(2) - raw[c][k].reshape(2))))
        Xw = np.asarray(Xw, np.float64)
        if not np.isfinite(Xw).all():
            continue
        rep = float(np.sqrt(np.mean(np.concatenate(
            [np.square(reproj[c]) for c in use]))))
        epi = None
        if (F is not None and len(first_pair) == 2
                and all(c in use for c in first_pair)):
            epi = float(np.median(st.epipolar_errors(
                raw[first_pair[0]], raw[first_pair[1]], F)))

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
        all_sp.extend((np.abs(sp_err) * 1000.0).tolist())

        cen = Xw.mean(axis=0)
        _u, _s, vt = np.linalg.svd(Xw - cen)
        normal = vt[2]
        planarity = float(np.sqrt(np.mean(((Xw - cen) @ normal) ** 2)))

        all_rep.append(rep)
        if epi is not None:
            all_epi.append(epi)
        centres.append(cen)
        suspect = (float(np.abs(sp_err).mean()) > 3 * spacing
                   or planarity > 3 * spacing)
        per.append({
            "id": pl.get("id"),
            "dt_ms": pl.get("dt_ms"),
            "cameras": use,
            "reprojection_rms_px": round(rep, 4),
            "reprojection_by_camera_px": {
                str(c): round(float(np.sqrt(np.mean(np.square(reproj[c])))), 4)
                for c in use},
            "epipolar_median_px": (round(epi, 4) if epi is not None else None),
            "spacing_mean_mm": round(float(d.mean() * 1000), 4),
            "spacing_mean_abs_error_mm": round(float(np.abs(sp_err).mean() * 1000), 4),
            "spacing_max_abs_error_mm": round(float(np.abs(sp_err).max() * 1000), 4),
            "spacing_rel_error_pct": round(float(100 * np.abs(sp_err).mean() / spacing), 4),
            "planarity_rms_mm": round(planarity * 1000, 4),
            "suspect_correspondence": bool(suspect),
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
        raise ValueError("没有可用的验证帧组（角点数不符或有效相机不足 2 台）")

    centres_arr = np.array(centres)
    agg = {
        "placements": len(per),
        "cameras": sorted({c for p in per for c in p["cameras"]}),
        "reprojection_rms_px": {
            "mean": round(float(np.mean(all_rep)), 4),
            "max": round(float(np.max(all_rep)), 4),
        },
        "spacing_abs_error_mm": {
            "mean": round(float(np.mean(all_sp)), 4),
            "p95": round(float(np.percentile(all_sp, 95)), 4),
            "max": round(float(np.max(all_sp)), 4),
            "expected_spacing_mm": round(spec.square_mm, 3),
            "unit": "mm",
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
    if all_epi:
        agg["epipolar_median_px"] = {
            "mean": round(float(np.mean(all_epi)), 4),
            "max": round(float(np.max(all_epi)), 4),
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
    sus = [p["id"] for p in per if p["suspect_correspondence"]]
    if sus:
        notes.append("位置 %s 的间距/平面度明显跑偏：可能棋盘格被转向后角点方向假设"
                     "不再适用，或该位置的角点检测有误；这些点不要拿来声称精度"
                     % "、".join(str(s) for s in sus))

    return {"aggregate": agg, "per_placement": per,
            "cross_checks": cross, "notes": notes}


def validate_rig(placements: list, spec: BoardSpec, rig: dict, K0, D0, K1, D1,
                 expected_centres=None) -> dict:
    """双相机兼容入口：内部走统一的 ``validate_rig_multi``。"""
    fixed = []
    for pl in placements:
        corners = {}
        if pl.get("corners0") is not None:
            corners[0] = pl["corners0"]
        if pl.get("corners1") is not None:
            corners[1] = pl["corners1"]
        if "corners" in pl:
            corners.update({int(c): v for c, v in pl["corners"].items()})
        fixed.append({"id": pl.get("id"), "dt_ms": pl.get("dt_ms"),
                      "corners": corners})
    return validate_rig_multi(fixed, spec, rig,
                              {0: (K0, D0), 1: (K1, D1)},
                              expected_centres=expected_centres)


__all__ = ["solve_pose", "cam_to_world", "relative_pose", "pair_consistency",
           "solve_rig", "solve_rig_multi", "validate_rig", "validate_rig_multi",
           "to_jsonable", "MIN_FRAMES", "RECOMMENDED_FRAMES"]
