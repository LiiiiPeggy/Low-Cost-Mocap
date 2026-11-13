"""双相机外参标定（直接调用 cv2.stereoCalibrate）。

坐标约定（务必与使用方一致）
--------------------------------------------------------------------------
固定内参，求解相机 0 → 相机 1 的刚体变换::

    X_cam1 = R_10 @ X_cam0 + T_10

这正是 ``cv2.stereoCalibrate`` 返回的 ``R``、``T`` 的定义（OpenCV 文档：
"the transformation from the first camera to the second"），不要再转置或取反。

* ``T`` 的单位与物点单位一致 —— 本模块物点由 ``BoardSpec`` 生成为**米**，
  所以 ``||T||`` 就是基线长度（米）。
* ``E = [T]_x R``，``F = K1^-T E K0^-1``，两者均由 OpenCV 返回，不自行推导。
* 三角化在**归一化相机坐标**下做（先 ``undistortPoints`` 再乘 [R|T]），
  这样得到的 3D 点直接是米制，不需要再乘 ``1/f``。

这里的同步只是"软件级近似同步"
--------------------------------------------------------------------------
输入帧对来自 ``sync.py`` 的主机单调时间戳配对，时间戳是**帧交付时刻**而不是
曝光中心时刻。对**静止**的标定板没有影响（静止目标不存在运动模糊之外的时序
误差），但绝不能把这套数据当成"硬件同步"的证据，也不能用它去论证高速运动
目标的三角化精度。
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from calibration.board import BoardSpec
from calibration.intrinsic import D_from, K_from, undistort_points

MIN_PAIRS = 3
RECOMMENDED_PAIRS = 10

DEFAULT_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 100, 1e-6)


# ---------------------------------------------------------------------------
# 几何工具
# ---------------------------------------------------------------------------
def _as_intr(x):
    """接受 {"K":..,"D":..} 字典或 (K, D) 元组。"""
    if isinstance(x, dict):
        return K_from(x), D_from(x)
    K, D = x
    return np.asarray(K, np.float64), np.asarray(D, np.float64).reshape(-1)


def triangulate(pts0, pts1, K0, D0, K1, D1, R, T) -> np.ndarray:
    """三角化，返回相机 0 坐标系下的 3D 点 (N,3)，单位米。"""
    x0 = undistort_points(pts0, K0, D0)      # 归一化坐标，只做一次去畸变
    x1 = undistort_points(pts1, K1, D1)
    P0 = np.hstack([np.eye(3), np.zeros((3, 1))])
    P1 = np.hstack([np.asarray(R, np.float64), np.asarray(T, np.float64).reshape(3, 1)])
    X = cv2.triangulatePoints(P0, P1, x0.T.astype(np.float64), x1.T.astype(np.float64))
    w = X[3]
    w[np.abs(w) < 1e-12] = 1e-12
    return (X[:3] / w).T


def epipolar_errors(pts0, pts1, F) -> np.ndarray:
    """点到极线的距离（像素），两个方向取平均。"""
    p0 = np.hstack([np.asarray(pts0, np.float64).reshape(-1, 2), np.ones((len(pts0), 1))])
    p1 = np.hstack([np.asarray(pts1, np.float64).reshape(-1, 2), np.ones((len(pts1), 1))])
    F = np.asarray(F, np.float64)

    l1 = p0 @ F.T                                   # F x0：图 1 中的极线
    d1 = np.abs(np.sum(l1 * p1, axis=1)) / np.maximum(
        np.linalg.norm(l1[:, :2], axis=1), 1e-12)

    l0 = p1 @ F                                     # F^T x1：图 0 中的极线
    d0 = np.abs(np.sum(l0 * p0, axis=1)) / np.maximum(
        np.linalg.norm(l0[:, :2], axis=1), 1e-12)
    return 0.5 * (d0 + d1)


def reproject(X, pts, K, D, R=None, T=None) -> np.ndarray:
    """把 3D 点投到某台相机，返回与检测角点的像素误差 (N,)。"""
    rvec = np.zeros((3, 1)) if R is None else cv2.Rodrigues(np.asarray(R, np.float64))[0]
    tvec = np.zeros((3, 1)) if T is None else np.asarray(T, np.float64).reshape(3, 1)
    proj, _ = cv2.projectPoints(np.asarray(X, np.float64).reshape(-1, 1, 3),
                                rvec, tvec,
                                np.asarray(K, np.float64), np.asarray(D, np.float64))
    return np.linalg.norm(proj.reshape(-1, 2) - np.asarray(pts, np.float64).reshape(-1, 2),
                          axis=1)


def in_front_ratio(X, R, T) -> float:
    """三角化点落在两台相机前方（Z>0）的比例，用来识别角点顺序反了的情况。"""
    X = np.asarray(X, np.float64)
    z0 = X[:, 2]
    X1 = X @ np.asarray(R, np.float64).T + np.asarray(T, np.float64).reshape(1, 3)
    z1 = X1[:, 2]
    return float(np.mean((z0 > 0) & (z1 > 0)))


def rotation_diagnostics(R) -> dict:
    R = np.asarray(R, np.float64)
    eye = np.eye(3)
    orth_err = float(np.max(np.abs(R @ R.T - eye)))
    det = float(np.linalg.det(R))
    ang = math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(R) - 1) / 2.0))))
    axis = cv2.Rodrigues(R)[0].reshape(3)
    n = float(np.linalg.norm(axis))
    axis_unit = (axis / n).tolist() if n > 1e-12 else [0.0, 0.0, 0.0]
    return {
        "orthogonality_error": round(orth_err, 9),
        "determinant": round(det, 9),
        "rotation_angle_deg": round(ang, 4),
        "rotation_axis": [round(v, 6) for v in axis_unit],
        "valid": bool(orth_err < 1e-6 and abs(det - 1.0) < 1e-6),
    }


# ---------------------------------------------------------------------------
# 标定
# ---------------------------------------------------------------------------
def calibrate(samples: list, spec: BoardSpec, image_size: tuple[int, int],
              intr0: dict, intr1: dict, *, flags: int = cv2.CALIB_FIX_INTRINSIC,
              criteria=DEFAULT_CRITERIA, auto_flip: bool = True,
              flip_b: bool | None = None) -> dict:
    """固定内参求解外参。返回含 R/T/E/F、基线、各项误差的完整结果。

    ``auto_flip``：棋盘格是中心对称图案，若某台相机倒装，角点标号可能整体
    旋转 180°，几何上无法从图案本身消歧。这里用"三角化后点必须落在两相机
    前方"作为判据，两种标号各解一次取更优者，并在结果里记录是否翻转。
    """
    if len(samples) < MIN_PAIRS:
        raise ValueError("双相机有效帧对只有 %d 对，至少需要 %d 对（建议 %d 对以上）"
                         % (len(samples), MIN_PAIRS, RECOMMENDED_PAIRS))

    K0, D0 = K_from(intr0), D_from(intr0)
    K1, D1 = K_from(intr1), D_from(intr1)

    if flip_b is None:
        attempts = [False, True] if auto_flip else [False]
    else:
        attempts = [bool(flip_b)]

    best = None
    tried = []
    for flip in attempts:
        try:
            res = _solve(samples, spec, image_size, K0, D0, K1, D1,
                         flags=flags, criteria=criteria, flip_b=flip)
        except cv2.error as exc:
            tried.append({"flip_b": flip, "error": str(exc).strip().splitlines()[-1]})
            continue
        # 判据：先看几何是否成立（点必须在两台相机前方），再比重投影 RMS。
        # 标号错 180° 时 RMS 会大一个量级（实测 20px vs 0.09px），而 in_front
        # 仍可能是 100%，所以不能只用 in_front 排序。
        score = (res["checks"]["in_front_ratio"] >= 0.99, -res["rms_px"])
        tried.append({"flip_b": flip, "rms_px": res["rms_px"],
                      "in_front_ratio": res["checks"]["in_front_ratio"]})
        if best is None or score > best[0]:
            best = (score, res)

    if best is None:
        raise ValueError("stereoCalibrate 全部失败：%s" % tried)

    result = best[1]
    result["search"] = {"attempts": tried, "auto_flip": bool(auto_flip),
                        "used_flip_b": result["flip_b"]}
    if result["flip_b"]:
        result["warnings"].append(
            "相机 1 的角点顺序被整体旋转了 180°（常见于相机倒装）。已按几何判据"
            "自动择向，R/T 基于此结果；若两台相机确实同向安装，请检查标定板是否被"
            "反相检测。")

    result["baseline_m"] = round(float(np.linalg.norm(result["T_10"])), 6)
    result["convention"] = "X_cam1 = R_10 @ X_cam0 + T_10"
    result["T_unit"] = "m"
    result["flags"] = ("CALIB_FIX_INTRINSIC" if flags & cv2.CALIB_FIX_INTRINSIC
                       else "flags=%d" % flags)
    result["board"] = spec.to_dict()
    result["num_pairs"] = len(samples)
    result["sample_ids"] = [s.id for s in samples]
    result["image_size"] = [int(image_size[0]), int(image_size[1])]
    return result


def _solve(samples, spec: BoardSpec, image_size, K0, D0, K1, D1, *,
           flags, criteria, flip_b: bool) -> dict:
    objp = spec.object_points().reshape(-1, 1, 3)
    obj_list, p0_list, p1_list, used = [], [], [], []
    for s in samples:
        c0 = s.corners(0)
        c1 = s.corners(1)
        if c0 is None or c1 is None:
            continue
        if len(c0) != spec.count or len(c1) != spec.count:
            continue
        if flip_b:
            c1 = c1[::-1].copy()
        obj_list.append(objp)
        p0_list.append(c0.astype(np.float32))
        p1_list.append(c1.astype(np.float32))
        used.append((s, c0, c1))

    if len(p0_list) < MIN_PAIRS:
        raise ValueError("可用帧对不足 %d 对" % MIN_PAIRS)

    rms, K0o, D0o, K1o, D1o, R, T, E, F = cv2.stereoCalibrate(
        obj_list, p0_list, p1_list,
        K0.copy(), D0.copy(), K1.copy(), D1.copy(),
        tuple(int(v) for v in image_size),
        flags=flags, criteria=criteria)

    ev = evaluate([u[0] for u in used], spec, image_size,
                  {"K": K0o, "D": D0o}, {"K": K1o, "D": D1o}, R, T, F,
                  flip_b=flip_b)

    warnings = []
    if not ev["checks"]["valid"]:
        warnings.append("旋转矩阵不满足正交性/行列式条件，结果不可信")
    if ev["checks"]["in_front_ratio"] < 0.99:
        warnings.append("仅 %.0f%% 的三角化点落在两相机前方，可能存在角点顺序或"
                        "帧对错配问题" % (100 * ev["checks"]["in_front_ratio"]))
    if len(used) < RECOMMENDED_PAIRS:
        warnings.append("只有 %d 对有效帧对，少于建议的 %d 对，外参可能不稳"
                        % (len(used), RECOMMENDED_PAIRS))
    if ev["epipolar"]["median_px"] is not None and ev["epipolar"]["median_px"] > 1.0:
        warnings.append("极线中位误差 %.2fpx 偏大，检查是否混入了不同姿态/不同板子的帧对"
                        % ev["epipolar"]["median_px"])
    if ev["checks"]["baseline_m"] < 0.01:
        warnings.append("基线只有 %.1fmm，双目三角化精度会很差" % (ev["checks"]["baseline_m"] * 1000))

    return {
        "R_10": np.asarray(R, np.float64).tolist(),
        "T_10": np.asarray(T, np.float64).reshape(3).tolist(),
        "E": np.asarray(E, np.float64).tolist(),
        "F": np.asarray(F, np.float64).tolist(),
        "rms_px": round(float(rms), 4),
        "flip_b": bool(flip_b),
        "flags_value": int(flags),
        "checks": ev["checks"],
        "epipolar": ev["epipolar"],
        "triangulation": ev["triangulation"],
        "known_distance": ev["known_distance"],
        "per_pair": ev["per_pair"],
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# 评估（同一套函数既评训练集，也评留出的验证集）
# ---------------------------------------------------------------------------
def evaluate(samples: list, spec: BoardSpec, image_size: tuple[int, int],
             intr0: dict, intr1: dict, R, T, F=None, *,
             flip_b: bool = False,
             expected_pairs: list[dict] | None = None) -> dict:
    """在给定帧对上评估外参质量：极线误差、三角化重投影、已知尺寸距离。"""
    K0, D0 = _as_intr(intr0)
    K1, D1 = _as_intr(intr1)
    R = np.asarray(R, np.float64)
    T = np.asarray(T, np.float64).reshape(3)
    if F is None:
        tx = np.array([[0, -T[2], T[1]], [T[2], 0, -T[0]], [-T[1], T[0], 0]])
        E = tx @ R
        F = np.linalg.inv(K1).T @ E @ np.linalg.inv(K0)
    F = np.asarray(F, np.float64)

    objp = spec.object_points()
    per_pair, epi_all, tri_all, dist_all = [], [], [], []

    for s in samples:
        c0 = s.corners(0)
        c1 = s.corners(1)
        if c0 is None or c1 is None or len(c0) != spec.count or len(c1) != spec.count:
            continue
        f0 = c0.reshape(-1, 2).astype(np.float64)
        f1 = c1.reshape(-1, 2).astype(np.float64)
        if flip_b:
            f1 = f1[::-1].copy()

        epi = float(np.median(epipolar_errors(f0, f1, F)))

        X = triangulate(f0, f1, K0, D0, K1, D1, R, T)
        e0 = reproject(X, f0, K0, D0)
        e1 = reproject(X, f1, K1, D1, R, T)
        tri = float(np.sqrt(np.mean(np.concatenate([e0, e1]) ** 2)))
        front = in_front_ratio(X, R, T)

        kd = known_distance_errors(X, spec, expected_pairs)
        for d in kd:
            # 相对误差按**该点对的期望距离**算，不是按一个方格边长
            exp = max(abs(d["expected_mm"]), 1e-9)
            dist_all.append((abs(d["error_mm"]), exp))

        epi_all.append(epi)
        tri_all.append(tri)
        per_pair.append({
            "sample": s.id,
            "epipolar_median_px": round(epi, 4),
            "triangulation_rms_px": round(tri, 4),
            "in_front_ratio": round(front, 4),
            "known_distance": kd,
        })

    epi = np.asarray(epi_all, dtype=np.float64)
    tri = np.asarray(tri_all, dtype=np.float64)
    checks = rotation_diagnostics(R)
    checks["baseline_m"] = round(float(np.linalg.norm(T)), 6)
    checks["in_front_ratio"] = round(float(np.mean(
        [p["in_front_ratio"] for p in per_pair])) if per_pair else 0.0, 4)
    checks["T_vector_m"] = [round(float(v), 6) for v in T]

    kd_stats = {}
    if dist_all:
        err = np.array([d[0] for d in dist_all], dtype=np.float64)
        exp = np.array([d[1] for d in dist_all], dtype=np.float64)
        rel = 100.0 * err / np.maximum(exp, 1e-9)
        kd_stats = {
            "n": int(err.size),
            "mean_error_mm": round(float(err.mean()), 4),
            "median_error_mm": round(float(np.median(err)), 4),
            "max_abs_error_mm": round(float(np.max(err)), 4),
            "mean_relative_error_pct": round(float(rel.mean()), 4),
            "max_relative_error_pct": round(float(rel.max()), 4),
            "definition": "相对误差 = |测量距离 - 期望距离| / 期望距离",
            "note": ("用棋盘格角点间的已知物理间距检验三角化的米制尺度；"
                     "这是同一批标定数据上的自洽检查，不能替代独立验证集"),
        }

    return {
        "num_pairs": len(per_pair),
        "checks": checks,
        "epipolar": {
            "median_px": round(float(np.median(epi)), 4) if epi.size else None,
            "mean_px": round(float(epi.mean()), 4) if epi.size else None,
            "p95_px": round(float(np.percentile(epi, 95)), 4) if epi.size else None,
            "max_px": round(float(epi.max()), 4) if epi.size else None,
        },
        "triangulation": {
            "median_px": round(float(np.median(tri)), 4) if tri.size else None,
            "mean_px": round(float(tri.mean()), 4) if tri.size else None,
            "p95_px": round(float(np.percentile(tri, 95)), 4) if tri.size else None,
            "max_px": round(float(tri.max()), 4) if tri.size else None,
        },
        "known_distance": kd_stats,
        "per_pair": per_pair,
    }


def known_distance_errors(X: np.ndarray, spec: BoardSpec,
                          expected_pairs: list[dict] | None = None) -> list[dict]:
    """用已知的板面角度间距检验三角化出来的米制尺度。

    默认检查两条边：角点 0 → (0, cols-1) 与 0 → (rows-1, 0)，
    期望距离分别为 (cols-1) 与 (rows-1) 个方格边长。
    """
    if X is None or len(X) != spec.count:
        return []
    pairs = expected_pairs or [
        {"i": 0, "j": spec.cols - 1, "expected_mm": (spec.cols - 1) * spec.square_mm,
         "label": "板面宽（%d 格）" % (spec.cols - 1)},
        {"i": 0, "j": (spec.rows - 1) * spec.cols,
         "expected_mm": (spec.rows - 1) * spec.square_mm,
         "label": "板面高（%d 格）" % (spec.rows - 1)},
    ]
    out = []
    for p in pairs:
        i, j = int(p["i"]), int(p["j"])
        if not (0 <= i < len(X) and 0 <= j < len(X)):
            continue
        measured = float(np.linalg.norm(np.asarray(X[i]) - np.asarray(X[j]))) * 1000.0
        exp = float(p["expected_mm"])
        out.append({
            "label": p.get("label", "%d-%d" % (i, j)),
            "i": i, "j": j,
            "expected_mm": round(exp, 4),
            "measured_mm": round(measured, 4),
            "error_mm": round(measured - exp, 4),
        })
    return out
