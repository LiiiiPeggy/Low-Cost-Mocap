"""合成数据自检：不接相机也能验证"检测 → 内参 → 外参 → 校验"整条链路。

用法（在 computer_code 目录下）::

    python -m calibration.selftest            # 全部用例
    python -m calibration.selftest --quick    # 少样本快速跑
    python -m calibration.selftest --keep     # 保留渲染出的临时图

原理：用已知的 K、D、R、T 把一块虚拟棋盘格**按真实畸变模型**反投影渲染成
灰度图（逐像素逆向映射，不是简单单应变换，所以畸变是真的），再走正式代码
检测角点、求解参数，最后和真值对比。

注意：这里验证的是**求解链路与坐标约定**，不是实机精度。合成图没有镜头
杂散光、板面不平、运动模糊，误差一定比实机小。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_API = os.path.dirname(_HERE)
if _API not in sys.path:
    sys.path.insert(0, _API)

from calibration import board as B            # noqa: E402
from calibration import extrinsic as E        # noqa: E402
from calibration import intrinsic as I        # noqa: E402
from calibration import stereo as S           # noqa: E402
from calibration.dataset import Session       # noqa: E402

RNG = np.random.default_rng(20261113)

TRUE_K = np.array([[912.0, 0.0, 643.5],
                   [0.0, 910.0, 358.2],
                   [0.0, 0.0, 1.0]], np.float64)
TRUE_D = np.array([-0.118, 0.092, 0.00042, -0.00031, -0.021], np.float64)
TRUE_R10 = cv2.Rodrigues(np.array([0.03, -0.26, 0.015]))[0]
TRUE_T10 = np.array([-0.152, 0.004, 0.007], np.float64)   # 基线 152mm

RESULT = []


def check(name: str, ok: bool, detail="") -> bool:
    detail = "" if detail == "" else str(detail)
    RESULT.append((name, bool(ok), detail))
    print(("  [PASS] " if ok else "  [FAIL] ") + name + ((" —— " + detail) if detail else ""))
    return ok


# ---------------------------------------------------------------------------
# 渲染
# ---------------------------------------------------------------------------
def board_texture(spec: B.BoardSpec, px_per_square: int = 44) -> tuple[np.ndarray, float]:
    cols, rows = spec.cols + 1, spec.rows + 1        # 方格数
    img = np.full((rows * px_per_square, cols * px_per_square), 30, np.uint8)
    for i in range(rows):
        for j in range(cols):
            if (i + j) % 2 == 0:
                img[i * px_per_square:(i + 1) * px_per_square,
                    j * px_per_square:(j + 1) * px_per_square] = 235
    # 白边（quiet zone），SB 需要
    pad = px_per_square
    img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=200)
    return img, pad / px_per_square


def render(spec: B.BoardSpec, R_b: np.ndarray, t_b: np.ndarray,
           K: np.ndarray, D: np.ndarray, size: tuple[int, int],
           tex: np.ndarray, tex_offset_sq: float,
           bg: int = 70, noise: float = 2.0) -> np.ndarray:
    """按真实畸变模型把板子渲染到图像平面（逐像素逆向映射）。"""
    w, h = size
    u, v = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    pixels = np.stack([u.ravel(), v.ravel()], axis=1).astype(np.float32)

    # 像素 → 去畸变归一化坐标（精确逆变换）
    norm = cv2.undistortPoints(pixels.reshape(-1, 1, 2), K.astype(np.float64),
                               D.astype(np.float64)).reshape(-1, 2)
    rays = np.stack([norm[:, 0], norm[:, 1], np.ones(len(norm))], axis=1)  # (P,3)

    n = R_b[:, 2]                       # 板面法向（相机系）
    d = float(n @ t_b)
    denom = rays @ n
    valid = np.abs(denom) > 1e-9
    s = np.zeros(len(rays))
    s[valid] = d / denom[valid]

    X = rays * s[:, None]                       # 相机系交点
    Xb = (X - t_b) @ R_b                        # 板坐标系（R^T (X-t) 的等价写法）

    sq = spec.square_m
    th, tw = tex.shape
    # 板坐标原点在图案中心。注意 Y 方向：纹理数组的行号向下增大，而世界 +Y
    # 向上，所以这里要取负号，否则渲染出的是"镜像板"，与世界点集手性相反，
    # solvePnP 会解出一个把相机放到板面下方的伪解。
    n_sq_x, n_sq_y = spec.squares_x, spec.squares_y
    tx = (Xb[:, 0] / sq + n_sq_x / 2.0 + tex_offset_sq) * (tw / (n_sq_x + 2 * tex_offset_sq))
    ty = (-Xb[:, 1] / sq + n_sq_y / 2.0 + tex_offset_sq) * (th / (n_sq_y + 2 * tex_offset_sq))
    ti = np.round(ty).astype(np.int64)
    tj = np.round(tx).astype(np.int64)

    inside = valid & (s > 0) & (ti >= 0) & (ti < th) & (tj >= 0) & (tj < tw)
    out = np.full(len(rays), bg, np.float64)
    out[inside] = tex[ti[inside], tj[inside]]

    img = out.reshape(h, w)
    img = cv2.GaussianBlur(img, (3, 3), 0.6)
    if noise > 0:
        img = img + RNG.normal(0, noise, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def pose_with_centered_board(spec, distance, yaw, pitch, roll, px=0.0, py=0.0):
    """板在相机系下的位姿：``X_cam = R_b @ X_board + t_b``。

    物点已以图案中心为原点，因此 ``t_b`` 就是图案中心在相机系中的位置。
    """
    R = cv2.Rodrigues(np.array([pitch, yaw, roll], np.float64))[0]
    return R, np.array([px, py, distance], np.float64)


def corners_in_image(spec: B.BoardSpec, R_b, t_b, K, D) -> np.ndarray:
    pts, _ = cv2.projectPoints(spec.object_points().reshape(-1, 1, 3),
                               cv2.Rodrigues(R_b)[0], t_b.reshape(3, 1), K, D)
    return pts.reshape(-1, 2)


def model_equivalent(K_est, D_est, K_true, D_true, size,
                     depths=(0.3, 0.6, 1.2)) -> float:
    """两组 (K, D) 在整幅画面上对同一批 3D 点的投影最大差异（像素）。

    k2/k3 之间存在强相关，系数不可逐项辨识，但"畸变模型的行为"是可以比较的。
    """
    w, h = size
    uv = np.array([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1],
                   [w / 2, h / 2], [w * 0.15, h * 0.8], [w * 0.85, h * 0.2]],
                  np.float32).reshape(-1, 1, 2)
    rays = cv2.undistortPoints(uv, np.asarray(K_true, np.float64),
                               np.asarray(D_true, np.float64)).reshape(-1, 2)
    worst = 0.0
    for z in depths:
        X = np.hstack([rays * z, np.full((len(rays), 1), z)]).reshape(-1, 1, 3).astype(np.float32)
        a = cv2.projectPoints(X, np.zeros(3), np.zeros(3), np.asarray(K_true, np.float64),
                              np.asarray(D_true, np.float64))[0].reshape(-1, 2)
        b = cv2.projectPoints(X, np.zeros(3), np.zeros(3), np.asarray(K_est, np.float64),
                              np.asarray(D_est, np.float64))[0].reshape(-1, 2)
        worst = max(worst, float(np.abs(a - b).max()))
    return worst


# ---------------------------------------------------------------------------
# 用例 1：内参
# ---------------------------------------------------------------------------
def test_intrinsic(quick: bool, keep_dir: str | None, spec: B.BoardSpec,
                   size: tuple[int, int]) -> dict | None:
    print("\n[1] 单相机内参（合成 %dx%d，真实 f=912/910, k1=-0.118）" % size)
    tex, off = board_texture(spec)
    n = 12 if quick else 26

    dets = []
    for i in range(n):
        # 刻意让板子出现在画面各个区域（含靠近边缘），这样畸变参数才有足够
        # 的半径覆盖 —— 这也是实机采集时给用户的建议。
        dist = RNG.uniform(0.62, 1.00)
        yaw = RNG.uniform(-0.42, 0.42)
        pitch = RNG.uniform(-0.32, 0.32)
        roll = RNG.uniform(-0.45, 0.45)
        px = RNG.uniform(-0.16, 0.16)
        py = RNG.uniform(-0.08, 0.08)
        R_b, t_b = pose_with_centered_board(spec, dist, yaw, pitch, roll, px, py)

        img = render(spec, R_b, t_b, TRUE_K, TRUE_D, size, tex, off)
        det = B.detect(img, spec)
        if not det.ok:
            continue
        truth = corners_in_image(spec, R_b, t_b, TRUE_K, TRUE_D)
        det_err = float(np.abs(det.flat - truth).max())
        if det_err > 2.0:
            print("    样本 %d 检测偏差 %.2fpx，丢弃" % (i, det_err))
            continue
        dets.append((det.corners, img, det_err))
        if keep_dir:
            cv2.imwrite(os.path.join(keep_dir, "intr_%02d.png" % i), img)

    check("角点检测成功率", len(dets) >= max(6, n // 2),
          "%d/%d 张成功" % (len(dets), n))
    if len(dets) < 6:
        return None

    check("检测角点与真值最大偏差 < 1px",
          max(d[2] for d in dets) < 1.0,
          "max=%.3fpx" % max(d[2] for d in dets))

    class Fake:
        def __init__(self, sid, corners):
            self.id, self.detections = sid, {"0": {"ok": True}}
            self._c = corners

        def corners(self, cam=0):
            return self._c

    samples = [Fake("s%03d" % i, c) for i, (c, _, _) in enumerate(dets)]
    res = I.calibrate(samples, spec, size, camera=0)

    fx_err = abs(res["focal_px"][0] - TRUE_K[0, 0]) / TRUE_K[0, 0]
    fy_err = abs(res["focal_px"][1] - TRUE_K[1, 1]) / TRUE_K[1, 1]
    cx_err = abs(res["principal_point_px"][0] - TRUE_K[0, 2])
    cy_err = abs(res["principal_point_px"][1] - TRUE_K[1, 2])
    k1_err = abs(res["D"][0] - TRUE_D[0])
    k2_err = abs(res["D"][1] - TRUE_D[1])

    check("焦距相对误差 < 0.5%", fx_err < 0.005 and fy_err < 0.005,
          "fx err=%.3f%%  fy err=%.3f%%" % (100 * fx_err, 100 * fy_err))
    check("主点误差 < 3px", cx_err < 3 and cy_err < 3,
          "cx err=%.2fpx  cy err=%.2fpx" % (cx_err, cy_err))
    check("k1 误差 < 0.02", k1_err < 0.02, "k1=%.4f (真值 %.4f)" % (res["D"][0], TRUE_D[0]))
    # k2/k3 在小样本下高度相关（互相吸收），单独比系数意义不大；
    # 改为比较两个畸变模型在整幅画面上的**实际投影差异**。
    emax = model_equivalent(res["K"], res["D"], TRUE_K, TRUE_D, size)
    cov = res["radius_coverage"]["ratio"]
    # 30mm 方格、11×8 内角点的板子在 1280×720 里相对较大，很难把角点铺到画面四角，
    # 所以这里不再用固定 px 阈值，而是"覆盖率 + 告警是否触发 + 外推量级"一起看。
    check("覆盖不足时给出外推告警",
          cov >= 0.75 or any("外推" in w for w in res["warnings"]),
          "半径覆盖 %.0f%%" % (100 * cov))
    check("畸变外推量级合理（< 15px）", emax < 15.0,
          "max=%.3fpx，半径覆盖 %.0f%%，k2=%.4f vs 真值 %.4f"
          % (emax, 100 * cov, res["D"][1], TRUE_D[1]))
    check("重投影 RMS < 0.5px", res["rms_px"] < 0.5, "rms=%.4fpx" % res["rms_px"])

    # 少样本必须被拒绝
    try:
        I.calibrate(samples[:2], spec, size, camera=0)
        check("样本不足时拒绝求解", False, "2 张样本竟然通过了")
    except ValueError as exc:
        check("样本不足时拒绝求解", True, str(exc)[:60])

    return res


# ---------------------------------------------------------------------------
# 用例 2：外参
# ---------------------------------------------------------------------------
class _Pair:
    def __init__(self, sid, c0, c1):
        self.id, self.detections = sid, {"0": {"ok": True}, "1": {"ok": True}}
        self._0, self._1 = c0, c1

    def corners(self, cam):
        return self._0 if int(cam) == 0 else self._1


def build_stereo_pairs(spec, size, n: int, tex, off, *, reverse_b: bool = False):
    """生成 n 对同步帧，返回 pairs。"""
    pairs = []
    for i in range(n):
        dist = RNG.uniform(0.85, 1.25)
        yaw = RNG.uniform(-0.30, 0.30)
        pitch = RNG.uniform(-0.22, 0.22)
        roll = RNG.uniform(-0.35, 0.35)
        px = RNG.uniform(-0.05, 0.05)
        py = RNG.uniform(-0.04, 0.04)
        R_b0, t_b0 = pose_with_centered_board(spec, dist, yaw, pitch, roll, px, py)

        # 相机 1 看到的同一块板
        R_b1 = TRUE_R10 @ R_b0
        t_b1 = TRUE_R10 @ t_b0 + TRUE_T10

        img0 = render(spec, R_b0, t_b0, TRUE_K, TRUE_D, size, tex, off)
        img1 = render(spec, R_b1, t_b1, TRUE_K, TRUE_D, size, tex, off)

        da, db, info = B.detect_pair(img0, img1, spec)
        if not (da.ok and db.ok):
            continue
        c0, c1 = da.corners, db.corners
        if reverse_b:
            c1 = c1[::-1].copy()

        pairs.append((c0, c1, img0, img1))
    return pairs


def test_stereo(spec: B.BoardSpec, size: tuple[int, int], intr0: dict, intr1: dict,
                quick: bool, keep_dir: str | None) -> dict | None:
    print("\n[2] 双相机外参（真实基线 %.1fmm，绕 y 轴 %.1f°）"
          % (np.linalg.norm(TRUE_T10) * 1000,
             math.degrees(math.acos(max(-1, min(1, (np.trace(TRUE_R10) - 1) / 2))))))
    tex, off = board_texture(spec)
    n = 8 if quick else 16
    raw = build_stereo_pairs(spec, size, n, tex, off)
    check("双相机同时检测成功", len(raw) >= max(4, n // 2),
          "%d/%d 对成功" % (len(raw), n))
    if len(raw) < 4:
        return None

    samples = [_Pair("p%03d" % i, c0, c1) for i, (c0, c1, _, _) in enumerate(raw)]
    res = S.calibrate(samples, spec, size, intr0, intr1, auto_flip=True)

    if keep_dir:
        for i, (_, _, img0, img1) in enumerate(raw[:4]):
            cv2.imwrite(os.path.join(keep_dir, "pair_%02d_A.png" % i), img0)
            cv2.imwrite(os.path.join(keep_dir, "pair_%02d_B.png" % i), img1)

    R = np.asarray(res["R_10"])
    T = np.asarray(res["T_10"])
    base_err = abs(res["baseline_m"] - float(np.linalg.norm(TRUE_T10))) * 1000
    ang_true = math.degrees(math.acos(max(-1, min(1, (np.trace(TRUE_R10) - 1) / 2))))
    ang_got = res["checks"]["rotation_angle_deg"]
    check("未发生误翻转", res["flip_b"] is False, "flip_b=%s" % res["flip_b"])
    check("基线误差 < 2mm", base_err < 2.0,
          "baseline=%.2fmm (真值 %.2fmm, 误差 %.3fmm)"
          % (res["baseline_m"] * 1000, np.linalg.norm(TRUE_T10) * 1000, base_err))
    check("相对旋转角误差 < 0.5°", abs(ang_got - ang_true) < 0.5,
          "angle=%.3f° (真值 %.3f°)" % (ang_got, ang_true))
    check("T 向量误差 < 3mm", float(np.linalg.norm(T - TRUE_T10)) * 1000 < 3.0,
          "|ΔT|=%.3fmm" % (float(np.linalg.norm(T - TRUE_T10)) * 1000))
    check("旋转矩阵正交且 det=1", res["checks"]["valid"],
          "orth_err=%.2e det=%.9f" % (res["checks"]["orthogonality_error"],
                                      res["checks"]["determinant"]))
    tol_epi = 2.0 if quick else 1.0
    check("极线中位误差 < %.0fpx" % tol_epi,
          (res["epipolar"]["median_px"] or 9) < tol_epi,
          "median=%.3fpx" % (res["epipolar"]["median_px"] or -1))
    check("三角化重投影 < 1px", (res["triangulation"]["median_px"] or 9) < 1.0,
          "median=%.3fpx" % (res["triangulation"]["median_px"] or -1))
    kd = res["known_distance"]
    check("已知间距尺度误差 < 0.5%", bool(kd) and kd["mean_relative_error_pct"] < 0.5,
          "mean=%.3fmm (%.2f%%)" % (kd.get("mean_error_mm", -1),
                                    kd.get("mean_relative_error_pct", -1))
          if kd else "无数据")
    check("全部点落在两相机前方", res["checks"]["in_front_ratio"] > 0.99,
          "%.1f%%" % (100 * res["checks"]["in_front_ratio"]))

    # 用留出集独立评估
    if len(samples) >= 8:
        val = S.evaluate(samples[-3:], spec, size, intr0, intr1, R, T,
                         np.asarray(res["F"]), flip_b=res["flip_b"])
        check("留出集极线中位误差 < 2px",
              (val["epipolar"]["median_px"] or 9) < 2.0,
              "holdout median=%.3fpx" % (val["epipolar"]["median_px"] or -1))

    return res


# ---------------------------------------------------------------------------
# 用例 3：角点顺序被整体翻转时能否自动纠正
# ---------------------------------------------------------------------------
def test_auto_flip(spec: B.BoardSpec, size: tuple[int, int],
                   intr0: dict, intr1: dict) -> None:
    print("\n[3] 相机 1 角点标号整体旋转 180°（模拟倒装）")
    tex, off = board_texture(spec)
    n = 8
    raw = build_stereo_pairs(spec, size, n, tex, off, reverse_b=True)
    check("反标号数据生成", len(raw) >= 5, "%d 对" % len(raw))
    if len(raw) < 5:
        return

    samples = [_Pair("p%03d" % i, c0, c1) for i, (c0, c1, _, _) in enumerate(raw)]

    forced_off = S.calibrate(samples, spec, size, intr0, intr1, flip_b=False)
    auto = S.calibrate(samples, spec, size, intr0, intr1, auto_flip=True)

    check("反标号在非翻转解下 RMS 明显变大",
          forced_off["rms_px"] > 5.0,
          "rms=%.2fpx（正确解应为亚像素）" % forced_off["rms_px"])
    check("auto_flip 选择了翻转", auto["flip_b"] is True, "flip_b=%s" % auto["flip_b"])
    check("翻转后 RMS 回到亚像素", auto["rms_px"] < 1.0, "rms=%.3fpx" % auto["rms_px"])
    err = float(np.linalg.norm(np.asarray(auto["T_10"]) - TRUE_T10)) * 1000
    check("翻转后 T 恢复正确 (< 3mm)", err < 3.0, "|ΔT|=%.3fmm" % err)
    ang = auto["checks"]["rotation_angle_deg"]
    ang_true = math.degrees(math.acos(max(-1, min(1, (np.trace(TRUE_R10) - 1) / 2))))
    check("翻转后旋转角恢复正确 (< 0.5°)", abs(ang - ang_true) < 0.5,
          "%.3f° vs %.3f°" % (ang, ang_true))


# ---------------------------------------------------------------------------
# 用例 4：固定棋盘格 + solvePnP 的世界系外参
# ---------------------------------------------------------------------------
def look_at(C, target, up=(0.0, 0.0, 1.0)):
    """构造 world→camera 位姿：相机在 C，看向 target，+Z 指向目标。"""
    C = np.asarray(C, np.float64)
    d = np.asarray(target, np.float64) - C
    z = d / np.linalg.norm(d)
    up = np.asarray(up, np.float64)
    x = np.cross(up, z)
    if np.linalg.norm(x) < 1e-9:
        x = np.cross([1.0, 0.0, 0.0], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])          # 相机三轴在世界系中的表示（按行）
    t = -R @ C
    return R, t


def render_rig(spec, R_wc, t_wc, world_offset, K, D, size, tex, off):
    """板平放在世界系 ``world_offset`` 处（无旋转），渲染到给定相机。"""
    d = np.asarray(world_offset, np.float64)
    return render(spec, R_wc, R_wc @ d + t_wc, K, D, size, tex, off)


def test_pnp(spec: B.BoardSpec, size: tuple[int, int], intr0: dict, intr1: dict,
             quick: bool, keep_dir: str | None):
    print("\n[4] 固定棋盘格 + solvePnP 世界系外参（%dx%d 内角点，%gmm）"
          % (spec.cols, spec.rows, spec.square_mm))
    tex, off = board_texture(spec)
    K0, D0 = np.asarray(intr0["K"]), np.asarray(intr0["D"])
    K1, D1 = np.asarray(intr1["K"]), np.asarray(intr1["D"])

    C0 = np.array([-1.05, -0.85, 1.20])
    C1 = np.array([0.98, -0.78, 1.28])
    target = np.array([0.0, 0.0, 0.02])
    R0t, t0t = look_at(C0, target)
    R1t, t1t = look_at(C1, target)

    # --- 单相机位姿 ---
    img0 = render_rig(spec, R0t, t0t, (0, 0, 0), K0, D0, size, tex, off)
    img1 = render_rig(spec, R1t, t1t, (0, 0, 0), K1, D1, size, tex, off)
    det0, det1 = B.detect(img0, spec), B.detect(img1, spec)
    check("两台相机都能检测到固定板", det0.ok and det1.ok,
          "%s / %s" % (det0.method or det0.reason, det1.method or det1.reason))
    if not (det0.ok and det1.ok):
        return None

    if keep_dir:
        cv2.imwrite(os.path.join(keep_dir, "pnp_cam0.png"), img0)
        cv2.imwrite(os.path.join(keep_dir, "pnp_cam1.png"), img1)

    p0 = E.solve_pose(det0.corners, spec, K0, D0)
    p1 = E.solve_pose(det1.corners, spec, K1, D1)
    d0 = float(p0["distance_to_pattern_centre_m"])
    d1 = float(p1["distance_to_pattern_centre_m"])
    true0 = float(np.linalg.norm(C0))
    true1 = float(np.linalg.norm(C1))
    # 矩形板的 D2 对称意味着"哪个角点是 0 号"本身不唯一，解出的相机位置可能是
    # 真值关于板中心的一种镜像。所以这里比对**对称不变量**：到板心的距离、高度。
    check("相机0 到板心距离误差 < 10mm", abs(d0 - true0) * 1000 < 10.0,
          "%.1fmm vs %.1fmm" % (d0 * 1000, true0 * 1000))
    check("相机1 到板心距离误差 < 10mm", abs(d1 - true1) * 1000 < 10.0,
          "%.1fmm vs %.1fmm" % (d1 * 1000, true1 * 1000))
    check("solvePnP 重投影 < 0.5px",
          max(p0["reprojection"]["rms_px"], p1["reprojection"]["rms_px"]) < 0.5,
          "%.4f / %.4f px" % (p0["reprojection"]["rms_px"], p1["reprojection"]["rms_px"]))
    check("相机都在板面之上（Z>0）",
          p0["height_above_floor_m"] > 0 and p1["height_above_floor_m"] > 0,
          "%.3f / %.3f m" % (p0["height_above_floor_m"], p1["height_above_floor_m"]))
    check("光轴朝下（Z 分量为负）",
          p0["optical_axis_world"][2] < 0 and p1["optical_axis_world"][2] < 0,
          "%.3f / %.3f" % (p0["optical_axis_world"][2], p1["optical_axis_world"][2]))
    check("自动选定的格点对应落在 Z>0 的可行集内",
          p0["correspondence"] in ("none", "row", "col", "both")
          and p1["correspondence"] in ("none", "row", "col", "both"),
          "cam0=%s cam1=%s" % (p0["correspondence"], p1["correspondence"]))

    # --- 双相机合成：相对位姿与基线（基线是 D2 不变量，可以作为硬指标） ---
    rig = E.solve_rig({0: [det0.corners] * 4, 1: [det1.corners] * 4}, spec,
                      K0, D0, K1, D1)
    base_true = float(np.linalg.norm(C1 - C0))
    base_err = abs(rig["relative"]["baseline_m"] - base_true) * 1000
    check("基线误差 < 5mm", base_err < 5.0,
          "%.2fmm（真值 %.2fmm）" % (rig["relative"]["baseline_m"] * 1000, base_true * 1000))
    check("三角化与已知世界点一致 < 2mm", rig["consistency"]["max_m"] < 0.002,
          "rms=%.4fmm max=%.4fmm" % (rig["consistency"]["rms_m"] * 1000,
                                     rig["consistency"]["max_m"] * 1000))
    check("相对旋转正交 det=1", rig["relative"]["rotation"]["valid"],
          "det=%.9f" % rig["relative"]["rotation"]["determinant"])

    # 世界→相机 / 相机→世界 互为逆
    Rw = np.asarray(rig["cameras"]["0"]["cam_to_world"]["R_wc"])
    tw = np.asarray(rig["cameras"]["0"]["cam_to_world"]["t_wc"])
    Rc = np.asarray(rig["cameras"]["0"]["R"])
    tc = np.asarray(rig["cameras"]["0"]["t"])
    check("cam_to_world 与 world_to_cam 互逆",
          np.abs(Rw @ Rc - np.eye(3)).max() < 1e-9 and
          np.abs(Rw @ tc + tw).max() < 1e-9,
          "max|R_wc R_cw - I| = %.2e" % np.abs(Rw @ Rc - np.eye(3)).max())

    # --- 180° 是"必须人工确认"的自由度：它只改世界系朝向，不改任何残差 ---
    a = E.solve_pose(det1.corners, spec, K1, D1)
    b = E.solve_pose(det1.corners, spec, K1, D1, flip180=True)
    Ca, Cb = np.array(a["camera_centre_world"]), np.array(b["camera_centre_world"])
    check("180° 翻转保持相机高度与到板心距离不变",
          abs(Ca[2] - Cb[2]) < 1e-6 and
          abs(np.linalg.norm(Ca) - np.linalg.norm(Cb)) < 1e-6,
          "Z %.4f/%.4f  |C| %.4f/%.4f" % (Ca[2], Cb[2], np.linalg.norm(Ca), np.linalg.norm(Cb)))
    check("180° 翻转即世界 XY 绕 Z 旋转 180°",
          float(np.linalg.norm(Ca[:2] + Cb[:2])) < 0.05,
          "|C_xy(a)+C_xy(b)| = %.4f m" % float(np.linalg.norm(Ca[:2] + Cb[:2])))
    check("180° 翻转不改变重投影误差（所以残差查不出来，必须人工确认）",
          abs(a["reprojection"]["rms_px"] - b["reprojection"]["rms_px"]) < 1e-6,
          "%.6f vs %.6f px" % (a["reprojection"]["rms_px"], b["reprojection"]["rms_px"]))
    rig_flip = E.solve_rig({0: [det0.corners] * 3, 1: [det1.corners] * 3}, spec,
                           K0, D0, K1, D1, flip180=True)
    check("全局 flip180 后基线不变（相对几何对整体 180° 免疫）",
          abs(rig_flip["relative"]["baseline_m"] - base_true) * 1000 < 5.0,
          "%.2fmm" % (rig_flip["relative"]["baseline_m"] * 1000))

    # --- 独立验证：板挪到 3 个不同位置 / 高度 ---
    placements = []
    offsets = [(0.00, 0.00, 0.00), (0.35, 0.12, 0.20), (-0.28, -0.18, 0.38)]
    for k, off_w in enumerate(offsets):
        a = render_rig(spec, R0t, t0t, off_w, K0, D0, size, tex, off)
        b = render_rig(spec, R1t, t1t, off_w, K1, D1, size, tex, off)
        da, db, _ = B.detect_pair(a, b, spec)
        if not (da.ok and db.ok):
            continue
        placements.append({"id": "v%d" % k, "corners0": da.corners,
                           "corners1": db.corners, "dt_ms": 0.3,
                           "_offset": off_w})
    check("独立验证数据采集（3 个位置/高度）", len(placements) >= 2,
          "%d/3 个位置检测成功" % len(placements))

    if placements:
        val = E.validate_rig(placements, spec, rig, K0, D0, K1, D1)
        agg = val["aggregate"]
        check("验证：间距均方误差 < 1mm", agg["spacing_abs_error_mm"]["mean"] < 1.0,
              "mean=%.3fmm p95=%.3fmm max=%.3fmm"
              % (agg["spacing_abs_error_mm"]["mean"], agg["spacing_abs_error_mm"]["p95"],
                 agg["spacing_abs_error_mm"]["max"]))
        check("验证：重投影 < 1px", agg["reprojection_rms_px"]["mean"] < 1.0,
              "mean=%.3fpx max=%.3fpx"
              % (agg["reprojection_rms_px"]["mean"], agg["reprojection_rms_px"]["max"]))
        # 汇总间距误差曾经把“米”当“毫米”输出（比逐位置小 1000 倍）。
        sp = agg["spacing_abs_error_mm"]
        per_max = max(p["spacing_max_abs_error_mm"] for p in val["per_placement"])
        check("验证：汇总间距误差单位与逐位置一致（mm）",
              sp.get("unit") == "mm" and abs(sp["max"] - per_max) < 1e-3
              and sp["expected_spacing_mm"] == round(spec.square_mm, 3),
              "汇总 max=%.4fmm  逐位置 max=%.4fmm" % (sp["max"], per_max))
        check("验证：极线中位 < 1.5px", agg["epipolar_median_px"]["mean"] < 1.5,
              "mean=%.3fpx" % agg["epipolar_median_px"]["mean"])
        check("验证：重建平面度 < 1mm",
              max(p["planarity_rms_mm"] for p in val["per_placement"]) < 1.0,
              "max=%.3fmm" % max(p["planarity_rms_mm"] for p in val["per_placement"]))
        # 高度覆盖：三个位置应出现不同的 centre_z
        zs = [p["centre_world_m"][2] for p in val["per_placement"]]
        check("验证覆盖了不同高度", len(set(round(z, 2) for z in zs)) >= 2,
              "centre_z = %s m" % [round(z, 3) for z in zs])
        check("验证结果标出每个位置的对应关系是否可疑",
              all("suspect_correspondence" in p for p in val["per_placement"])
              and not any(p["suspect_correspondence"] for p in val["per_placement"]),
              [p["suspect_correspondence"] for p in val["per_placement"]])
        # 板框尺寸检查（已知 300mm × 210mm 内角点跨度）
        bsc = val["per_placement"][0]["board_size_check_mm"]
        check("板面跨距与理论值一致 < 2mm",
              abs(bsc["x"] - bsc["expected_x"]) < 2 and abs(bsc["y"] - bsc["expected_y"]) < 2,
              "x=%.2f/%.2f y=%.2f/%.2f mm" % (bsc["x"], bsc["expected_x"],
                                              bsc["y"], bsc["expected_y"]))

    return rig


# ---------------------------------------------------------------------------
# 用例 6：代表帧对应 / 采集去重 / 结果失效与恢复 / CLI 保存
# ---------------------------------------------------------------------------
def _two_camera_poses():
    C0 = np.array([-1.05, -0.85, 1.20])
    C1 = np.array([0.98, -0.78, 1.28])
    target = np.array([0.0, 0.0, 0.02])
    R0, t0 = look_at(C0, target)
    R1, t1 = look_at(C1, target)
    return C0, C1, (R0, t0), (R1, t1)


def test_frame_pairing(spec: B.BoardSpec, size, tex, off) -> None:
    """代表位姿与一致性检查必须来自**同一帧**（曾经错配成首帧）。"""
    print("\n[6] 外参代表帧与角点对应")
    C0, C1, (R0, t0), (R1, t1) = _two_camera_poses()
    img0 = render(spec, R0, t0, TRUE_K, TRUE_D, size, tex, off)
    img1 = render(spec, R1, t1, TRUE_K, TRUE_D, size, tex, off)
    d0, d1 = B.detect(img0, spec), B.detect(img1, spec)
    if not (d0.ok and d1.ok):
        check("代表帧用例：检测", False, "检测失败")
        return

    # 人为把 cam0 的第 0 帧角点推偏，使"最好的帧"是第 1 帧
    bad = (d0.corners.reshape(-1, 2) + np.array([16.0, -13.0])).reshape(-1, 1, 2)
    frames0 = [bad.astype(np.float32), d0.corners, d0.corners]
    frames1 = [d1.corners, d1.corners, d1.corners]

    rig = E.solve_rig({0: frames0, 1: frames1}, spec, TRUE_K, TRUE_D, TRUE_K, TRUE_D)
    used = rig["stability"]["0"]["used_frame_index"]
    check("代表位姿取重投影最好的那一帧", used == 1, "used_frame_index=%s" % used)
    check("一致性检查使用同一帧角点（不被首帧带偏）",
          rig["consistency"]["max_m"] < 0.003,
          "max=%.2fmm（若错用首帧会明显变大）" % (rig["consistency"]["max_m"] * 1000))

    # IPPE 候选：必须标出物理合理性，且选中解满足相机在板面上方
    p = E.solve_pose(d0.corners, spec, TRUE_K, TRUE_D)
    cands = p["ippe_candidates"]
    check("IPPE 候选带物理合理性标记",
          len(cands) >= 1 and all("plausible" in c for c in cands),
          "%d 个候选" % len(cands))
    check("选中解满足相机在板面上方且角点全在前方",
          p["height_above_pattern_m"] > 0 and p["points_in_front"] >= 0.99,
          "Z=%.3f  in_front=%.3f" % (p["height_above_pattern_m"], p["points_in_front"]))
    check("自动格点对应不按重投影误差挑选（保留歧义提示）",
          p["orientation_ambiguous"] is True,
          "alternatives=%s" % (p["correspondence_alternatives"],))


def _build_session(root: str, spec: B.BoardSpec, size, tex, off, *, n_intr=6):
    """造一个可离线求解的完整会话（内参/外参/验证三类样本）。"""
    shutil.rmtree(root, ignore_errors=True)
    sess = Session.create(root, spec, [0, 1], size, "NV12", "自检")
    for cam in (0, 1):
        k = 0
        guard = 0
        while k < n_intr and guard < n_intr * 6:
            guard += 1
            R_b, t_b = pose_with_centered_board(
                spec, RNG.uniform(0.62, 1.0), RNG.uniform(-0.4, 0.4),
                RNG.uniform(-0.3, 0.3), RNG.uniform(-0.42, 0.42),
                RNG.uniform(-0.16, 0.16), RNG.uniform(-0.08, 0.08))
            img = render(spec, R_b, t_b, TRUE_K, TRUE_D, size, tex, off)
            det = B.detect(img, spec)
            if not det.ok:
                continue
            sess.add_single(cam, img, det, {"seq": k},
                            signature=B.pose_signature(det, spec), phase="intrinsic")
            k += 1

    _, _, (R0, t0), (R1, t1) = _two_camera_poses()
    for cam, (R, t) in ((0, (R0, t0)), (1, (R1, t1))):
        img = render(spec, R, t, TRUE_K, TRUE_D, size, tex, off)
        det = B.detect(img, spec)
        for k in range(4):
            sess.add_single(cam, img, det, {"seq": 50 + k},
                            signature=B.pose_signature(det, spec), phase="extrinsic")

    for off_w in ((0.0, 0.0, 0.0), (0.32, 0.10, 0.22)):
        a = render(spec, R0, R0 @ np.array(off_w) + t0, TRUE_K, TRUE_D, size, tex, off)
        b = render(spec, R1, R1 @ np.array(off_w) + t1, TRUE_K, TRUE_D, size, tex, off)
        da, db, _ = B.detect_pair(a, b, spec)
        if da.ok and db.ok:
            sess.add_stereo({"0": {"seq": 0}, "1": {"seq": 0}}, {0: a, 1: b},
                            {0: da, 1: db}, dt_ms=0.3,
                            signature=B.pose_signature(da, spec), phase="validation")
    return sess


def test_cli_and_state(spec: B.BoardSpec, size, tex, off, tmp: str) -> None:
    print("\n[7] CLI 保存 / 重载恢复 / 新增采集后的失效")
    import calibration.service as SVC

    root = os.path.join(tmp, "cli_session")
    _build_session(root, spec, size, tex, off)
    out = os.path.join(tmp, "cli_calibration.json")

    # (1) CLI 保存：既验证没有重复 cmd_save，也验证参数齐全
    cmd = [sys.executable, os.path.join(_API, "calib_cli.py"), "save",
           "--session", root, "--out", out, "--validate", "--orientation-confirmed"]
    proc = subprocess.run(cmd, cwd=_API, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    ok = proc.returncode == 0 and os.path.isfile(out)
    check("CLI save 正常退出并写出结果", ok,
          (proc.stderr or proc.stdout).strip().splitlines()[-1] if not ok else out)
    if ok:
        with open(out, encoding="utf-8") as fh:
            data = json.load(fh)
        ext = data.get("extrinsics") or {}
        check("CLI 结果 schema v2 且含世界系外参",
              data.get("version") == 2 and ext.get("method", "").startswith("solvePnP"),
              ext.get("method"))
        check("结果记录方向确认状态",
              ext.get("orientation_confirmed") is True
              and ext.get("cameras", {}).get("0", {}).get("R_world_to_cam") is not None,
              "orientation_confirmed=%s" % ext.get("orientation_confirmed"))

    # (2) 重载会话要能恢复已保存的结果（不需要重算）
    svc = SVC.service()
    svc.reload_session(root)
    check("重载会话恢复内参结果", bool(svc._intrinsics),
          "相机 %s" % ",".join(sorted(svc._intrinsics)))
    check("重载会话恢复外参结果", svc._extrinsic is not None,
          (svc._extrinsic or {}).get("method"))
    check("重载后 build_result 可用", svc.build_result().get("version") == 2)

    # (3) 新增采集后旧结果必须失效（不能拿旧外参当新结果保存）
    before = svc._extrinsic
    svc._invalidate("extrinsic")
    check("新增外参样本后旧外参被作废", before is not None and svc._extrinsic is None)


def _frame(gray, seq):
    from camera_manager import Frame
    return Frame(gray=gray, seq=seq, mf_ts=seq * 333333,
                 host_ts=time.time(), mono_ts=time.perf_counter())


class _FakeWorker:
    """最小相机替身：只有采集用得到的那几个属性/方法。"""

    state = "streaming"
    device_name = "fake-cam"
    format_info = "NV12"
    requested = {}

    def __init__(self, frame, index=0):
        self.index = int(index)
        self.device_key = "fake-cam-%d" % self.index
        self.frames = [frame]

    def push(self, frame):
        self.frames.append(frame)

    def latest(self):
        return self.frames[-1]

    def wait_frame(self, after_seq, timeout=1.0):
        f = self.frames[-1]
        return f if f.seq > after_seq else None

    @property
    def buffered(self):
        return len(self.frames)


class _FakeManager:
    def __init__(self, workers):
        self._workers = workers

    def worker(self, index):
        return self._workers.get(int(index))


class _FakeSync:
    """最小同步服务替身：每次返回同一对新帧，用于测"重复帧对"拒绝。"""

    max_dt_ms = 20.0

    def __init__(self, fa, fb, pair_seq: int = 1):
        self._fa, self._fb = fa, fb
        self.pair_seq = int(pair_seq)
        self.calls = 0

    def recent_groups(self, count=1):
        from sync import GroupRecord
        self.calls += 1
        return [GroupRecord(group_seq=self.pair_seq, span_ms=0.3,
                            frames={0: self._fa, 1: self._fb},
                            created_mono=time.perf_counter())]

    # 兼容旧调用
    def recent_pairs(self, count=1):
        return self.recent_groups(count)


def test_capture_http(spec: B.BoardSpec, size, tex, off, tmp: str) -> None:
    """HTTP 采集 + 重复帧/重复帧对拒绝（用假相机，不碰真实设备）。"""
    print("\n[8] HTTP 采集与重复帧处理（模拟相机）")
    import calibration.service as SVC
    from preview_server import build_app

    _, _, (R0, t0), (R1, t1) = _two_camera_poses()
    img0 = render(spec, R0, t0, TRUE_K, TRUE_D, size, tex, off)
    img1 = render(spec, R1, t1, TRUE_K, TRUE_D, size, tex, off)

    w0, w1 = _FakeWorker(_frame(img0, 100), 0), _FakeWorker(_frame(img1, 200), 1)
    orig_manager, orig_sync = SVC.manager, SVC.sync_service
    SVC.manager = lambda: _FakeManager({0: w0, 1: w1})
    try:
        svc = SVC.service()
        root = os.path.join(tmp, "http_session")
        shutil.rmtree(root, ignore_errors=True)
        client = build_app().test_client()
        r = client.post("/api/calib/session", json={
            "cols": spec.cols, "rows": spec.rows, "square_mm": spec.square_mm,
            "cameras": [0, 1], "root": root})
        check("HTTP 建会话", r.status_code == 200, r.json if r.status_code != 200 else "")

        r1 = client.post("/api/calib/capture/single", json={"camera": 0})
        check("HTTP 采集第一帧", r1.status_code == 200 and r1.json.get("ok"),
              r1.json.get("message") if r1.status_code == 200 else r1.json)
        r2 = client.post("/api/calib/capture/single", json={"camera": 0})
        check("同一帧不会被当成第二次观测", r2.status_code == 409,
              (r2.json or {}).get("error", "")[:60])

        w0.push(_frame(img0, 101))
        r3 = client.post("/api/calib/capture/single", json={"camera": 0})
        check("新帧到达后采集被接受（姿态判重仍可能拦）", r3.status_code == 200,
              (r3.json or {}).get("message"))
        check("采集后旧外参被作废", svc._extrinsic is None)

        # 同步帧对：同一对不会被重复入库
        fa, fb = _frame(img0, 300), _frame(img1, 400)
        SVC.sync_service = lambda: _FakeSync(fa, fb)
        r4 = client.post("/api/calib/capture/stereo", json={})
        check("HTTP 采集同步帧对", r4.status_code == 200, (r4.json or {}).get("message"))
        t0_ts = time.perf_counter()
        r5 = client.post("/api/calib/capture/stereo", json={})
        check("同一对同步帧被拒绝", r5.status_code == 409,
              (r5.json or {}).get("error", "")[:60])
        check("重复帧对检测确实等待过新帧", time.perf_counter() - t0_ts > 1.0,
              "%.1fs" % (time.perf_counter() - t0_ts))
    finally:
        SVC.manager, SVC.sync_service = orig_manager, orig_sync
        try:
            SVC.service().close_session()
        except Exception:
            pass



# ---------------------------------------------------------------------------
# 用例 9：会话落盘 / 读回 / 结果 JSON / 坐标变换
# ---------------------------------------------------------------------------
def test_persistence(spec: B.BoardSpec, size: tuple[int, int],
                     intr: dict, stereo_res: dict, tmp: str) -> None:
    print("\n[9] 会话落盘、读回与结果文件")
    root = os.path.join(tmp, "session_test")
    sess = Session.create(root, spec, [0, 1], size, "NV12", "自检会话")
    tex, off = board_texture(spec)
    R_b, t_b = pose_with_centered_board(spec, 0.7, 0.1, -0.05, 0.0)
    img = render(spec, R_b, t_b, TRUE_K, TRUE_D, size, tex, off)
    det = B.detect(img, spec)
    sess.add_single(0, img, det, {"seq": 1, "mf_ts": 0, "host_ts": 0.0, "mono_ts": 0.0},
                    overlay=B.draw_overlay(img, det, spec, scale=0.5))
    check("单相机样本写入", os.path.isfile(os.path.join(sess.img_dir, "s0001_cam0.png")))

    sess2 = Session.load(root)
    check("会话读回", len(sess2.samples) == 1 and sess2.spec.cols == spec.cols,
          "%d 个样本" % len(sess2.samples))
    c = sess2.samples[0].corners(0)
    check("读回后角点坐标不变", c is not None and np.abs(c - det.corners).max() < 1e-6)

    # 图像尺寸一致性：读回的 PNG 必须和角点坐标系一致
    back = cv2.imread(os.path.join(sess.img_dir, "s0001_cam0.png"), cv2.IMREAD_UNCHANGED)
    check("落盘 PNG 为 8bit 单通道且尺寸不变",
          back is not None and back.dtype == np.uint8 and back.ndim == 2
          and (back.shape[1], back.shape[0]) == size,
          "%s %s" % (None if back is None else back.shape, None if back is None else back.dtype))

    # 结果 JSON
    result = {
        "schema": "lowcost-mocap/calibration", "version": 1,
        "intrinsics": {"0": intr, "1": intr},
        "extrinsics": {"R_10": stereo_res["R_10"], "T_10": stereo_res["T_10"],
                       "baseline_m": stereo_res["baseline_m"]},
    }
    path = os.path.join(tmp, "calibration.json")
    import json
    from calibration.board import to_jsonable
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(to_jsonable(result), fh, ensure_ascii=False, indent=2)

    try:
        from calibration.service import load_calibration, stereo_matrices
        data = load_calibration(path)
        check("结果文件加载 + 校验", True, os.path.basename(path))
        K0, D0, K1, D1, R, T = stereo_matrices(data)
        check("矩阵取出", K0.shape == (3, 3) and R.shape == (3, 3) and T.shape == (3,),
              "baseline=%.2fmm" % (np.linalg.norm(T) * 1000))
        K0s, _, _, _, _, _ = stereo_matrices(data, scaled_size=(640, 360))
        check("按分辨率缩放 K 正确", abs(K0s[0, 0] - K0[0, 0] / 2) < 1e-6,
              "fx: %.2f → %.2f" % (K0[0, 0], K0s[0, 0]))
    except Exception as exc:              # pragma: no cover
        check("结果文件加载 + 校验", False, "%s: %s" % (type(exc).__name__, exc))

    # 坐标约定自检：把板原点的 3D 点用 R/T 变到相机 1，再投影，应与真值一致
    X0 = np.array([[0.0, 0.0, 0.6]])
    X1 = X0 @ np.asarray(stereo_res["R_10"]).T + np.asarray(stereo_res["T_10"])
    p0 = cv2.projectPoints(X0, np.zeros(3), np.zeros(3), TRUE_K, TRUE_D)[0].reshape(-1, 2)
    p1 = cv2.projectPoints(X1, np.zeros(3), np.zeros(3), TRUE_K, TRUE_D)[0].reshape(-1, 2)
    # 用估计的外参重投影，应与真值投影接近
    p1_est = cv2.projectPoints(X1, np.zeros(3), np.zeros(3),
                               np.asarray(intr["K"]), np.asarray(intr["D"]))[0].reshape(-1, 2)
    check("X_cam1 = R@X_cam0 + T 约定自洽", float(np.linalg.norm(p1_est - p1)) < 5.0,
          "重投影差 %.2fpx" % float(np.linalg.norm(p1_est - p1)))


# ---------------------------------------------------------------------------
# 用例 10：内参覆盖率按相机 / 阶段隔离
# ---------------------------------------------------------------------------
def _add_intrinsic_samples(sess: Session, spec: B.BoardSpec, size, tex, off,
                           n: int = 6) -> dict:
    """给两台相机各写 n 张 intrinsic 阶段的合成样本。"""
    counts = {}
    for cam in (0, 1):
        k, guard = 0, 0
        while k < n and guard < n * 10:
            guard += 1
            R_b, t_b = pose_with_centered_board(
                spec, RNG.uniform(0.62, 1.0), RNG.uniform(-0.4, 0.4),
                RNG.uniform(-0.3, 0.3), RNG.uniform(-0.42, 0.42),
                RNG.uniform(-0.16, 0.16), RNG.uniform(-0.08, 0.08))
            img = render(spec, R_b, t_b, TRUE_K, TRUE_D, size, tex, off)
            det = B.detect(img, spec)
            if not det.ok:
                continue
            sess.add_single(cam, img, det, {"seq": k},
                            signature=B.pose_signature(det, spec), phase="intrinsic")
            k += 1
        counts[cam] = k
    return counts


def test_intrinsic_coverage(spec: B.BoardSpec, size, tex, off, tmp: str) -> None:
    print("\n[10] 内参覆盖率：只统计本相机 intrinsic 阶段")
    import calibration.service as SVC

    root = os.path.join(tmp, "coverage_session")
    shutil.rmtree(root, ignore_errors=True)
    sess = Session.create(root, spec, [0, 1], size, "NV12", "覆盖率自检")
    counts = _add_intrinsic_samples(sess, spec, size, tex, off, 6)
    check("覆盖率用例：两相机各有 6 张内参图",
          counts == {0: 6, 1: 6}, counts)

    # 再塞一张 extrinsic 图：它绝不能进入内参覆盖率统计
    R_b, t_b = pose_with_centered_board(spec, 0.9, 0.2, -0.1, 0.3, 0.15, 0.10)
    img = render(spec, R_b, t_b, TRUE_K, TRUE_D, size, tex, off)
    det = B.detect(img, spec)
    if det.ok:
        sess.add_single(0, img, det, {"seq": 50},
                        signature=B.pose_signature(det, spec), phase="extrinsic")

    svc = SVC.service()
    svc.reload_session(root)
    try:
        res = svc.run_intrinsic(None)
        ok_counts, ok_cov = True, True
        detail = []
        for cam in (0, 1):
            d = res["intrinsics"].get(str(cam), {})
            cov = d.get("coverage") or {}
            hits = cov.get("corner_hits")
            ok_counts = ok_counts and d.get("num_views") == 6
            ok_cov = ok_cov and hits == 6 * spec.count and 0 <= cov.get(
                "covered_ratio", -1) <= 1
            detail.append("cam%s views=%s hits=%s cov=%.0f%%" % (
                cam, d.get("num_views"), hits,
                100 * cov.get("covered_ratio", 0)))
        check("内参样本数只算 intrinsic（外参图不混入）", ok_counts, "；".join(detail))
        check("两台相机分别给出 6×6 网格覆盖率", ok_cov, "；".join(detail))
    finally:
        svc.close_session()


# ---------------------------------------------------------------------------
# 用例 11：双相机同帧外参采集 / 单张求解 / 采集并自动验证
# ---------------------------------------------------------------------------
def test_extrinsic_pair_and_validation(spec: B.BoardSpec, size, tex, off,
                                       tmp: str) -> None:
    print("\n[11] 双相机同帧外参采集 + 单张 solvePnP + 自动验证")
    import calibration.service as SVC

    root = os.path.join(tmp, "pair_session")
    shutil.rmtree(root, ignore_errors=True)
    sess = Session.create(root, spec, [0, 1], size, "NV12", "同帧采集自检")
    counts = _add_intrinsic_samples(sess, spec, size, tex, off, 6)
    if counts != {0: 6, 1: 6}:
        check("同帧采集用例：内参数据准备", False, counts)
        return

    svc = SVC.service()
    svc.reload_session(root)
    intr = svc.run_intrinsic(None)
    check("同帧采集用例：先解出两台相机内参",
          len(intr.get("intrinsics", {})) == 2, list(intr.get("intrinsics", {})))

    C0, C1, (R0, t0), (R1, t1) = _two_camera_poses()

    img0 = render(spec, R0, t0, TRUE_K, TRUE_D, size, tex, off)
    img1 = render(spec, R1, t1, TRUE_K, TRUE_D, size, tex, off)
    w0, w1 = _FakeWorker(_frame(img0, 1000), 0), _FakeWorker(_frame(img1, 2000), 1)

    orig_manager, orig_sync = SVC.manager, SVC.sync_service
    SVC.manager = lambda: _FakeManager({0: w0, 1: w1})
    try:
        SVC.sync_service = lambda: _FakeSync(w0.latest(), w1.latest(), pair_seq=10)
        cap = svc.capture_extrinsic_pair()
        check("一次采集同时保存两台相机的外参图像",
              cap.get("ok") and all(v.get("ok") for v in cap["cameras"].values()),
              cap.get("message"))

        st = svc.session.stats()["by_phase"]
        check("外参计数各 1，内参计数不变",
              st.get("extrinsic") == {"0": 1, "1": 1}
              and st.get("intrinsic") == {"0": 6, "1": 6}, st)

        rig = svc.solve_extrinsic()
        base_true = float(np.linalg.norm(C1 - C0))
        check("每台相机仅 1 张有效外参图即可 solvePnP",
              abs(rig["relative"]["baseline_m"] - base_true) * 1000 < 5.0,
              "baseline %.2fmm（真值 %.2fmm）"
              % (rig["relative"]["baseline_m"] * 1000, base_true * 1000))
        check("单帧不输出无意义的帧间标准差",
              rig["stability"]["0"]["frames"] == 1
              and rig["stability"]["0"]["centre_std_m"] is None,
              rig["stability"]["0"].get("note"))

        # 板挪到别处，采一对新同步帧并自动验证
        off_w = np.array([0.32, 0.10, 0.22])
        va = render(spec, R0, R0 @ off_w + t0, TRUE_K, TRUE_D, size, tex, off)
        vb = render(spec, R1, R1 @ off_w + t1, TRUE_K, TRUE_D, size, tex, off)
        wa, wb = _FakeWorker(_frame(va, 1100), 0), _FakeWorker(_frame(vb, 2100), 1)
        SVC.manager = lambda: _FakeManager({0: wa, 1: wb})
        SVC.sync_service = lambda: _FakeSync(wa.latest(), wb.latest(), pair_seq=11)

        val = svc.capture_and_validate()
        check("采集并自动验证（一次操作）",
              bool(val.get("ok")) and val.get("validation") is not None,
              val.get("validation_error") or "")
        if val.get("validation"):
            agg = val["validation"]["aggregate"]
            sp = agg["spacing_abs_error_mm"]
            check("自动验证给出 1 个有效位置",
                  agg["placements"] == 1, agg["placements"])
            check("自动验证汇总间距误差为 mm（0–100 mm 合理区间）",
                  sp.get("unit") == "mm" and 0 <= sp["mean"] <= 100,
                  "mean=%.4fmm max=%.4fmm" % (sp["mean"], sp["max"]))
            st2 = svc.session.stats()["by_phase"]
            check("验证采集只增加 validation 帧对计数",
                  st2.get("validation_pairs") == 1
                  and st2.get("extrinsic") == {"0": 1, "1": 1}
                  and st2.get("intrinsic") == {"0": 6, "1": 6}, st2)

        # 低覆盖但高于外参下限：应接受（并告警），不能沿用内参 0.8% 门限
        real_detect = SVC.board_mod.detect
        base_det = B.detect(img0, spec)
        fake = B.Detection(ok=True, corners=base_det.corners,
                           method=base_det.method, polarity=base_det.polarity,
                           reason="",
                           quality=dict(base_det.quality, coverage=0.0015))
        SVC.board_mod.detect = lambda gray, spec, **kw: fake
        try:
            wx = _FakeWorker(_frame(img0, 1200), 0)
            wy = _FakeWorker(_frame(img1, 2200), 1)
            SVC.manager = lambda: _FakeManager({0: wx, 1: wy})
            SVC.sync_service = lambda: _FakeSync(wx.latest(), wy.latest(), pair_seq=12)
            cap2 = svc.capture_extrinsic_pair()
            check("0.15% 的外参图高于 0.1% 下限，被接受并带告警",
                  bool(cap2.get("ok")) and all(
                      v.get("ok") and v.get("warning")
                      for v in cap2["cameras"].values()),
                  {k: (v.get("status"), v.get("warning"))
                   for k, v in cap2["cameras"].items()})
            st3 = svc.session.stats()["by_phase"]
            check("低覆盖外参图计入有效外参数量",
                  st3.get("extrinsic") == {"0": 2, "1": 2}, st3)
        finally:
            SVC.board_mod.detect = real_detect
    finally:
        SVC.manager, SVC.sync_service = orig_manager, orig_sync
        try:
            SVC.service().close_session()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# 用例 13：最新内外参自动保存、跨会话载入与会话结果
# ---------------------------------------------------------------------------
def test_latest_result_files(spec: B.BoardSpec, size, tex, off, tmp: str) -> None:
    print("\n[13] 最新内外参自动保存与跨会话载入")
    import calibration.service as SVC
    from calibration.service import (
        DEFAULT_EXTRINSICS_PATH, DEFAULT_INTRINSICS_PATH,
        load_calibration, stereo_matrices,
    )

    root_a = os.path.join(tmp, "latest_a")
    shutil.rmtree(root_a, ignore_errors=True)
    sess_a = Session.create(root_a, spec, [0, 1], size, "NV12")
    counts = _add_intrinsic_samples(sess_a, spec, size, tex, off, 6)
    check("最新结果用例：内参样本准备", counts == {0: 6, 1: 6}, counts)

    svc = SVC.service()
    svc.reload_session(root_a)
    r = svc.run_intrinsic(None)
    check("内参计算后自动保存最新 intrinsics.json",
          bool(r.get("latest_saved")) and os.path.isfile(DEFAULT_INTRINSICS_PATH),
          r.get("latest_error"))
    with open(DEFAULT_INTRINSICS_PATH, encoding="utf-8") as fh:
        idata = json.load(fh)
    icams = idata.get("cameras") or {}
    check("最新内参文件含两台相机的 K/D/RMS/尺寸/样本数/覆盖率/时间",
          idata.get("schema") == "lowcost-mocap/intrinsics"
          and all(k in icams for k in ("0", "1"))
          and all(all(f in icams[k] for f in (
              "K", "D", "rms_px", "image_size", "num_views",
              "coverage_ratio", "created")) for k in ("0", "1")),
          sorted(icams))
    check("内参计算不再返回/写全局兼容文件",
          "compat" not in r
          and not os.path.isfile(os.path.join(SVC.CALIB_DIR, "calibration.json")))

    # 保存失败：把目标路径换成一个目录，计算仍应返回结果
    real_intr_path = SVC.DEFAULT_INTRINSICS_PATH
    bad_path = os.path.join(tmp, "intrinsics_as_dir.json")
    shutil.rmtree(bad_path, ignore_errors=True)
    os.makedirs(bad_path)
    SVC.DEFAULT_INTRINSICS_PATH = bad_path
    try:
        rf = svc.run_intrinsic(None)
        check("最新内参保存失败时仍返回计算结果",
              rf.get("latest_saved") is False and bool(rf.get("intrinsics"))
              and bool(rf.get("latest_error")), rf.get("latest_error"))
    finally:
        SVC.DEFAULT_INTRINSICS_PATH = real_intr_path

    # 新会话、零样本，直接载入内参
    root_b = os.path.join(tmp, "latest_b")
    shutil.rmtree(root_b, ignore_errors=True)
    Session.create(root_b, spec, [0, 1], size, "NV12")
    svc.reload_session(root_b)
    before_phase = svc.session.stats()["by_phase"]
    rl = svc.load_intrinsics()
    check("新会话无样本也能载入两台内参",
          set(rl.get("loaded") or {}) == {"0", "1"}
          and all(v.get("source") == "loaded"
                  for v in svc._intrinsics.values()),
          rl.get("errors"))
    check("载入内参不改变采集计数",
          svc.session.stats()["by_phase"] == before_phase)
    with open(os.path.join(root_b, "result_intrinsic.json"), encoding="utf-8") as fh:
        b_intr = json.load(fh)
    check("载入内参在会话结果里标记 loaded_from（不冒充重新计算）",
          b_intr.get("loaded_from") == rl.get("path"), b_intr.get("loaded_from"))

    # 外参：未确认/已确认都只写会话结果和最新外参文件
    C0, C1, (R0, t0), (R1, t1) = _two_camera_poses()
    img0 = render(spec, R0, t0, TRUE_K, TRUE_D, size, tex, off)
    img1 = render(spec, R1, t1, TRUE_K, TRUE_D, size, tex, off)

    w0 = _FakeWorker(_frame(img0, 3100), 0)
    w1 = _FakeWorker(_frame(img1, 3200), 1)
    orig_manager, orig_sync = SVC.manager, SVC.sync_service
    SVC.manager = lambda: _FakeManager({0: w0, 1: w1})
    try:
        SVC.sync_service = lambda: _FakeSync(w0.latest(), w1.latest(), pair_seq=30)
        cap = svc.capture_extrinsic_pair()
        check("最新结果用例：外参图像采集", bool(cap.get("ok")), cap.get("message"))

        pending = svc.solve_extrinsic(orientation_confirmed=False,
                                      flip_cam={"0": False, "1": False})
        with open(DEFAULT_EXTRINSICS_PATH, encoding="utf-8") as fh:
            pdata = json.load(fh)
        check("未确认方向的外参保存为待确认，且不写全局兼容文件",
              pdata.get("orientation_confirmed") is False
              and "compat" not in pending
              and not os.path.isfile(os.path.join(SVC.CALIB_DIR,
                                                  "calibration.json")))

        rig = svc.solve_extrinsic(orientation_confirmed=True,
                                  flip_cam={"0": False, "1": False})
        check("已确认外参自动保存 world_extrinsics.json",
              bool(rig.get("latest_saved"))
              and os.path.isfile(DEFAULT_EXTRINSICS_PATH),
              rig.get("latest_error"))
        with open(DEFAULT_EXTRINSICS_PATH, encoding="utf-8") as fh:
            edata = json.load(fh)
        check("最新外参文件含相机位姿/相对位姿/基线/世界系/内参快照",
              edata.get("schema") == "lowcost-mocap/world-extrinsics"
              and all(k in (edata.get("cameras") or {}) for k in ("0", "1"))
              and edata.get("orientation_confirmed") is True
              and edata.get("baseline_m") is not None
              and edata.get("board") and edata.get("world_frame")
              and (edata.get("relative") or {}).get("R_10") is not None
              and (edata.get("intrinsics") or {}).get("0") is not None,
              edata.get("baseline_m"))
        check("已确认外参不再写全局兼容文件",
              "compat" not in rig
              and not os.path.isfile(os.path.join(SVC.CALIB_DIR,
                                                  "calibration.json")))

        saved = svc.save_result(None)
        session_result = os.path.join(svc.session.root, "calibration.json")
        check("save_result 默认只写当前会话副本",
              os.path.isfile(session_result)
              and saved.get("path") == os.path.relpath(
                  session_result, SVC.PROJECT_DIR).replace(os.sep, "/"),
              saved.get("path"))
        cdata = load_calibration(session_result)
        K0, D0, K1, D1, R, T = stereo_matrices(cdata)
        check("会话 calibration.json 的下游矩阵读取正常",
              K0.shape == (3, 3) and R.shape == (3, 3) and T.shape == (3,)
              # baseline_m 落盘保留 6 位小数，T_10 未取整；差 <0.1mm 即一致
              and abs(float(np.linalg.norm(T)) - float(edata["baseline_m"])) < 1e-4,
              "baseline=%.2fmm" % (np.linalg.norm(T) * 1000))
        try:
            load_calibration(None)
            no_path_error = ""
        except ValueError as exc:
            no_path_error = str(exc)
        check("load_calibration 不再回退默认兼容路径",
              "必须提供" in no_path_error, no_path_error)

        from preview_server import build_app
        mr = build_app().test_client().get("/api/calib/matrices")
        check("GET /api/calib/matrices 读取当前会话结果",
              mr.status_code == 200
              and abs(float(mr.json.get("baseline_m", 0))
                      - float(edata["baseline_m"])) < 1e-4,
              mr.status_code)

        # 最新外参保存失败时仍保留内存结果
        real_world_path = SVC.DEFAULT_EXTRINSICS_PATH
        bad_world_dir = os.path.join(tmp, "world_as_dir.json")
        shutil.rmtree(bad_world_dir, ignore_errors=True)
        os.makedirs(bad_world_dir)
        SVC.DEFAULT_EXTRINSICS_PATH = bad_world_dir
        try:
            rf = svc.solve_extrinsic(orientation_confirmed=True,
                                     flip_cam={"0": False, "1": False})
            check("最新外参保存失败时仍返回计算结果",
                  rf.get("latest_saved") is False and bool(rf.get("latest_error"))
                  and svc._extrinsic is not None, rf.get("latest_error"))
        finally:
            SVC.DEFAULT_EXTRINSICS_PATH = real_world_path

        # 模拟重启 + 新会话：先后载入内外参
        root_c = os.path.join(tmp, "latest_c")
        shutil.rmtree(root_c, ignore_errors=True)
        Session.create(root_c, spec, [0, 1], size, "NV12")
        svc.reload_session(root_c)
        rli = svc.load_intrinsics()
        rle = svc.load_extrinsics()
        check("新会话载入内外参后可直接使用",
              set(rli.get("loaded") or {}) == {"0", "1"}
              and svc._extrinsic is not None
              and all(k in svc._extrinsic["cameras"] for k in ("0", "1")),
              rle.get("path"))
        check("载入外参恢复基线、flip 与方向确认状态",
              abs(svc._extrinsic["relative"]["baseline_m"]
                  - edata["baseline_m"]) < 1e-9
              and svc._extrinsic["orientation_confirmed"] is True
              and svc._extrinsic.get("source") == "loaded"
              and isinstance(svc._extrinsic.get("flip_cam"), dict))

        # 不兼容 / 缺失不得破坏当前结果
        before_rig = svc._extrinsic
        bad_world = os.path.join(tmp, "bad_world.json")
        bad_data = json.loads(json.dumps(edata))
        bad_data["board"]["square_mm"] = 25.0
        with open(bad_world, "w", encoding="utf-8") as fh:
            json.dump(bad_data, fh, ensure_ascii=False)
        try:
            svc.load_extrinsics(bad_world)
            msg = ""
        except ValueError as exc:
            msg = str(exc)
        check("不兼容外参文件被拒绝且不清除当前外参",
              "不兼容" in msg and svc._extrinsic is before_rig, msg[:100])

        before_intr = svc._intrinsics["0"]["K"]
        bad_intr = os.path.join(tmp, "bad_intrinsics_data.json")
        bidata = json.loads(json.dumps(idata))
        bidata["cameras"]["0"]["image_size"] = [640, 480]
        bidata["cameras"]["1"]["image_size"] = [640, 480]
        with open(bad_intr, "w", encoding="utf-8") as fh:
            json.dump(bidata, fh, ensure_ascii=False)
        try:
            svc.load_intrinsics(bad_intr)
            msg2 = ""
        except ValueError as exc:
            msg2 = str(exc)
        check("不兼容内参文件被拒绝且不清空当前内参",
              "不兼容" in msg2 and svc._intrinsics["0"]["K"] == before_intr,
              msg2[:100])

        before_rig2 = svc._extrinsic
        try:
            svc.load_extrinsics(os.path.join(tmp, "no_such_world.json"))
            msg3 = ""
        except FileNotFoundError as exc:
            msg3 = str(exc)
        check("缺失外参文件不清除当前外参",
              "不存在" in msg3 and svc._extrinsic is before_rig2, msg3[:100])

        inv = svc.invalidate_extrinsic()
        check("修改 180° 后旧外参与方向确认失效",
              inv.get("invalidated") is True and svc._extrinsic is None
              and svc._validation is None, inv.get("message"))
    finally:
        SVC.manager, SVC.sync_service = orig_manager, orig_sync
        try:
            svc.close_session()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# 用例 12：网页入口与项目相对路径
# ---------------------------------------------------------------------------
def test_preview_ui_and_paths(tmp: str | None = None) -> None:
    print("\n[12] 网页入口与项目相对路径")
    from calibration.service import (
        CALIB_DIR, DEFAULT_SESSION_ROOT, MIN_COVERAGE,
        MIN_COVERAGE_EXTRINSIC, PROJECT_DIR, _normalize_validation_units,
        coverage_limit_for, project_relative,
    )
    import calibration.service as SVC

    check("外参/验证覆盖率为 0.1%，内参仍为 0.8%",
          abs(MIN_COVERAGE_EXTRINSIC - 0.001) < 1e-12
          and coverage_limit_for("intrinsic") == MIN_COVERAGE
          and coverage_limit_for("extrinsic") == MIN_COVERAGE_EXTRINSIC,
          "内参 %.1f%% / 外参 %.1f%%"
          % (100 * MIN_COVERAGE, 100 * MIN_COVERAGE_EXTRINSIC))

    check("项目相对路径 helper 以 computer_code 为基准",
          project_relative(os.path.join(PROJECT_DIR, "captures", "calibration"))
          == "captures/calibration"
          and project_relative(os.path.join(CALIB_DIR, "service.py"))
          == "api/calibration/service.py",
          project_relative(os.path.join(CALIB_DIR, "service.py")))
    if tmp:
        check("自检把跨会话结果重定向到临时目录",
              os.path.abspath(DEFAULT_SESSION_ROOT) == os.path.abspath(tmp),
              DEFAULT_SESSION_ROOT)
    check("旧版全局兼容结果常量与文件已移除",
          not hasattr(SVC, "DEFAULT_RESULT_PATH")
          and not os.path.isfile(os.path.join(CALIB_DIR, "calibration.json")))

    legacy = {"aggregate": {"spacing_abs_error_mm": {
        "mean": 0.0038, "p95": 0.0112, "max": 0.2028,
        "expected_spacing_mm": 30.0}}}
    _normalize_validation_units(legacy)
    _normalize_validation_units(legacy)          # 重复调用不能再次放大
    sp = legacy["aggregate"]["spacing_abs_error_mm"]
    check("旧验证结果的汇总间距误差按米→毫米迁移一次",
          sp.get("unit") == "mm" and abs(sp["mean"] - 3.8) < 1e-6
          and abs(sp["max"] - 202.8) < 1e-6, sp)

    html_path = os.path.join(_API, "static", "preview.html")
    with open(html_path, encoding="utf-8") as fh:
        html = fh.read()
    check("可视化模式选择框（三种模式）",
          'id="c-vis-mode"' in html and "棋盘格角点可视化" in html
          and "世界坐标轴可视化" in html and "无可视化" in html)
    check("可视化模式默认选中棋盘格角点",
          'value="corners" selected' in html)
    check("三维视图与重置/俯视控制",
          'id="c-scene"' in html and 'id="c-reset-view"' in html
          and 'id="c-top-view"' in html and "R_cam_to_world" in html)
    check("合并后的采集/验证入口",
          'id="c-ex-pair"' in html and 'id="c-val-capture"' in html
          and 'id="c-val-recalc"' in html)
    check("方向确认标签已更名",
          "已确认世界坐标轴方向" in html and "已用轴叠加确认" not in html)
    check("已移除网页双目标定/覆盖检查/分散采集入口",
          all(x not in html for x in (
              'id="c-stereo"', 'id="c-cover"', 'id="c-ex0"', 'id="c-ex1"',
              'id="c-cap-pair"', 'id="c-valrig"')))
    check("内参自动保存/载入入口与数值卡片",
          'id="c-intr-load"' in html and 'id="c-intr-out"' in html)
    check("外参自动保存/载入入口与数值卡片",
          'id="c-ex-load"' in html and 'id="c-ex-out"' in html)
    check("已移除手动查看结果/保存 calibration.json 按钮",
          'id="c-view"' not in html and 'id="c-save"' not in html)
    check("红外 LED 检测/定位入口",
          'id="led-section"' in html and 'id="l-detect-start"' in html
          and 'id="l-locate-start"' in html and 'id="l-view"' in html
          and 'value="led"' in html)
    check("红外参数表含阈值/面积/峰距/分离/极线/三角容差",
          all(x in html for x in (
              'id="l-thr"', 'id="l-mina"', 'id="l-maxa"', 'id="l-peak"',
              'id="l-split"', 'id="l-blur"', 'id="l-epi"',
              'id="l-reproj"', 'id="l-tri"', 'id="l-thr-0"')))
    check("版本横幅与刷新按钮",
          'id="version-banner"' in html and 'id="btn-reload"' in html)
    check("前端不再写死版本或回退到其他端口",
          'DIRECT_BASE' not in html and 'API_BASE' not in html
          and 'v6 ·' not in html)

    from preview_server import build_app
    client = build_app().test_client()
    vr = client.get("/api/version")
    vd = vr.json or {}
    check("后端 /api/version 返回真实 Git 版本",
          vr.status_code == 200 and vd.get("git_sha") and vd.get("started_at"),
          "%s / %s" % (vd.get("git_sha"), vd.get("started_at")))
    sr = client.get("/api/led/state")
    sd = sr.json or {}
    check("LED state 空闲状态可用",
          sr.status_code == 200 and sd.get("status") == "idle", sr.status_code)
    pr = client.get("/api/led/params")
    pd = pr.json or {}
    check("LED 参数读取默认值",
          pr.status_code == 200
          and (pd.get("params") or {}).get("threshold") is not None,
          pr.status_code)
    wr = client.post("/api/led/params", json={
        "params": {"threshold": 180, "min_area": 4, "max_area": 500,
                   "min_peak_distance": 5.5, "split_blobs": True,
                   "blur_kernel": 3, "epipolar_tol_px": 2.5,
                   "reprojection_tol_px": 2.5, "triangle_tol_mm": 5},
        "overrides": {"0": {"threshold": 190}}})
    wd = wr.json or {}
    check("LED 参数保存到临时目录",
          wr.status_code == 200 and wd.get("version") == 1, wd.get("path"))
    rs = client.post("/api/led/params/reset", json={})
    rd = rs.json or {}
    check("LED 参数恢复默认", rs.status_code == 200
          and (rd.get("params") or {}).get("threshold") == 200, rs.status_code)
    check("网页使用 34/79/79 mm 等腰三角形与灯板方向箭头",
          "34/79/79" in html and "drawBoardDirection" in html
          and "灯板方向" in html)
    check("相机模型与视锥体绘制缩小 50%",
          "const d = 0.14" in html and "0.0175" in html
          and "const d = 0.28" not in html)
    check("外参采集按钮改为多相机措辞",
          "采集相机图片" in html and "采集双相机图片" not in html)
    check("内参与外参计数按相机动态生成",
          'data-cap="${cam}"' in html and "c-nx-line" in html
          and "外参帧数 " in html)
    check("相机2/3 的 180° 复选框插在已确认之前",
          "confirmedRow.parentNode.insertBefore(node, confirmedRow)" in html)
    check("标定实时图为相机2/3 克隆相机1 的 figure",
          "figs[1]" in html and "c-img1" in html)


def _selftest_dir() -> str:
    """建自检用的临时目录。

    刻意**不用** ``%TEMP%``：Windows 上杀软/索引会临时占用那里的 .tmp 文件，
    导致写 session.json 时报 PermissionError。放在工作区 ``captures/`` 下
    （已 gitignore）更稳。
    """
    root = os.path.join(os.path.dirname(_API), "captures")
    os.makedirs(root, exist_ok=True)
    return tempfile.mkdtemp(prefix="calib_selftest_", dir=root)


def _write_fake_camera_config(root: str) -> None:
    """自检用的逻辑编号↔设备绑定（全部已确认），避免历史文件触发身份校验。"""
    path = os.path.join(root, "camera_config.json")
    payload = {
        "schema": "lowcost-mocap/camera-config", "version": 1,
        "cameras": [{"index": i, "device_key": "fake-cam-%d" % i,
                     "device_name": "fake-cam", "confirmed": True,
                     "width": 1280, "height": 720, "fps": 30.0,
                     "format": "NV12"} for i in range(4)],
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False)


def test_camera_identity(tmp: str) -> None:
    """逻辑编号 ↔ 设备实例绑定的确认/拒绝规则。"""
    print("\n[14] 相机身份绑定与标定归属校验")
    import camera_config as CC

    path = os.path.join(tmp, "identity_config.json")
    old_path = CC.CONFIG_PATH
    CC.CONFIG_PATH = path
    try:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"schema": CC.SCHEMA, "version": 1, "cameras": [
                {"index": 0, "device_key": "usb#port0", "confirmed": True,
                 "width": 1280, "height": 720, "fps": 30, "format": "NV12"},
                {"index": 1, "device_key": "usb#port1", "confirmed": False,
                 "width": 1280, "height": 720, "fps": 30, "format": "NV12"},
            ]}, fh)
        cfg = CC.load()
        check("读取固定绑定配置", [c["index"] for c in cfg["cameras"]] == [0, 1],
              [c["device_key"] for c in cfg["cameras"]])
        ok, why = CC.calibration_check(0, {"device_key": "usb#port0"})
        check("设备标识一致时允许套用标定", ok, why)
        ok, why = CC.calibration_check(1, {"device_key": "usb#other"})
        check("设备标识不一致且未确认时拒绝套用标定",
              not ok and "属于设备" in why, why)
        ok, why = CC.calibration_check(0, {"device_key": "usb#other"})
        check("设备标识不一致但已确认时允许沿用标定（换 USB 口/延长线）", ok, why)
        ok, why = CC.calibration_check(1, {})
        check("旧标定无设备信息且未确认时拒绝", not ok and "人工确认" in why, why)
        ok, why = CC.calibration_check(0, {})
        check("旧标定无设备信息但已确认时允许", ok, why)
        try:
            CC.update(1, {"device_key": "usb#port0", "confirmed": True})
            dup_ok = False
        except ValueError:
            dup_ok = True
        check("同一物理设备不能绑定两个逻辑编号", dup_ok)
        CC.update(1, {"device_key": "usb#port1", "confirmed": True})
        ok, _why = CC.calibration_check(1, {})
        check("人工确认后旧标定可用", ok)
    finally:
        CC.CONFIG_PATH = old_path


def test_multi_camera_rig(spec: B.BoardSpec, size, tex, off, tmp: str) -> None:
    """3/4 台不同分辨率相机的统一外参、结果结构与联合验证。"""
    print("\n[15] 多相机（4 台、不同分辨率）统一外参与联合验证")
    import calibration.service as SVC

    cams = [0, 1, 2, 3]
    sizes = {0: (1280, 720), 1: (1280, 720),
             2: (960, 540), 3: (640, 360)}
    root = os.path.join(tmp, "multi_rig")
    shutil.rmtree(root, ignore_errors=True)
    sess = Session.create(root, spec, cams, sizes, "NV12")
    check("会话保存每台相机自己的图像尺寸",
          sess.image_size_for(2) == (960, 540)
          and sess.image_size_for(3) == (640, 360),
          sess.image_sizes)

    centres = [(-1.05, -0.85, 1.20), (0.98, -0.78, 1.28),
               (-0.95, 0.85, 1.15), (1.00, 0.90, 1.25)]
    poses, intr = {}, {}
    for i, c in enumerate(cams):
        w, h = sizes[c]
        sx, sy = w / 1280.0, h / 720.0
        K = np.array([[TRUE_K[0, 0] * sx, 0.0, TRUE_K[0, 2] * sx],
                      [0.0, TRUE_K[1, 1] * sy, TRUE_K[1, 2] * sy],
                      [0.0, 0.0, 1.0]])
        D = np.asarray(TRUE_D, np.float64)
        intr[str(c)] = {"K": K.tolist(), "D": D.tolist(),
                        "image_size": [w, h], "rms_px": 0.1, "num_views": 1}
        poses[c] = look_at(np.array(centres[i], np.float64),
                           np.array([0.0, 0.0, 0.02]))

    svc = SVC.CalibrationService()
    svc._session = sess
    svc._intrinsics = {k: dict(v) for k, v in intr.items()}

    for c in cams:
        R, t = poses[c]
        K = np.asarray(intr[str(c)]["K"])
        D = np.asarray(intr[str(c)]["D"])
        img = render_rig(spec, R, t, (0, 0, 0), K, D, sizes[c], tex, off)
        det = B.detect(img, spec)
        sess.add_single(c, img, det,
                        {"seq": 1, "mf_ts": 0, "host_ts": 0.0,
                         "mono_ts": 0.0, "group_seq": 1},
                        phase="extrinsic")
    rig = svc.solve_extrinsic(orientation_confirmed=True)
    check("四台相机各自 solvePnP 得到外参",
          sorted(rig["cameras"], key=int) == ["0", "1", "2", "3"],
          sorted(rig["cameras"]))
    check("多视角联合三角化一致性 < 2 mm",
          rig["multi_view"]["max_m"] is not None
          and rig["multi_view"]["max_m"] < 0.002,
          rig["multi_view"]["max_m"])
    check("外参记录每台相机的独立分辨率",
          all(rig["cameras"][str(c)]["reprojection"]["rms_px"] < 1.0
              for c in cams))

    result = svc.build_result()
    check("结果包含全部 4 台相机与各自 image_size",
          len(result["cameras"]) == 4
          and result["cameras"]["2"]["width"] == 960
          and result["cameras"]["3"]["height"] == 360
          and len(result["intrinsics"]) == 4,
          {k: (v["width"], v["height"]) for k, v in result["cameras"].items()})

    # 联合验证：板挪到另一个位置，只有 3 台相机看到
    placements = []
    for k, offset in enumerate([(0.08, -0.05, 0.15), (-0.06, 0.10, 0.25)]):
        corners = {}
        for c in (0, 1, 2):
            R, t = poses[c]
            K = np.asarray(intr[str(c)]["K"])
            D = np.asarray(intr[str(c)]["D"])
            img = render_rig(spec, R, t, offset, K, D, sizes[c], tex, off)
            det = B.detect(img, spec)
            corners[c] = det.corners if det.ok else None
        placements.append({"id": "v%d" % k, "corners": corners, "dt_ms": 0.3})
    val = E.validate_rig_multi(placements, spec, rig,
                               {c: (np.asarray(intr[str(c)]["K"]),
                                    np.asarray(intr[str(c)]["D"]))
                                for c in cams})
    check("三相机联合验证给出有效位置",
          val["aggregate"]["placements"] == 2
          and val["aggregate"]["cameras"] == [0, 1, 2],
          val["aggregate"].get("cameras"))
    check("联合验证间距误差 < 1 mm",
          val["aggregate"]["spacing_abs_error_mm"]["mean"] < 1.0,
          val["aggregate"]["spacing_abs_error_mm"])
    check("联合验证平面度 < 1 mm",
          max(p["planarity_rms_mm"] for p in val["per_placement"]) < 1.0,
          [p["planarity_rms_mm"] for p in val["per_placement"]])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="标定模块合成数据自检")
    ap.add_argument("--quick", action="store_true", help="减少样本数，快速跑")
    ap.add_argument("--keep", action="store_true", help="保留渲染图便于肉眼检查")
    args = ap.parse_args(argv)

    size = (1280, 720)
    # 现场配置：12×9 个方格 → 11×8 内角点，单格 30mm
    spec = B.BoardSpec(11, 8, 30.0)

    tmp = _selftest_dir()
    # 自检不能覆盖本机真实的最新内参/外参；在第一次 import service 前
    # 把跨会话结果目录重定向到临时目录。
    os.environ["MOCAP_CALIB_DIR"] = tmp
    _write_fake_camera_config(tmp)
    keep_dir = tmp if args.keep else None
    print("自检临时目录：%s" % tmp)
    print("标定板 %dx%d 内角点（%d×%d 方格），方格 %gmm，图像 %dx%d"
          % (spec.cols, spec.rows, spec.squares_x, spec.squares_y, spec.square_mm, *size))

    intr = test_intrinsic(args.quick, keep_dir, spec, size)
    if intr is None:
        print("\n内参用例未通过，后续用例跳过")
        return 1

    if keep_dir:
        keep_dir = tmp
    st = test_stereo(spec, size, intr, intr, args.quick, keep_dir)
    if st:
        test_auto_flip(spec, size, intr, intr)
    test_pnp(spec, size, intr, intr, args.quick, keep_dir)
    _tex, _off = board_texture(spec)
    test_frame_pairing(spec, size, _tex, _off)
    test_cli_and_state(spec, size, _tex, _off, tmp)
    test_capture_http(spec, size, _tex, _off, tmp)
    test_intrinsic_coverage(spec, size, _tex, _off, tmp)
    test_extrinsic_pair_and_validation(spec, size, _tex, _off, tmp)
    test_latest_result_files(spec, size, _tex, _off, tmp)
    test_camera_identity(tmp)
    test_multi_camera_rig(spec, size, _tex, _off, tmp)
    test_preview_ui_and_paths(tmp)
    if st:
        test_persistence(spec, size, intr, st, tmp)

    print("\n" + "=" * 66)
    passed = sum(1 for _, ok, _ in RESULT if ok)
    failed = [(n, d) for n, ok, d in RESULT if not ok]
    print("自检结果：%d/%d 通过" % (passed, len(RESULT)))
    for n, d in failed:
        print("  FAILED: %s —— %s" % (n, d))
    if not args.keep:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)
    else:
        print("渲染图保留在：%s" % tmp)
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
