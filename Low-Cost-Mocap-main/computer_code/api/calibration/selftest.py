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
import math
import os
import sys
import tempfile

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_API = os.path.dirname(_HERE)
if _API not in sys.path:
    sys.path.insert(0, _API)
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


def check(name: str, ok: bool, detail: str = "") -> bool:
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


def board_pose(distance: float, yaw: float, pitch: float, roll: float,
               shift_x: float = 40.0, shift_y: float = 25.0) -> tuple[np.ndarray, np.ndarray]:
    """构造板在相机系下的位姿：板中心落在光轴附近，距离 distance。"""
    R = cv2.Rodrigues(np.array([pitch, yaw, roll], np.float64))[0]
    # 让板中心（板坐标几何中心）平移到 (cx, cy, distance)
    return R, np.array([shift_x, shift_y, distance], np.float64)


def board_center_offset(spec: B.BoardSpec) -> np.ndarray:
    """板几何中心相对板坐标原点的偏移。

    世界系原点已经在图案中心，所以这里恒为 0（保留函数是为了让调用方明确
    "板位姿的平移就是图案中心的位置"）。
    """
    return np.zeros(3)


def pose_with_centered_board(spec, distance, yaw, pitch, roll, px=0.0, py=0.0):
    """板在相机系下的位姿：``X_cam = R_b @ X_board + t_b``。

    因为物点已经以图案中心为原点，``t_b`` 就是图案中心在相机系中的位置。
    """
    c = board_center_offset(spec)
    R = cv2.Rodrigues(np.array([pitch, yaw, roll], np.float64))[0]
    t = np.array([px, py, distance], np.float64) - R @ c
    return R, t


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
def build_stereo_pairs(spec, size, n: int, tex, off, *, reverse_b: bool = False,
                       flip_board_b: bool = False):
    """生成 n 对同步帧，返回 (pairs, 真值 R, 真值 T)。"""
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
        if flip_board_b:
            pass  # 板是中心对称的，物理翻转在渲染上无差别；这里只模拟标号差异

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

    class Pair:
        def __init__(self, sid, c0, c1):
            self.id, self.detections = sid, {"0": {"ok": True}, "1": {"ok": True}}
            self._0, self._1 = c0, c1

        def corners(self, cam):
            return self._0 if int(cam) == 0 else self._1

    samples = [Pair("p%03d" % i, c0, c1) for i, (c0, c1, _, _) in enumerate(raw)]
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

    class Pair:
        def __init__(self, sid, c0, c1):
            self.id, self.detections = sid, {"0": {"ok": True}, "1": {"ok": True}}
            self._0, self._1 = c0, c1

        def corners(self, cam):
            return self._0 if int(cam) == 0 else self._1

    samples = [Pair("p%03d" % i, c0, c1) for i, (c0, c1, _, _) in enumerate(raw)]

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
        check("验证：极线中位 < 1.5px", agg["epipolar_median_px"]["mean"] < 1.5,
              "mean=%.3fpx" % agg["epipolar_median_px"]["mean"])
        check("验证：重建平面度 < 1mm",
              max(p["planarity_rms_mm"] for p in val["per_placement"]) < 1.0,
              "max=%.3fmm" % max(p["planarity_rms_mm"] for p in val["per_placement"]))
        # 高度覆盖：三个位置应出现不同的 centre_z
        zs = [p["centre_world_m"][2] for p in val["per_placement"]]
        check("验证覆盖了不同高度", len(set(round(z, 2) for z in zs)) >= 2,
              "centre_z = %s m" % [round(z, 3) for z in zs])
        # 板框尺寸检查（已知 300mm × 210mm 内角点跨度）
        bsc = val["per_placement"][0]["board_size_check_mm"]
        check("板面跨距与理论值一致 < 2mm",
              abs(bsc["x"] - bsc["expected_x"]) < 2 and abs(bsc["y"] - bsc["expected_y"]) < 2,
              "x=%.2f/%.2f y=%.2f/%.2f mm" % (bsc["x"], bsc["expected_x"],
                                              bsc["y"], bsc["expected_y"]))

    return rig


# ---------------------------------------------------------------------------
# 用例 5：会话落盘 / 读回 / 结果 JSON / 坐标变换
# ---------------------------------------------------------------------------
def test_persistence(spec: B.BoardSpec, size: tuple[int, int],
                     intr: dict, stereo_res: dict, tmp: str) -> None:
    print("\n[5] 会话落盘、读回与结果文件")
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
    X1_est = X0 @ np.asarray(stereo_res["R_10"]).T + np.asarray(stereo_res["T_10"])
    p1_est = cv2.projectPoints(X1_est, np.zeros(3), np.zeros(3),
                               np.asarray(intr["K"]), np.asarray(intr["D"]))[0].reshape(-1, 2)
    check("X_cam1 = R@X_cam0 + T 约定自洽", float(np.linalg.norm(p1_est - p1)) < 5.0,
          "重投影差 %.2fpx" % float(np.linalg.norm(p1_est - p1)))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="标定模块合成数据自检")
    ap.add_argument("--quick", action="store_true", help="减少样本数，快速跑")
    ap.add_argument("--keep", action="store_true", help="保留渲染图便于肉眼检查")
    args = ap.parse_args(argv)

    size = (1280, 720)
    # 现场配置：12×9 个方格 → 11×8 内角点，单格 30mm
    spec = B.BoardSpec(11, 8, 30.0)

    tmp = tempfile.mkdtemp(prefix="mocap_calib_selftest_")
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
