"""棋盘格规格、角点检测、检测质量评估与可视化。

坐标系约定（全模块一致，务必遵守）
--------------------------------------------------------------------------
* 输入图像一律是**相机原始灰度图**（默认 1280x720），不做 ``np.rot90`` /
  ``make_square`` / 裁剪 / 缩放 / ``undistort`` / 滤波增强。
* 返回的角点坐标就是该原始图像的像素坐标，可直接喂给
  ``cv2.calibrateCamera`` / ``cv2.stereoCalibrate``。
* 标定板三维坐标单位为**米**，Z=0 平面；角点顺序为行优先
  （index = row * cols + col），原点在第一个内角点。
* 具体 OpenCV 坐标约定见 ``intrinsic.py`` / ``stereo.py`` 顶部注释。

关于"极性"（这一节是踩过的坑，别删）
--------------------------------------------------------------------------
红外相机下打印的棋盘格可能整体反相（白格变暗、黑格变亮）。我们对
``255-gray`` 再检测一次作为兜底。实测：``findChessboardCornersSB`` 对反相图
返回的角点**顺序会整体旋转 180°**（原本 index 0 的角点跑到了 index N-1）。

* 单相机内参标定：这种 180° 重标号等价于把标定板坐标系绕自身 Z 轴转 180°，
  而棋盘格对该旋转是对称的，所以对 K、D 没有影响，可以放心混用。
* 双相机外参标定：**同一物理姿态的两张图必须用同一种极性/同一种检测方法**，
  否则两台相机的角点对应关系差 180°，R、T 会吸收这个错误的板面旋转。
  因此这里提供 ``detect_pair()``，只在"两张图都成功"的前提下接受某个极性，
  绝不一边正相、一边反相。
* 即便是同极性，两台相机若有一台倒装，SB 仍可能给出 180° 不同的标号。
  这个二义性靠棋盘格本身无法消除（对称图案），由 ``stereo.py`` 用
  "三角化后点是否落在两相机前方"做几何判据自动择向，并在结果里留痕。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

# 现场实测配置：12×9 个方格 → 11×8 内角点，单格 30 mm。
# 内角点数 = 方格数 - 1，别把 12×9 当成内角点数。
DEFAULT_COLS = 11         # 内角点列数（沿 X / 12 格长边）= 12 - 1
DEFAULT_ROWS = 8          # 内角点行数（沿 Y / 9 格短边）  = 9 - 1
DEFAULT_SQUARE_MM = 30.0  # 方格边长（毫米）


# ---------------------------------------------------------------------------
# 标定板规格
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BoardSpec:
    """内角点行列数 + 方格实际边长（毫米）+ 板面相对地面的高度。

    世界坐标系（与需求一致）
    ------------------------------------------------------------------
    * 原点：**棋盘格图案中心**（内角点阵列的几何中心，不是某个角点）。
    * X 轴：沿 12 格**长边**方向，与 ``cols`` 方向一致。
    * Y 轴：沿 9 格**短边**方向，与 ``rows`` 方向一致。
    * Z 轴：板面法向，指向图案正面外侧；板正面朝上平放时即"垂直地面向上"。
    * 右手系 ``Z = X × Y``：俯视地面上的板，X 向右、Y 向里（远离自己）时，
      Z 指向自己，也就是**朝上**。
    * ``surface_height_m``：图案所在平面高于真实地面的高度。世界原点在板面上，
      所以**地面平面是 Z = -surface_height_m**；要地面基准坐标就把 Z 加上该值。
    """

    cols: int = DEFAULT_COLS
    rows: int = DEFAULT_ROWS
    square_mm: float = DEFAULT_SQUARE_MM
    surface_height_m: float = 0.0

    def __post_init__(self) -> None:
        if int(self.cols) < 2 or int(self.rows) < 2:
            raise ValueError("内角点行列数都必须 >= 2，当前 %dx%d" % (self.cols, self.rows))
        if not (float(self.square_mm) > 0):
            raise ValueError("方格边长必须为正数（单位 mm），当前 %r" % (self.square_mm,))
        if float(self.surface_height_m) < 0:
            raise ValueError("板面高度不能为负（板面在地面之上时取正值），当前 %r"
                             % (self.surface_height_m,))

    @property
    def count(self) -> int:
        return int(self.cols) * int(self.rows)

    @property
    def square_m(self) -> float:
        return float(self.square_mm) / 1000.0

    @property
    def grid_size(self) -> tuple[int, int]:
        """OpenCV 需要的 (cols, rows) 元组。"""
        return (int(self.cols), int(self.rows))

    @property
    def squares_x(self) -> int:
        return int(self.cols) + 1

    @property
    def squares_y(self) -> int:
        return int(self.rows) + 1

    def world_points(self) -> np.ndarray:
        """88 个世界坐标角点 (N,3) float32，单位**米**，Z=0。

        编号规则（与检测顺序严格对应，勿改）::

            index = row * cols + col          # 行优先
            x = (col - (cols-1)/2) * square    # 沿 X（长边）
            y = (row - (rows-1)/2) * square    # 沿 Y（短边）
            z = 0

        原点在图案中心，所以 x、y 正负对称；11×8、30mm 时 x ∈ [-0.15, 0.15]，
        y ∈ [-0.105, 0.105]。
        """
        rows, cols = int(self.rows), int(self.cols)
        sq = self.square_m
        cx = (cols - 1) / 2.0
        cy = (rows - 1) / 2.0
        pts = np.zeros((rows * cols, 3), dtype=np.float32)
        for row in range(rows):
            for col in range(cols):
                i = row * cols + col
                pts[i, 0] = (col - cx) * sq
                pts[i, 1] = (row - cy) * sq
        return pts

    def object_points(self) -> np.ndarray:
        """等价于 :meth:`world_points`。

        ``calibrateCamera`` / ``stereoCalibrate`` 对物点坐标系的原点与朝向不敏感
        （每视图位姿自由），所以内参与双目外参统一用世界系，避免两套约定。
        """
        return self.world_points()

    def flip180(self, corners: np.ndarray) -> np.ndarray:
        """把角点序列整体反转（等价于图案绕中心旋转 180°）。

        ``corners[::-1]`` 正好把 (row, col) 映射到 (rows-1-row, cols-1-col)，
        与世界点编号的 180° 对称严格对应。用于修正两台相机从相反方向拍同一块板
        时的角点标号反转。
        """
        return np.asarray(corners)[::-1].copy()

    def world_frame(self) -> dict:
        """世界坐标系定义（写进标定文件，供下游三维动捕对齐）。"""
        sx, sy = self.board_size_m()
        return {
            "origin": "board pattern centre (geometric centre of the inner-corner grid)",
            "origin_on_pattern_plane": True,
            "x_axis": "along the %d-square long edge (cols direction)" % self.squares_x,
            "y_axis": "along the %d-square short edge (rows direction)" % self.squares_y,
            "z_axis": "pattern normal, out of the printed face; up when the board lies flat",
            "handedness": "right-handed, Z = X x Y",
            "unit": "m",
            "index_order": "row-major: index = row * cols + col",
            "board_extent_x_m": round(sx, 6),
            "board_extent_y_m": round(sy, 6),
            "surface_height_m": round(float(self.surface_height_m), 6),
            "floor_z_m": round(-float(self.surface_height_m), 6),
            "floor_note": "地面平面为 Z = -surface_height_m；要地面基准坐标请把 Z 加上 surface_height_m",
        }

    def board_size_m(self) -> tuple[float, float]:
        """整块板（含外圈方格）的物理尺寸，米。"""
        return (self.squares_x * self.square_m, self.squares_y * self.square_m)

    def to_dict(self) -> dict:
        return {
            "cols": int(self.cols),
            "rows": int(self.rows),
            "inner_corners": self.count,
            "square_mm": float(self.square_mm),
            "square_m": self.square_m,
            "squares": [self.squares_x, self.squares_y],
            "surface_height_m": round(float(self.surface_height_m), 6),
            "object_point_order": "row-major, index = row*cols+col, origin at pattern centre, Z=0",
            "world_frame": self.world_frame(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BoardSpec":
        return cls(cols=int(d.get("cols", DEFAULT_COLS)),
                   rows=int(d.get("rows", DEFAULT_ROWS)),
                   square_mm=float(d.get("square_mm", DEFAULT_SQUARE_MM)),
                   surface_height_m=float(d.get("surface_height_m", 0.0)))


# ---------------------------------------------------------------------------
# 检测结果
# ---------------------------------------------------------------------------
@dataclass
class Detection:
    ok: bool
    corners: np.ndarray | None = None      # (N,1,2) float32，原始图像像素坐标
    method: str = ""                       # sb / sb-inverted / classic / classic-inverted
    polarity: str = "normal"               # normal / inverted
    reason: str = ""                       # 失败原因（中文，直接给用户看）
    quality: dict = field(default_factory=dict)

    @property
    def flat(self) -> np.ndarray | None:
        return None if self.corners is None else self.corners.reshape(-1, 2)

    def corners_list(self) -> list[list[float]]:
        if self.corners is None:
            return []
        return [[float(x), float(y)] for x, y in self.corners.reshape(-1, 2)]

    def to_dict(self) -> dict:
        return {
            "ok": bool(self.ok),
            "method": self.method,
            "polarity": self.polarity,
            "reason": self.reason,
            "quality": self.quality,
            "corners": self.corners_list(),
        }


def _as_gray(gray: np.ndarray) -> np.ndarray:
    if gray is None:
        raise ValueError("图像为空")
    arr = np.asarray(gray)
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    if arr.dtype != np.uint8:
        # 标定不做隐式拉伸：只接受已经是 8bit 的数据
        raise ValueError("期望 uint8 灰度图，收到 %s（标定不做隐式类型/范围转换）" % arr.dtype)
    return np.ascontiguousarray(arr)


# ---------------------------------------------------------------------------
# 检测
# ---------------------------------------------------------------------------
def _sb(gray: np.ndarray, spec: BoardSpec, fast: bool):
    flags = 0 if fast else (cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)
    ok, corners = cv2.findChessboardCornersSB(gray, spec.grid_size, flags=flags)
    if ok and corners is not None and len(corners) == spec.count:
        return corners.astype(np.float32)
    return None


def _classic(gray: np.ndarray, spec: BoardSpec):
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    ok, corners = cv2.findChessboardCorners(gray, spec.grid_size, flags=flags)
    if not ok or corners is None or len(corners) != spec.count:
        return None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 40, 1e-4)
    # 亚像素精化用**原始灰度**，不用反相图：cornerSubPix 对大梯度方向敏感，
    # 反相后梯度符号翻转但不影响极值位置，这里统一在原始图上做，减少歧义。
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return corners.astype(np.float32)


def detect(gray: np.ndarray, spec: BoardSpec, *, fast: bool = False,
           allow_inverted: bool = True) -> Detection:
    """单张图检测。返回的角点坐标对应传入图像的原始像素坐标。"""
    try:
        g = _as_gray(gray)
    except ValueError as exc:
        return Detection(False, reason=str(exc))

    inv = np.ascontiguousarray(255 - g)
    attempts = [("sb", g, "normal"), ("classic", g, "normal")]
    if allow_inverted:
        attempts += [("sb-inverted", inv, "inverted"), ("classic-inverted", inv, "inverted")]

    reasons = []
    for name, img, polarity in attempts:
        corners = _sb(img, spec, fast) if name.startswith("sb") else _classic(img, spec)
        if corners is None:
            reasons.append("%s: 未找到 %dx%d 内角点" % (name, spec.cols, spec.rows))
            continue
        q = quality(g, corners, spec)
        if q["uniformity"] > 0.5:
            # 相邻角点间距忽大忽小，基本可以断定检测到了错误的结构
            reasons.append("%s: 角点间距不均匀 (%.2f)，疑似误检" % (name, q["uniformity"]))
            continue
        return Detection(True, corners, name, polarity, "", q)

    return Detection(False, reason="；".join(reasons) or "检测失败")


def detect_pair(gray_a: np.ndarray, gray_b: np.ndarray, spec: BoardSpec,
                *, fast: bool = False) -> tuple[Detection, Detection, dict]:
    """双相机同步帧对检测：**只接受两张图同时成功的同一种检测方式**。

    这样做是为了保证同一物理姿态在两台相机里的角点标号一致（见模块 docstring
    里"极性"一节）。绝不允许一边正相、一边反相。
    """
    try:
        ga = _as_gray(gray_a)
        gb = _as_gray(gray_b)
    except ValueError as exc:
        msg = str(exc)
        return Detection(False, reason=msg), Detection(False, reason=msg), {"attempts": []}

    variants = [("sb", "normal"), ("classic", "normal"),
                ("sb-inverted", "inverted"), ("classic-inverted", "inverted")]
    attempts = []
    for name, polarity in variants:
        ia = 255 - ga if polarity == "inverted" else ga
        ib = 255 - gb if polarity == "inverted" else gb
        if name.startswith("sb"):
            ca = _sb(np.ascontiguousarray(ia), spec, fast)
            cb = _sb(np.ascontiguousarray(ib), spec, fast)
        else:
            ca = _classic(np.ascontiguousarray(ia), spec)
            cb = _classic(np.ascontiguousarray(ib), spec)

        ok = ca is not None and cb is not None
        attempts.append({"method": name, "polarity": polarity, "ok": bool(ok),
                         "a": ca is not None, "b": cb is not None})
        if not ok:
            continue

        qa = quality(ga, ca, spec)
        qb = quality(gb, cb, spec)
        if qa["uniformity"] > 0.5 or qb["uniformity"] > 0.5:
            attempts[-1]["ok"] = False
            attempts[-1]["note"] = "角点间距不均匀，疑似误检"
            continue

        # 同一个物理姿态，两视角的板面网格手性必须一致。180° 重标号不会改变
        # 叉积符号（ex、ey 同时反向），所以这个判据挡不住倒装，但能挡住
        # 左右镜像之类的错误检测。
        hand_a = _handedness(ca, spec)
        hand_b = _handedness(cb, spec)
        consistent = (hand_a == hand_b)
        info = {"method": name, "polarity": polarity, "handedness": [hand_a, hand_b],
                "ordering_consistent": bool(consistent), "attempts": attempts}
        if not consistent:
            attempts[-1]["note"] = "两视角手性相反"
            da = Detection(True, ca, name, polarity, "", qa)
            db = Detection(True, cb, name, polarity, "", qb)
            continue
        return (Detection(True, ca, name, polarity, "", qa),
                Detection(True, cb, name, polarity, "", qb), info)

    msg = "两台相机未能同时检测到同一块 %dx%d 棋盘格" % (spec.cols, spec.rows)
    return Detection(False, reason=msg), Detection(False, reason=msg), {
        "attempts": attempts, "reason": msg}


def _handedness(corners: np.ndarray, spec: BoardSpec) -> int:
    """用板面两条网格轴的叉积符号表示手性（+1 / -1）。"""
    f = corners.reshape(-1, 2)
    ex = f[spec.cols - 1] - f[0]
    ey = f[(spec.rows - 1) * spec.cols] - f[0]
    return 1 if (ex[0] * ey[1] - ex[1] * ey[0]) > 0 else -1


# ---------------------------------------------------------------------------
# 质量评估
# ---------------------------------------------------------------------------
def quality(gray: np.ndarray, corners: np.ndarray, spec: BoardSpec) -> dict:
    """角点几何质量 + 图像清晰度 + 画面覆盖。全部基于原始图像坐标。"""
    f = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
    grid = f.reshape(spec.rows, spec.cols, 2)

    ex = np.linalg.norm(np.diff(grid, axis=1), axis=2)   # (rows, cols-1)
    ey = np.linalg.norm(np.diff(grid, axis=0), axis=2)   # (rows-1, cols)
    ex_mean, ey_mean = float(ex.mean()), float(ey.mean())
    uniformity = max(float(ex.std() / ex_mean) if ex_mean else 1.0,
                     float(ey.std() / ey_mean) if ey_mean else 1.0)

    h, w = gray.shape[:2]
    hull = cv2.convexHull(f.astype(np.float32)).reshape(-1, 2)
    area = float(abs(cv2.contourArea(hull)))
    coverage = area / float(w * h)

    # 板面轴与世界坐标 x 轴的夹角（用于判断姿态是否重复）
    ax = grid[0, -1] - grid[0, 0]
    angle = math.degrees(math.atan2(float(ax[1]), float(ax[0])))
    centroid = f.mean(axis=0)
    diag = math.hypot(w, h)

    # 3x3 网格占用：角点落在多少个格子区域里
    cells = set()
    for x, y in f:
        cells.add((min(2, int(x / w * 3)), min(2, int(y / h * 3))))
    occupancy = len(cells) / 9.0

    sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    return {
        "mean_edge_px": round((ex_mean + ey_mean) / 2.0, 3),
        "uniformity": round(uniformity, 4),
        "coverage": round(coverage, 4),
        "occupancy": round(occupancy, 3),
        "sharpness": round(sharp, 2),
        "angle_deg": round(angle, 2),
        "centroid": [round(float(centroid[0]), 2), round(float(centroid[1]), 2)],
        "image_size": [int(w), int(h)],
    }


def pose_signature(det: Detection, spec: BoardSpec) -> dict:
    """归一化姿态签名，用于判重：位置 / 尺度 / 转角。"""
    q = det.quality
    w, h = q["image_size"]
    diag = math.hypot(w, h)
    return {
        "cx": q["centroid"][0] / w,
        "cy": q["centroid"][1] / h,
        "scale": q["mean_edge_px"] / diag,
        "angle": q["angle_deg"],
    }


def is_duplicate(sig: dict, others: list[dict], *,
                 pos_tol: float = 0.06, scale_tol: float = 0.12,
                 angle_tol_deg: float = 8.0) -> dict | None:
    """判断姿态是否与已有样本几乎相同。返回命中的样本信息或 None。"""
    for o in others:
        osig = o.get("signature") or {}
        if not osig:
            continue
        dpos = math.hypot(sig["cx"] - osig.get("cx", 0), sig["cy"] - osig.get("cy", 0))
        s0, s1 = sig["scale"], osig.get("scale") or 1e-9
        dscl = abs(math.log(s1 / s0)) if s0 and s1 else 9.9
        da = abs(((sig["angle"] - osig.get("angle", 0) + 90) % 180) - 90)
        if dpos <= pos_tol and dscl <= scale_tol and da <= angle_tol_deg:
            return {"sample": o.get("id"), "dpos": round(dpos, 4),
                    "dscale": round(dscl, 4), "dangle": round(da, 2)}
    return None


# ---------------------------------------------------------------------------
# 可视化
# ---------------------------------------------------------------------------
def draw_overlay(gray: np.ndarray, det: Detection, spec: BoardSpec,
                 *, scale: float = 1.0, label_step: int = 1) -> np.ndarray:
    """把角点、编号、网格线画到 BGR 图上，仅供显示，不参与计算。"""
    canvas = cv2.cvtColor(_as_gray(gray), cv2.COLOR_GRAY2BGR)
    if scale != 1.0:
        canvas = cv2.resize(canvas, None, fx=scale, fy=scale,
                            interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    if det.corners is None:
        cv2.putText(canvas, "NO BOARD", (12, 32), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 0, 255), 2, cv2.LINE_AA)
        return canvas

    pts = det.corners.reshape(-1, 2).astype(np.float32) * scale
    cv2.drawChessboardCorners(canvas, spec.grid_size, pts.reshape(-1, 1, 2), True)

    # 网格连线（看清行列关系）
    grid = pts.reshape(spec.rows, spec.cols, 2)
    for r in range(spec.rows):
        for c in range(spec.cols):
            if c + 1 < spec.cols:
                cv2.line(canvas, tuple(grid[r, c].astype(int)),
                         tuple(grid[r, c + 1].astype(int)), (0, 200, 255), 1, cv2.LINE_AA)
            if r + 1 < spec.rows:
                cv2.line(canvas, tuple(grid[r, c].astype(int)),
                         tuple(grid[r + 1, c].astype(int)), (0, 200, 255), 1, cv2.LINE_AA)

    # 角点编号（默认全画，板子角点多时建议 label_step>1）
    for idx in range(0, len(pts), max(1, label_step)):
        x, y = int(pts[idx, 0]), int(pts[idx, 1])
        cv2.putText(canvas, str(idx), (x + 3, y - 3), cv2.FONT_HERSHEY_SIMPLEX,
                    0.38, (0, 255, 0), 1, cv2.LINE_AA)

    q = det.quality or {}
    text = "%s  %dx%d  edge=%.1fpx  cov=%.0f%%  sharp=%.0f" % (
        det.method, spec.cols, spec.rows,
        q.get("mean_edge_px", 0), 100 * q.get("coverage", 0), q.get("sharpness", 0))
    cv2.putText(canvas, text, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 255, 0), 2, cv2.LINE_AA)
    return canvas


def coverage_map(samples: list[Detection], image_size: tuple[int, int],
                 spec: BoardSpec, cells: int = 6) -> dict:
    """角点在画面里的累积覆盖热度（用于"角点空间覆盖"提示）。"""
    w, h = image_size
    heat = np.zeros((cells, cells), dtype=np.int32)
    for det in samples:
        if det.corners is None:
            continue
        for x, y in det.corners.reshape(-1, 2):
            cx = min(cells - 1, max(0, int(x / w * cells)))
            cy = min(cells - 1, max(0, int(y / h * cells)))
            heat[cy, cx] += 1
    total = int(heat.sum())
    covered = int((heat > 0).sum())
    return {
        "cells": int(cells),
        "heat": heat.tolist(),
        "covered_ratio": round(covered / float(cells * cells), 3),
        "corner_hits": total,
        "suggestion": ("角点覆盖较集中，建议让标定板出现在画面四角与中心"
                       if covered / float(cells * cells) < 0.5 else "覆盖良好"),
    }


# 世界系坐标轴配色（BGR）：X 红、Y 绿、Z 蓝
AXIS_COLORS = {"x": (55, 55, 235), "y": (60, 195, 60), "z": (235, 145, 50)}


def board_outline_world(spec: BoardSpec) -> np.ndarray:
    """板面四角（世界系，米），用于画板框。"""
    sx, sy = spec.board_size_m()
    return np.array([[-sx / 2, -sy / 2, 0.0], [sx / 2, -sy / 2, 0.0],
                     [sx / 2, sy / 2, 0.0], [-sx / 2, sy / 2, 0.0]], np.float32)


def project_world(points_w: np.ndarray, K: np.ndarray, D: np.ndarray,
                  R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """世界点 → 像素（``X_cam = R @ X_world + t``）。"""
    rvec = cv2.Rodrigues(np.asarray(R, np.float64))[0]
    proj, _ = cv2.projectPoints(np.asarray(points_w, np.float64).reshape(-1, 1, 3),
                                rvec, np.asarray(t, np.float64).reshape(3, 1),
                                np.asarray(K, np.float64), np.asarray(D, np.float64))
    return proj.reshape(-1, 2)


def draw_world_overlay(gray: np.ndarray, spec: BoardSpec, K, D, R, t, *,
                       corners: np.ndarray | None = None,
                       scale: float = 1.0, axis_len_m: float | None = None,
                       label_step: int = 6, flipped: bool = False) -> np.ndarray:
    """在图像上画：板框、世界原点、X/Y/Z 轴、角点编号（可选）。

    ``R``、``t`` 是 world→camera（``X_cam = R @ X_world + t``），即 ``solvePnP``
    对世界点求解的直接输出。``corners`` 必须已经**按世界点顺序排好**（即做过
    格点对应重排），这样图上编号与世界点编号一一对应。

    这是人工确认角点物理对应关系的主要手段：把解出的世界系轴投影回原始图像，
    如果画出的 X 轴与实物 12 格长边不重合或反了，就说明需要切换 180°。
    """
    canvas = cv2.cvtColor(_as_gray(gray), cv2.COLOR_GRAY2BGR)
    if scale != 1.0:
        canvas = cv2.resize(canvas, None, fx=scale, fy=scale,
                            interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    K = np.asarray(K, np.float64)
    D = np.asarray(D, np.float64).reshape(-1)
    R = np.asarray(R, np.float64)
    t = np.asarray(t, np.float64).reshape(3)

    # 板框
    outline = project_world(board_outline_world(spec), K, D, R, t) * scale
    cv2.polylines(canvas, [outline.astype(np.int32)], True, (200, 200, 200), 1, cv2.LINE_AA)

    # 原点 + 三轴
    if axis_len_m is None:
        axis_len_m = spec.square_m * 3.0
    axes_w = np.array([[0, 0, 0],
                       [axis_len_m, 0, 0],
                       [0, axis_len_m, 0],
                       [0, 0, axis_len_m]], np.float32)
    pts = project_world(axes_w, K, D, R, t) * scale
    o = tuple(int(v) for v in pts[0])
    for name, p, color in (("X", pts[1], AXIS_COLORS["x"]),
                           ("Y", pts[2], AXIS_COLORS["y"]),
                           ("Z", pts[3], AXIS_COLORS["z"])):
        q = tuple(int(v) for v in p)
        cv2.arrowedLine(canvas, o, q, color, 2, cv2.LINE_AA, tipLength=0.18)
        cv2.putText(canvas, name, (q[0] + 4, q[1] - 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, color, 2, cv2.LINE_AA)
    cv2.circle(canvas, o, 5, (0, 255, 255), -1, cv2.LINE_AA)
    cv2.putText(canvas, "O", (o[0] + 7, o[1] + 16), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 255), 2, cv2.LINE_AA)

    # 角点 + 编号（顺序已与世界点一致）
    if corners is not None:
        cp = np.asarray(corners, np.float32).reshape(-1, 2) * scale
        cv2.drawChessboardCorners(canvas, spec.grid_size, cp.reshape(-1, 1, 2), True)
        for idx in range(0, len(cp), max(1, label_step)):
            x, y = int(cp[idx, 0]), int(cp[idx, 1])
            cv2.putText(canvas, str(idx), (x + 3, y - 3), cv2.FONT_HERSHEY_SIMPLEX,
                        0.36, (0, 220, 255), 1, cv2.LINE_AA)
        # 高亮 0 号角点：确认物理对应关系时最直接的锚点
        cv2.circle(canvas, tuple(int(v) for v in cp[0]), 9, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, "0", (int(cp[0, 0]) + 8, int(cp[0, 1]) + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

    if flipped:
        cv2.putText(canvas, "flip 180deg ON", (12, 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 2, cv2.LINE_AA)
    return canvas


def to_jsonable(obj: Any) -> Any:
    """把 numpy 递归转成可 json.dump 的结构。"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    return obj
