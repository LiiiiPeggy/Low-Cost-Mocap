"""单相机内参标定（直接调用 cv2.calibrateCamera）。

坐标 / 模型约定
--------------------------------------------------------------------------
* 针孔模型 + 五参数畸变 ``[k1, k2, p1, p2, k3]``（OpenCV 默认）。
* 三维物点单位为**米**（由 BoardSpec 生成），所以标定得到的内参对"米制三角化"
  直接可用；K 的单位是像素，与物点单位无关。
* 图像坐标 = 采集到的**原始灰度图**坐标（1280x720，未旋转未缩放）。
* 返回的 K、D 只对同一分辨率、同一裁剪方式、同一朝向的图像有效。如果以后对
  图像做缩放/裁剪，必须按比例换算 K（见 ``docs/calibration.md``），不能直接复用。

不做的事
--------------------------------------------------------------------------
不手写张正友标定、不手写非线性优化、不自定义残差。全部交给
``cv2.calibrateCamera``（内部就是张氏初值 + LM 精化）。
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from calibration.board import BoardSpec, to_jsonable

# 收敛条件与 OpenCV 官方 sample 一致
DEFAULT_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 100, 1e-6)

MIN_VIEWS = 4          # 少于此数量直接拒绝：5 个畸变参数根本约束不住
RECOMMENDED_VIEWS = 15  # 低于此数量给"建议补采"提示


def calibrate(samples: list, spec: BoardSpec, image_size: tuple[int, int],
              *, camera: int | None = None, flags: int = 0,
              criteria=DEFAULT_CRITERIA) -> dict:
    """用若干样本求解一台相机的 K、D。

    ``samples`` 需带 ``corners(camera)`` 方法（见 dataset.Sample）；调用方负责
    只传同一台相机、同一棋盘格、同一分辨率的有效样本。
    """
    if len(samples) < MIN_VIEWS:
        raise ValueError(
            "有效样本只有 %d 张，至少需要 %d 张（建议 %d 张以上）才能稳定求解 "
            "K 与 5 个畸变参数" % (len(samples), MIN_VIEWS, RECOMMENDED_VIEWS))

    objp = spec.object_points().reshape(-1, 1, 3)
    object_points, image_points, used_ids = [], [], []
    for s in samples:
        if camera is not None:
            corners = s.corners(camera)
        else:
            corners = None
            for key in s.detections:
                corners = s.corners(key)
                if corners is not None:
                    break
        if corners is None:
            continue
        if len(corners) != spec.count:
            raise ValueError("样本 %s 的角点数 %d 与规格 %d 不符"
                             % (s.id, len(corners), spec.count))
        image_points.append(corners.astype(np.float32))
        object_points.append(objp)
        used_ids.append(s.id)

    if len(image_points) < MIN_VIEWS:
        raise ValueError("实际可用样本只有 %d 张，至少需要 %d 张"
                         % (len(image_points), MIN_VIEWS))

    rms, K, D, rvecs, tvecs = cv2.calibrateCamera(
        object_points, image_points, tuple(int(v) for v in image_size),
        None, None, flags=flags, criteria=criteria)

    D = np.asarray(D).reshape(-1)
    if D.size < 5:                      # 统一成 5 参数，便于落盘和后续使用
        D = np.pad(D, (0, 5 - D.size))
    D = D[:5]

    # 逐张重投影误差：用标定出的位姿把所有物点投回图像，与检测角点比
    per_view = []
    for sid, (rvec, tvec, imgp) in zip(used_ids, zip(rvecs, tvecs, image_points)):
        proj, _ = cv2.projectPoints(objp, rvec, tvec, K, D)
        err = np.linalg.norm(proj.reshape(-1, 2) - imgp.reshape(-1, 2), axis=1)
        per_view.append({
            "sample": sid,
            "rms_px": round(float(np.sqrt(np.mean(err ** 2))), 4),
            "mean_px": round(float(err.mean()), 4),
            "max_px": round(float(err.max()), 4),
            "board_distance_m": round(float(np.linalg.norm(np.asarray(tvec).reshape(3))), 4),
        })

    errors = np.array([p["rms_px"] for p in per_view], dtype=np.float64)
    suggestion = suggest_exclusions(per_view)

    fx, fy = float(K[0, 0]), float(K[1, 1])
    coverage = _radius_coverage(image_points, K, image_size)
    return {
        "camera": None,                # 由调用方填
        "image_size": [int(image_size[0]), int(image_size[1])],
        "radius_coverage": coverage,
        "num_views": len(image_points),
        "sample_ids": used_ids,
        "K": np.asarray(K).tolist(),
        "D": D.tolist(),
        "rms_px": round(float(rms), 4),
        "reprojection": {
            "mean_px": round(float(errors.mean()), 4),
            "median_px": round(float(np.median(errors)), 4),
            "p95_px": round(float(np.percentile(errors, 95)), 4),
            "max_px": round(float(errors.max()), 4),
        },
        "per_view": per_view,
        "focal_px": [round(fx, 4), round(fy, 4)],
        "principal_point_px": [round(float(K[0, 2]), 4), round(float(K[1, 2]), 4)],
        "skew": round(float(K[0, 1]), 6),
        "distortion_model": "pinhole + [k1,k2,p1,p2,k3]",
        "outlier_suggestion": suggestion,
        "warnings": _warnings(K, D, float(rms), len(image_points), coverage),
        "board": spec.to_dict(),
    }


def _radius_coverage(image_points: list, K: np.ndarray,
                     image_size: tuple[int, int]) -> dict:
    """角点相对主点的半径覆盖：畸变模型只在有数据的半径范围内可信。

    实测（合成数据）：角点只覆盖画面中心区域时，k2/k3 会互相补偿，重投影 RMS
    仍可低到 0.07px，但把模型外推到画面四角会出现十几像素的偏差。所以这个指标
    必须单独告诉用户，不能只看 RMS。
    """
    w, h = int(image_size[0]), int(image_size[1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    r_max = 0.0
    for pts in image_points:
        d = np.linalg.norm(np.asarray(pts).reshape(-1, 2) - np.array([cx, cy]), axis=1)
        r_max = max(r_max, float(d.max()))
    diag = float(math.hypot(w, h)) / 2.0
    ratio = r_max / diag if diag else 0.0
    return {
        "max_corner_radius_px": round(r_max, 1),
        "half_diagonal_px": round(diag, 1),
        "ratio": round(ratio, 3),
        "note": "角点最远半径 / 画面半对角；<0.75 时画面边缘的畸变属于外推",
    }


def suggest_exclusions(per_view: list[dict], *, factor: float = 2.0,
                       absolute_px: float = 1.0) -> dict:
    """给出"哪些图建议剔除"的建议（不自动删，交给人决定）。"""
    if not per_view:
        return {"threshold_px": None, "exclude": [], "note": "没有样本"}
    vals = np.array([p["rms_px"] for p in per_view], dtype=np.float64)
    med = float(np.median(vals))
    # 阈值 = max(中位数*factor, 中位数+absolute_px)。不用固定 px 阈值卡所有设备。
    thr = max(med * factor, med + absolute_px)
    bad = [p for p in per_view if p["rms_px"] > thr]
    return {
        "threshold_px": round(thr, 3),
        "median_px": round(med, 3),
        "exclude": [{"sample": p["sample"], "rms_px": p["rms_px"]} for p in bad],
        "note": ("阈值 = max(中位数×%.1f, 中位数+%.1fpx)，只是建议；"
                 "剔除后请重新标定并对比 RMS 是否真的下降" % (factor, absolute_px)),
    }


def _warnings(K, D, rms: float, n: int, coverage: dict | None = None) -> list[str]:
    out = []
    if n < RECOMMENDED_VIEWS:
        out.append("有效样本 %d 张，少于建议的 %d 张，畸变参数可能不稳" % (n, RECOMMENDED_VIEWS))

    if coverage and coverage.get("ratio", 1.0) < 0.75:
        out.append("角点只覆盖到画面中心 %.0f%% 半径，画面边缘的畸变属于外推结果；"
                   "建议补采让标定板出现在四角的姿态" % (100 * coverage["ratio"]))

    fx, fy = float(K[0, 0]), float(K[1, 1])
    if fx <= 1 or fy <= 1:
        out.append("焦距异常（fx=%.1f, fy=%.1f），检查棋盘格边长单位或角点顺序" % (fx, fy))
    if abs(fx - fy) / max(fx, fy) > 0.02:
        out.append("fx 与 fy 相差 %.1f%%，若非故意的非方形像素，可能是角点检测问题"
                   % (100 * abs(fx - fy) / max(fx, fy)))

    k1 = float(D[0])
    if abs(k1) > 0.6:
        out.append("k1=%.3f 偏大，若镜头是大广角/鱼眼，常规针孔模型不适用，"
                   "需要改用 cv2.fisheye 标定模型" % k1)
    if rms > 1.0:
        out.append("整体 RMS=%.3fpx 偏大，常见原因：运动模糊、板面不平、"
                   "角点误检、混入了不同分辨率的图" % rms)
    if rms < 0.05 and n < 8:
        out.append("RMS 异常低且样本少，可能是过拟合，请补采更多姿态")
    return out


def undistort_points(pts: np.ndarray, K: np.ndarray, D: np.ndarray) -> np.ndarray:
    """像素坐标 → 去畸变后的归一化坐标。**只对原始（未去畸变）像素点调用一次。**

    如果上游已经把图 ``cv2.undistort`` 过了，再调这个就是"二次去畸变"，
    结果会偏，务必确认输入来源。
    """
    p = np.asarray(pts, dtype=np.float32).reshape(-1, 1, 2)
    out = cv2.undistortPoints(p, np.asarray(K, np.float64), np.asarray(D, np.float64))
    return out.reshape(-1, 2)


def reprojection_error(object_points: np.ndarray, image_points: np.ndarray,
                       rvec, tvec, K, D) -> dict:
    """给定位姿，算一张图的重投影误差（像素）。"""
    proj, _ = cv2.projectPoints(np.asarray(object_points, np.float32).reshape(-1, 1, 3),
                                np.asarray(rvec, np.float64).reshape(3, 1),
                                np.asarray(tvec, np.float64).reshape(3, 1),
                                np.asarray(K, np.float64), np.asarray(D, np.float64))
    err = np.linalg.norm(proj.reshape(-1, 2) - np.asarray(image_points, np.float32).reshape(-1, 2),
                         axis=1)
    return {"rms_px": round(float(np.sqrt(np.mean(err ** 2))), 4),
            "mean_px": round(float(err.mean()), 4),
            "max_px": round(float(err.max()), 4)}


def K_from(result: dict) -> np.ndarray:
    return np.asarray(result["K"], dtype=np.float64)


def D_from(result: dict) -> np.ndarray:
    return np.asarray(result["D"], dtype=np.float64).reshape(-1)


def validate_result(d: dict) -> tuple[bool, list[str]]:
    """检查一份内参结果结构是否自洽（离线加载后先跑一遍再用）。"""
    problems = []
    for key in ("K", "D", "image_size", "num_views"):
        if key not in d:
            problems.append("缺少字段 %s" % key)
    if problems:
        return False, problems

    K = np.asarray(d["K"], dtype=np.float64)
    if K.shape != (3, 3):
        problems.append("K 形状应为 3x3，实际 %s" % (K.shape,))
    if len(np.asarray(d["D"]).reshape(-1)) != 5:
        problems.append("D 应为 5 个参数")
    if K[0, 0] <= 0 or K[1, 1] <= 0:
        problems.append("焦距必须为正")
    if not math.isfinite(float(np.sum(K))):
        problems.append("K 含非有限值")
    return (not problems), problems


__all__ = ["calibrate", "suggest_exclusions", "undistort_points", "reprojection_error",
           "K_from", "D_from", "validate_result", "MIN_VIEWS", "RECOMMENDED_VIEWS",
           "to_jsonable"]
