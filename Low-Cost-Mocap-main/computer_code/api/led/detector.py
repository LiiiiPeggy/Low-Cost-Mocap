"""单相机红外 LED 亮点检测。

不假设“一个连通域 = 一个 LED”。流程：
1. 可选高斯模糊；
2. 灰度阈值二值化；
3. 连通域 + 面积/形状过滤；
4. 对每个候选亮区用**距离变换峰值**判断是否包含多个光斑；
5. 多个峰值时用最近峰值分配（watershed 的简化等价）拆分；
6. 用原始灰度做强度加权亚像素中心。

完全饱和、峰值无法分开的亮区会标记为 ``unresolved``，不会伪造三个点。
"""

from __future__ import annotations

import copy
import math

import cv2
import numpy as np

# 参数默认值。面积单位是原始图像像素；三角形容差单位是 mm。
DEFAULT_PARAMS = {
    "threshold": 200,             # 0-255，二值化阈值
    "min_area": 3,                # 单个 LED 的最小亮区面积（px2）
    "max_area": 600,              # 超过此面积仍无法拆分则视为过大亮区
    "min_peak_distance": 6.0,     # 峰值最小间距（px）
    "split_blobs": True,          # 是否做光斑分离
    "blur_kernel": 3,             # 0/3/5/7/9，0 表示不模糊
    "peak_rel_threshold": 0.50,   # 峰值至少达到最大距离变换值的比例
    "min_solidity": 0.25,         # 面积 / 凸包面积，过滤细长噪声
    "max_candidates": 12,         # 单相机最多输出候选点
    "saturation_value": 250,      # 视为饱和的像素值
    "epipolar_tol_px": 3.0,       # 双相机极线距离阈值（去畸变理想像素）
    "reprojection_tol_px": 3.0,   # 三角化点的重投影误差阈值
    "triangle_tol_mm": 6.0,       # 33 mm 等边三角形的边长容差
}

# 每个数值参数的允许范围；用于 API 校验，避免误输入把检测拖死。
PARAM_LIMITS = {
    "threshold": (0.0, 255.0),
    "min_area": (1.0, 100000.0),
    "max_area": (2.0, 1000000.0),
    "min_peak_distance": (1.0, 200.0),
    "blur_kernel": (0.0, 9.0),
    "peak_rel_threshold": (0.05, 0.99),
    "min_solidity": (0.02, 1.0),
    "max_candidates": (3.0, 64.0),
    "saturation_value": (128.0, 255.0),
    "epipolar_tol_px": (0.25, 30.0),
    "reprojection_tol_px": (0.25, 30.0),
    "triangle_tol_mm": (1.0, 30.0),
}

TRIANGLE_SIDE_M = 0.033          # 三个 LED 中心间距，米


def default_params() -> dict:
    return copy.deepcopy(DEFAULT_PARAMS)


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    s = str(value).strip().lower()
    if s in ("1", "true", "yes", "on", "开", "是"):
        return True
    if s in ("0", "false", "no", "off", "关", "否", ""):
        return False
    raise ValueError("布尔参数取值无法识别：%r" % (value,))


def validate_params(raw: dict | None, *, partial: bool = False) -> dict:
    """校验并规范化检测参数。``partial=True`` 用于相机覆盖项。"""
    raw = raw or {}
    if not isinstance(raw, dict):
        raise ValueError("参数必须是 JSON 对象")
    out = {} if partial else default_params()
    for key, value in raw.items():
        if key not in DEFAULT_PARAMS:
            continue                       # 忽略未知字段，保持向前兼容
        if key == "split_blobs":
            out[key] = _as_bool(value)
            continue
        try:
            num = float(value)
        except (TypeError, ValueError):
            raise ValueError("参数 %s 必须是数字" % key)
        lo, hi = PARAM_LIMITS[key]
        if not (lo <= num <= hi):
            raise ValueError("参数 %s 超出范围 [%g, %g]：%g" % (key, lo, hi, num))
        if key in ("threshold", "min_area", "max_area", "blur_kernel",
                   "max_candidates", "saturation_value"):
            out[key] = int(round(num))
        else:
            out[key] = float(num)
    if "blur_kernel" in out and out["blur_kernel"] % 2 == 0:
        raise ValueError("blur_kernel 必须是奇数（0/3/5/7/9）")
    if out.get("max_area", 1) <= out.get("min_area", 0):
        raise ValueError("max_area 必须大于 min_area")
    return out


def merge_params(base: dict, override: dict | None) -> dict:
    out = default_params()
    out.update(base or {})
    out.update(override or {})
    return validate_params(out)


def _as_gray(gray: np.ndarray) -> np.ndarray:
    if gray is None:
        raise ValueError("图像为空")
    arr = np.asarray(gray)
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    if arr.dtype != np.uint8:
        raise ValueError("期望 uint8 灰度图，收到 %s" % arr.dtype)
    return np.ascontiguousarray(arr)


def _weighted_center(gray_roi: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return 0.0, 0.0
    w = gray_roi[ys, xs].astype(np.float64)
    s = float(w.sum())
    if s <= 1e-9:
        return float(xs.mean()), float(ys.mean())
    return float((xs * w).sum() / s), float((ys * w).sum() / s)


def _contour_of(mask: np.ndarray) -> np.ndarray | None:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def _peak_points(dt: np.ndarray, mask: np.ndarray, params: dict) -> np.ndarray:
    """距离变换局部极大值；返回 (N,2) 的 (x,y) 峰值坐标。"""
    max_dt = float(dt.max())
    if max_dt < 1.0:
        return np.empty((0, 2), np.float64)
    k = max(3, int(round(float(params["min_peak_distance"]))))
    if k % 2 == 0:
        k += 1
    kernel = np.ones((k, k), np.uint8)
    local_max = cv2.dilate(dt, kernel)
    peak_mask = ((dt >= local_max - 1e-4)
                 & (dt >= max(1.0, params["peak_rel_threshold"] * max_dt))
                 & mask).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(peak_mask, 8)
    peaks = []
    for label in range(1, n):
        ys, xs = np.nonzero(labels == label)
        if len(xs) == 0:
            continue
        w = dt[ys, xs]
        s = float(w.sum())
        if s <= 1e-9:
            peaks.append([float(xs.mean()), float(ys.mean())])
        else:
            peaks.append([float((xs * w).sum() / s),
                          float((ys * w).sum() / s)])
    return np.asarray(peaks, np.float64)


def _intensity_peaks(gray_roi: np.ndarray, mask: np.ndarray,
                     threshold: float, params: dict) -> np.ndarray:
    """距离变换分不开时的兜底：灰度局部极大值。"""
    k = max(3, int(round(float(params["min_peak_distance"]))))
    if k % 2 == 0:
        k += 1
    smooth = cv2.GaussianBlur(gray_roi, (k, k), 0)
    local_max = cv2.dilate(smooth, np.ones((k, k), np.uint8))
    vals = smooth[mask]
    if vals.size == 0:
        return np.empty((0, 2), np.float64)
    lo = threshold + 0.10 * (float(vals.max()) - threshold)
    peak_mask = (mask & (smooth >= local_max - 1e-3)
                 & (smooth >= lo)).astype(np.uint8)
    n, labels, _, _ = cv2.connectedComponentsWithStats(peak_mask, 8)
    peaks = []
    for label in range(1, n):
        ys, xs = np.nonzero(labels == label)
        if len(xs):
            w = smooth[ys, xs].astype(np.float64)
            s = float(w.sum())
            peaks.append([float((xs * w).sum() / s) if s else float(xs.mean()),
                          float((ys * w).sum() / s) if s else float(ys.mean())])
    return np.asarray(peaks, np.float64)


def _split_component(gray_roi: np.ndarray, mask: np.ndarray,
                     params: dict) -> list[np.ndarray]:
    """按距离变换峰值把一个连通亮区拆成 1~N 个部分。

    返回的 part mask 都是 ROI 坐标。只有 1 个 part 表示没有分离出多个光斑。
    """
    dt = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    peaks = _peak_points(dt, mask, params)
    if len(peaks) < 2:
        # 距离变换只有一个峰时，再尝试灰度局部极大值（薄桥粘连的情况）
        alt = _intensity_peaks(gray_roi, mask,
                               float(params["threshold"]), params)
        if len(alt) >= 2:
            peaks = alt
    if len(peaks) < 2:
        return [mask]

    ys, xs = np.nonzero(mask)
    # 最近峰值分配（Voronoi）。对圆形/近圆形 LED 光斑，效果与 watershed
    # 的分水岭边界基本一致，但实现更简单、没有 OpenCV 边界 -1 的问题。
    d2 = np.stack([(xs - px) ** 2 + (ys - py) ** 2 for px, py in peaks])
    assign = np.argmin(d2, axis=0)
    parts = []
    for k in range(len(peaks)):
        sel = assign == k
        if int(sel.sum()) < max(1, int(params["min_area"])):
            continue
        part = np.zeros_like(mask, dtype=bool)
        part[ys[sel], xs[sel]] = True
        parts.append(part)
    return parts if parts else [mask]


def detect_leds(gray: np.ndarray, params: dict | None = None) -> dict:
    """检测单张原始灰度图中的红外 LED 亮点。

    返回的 ``leds`` 坐标是原始图像像素坐标；``mask`` 是二值图（供可视化）。
    """
    p = validate_params(params or default_params())
    g = _as_gray(gray)
    k = int(p["blur_kernel"])
    processed = cv2.GaussianBlur(g, (k, k), 0) if k > 0 else g
    _, mask = cv2.threshold(processed, int(p["threshold"]), 255,
                            cv2.THRESH_BINARY)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8, cv2.CV_32S)
    leds, rejected, warnings = [], [], []
    unresolved = 0
    overexposed = 0

    for label in range(1, n):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < int(p["min_area"]):
            continue
        x, y, w, h = (int(stats[label, cv2.CC_STAT_LEFT]),
                      int(stats[label, cv2.CC_STAT_TOP]),
                      int(stats[label, cv2.CC_STAT_WIDTH]),
                      int(stats[label, cv2.CC_STAT_HEIGHT]))
        roi_mask = labels[y:y + h, x:x + w] == label
        roi_gray = g[y:y + h, x:x + w]
        vals = roi_gray[roi_mask]
        mean_v = float(vals.mean()) if vals.size else 0.0
        max_v = int(vals.max()) if vals.size else 0
        sat_ratio = (float((vals >= int(p["saturation_value"])).mean())
                     if vals.size else 0.0)
        contour = _contour_of(roi_mask)
        hull_area = (float(cv2.contourArea(cv2.convexHull(contour)))
                     if contour is not None else float(area))
        solidity = float(area) / hull_area if hull_area > 0 else 0.0

        common = {
            "parent_bbox": [x, y, w, h],
            "parent_area": area,
            "mean_intensity": round(mean_v, 1),
            "max_intensity": max_v,
            "saturated_ratio": round(sat_ratio, 3),
            "solidity": round(solidity, 3),
        }

        if solidity < float(p["min_solidity"]):
            rejected.append(dict(common, reason="形状不规则（solidity %.2f）"
                                 % solidity,
                                 contour=None if contour is None
                                 else (contour.reshape(-1, 2) + [x, y]).tolist()))
            continue

        parts = (_split_component(roi_gray, roi_mask, p)
                 if p["split_blobs"] else [roi_mask])
        if len(parts) >= 2:
            split_leds = []
            for part in parts:
                pa = int(part.sum())
                if pa < int(p["min_area"]):
                    continue
                cx, cy = _weighted_center(roi_gray, part)
                pc = _contour_of(part)
                pvals = roi_gray[part]
                split_leds.append({
                    "x": round(float(cx + x), 3),
                    "y": round(float(cy + y), 3),
                    "area": pa,
                    "mean_intensity": round(float(pvals.mean()), 1),
                    "max_intensity": int(pvals.max()),
                    "bbox": [x, y, w, h],
                    "contour": (None if pc is None
                                else (pc.reshape(-1, 2) + [x, y]).tolist()),
                    "split": True,
                    "parent_area": area,
                    "saturated_ratio": round(
                        float((pvals >= int(p["saturation_value"])).mean()), 3),
                })
            if len(split_leds) >= 2:
                leds.extend(split_leds)
                continue
            # 拆出来的有效部分不足 2 个：不要既加 part 又加整个连通域


        # 没有拆出多个光斑
        if area > int(p["max_area"]):
            unresolved += 1
            reason = "面积过大且无法分离"
            if sat_ratio >= 0.8:
                overexposed += 1
                reason += "（过曝/峰值不可分）"
            rejected.append(dict(common, reason=reason,
                                 contour=None if contour is None
                                 else (contour.reshape(-1, 2) + [x, y]).tolist()))
            continue

        if (sat_ratio >= 0.9
                and area >= max(4 * int(p["min_area"]),
                                int(0.5 * int(p["max_area"])))):
            overexposed += 1
            warnings.append("亮区高度饱和且较大（面积 %d px2），可能包含多个粘连 "
                            "LED；请降低 LED 亮度/曝光或调整阈值" % area)
        cx, cy = _weighted_center(roi_gray, roi_mask)
        leds.append({
            "x": round(float(cx + x), 3),
            "y": round(float(cy + y), 3),
            "area": area,
            "mean_intensity": round(mean_v, 1),
            "max_intensity": max_v,
            "bbox": [x, y, w, h],
            "contour": (None if contour is None
                        else (contour.reshape(-1, 2) + [x, y]).tolist()),
            "split": False,
            "parent_area": area,
            "saturated_ratio": round(sat_ratio, 3),
        })

    leds.sort(key=lambda d: (d["x"], d["y"]))
    if len(leds) > int(p["max_candidates"]):
        extra = leds[int(p["max_candidates"]):]
        leds = leds[:int(p["max_candidates"])]
        warnings.append("候选点超过 %d 个，已截断 %d 个"
                        % (p["max_candidates"], len(extra)))
    if unresolved:
        warnings.append("有 %d 个亮区无法分离成独立光斑%s"
                        % (unresolved, "，可能存在过曝" if overexposed else ""))
    if overexposed:
        warnings.append("检测到过曝/饱和亮区；请降低 LED 亮度或调整阈值，"
                        "不要把这些区域当作有效三点")

    status = "ok" if leds else ("unresolved" if unresolved else "no_leds")
    reason = ""
    if status == "no_leds":
        reason = "没有找到超过阈值的有效亮区"
    elif status == "unresolved":
        reason = "亮区无法分离为独立光斑（可能过曝或粘连）"
    for wmsg in warnings:
        if "过曝" in wmsg:
            reason = reason or wmsg

    return {
        "status": status,
        "reason": reason,
        "count": len(leds),
        "leds": leds,
        "rejected": rejected,
        "warnings": warnings,
        "unresolved_count": unresolved,
        "overexposed": bool(overexposed),
        "mask": mask,
        "params": p,
        "image_size": [int(g.shape[1]), int(g.shape[0])],
    }


__all__ = ["DEFAULT_PARAMS", "PARAM_LIMITS", "TRIANGLE_SIDE_M",
           "default_params", "validate_params", "merge_params",
           "detect_leds", "_as_gray"]
