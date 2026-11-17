"""单相机红外 LED 亮点检测。

设计原则
--------------------------------------------------------------------------
1. **不要假设“一个连通域 = 一个 LED”**。三个 LED 可能因为过曝/光晕粘成一个
   亮区；环境里的灯带、地面反光也可能形成细长/不规则的亮区。
2. 先做**原始连通域**统计，再依次做面积、形状、亮度过滤；通过过滤的才是
   “LED 候选”，与“原始候选点”分开返回，网页可以分别画出来。
3. 粘连亮区用**距离变换局部峰值 + 最近峰值分配**（watershed 的简化等价）
   拆分；距离变换分不开时再用灰度局部极大值兜底。饱和平台没有可区分峰值时
   明确报告 ``unresolved``，绝不凭面积估算三个虚假中心。
4. 中心用原始灰度的强度加权亚像素中心；坐标始终是**整幅原始图像**坐标，
   即使检测只在一个 ROI 内进行，也会加回 ROI 偏移。

不引入深度学习、不引入新的第三方依赖，只使用 OpenCV/NumPy 的经典操作。
"""

from __future__ import annotations

import copy
import math

import cv2
import numpy as np

# 参数默认值。面积单位是原始图像像素；三角形容差单位是 mm。
# 这里的目标工作距离是 1–2.5 m、1280x720：单个 LED 光斑通常几十~几百 px²；
# 三个 LED 粘连后的父亮区可能上千 px²，因此 max_area 只限制“无法拆分的父
# 亮区”，拆分成功后按子区域面积判断。
DEFAULT_PARAMS = {
    "threshold": 200,             # 0-255，二值化阈值
    "min_area": 3,                # 单个 LED 的最小亮区面积（px2）
    "max_area": 800,              # 无法拆分时允许的最大父亮区面积
    "min_peak_distance": 6.0,     # 峰值最小间距（px）
    "split_blobs": True,          # 是否做光斑分离
    "split_min_area_ratio": 0.15,  # 拆分后每块至少占父亮区的比例，防止单灯被拆碎
    "blur_kernel": 3,             # 0/3/5/7/9，0 表示不模糊
    "peak_rel_threshold": 0.50,   # 峰值至少达到最大距离变换值的比例
    "min_solidity": 0.35,         # 面积 / 凸包面积，过滤细长/锯齿噪声
    "min_fill": 0.20,             # 面积 / 外接矩形面积，过滤细长反光
    "max_aspect": 5.0,            # 最小外接矩形长宽比上限，过滤灯带/地面反光
    "max_candidates": 24,         # 单相机最多输出的筛选后候选点
    "saturation_value": 250,      # 视为饱和的像素值
    "epipolar_tol_px": 5.0,       # 双相机极线距离阈值（去畸变理想像素）
    "reprojection_tol_px": 4.0,   # 三角化点的重投影误差阈值
    "triangle_tol_mm": 10.0,      # 33 mm 等边三角形的边长容差；实机质心噪声下先给 10mm
}

# 每个数值参数的允许范围；用于 API 校验，避免误输入把检测拖死。
PARAM_LIMITS = {
    "threshold": (0.0, 255.0),
    "min_area": (1.0, 100000.0),
    "max_area": (2.0, 1000000.0),
    "min_peak_distance": (1.0, 200.0),
    "split_min_area_ratio": (0.02, 0.90),
    "blur_kernel": (0.0, 9.0),
    "peak_rel_threshold": (0.05, 0.99),
    "min_solidity": (0.02, 1.0),
    "min_fill": (0.02, 1.0),
    "max_aspect": (1.0, 50.0),
    "max_candidates": (3.0, 64.0),
    "saturation_value": (128.0, 255.0),
    "epipolar_tol_px": (0.25, 30.0),
    "reprojection_tol_px": (0.25, 30.0),
    "triangle_tol_mm": (1.0, 30.0),
}

TRIANGLE_SIDE_M = 0.033          # 三个 LED 中心间距，米
MAX_RAW_COMPONENTS = 64           # 返回给网页的原始连通域上限（防止极端噪声刷屏）


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
    # 只有两个值都出现才比较，避免 partial=True 时用默认值误判：
    # 例如只覆盖 min_area=50 不应该因为 max_area 缺省为 1 而报错。
    if ("max_area" in out and "min_area" in out
            and out["max_area"] <= out["min_area"]):
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


def _normalize_roi(roi, height: int, width: int):
    """把 ROI 规范成 (x, y, w, h)；返回 None 表示整幅图。"""
    if not roi:
        return None
    if isinstance(roi, dict):
        vals = [roi.get(k) for k in ("x", "y", "w", "h")]
    else:
        try:
            vals = list(roi)
        except TypeError:
            return None
    if len(vals) != 4 or any(v is None for v in vals):
        return None
    try:
        x, y, w, h = [int(round(float(v))) for v in vals]
    except (TypeError, ValueError):
        return None
    x = max(0, min(width - 1, x))
    y = max(0, min(height - 1, y))
    w = max(0, min(width - x, w))
    h = max(0, min(height - y, h))
    if w < 4 or h < 4:
        return None
    return x, y, w, h


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


def _shape_metrics(roi_mask: np.ndarray, width: int, height: int,
                   area: int) -> dict:
    """solidity / 长宽比 / 外接矩形填充率。"""
    contour = _contour_of(roi_mask)
    solidity = 1.0
    aspect = 1.0
    if contour is not None:
        hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
        if hull_area > 0:
            # 小连通域的 contourArea 会低估面积，solidity 可能 >1；夹到 1 便于阅读
            solidity = min(1.0, float(area) / hull_area)
        try:
            (rw, rh) = cv2.minAreaRect(contour)[1]
            rw, rh = float(rw), float(rh)
            if min(rw, rh) > 1e-6:
                aspect = max(rw, rh) / min(rw, rh)
            elif max(rw, rh) > 1e-6:
                aspect = 50.0
        except Exception:
            aspect = 1.0
    fill = float(area) / max(1.0, float(width * height))
    return {"solidity": round(float(solidity), 4),
            "aspect": round(float(aspect), 4),
            "fill": round(float(fill), 4),
            "contour": contour}


def _merge_close_peaks(peaks: np.ndarray, min_distance: float) -> np.ndarray:
    """按距离合并过近的峰，避免同一个 LED 被拆成两个。"""
    if len(peaks) <= 1:
        return peaks
    out = []
    for p in peaks:
        if all(math.hypot(float(p[0] - q[0]), float(p[1] - q[1])) >= min_distance
               for q in out):
            out.append(p)
    return np.asarray(out, np.float64)


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
    n, labels, _, _ = cv2.connectedComponentsWithStats(peak_mask, 8)
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
    return _merge_close_peaks(np.asarray(peaks, np.float64),
                              float(params["min_peak_distance"]))


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
    return _merge_close_peaks(np.asarray(peaks, np.float64),
                              float(params["min_peak_distance"]))


def _split_component(gray_roi: np.ndarray, mask: np.ndarray,
                     params: dict) -> list[np.ndarray]:
    """按峰值把一个连通亮区拆成 1~N 个部分（ROI 坐标）。

    每个 part 至少占父亮区 ``split_min_area_ratio``，避免单个 LED 的亮核/
    光晕被误拆成两个候选。
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
    parent_area = int(mask.sum())
    min_part = max(1, int(params["min_area"]),
                   int(round(float(params["split_min_area_ratio"])
                             * parent_area)))
    parts = []
    for k in range(len(peaks)):
        sel = assign == k
        if int(sel.sum()) < min_part:
            continue
        part = np.zeros_like(mask, dtype=bool)
        part[ys[sel], xs[sel]] = True
        parts.append(part)
    return parts if len(parts) >= 2 else [mask]


def _part_record(gray_roi: np.ndarray, part: np.ndarray, params: dict):
    """把拆分后的 part 转成候选记录；不满足形状/面积时返回 None。"""
    pa = int(part.sum())
    if pa < int(params["min_area"]):
        return None
    cx, cy = _weighted_center(gray_roi, part)
    pc = _contour_of(part)
    ys, xs = np.nonzero(part)
    if len(xs) == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    w, h = x1 - x0 + 1, y1 - y0 + 1
    vals = gray_roi[part]
    metrics = _shape_metrics(part, w, h, pa)
    if metrics["solidity"] < float(params["min_solidity"]):
        return None
    if metrics["aspect"] > float(params["max_aspect"]):
        return None
    if metrics["fill"] < float(params["min_fill"]):
        return None
    return {
        "cx": cx, "cy": cy, "area": pa,
        "bbox_local": [x0, y0, w, h],
        "mean_intensity": float(vals.mean()) if vals.size else 0.0,
        "max_intensity": int(vals.max()) if vals.size else 0,
        "saturated_ratio": (float((vals >= int(params["saturation_value"])).mean())
                            if vals.size else 0.0),
        "contour": pc,
        "solidity": metrics["solidity"],
        "aspect": metrics["aspect"],
        "fill": metrics["fill"],
    }


def _rank_score(rec: dict, params: dict) -> float:
    mean = float(rec.get("mean_intensity") or 0.0) / 255.0
    solidity = float(rec.get("solidity") or 0.0)
    fill = float(rec.get("fill") or 0.0)
    aspect = float(rec.get("aspect") or 1.0)
    # 亮度/紧凑度/填充率为主，细长形状扣分；不按面积硬排，避免把远处小 LED
    # 排在近处大反光后面。
    return (2.0 * min(1.0, mean) + 1.0 * solidity + 0.5 * fill
            - 0.10 * max(0.0, aspect - 1.0))


def detect_leds(gray: np.ndarray, params: dict | None = None, *,
                roi=None) -> dict:
    """检测单张原始灰度图中的红外 LED 亮点。

    ``roi`` 可以是 (x, y, w, h) 像素或 {"x","y","w","h"}；只在该区域内检测，
    但返回的坐标、bbox、contour 都已加回 ROI 偏移，仍是整图坐标。
    ``leds`` 是通过面积/形状/分离筛选的 LED 候选；``raw_components`` 是全部
    原始连通域（含被拒绝的），供网页区分绘制。
    """
    p = validate_params(params or default_params())
    g = _as_gray(gray)
    height, width = g.shape[:2]
    roi_box = _normalize_roi(roi, height, width)
    if roi_box is None:
        x_off, y_off = 0, 0
        work = g
    else:
        x_off, y_off, rw, rh = roi_box
        work = g[y_off:y_off + rh, x_off:x_off + rw]

    k = int(p["blur_kernel"])
    processed = cv2.GaussianBlur(work, (k, k), 0) if k > 0 else work
    _, mask_work = cv2.threshold(processed, int(p["threshold"]), 255,
                                 cv2.THRESH_BINARY)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_work, 8, cv2.CV_32S)
    leds, rejected, warnings = [], [], []
    raw_components = []
    unresolved = 0
    overexposed = 0

    def warn_once(msg: str) -> None:
        if msg not in warnings:
            warnings.append(msg)

    for label in range(1, n):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < int(p["min_area"]):
            continue
        x, y, w, h = (int(stats[label, cv2.CC_STAT_LEFT]),
                      int(stats[label, cv2.CC_STAT_TOP]),
                      int(stats[label, cv2.CC_STAT_WIDTH]),
                      int(stats[label, cv2.CC_STAT_HEIGHT]))
        roi_mask = labels[y:y + h, x:x + w] == label
        roi_gray = work[y:y + h, x:x + w]
        vals = roi_gray[roi_mask]
        mean_v = float(vals.mean()) if vals.size else 0.0
        max_v = int(vals.max()) if vals.size else 0
        sat_ratio = (float((vals >= int(p["saturation_value"])).mean())
                     if vals.size else 0.0)
        cx, cy = _weighted_center(roi_gray, roi_mask)
        metrics = _shape_metrics(roi_mask, w, h, area)
        comp_id = len(raw_components)
        # 连通域中心是 ROI 内的局部坐标：先加 bbox 偏移，再加 ROI 偏移。
        gx, gy = cx + x + x_off, cy + y + y_off
        common = {
            "id": comp_id,
            "x": round(float(gx), 3), "y": round(float(gy), 3),
            "bbox": [x + x_off, y + y_off, w, h],
            "area": area,
            "mean_intensity": round(mean_v, 1),
            "max_intensity": max_v,
            "saturated_ratio": round(sat_ratio, 3),
            "solidity": metrics["solidity"],
            "aspect": metrics["aspect"],
            "fill": metrics["fill"],
            "accepted": False,
            "reason": "",
            "split_into": 0,
        }
        raw_components.append(common)

        def reject(reason: str) -> None:
            common["reason"] = reason
            rejected.append(dict(common, parent_bbox=common["bbox"],
                                 parent_area=area))

        # ---- 形状过滤：细长灯带 / 地面反光 / 锯齿噪声 ----
        if metrics["solidity"] < float(p["min_solidity"]):
            reject("形状不规则（solidity %.2f）" % metrics["solidity"])
            continue
        if metrics["aspect"] > float(p["max_aspect"]):
            reject("细长亮区（长宽比 %.1f，疑似灯带/反光）"
                   % metrics["aspect"])
            continue
        if metrics["fill"] < float(p["min_fill"]):
            reject("外接矩形填充率过低（%.2f，疑似细长反光）"
                   % metrics["fill"])
            continue

        # ---- 粘连分离 ----
        parts = (_split_component(roi_gray, roi_mask, p)
                 if p["split_blobs"] else [roi_mask])
        split_leds = []
        if len(parts) >= 2:
            for part in parts:
                rec = _part_record(roi_gray, part, p)
                if rec is None:
                    continue
                px, py, pw, ph = rec["bbox_local"]
                vals_p = roi_gray[part]
                split_leds.append({
                    "component_id": comp_id,
                    "part_index": len(split_leds),
                    "x": round(float(rec["cx"] + x + x_off), 3),
                    "y": round(float(rec["cy"] + y + y_off), 3),
                    "area": rec["area"],
                    "mean_intensity": round(rec["mean_intensity"], 1),
                    "max_intensity": rec["max_intensity"],
                    "bbox": [px + x + x_off, py + y + y_off, pw, ph],
                    "contour": (None if rec["contour"] is None
                                else (rec["contour"].reshape(-1, 2)
                                      + [x + x_off, y + y_off]).tolist()),
                    "split": True,
                    "parent_area": area,
                    "parent_bbox": [x + x_off, y + y_off, w, h],
                    "saturated_ratio": round(rec["saturated_ratio"], 3),
                    "solidity": rec["solidity"],
                    "aspect": rec["aspect"],
                    "fill": rec["fill"],
                })
            if len(split_leds) >= 2:
                for rec in split_leds:
                    rec["rank_score"] = round(_rank_score(rec, p), 4)
                    leds.append(rec)
                common["accepted"] = True
                common["split_into"] = len(split_leds)
                common["reason"] = "拆分为 %d 个光斑" % len(split_leds)
                continue
            # 拆出来的有效部分不足 2 个：不要既加 part 又加整个连通域

        # ---- 没有拆出多个光斑 ----
        if area > int(p["max_area"]):
            unresolved += 1
            reason = "面积过大且无法分离"
            if sat_ratio >= 0.8:
                overexposed += 1
                reason += "（过曝/峰值不可分）"
            reject(reason)
            continue

        if (sat_ratio >= 0.9
                and area >= max(4 * int(p["min_area"]),
                                int(0.5 * int(p["max_area"])))):
            overexposed += 1
            warn_once("亮区高度饱和且较大（面积 %d px2），可能包含多个粘连 "
                      "LED；请降低 LED 亮度/曝光或调整阈值" % area)
        led = {
            "component_id": comp_id,
            "part_index": 0,
            "x": round(float(gx), 3),
            "y": round(float(gy), 3),
            "area": area,
            "mean_intensity": round(mean_v, 1),
            "max_intensity": max_v,
            "bbox": [x + x_off, y + y_off, w, h],
            "contour": (None if metrics["contour"] is None
                        else (metrics["contour"].reshape(-1, 2)
                              + [x + x_off, y + y_off]).tolist()),
            "split": False,
            "parent_area": area,
            "parent_bbox": [x + x_off, y + y_off, w, h],
            "saturated_ratio": round(sat_ratio, 3),
            "solidity": metrics["solidity"],
            "aspect": metrics["aspect"],
            "fill": metrics["fill"],
        }
        led["rank_score"] = round(_rank_score(led, p), 4)
        leds.append(led)
        common["accepted"] = True
        common["reason"] = "通过筛选"

    # 通过筛选的候选按“亮度/紧凑度/填充率”排序；截断只影响匹配候选数量，
    # 不会像以前那样按 x/y 排序把真正的 LED 挤掉。
    leds.sort(key=lambda d: (-float(d.get("rank_score") or 0.0),
                             float(d["x"]), float(d["y"])))
    if len(leds) > int(p["max_candidates"]):
        extra = len(leds) - int(p["max_candidates"])
        leds = leds[:int(p["max_candidates"])]
        warn_once("候选点超过 %d 个，已按亮度/形状截断 %d 个"
                  % (p["max_candidates"], extra))

    if unresolved:
        warn_once("有 %d 个亮区无法分离成独立光斑%s"
                  % (unresolved, "，可能存在过曝" if overexposed else ""))
    if overexposed:
        warn_once("检测到过曝/饱和亮区；请降低 LED 亮度或调整阈值，"
                  "不要把这些区域当作有效三点")

    # 有可用候选时仍返回 ok；无法分离的大亮区会出现在 warnings/unresolved_count，
    # 但不把整个检测状态标成失败（实景里天花板灯经常就是这种大饱和区）。
    status = "ok" if leds else ("unresolved" if unresolved else "no_leds")
    reason = ""
    if status == "no_leds":
        reason = "没有找到超过阈值的有效亮区"
    elif status == "unresolved":
        reason = "亮区无法分离为独立光斑（可能过曝或粘连）"
    for wmsg in warnings:
        if "无法分离" in wmsg or "过曝" in wmsg:
            reason = reason or wmsg
            break

    # 原始连通域只返回前 MAX_RAW_COMPONENTS 个（按面积降序），避免极端噪声
    raw_components.sort(key=lambda d: (-int(d["area"]), int(d["id"])))
    raw_total = len(raw_components)
    raw_components = raw_components[:MAX_RAW_COMPONENTS]
    raw_components.sort(key=lambda d: int(d["id"]))

    if roi_box is None:
        mask = mask_work
        roi_out = None
    else:
        mask = np.zeros((height, width), np.uint8)
        mask[roi_box[1]:roi_box[1] + roi_box[3],
             roi_box[0]:roi_box[0] + roi_box[2]] = mask_work
        roi_out = [int(v) for v in roi_box]

    return {
        "status": status,
        "reason": reason,
        "count": len(leds),
        "leds": leds,
        "raw_count": raw_total,
        "raw_components": raw_components,
        "rejected": rejected,
        "warnings": warnings,
        "unresolved_count": unresolved,
        "overexposed": bool(overexposed),
        "mask": mask,
        "params": p,
        "image_size": [int(width), int(height)],
        "roi": roi_out,
        "roi_fallback": False,
    }


__all__ = ["DEFAULT_PARAMS", "PARAM_LIMITS", "TRIANGLE_SIDE_M",
           "default_params", "validate_params", "merge_params",
           "detect_leds", "_as_gray", "_normalize_roi"]
