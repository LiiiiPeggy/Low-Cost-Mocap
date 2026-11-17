"""双相机极线匹配、世界系三角化与 33 mm 等边三角形识别。

坐标约定与 calibration 模块完全一致：
    X_cam = R @ X_world + t          （R、t 来自标定文件/结果）
    P     = K @ [R | t]              （世界 → 原始像素，含 K）
匹配/三角化使用 cv2.undistortPoints(..., P=K) 得到的**理想像素**坐标，
因此投影矩阵也使用同一个 K；原始检测点只用于重投影误差检查，且只去畸变一次。
"""

from __future__ import annotations

import itertools
import math

import cv2
import numpy as np

from led.detector import TRIANGLE_SIDE_M


def skew(v: np.ndarray) -> np.ndarray:
    x, y, z = [float(t) for t in np.asarray(v).reshape(3)]
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], np.float64)


def relative_pose(R0, t0, R1, t1):
    R10 = R1 @ R0.T
    T10 = t1 - R10 @ t0
    return R10, T10


def fundamental_from_pose(K0, K1, R10, T10) -> np.ndarray:
    E = skew(T10) @ R10
    return np.linalg.inv(K1).T @ E @ np.linalg.inv(K0)


def ideal_pixels(points: np.ndarray, K, D) -> np.ndarray:
    """原始像素 → 去畸变理想像素。**只调用一次**，之后不要再 undistort。"""
    pts = np.asarray(points, np.float64).reshape(-1, 1, 2)
    # 第 5 个参数 P=K：输出“理想像素”坐标，而不是归一化坐标。
    out = cv2.undistortPoints(pts, np.asarray(K, np.float64),
                              np.asarray(D, np.float64).reshape(-1),
                              None, np.asarray(K, np.float64))
    return out.reshape(-1, 2)


def _project_world(Xw, K, D, R, t):
    rvec = cv2.Rodrigues(np.asarray(R, np.float64))[0]
    proj, _ = cv2.projectPoints(np.asarray(Xw, np.float64).reshape(1, 1, 3),
                                rvec, np.asarray(t, np.float64).reshape(3, 1),
                                np.asarray(K, np.float64),
                                np.asarray(D, np.float64).reshape(-1))
    return proj.reshape(2)


def _epipolar_distance(p0, p1, F):
    h0 = np.array([p0[0], p0[1], 1.0], np.float64)
    h1 = np.array([p1[0], p1[1], 1.0], np.float64)
    l1 = F @ h0
    l0 = F.T @ h1
    n1 = math.hypot(l1[0], l1[1])
    n0 = math.hypot(l0[0], l0[1])
    d1 = abs(float(l1 @ h1)) / n1 if n1 > 1e-12 else 1e9
    d0 = abs(float(l0 @ h0)) / n0 if n0 > 1e-12 else 1e9
    return max(d0, d1), d0, d1


def _labels_from_previous(points: np.ndarray, previous_points):
    """等边三角形三个顶点完全等价；仅用时间连续性保持显示编号稳定。

    返回 (重排后的点, 方式, 原索引顺序)。第三个返回值必须同时用于匹配记录，
    否则 LED 的 label 与 cam0/cam1 检测点会错位。
    """
    n = len(points)
    if previous_points is not None:
        prev = np.asarray(previous_points, np.float64).reshape(-1, 3)
        if len(prev) == n and n == 3:
            best, best_cost = None, float("inf")
            for perm in itertools.permutations(range(n)):
                cost = float(sum(np.linalg.norm(points[perm[k]] - prev[k])
                                 for k in range(n)))
                if cost < best_cost:
                    best_cost, best = cost, perm
            if best is not None:
                order = list(best)
                return points[order], "previous", order
    # 没有历史时按 x、y 排序，给出一个确定的临时编号
    order = sorted(range(n), key=lambda i: (points[i][0], points[i][1]))
    return points[order], "spatial", order


def match_and_triangulate(det0: dict, det1: dict, calib: dict,
                          params: dict, *, previous_points=None,
                          predicted_center=None,
                          max_match_candidates: int = 48) -> dict:
    """极线候选 → 三角化 → 重投影筛选 → 33 mm 等边三角形选择。

    ``calib`` 需包含 K0/D0/K1/D1/R0/t0/R1/t1（world→camera）和 image_size。
    """
    leds0 = list((det0 or {}).get("leds") or [])
    leds1 = list((det1 or {}).get("leds") or [])
    if len(leds0) < 3 or len(leds1) < 3:
        return {
            "ok": False, "status": "insufficient_detections",
            "reason": "两台相机都需要至少 3 个候选点（当前 %d / %d）"
                      % (len(leds0), len(leds1)),
            "matches": [], "leds": [], "raw_center": None,
            "detected_counts": [len(leds0), len(leds1)],
        }

    K0 = np.asarray(calib["K0"], np.float64)
    D0 = np.asarray(calib["D0"], np.float64).reshape(-1)
    K1 = np.asarray(calib["K1"], np.float64)
    D1 = np.asarray(calib["D1"], np.float64).reshape(-1)
    R0 = np.asarray(calib["R0"], np.float64)
    t0 = np.asarray(calib["t0"], np.float64).reshape(3)
    R1 = np.asarray(calib["R1"], np.float64)
    t1 = np.asarray(calib["t1"], np.float64).reshape(3)
    R10, T10 = relative_pose(R0, t0, R1, t1)
    F = fundamental_from_pose(K0, K1, R10, T10)
    P0 = K0 @ np.c_[R0, t0]
    P1 = K1 @ np.c_[R1, t1]

    pts0 = np.array([[d["x"], d["y"]] for d in leds0], np.float64)
    pts1 = np.array([[d["x"], d["y"]] for d in leds1], np.float64)
    ip0 = ideal_pixels(pts0, K0, D0)
    ip1 = ideal_pixels(pts1, K1, D1)

    epi_tol = float(params["epipolar_tol_px"])
    rep_tol = float(params["reprojection_tol_px"])
    tol_m = float(params["triangle_tol_mm"]) / 1000.0

    candidates = []
    for i in range(len(leds0)):
        for j in range(len(leds1)):
            epi, d0, d1 = _epipolar_distance(ip0[i], ip1[j], F)
            if epi > epi_tol:
                continue
            Xh = cv2.triangulatePoints(P0, P1,
                                       ip0[i].reshape(2, 1),
                                       ip1[j].reshape(2, 1))
            w = float(Xh[3, 0])
            if abs(w) < 1e-12:
                continue
            Xw = (Xh[:3, 0] / w).astype(np.float64)
            if not np.all(np.isfinite(Xw)):
                continue
            z0 = float((R0 @ Xw + t0)[2])
            z1 = float((R1 @ Xw + t1)[2])
            if z0 <= 0.1 or z1 <= 0.1:
                continue
            e0 = float(np.linalg.norm(_project_world(Xw, K0, D0, R0, t0) - pts0[i]))
            e1 = float(np.linalg.norm(_project_world(Xw, K1, D1, R1, t1) - pts1[j]))
            if max(e0, e1) > rep_tol:
                continue
            candidates.append({
                "i0": i, "i1": j,
                "xyz": Xw.tolist(),
                "epipolar_px": round(epi, 4),
                "epipolar_0_px": round(d0, 4),
                "epipolar_1_px": round(d1, 4),
                "reproj_0_px": round(e0, 4),
                "reproj_1_px": round(e1, 4),
                "score": epi + e0 + e1,
            })

    if len(candidates) < 3:
        return {
            "ok": False, "status": "insufficient_matches",
            "reason": "极线/重投影过滤后只有 %d 个候选匹配（至少需要 3 个）"
                      % len(candidates),
            "matches": [], "leds": [], "raw_center": None,
            "detected_counts": [len(leds0), len(leds1)],
            "candidate_count": len(candidates),
        }

    candidates.sort(key=lambda c: c["score"])
    candidates = candidates[:max_match_candidates]
    points = np.array([c["xyz"] for c in candidates], np.float64)

    best, best_bad = None, None
    for a, b, c in itertools.combinations(range(len(candidates)), 3):
        ca, cb, cc = candidates[a], candidates[b], candidates[c]
        if len({ca["i0"], cb["i0"], cc["i0"]}) != 3:
            continue
        if len({ca["i1"], cb["i1"], cc["i1"]}) != 3:
            continue
        tri = points[[a, b, c]]
        d01 = float(np.linalg.norm(tri[0] - tri[1]))
        d12 = float(np.linalg.norm(tri[1] - tri[2]))
        d20 = float(np.linalg.norm(tri[2] - tri[0]))
        edges = np.array([d01, d12, d20], np.float64)
        errs = np.abs(edges - TRIANGLE_SIDE_M)
        max_err = float(errs.max())
        center = tri.mean(axis=0)
        cand = {
            "indices": (a, b, c),
            "points": tri,
            "edges": edges,
            "errs": errs,
            "center": center,
            "mean_reproj": float(np.mean([
                ca["reproj_0_px"], ca["reproj_1_px"],
                cb["reproj_0_px"], cb["reproj_1_px"],
                cc["reproj_0_px"], cc["reproj_1_px"]])),
            "mean_epi": float(np.mean([ca["epipolar_px"],
                                       cb["epipolar_px"], cc["epipolar_px"]])),
        }
        if max_err > tol_m:
            if best_bad is None or max_err < best_bad[0]:
                best_bad = (max_err, cand)
            continue
        # 边长误差为主，重投影/极线为辅；有历史时轻微偏向预测中心
        score = (2.0 * float(errs.mean()) / max(tol_m, 1e-9)
                 + cand["mean_reproj"] / max(rep_tol, 1e-9)
                 + 0.5 * cand["mean_epi"] / max(epi_tol, 1e-9))
        if predicted_center is not None:
            pred = np.asarray(predicted_center, np.float64).reshape(3)
            jump = float(np.linalg.norm(center - pred))
            score += 0.15 * jump
        if best is None or score < best[0]:
            best = (score, cand)

    if best is None:
        bad = best_bad[1] if best_bad else None
        return {
            "ok": False, "status": "geometry_mismatch",
            "reason": "没有三点组合满足 %.1f mm 等边三角形（容差 %.1f mm）"
                      % (TRIANGLE_SIDE_M * 1000, params["triangle_tol_mm"]),
            "matches": [], "leds": [], "raw_center": None,
            "detected_counts": [len(leds0), len(leds1)],
            "best_edges_mm": None if bad is None
                             else [round(float(v * 1000), 3) for v in bad["edges"]],
            "best_edge_errors_mm": None if bad is None
                                   else [round(float(v * 1000), 3)
                                         for v in bad["errs"]],
        }

    chosen = best[1]
    tri = chosen["points"]
    base_cands = [candidates[i] for i in chosen["indices"]]
    # 用时间连续性保持显示编号稳定；这只是临时编号，不是物理顶点身份。
    labeled, assignment, order = _labels_from_previous(tri, previous_points)
    ordered_cands = [base_cands[i] for i in order]
    edges = [float(np.linalg.norm(labeled[0] - labeled[1])),
             float(np.linalg.norm(labeled[1] - labeled[2])),
             float(np.linalg.norm(labeled[2] - labeled[0]))]
    errs = [abs(v - TRIANGLE_SIDE_M) for v in edges]
    center = labeled.mean(axis=0)
    leds_out = []
    for k, (pt, cand) in enumerate(zip(labeled, ordered_cands)):
        leds_out.append({
            "label": k,
            "xyz": [round(float(v), 6) for v in pt],
            "reproj_0_px": cand["reproj_0_px"],
            "reproj_1_px": cand["reproj_1_px"],
            "epipolar_px": cand["epipolar_px"],
            "cam0_index": cand["i0"],
            "cam1_index": cand["i1"],
        })
    matches = [{"led": k,
                "cam0_index": c["i0"], "cam1_index": c["i1"],
                "epipolar_px": c["epipolar_px"],
                "reproj_0_px": c["reproj_0_px"],
                "reproj_1_px": c["reproj_1_px"]}
               for k, c in enumerate(ordered_cands)]
    return {
        "ok": True,
        "status": "ok",
        "reason": "",
        "matches": matches,
        "leds": leds_out,
        "raw_center": [round(float(v), 6) for v in center],
        "edges_mm": [round(v * 1000, 3) for v in edges],
        "edge_errors_mm": [round(v * 1000, 3) for v in errs],
        "mean_reprojection_px": round(chosen["mean_reproj"], 4),
        "mean_epipolar_px": round(chosen["mean_epi"], 4),
        "triangle_side_m": TRIANGLE_SIDE_M,
        "triangle_tol_mm": params["triangle_tol_mm"],
        "labels_are_temporary": True,
        "label_assignment": assignment,
        "detected_counts": [len(leds0), len(leds1)],
        "candidate_count": len(candidates),
    }


__all__ = ["skew", "relative_pose", "fundamental_from_pose", "ideal_pixels",
           "match_and_triangulate"]
