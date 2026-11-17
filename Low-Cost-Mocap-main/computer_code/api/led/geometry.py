"""双相机极线匹配、世界系三角化与 34/79/79 mm 等腰三角形识别。

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

from led.detector import TRIANGLE_LONG_M, TRIANGLE_SHORT_M, TRIANGLE_SIDES_M

_EXPECTED_EDGES_M = np.sort(np.asarray(TRIANGLE_SIDES_M, np.float64))


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


def _signed_area(points) -> float:
    """二维三点有向面积；用于检查两台相机的三角形绕向是否一致。"""
    p = np.asarray(points, np.float64)
    if p.shape != (3, 2):
        return 0.0
    return 0.5 * float(
        (p[1, 0] - p[0, 0]) * (p[2, 1] - p[0, 1])
        - (p[2, 0] - p[0, 0]) * (p[1, 1] - p[0, 1]))


def _triangle_order(points: np.ndarray, previous_points=None):
    """按 34/79/79 mm 几何把三点排成 [L0, L1, L2]。

    L0 是两条 79 mm 长边的公共顶点；L1/L2 是 34 mm 短边端点，物理身份不唯一，
    因此只在与上一帧不冲突时用时间连续性保持顺序。

    返回 (重排后的点, 方式, 原索引顺序)。第三个返回值必须同时用于匹配记录，
    否则 LED 的 label 与 cam0/cam1 检测点会错位。
    """
    p = np.asarray(points, np.float64).reshape(3, 3)
    d = [float(np.linalg.norm(p[0] - p[1])),
         float(np.linalg.norm(p[1] - p[2])),
         float(np.linalg.norm(p[2] - p[0]))]
    # (短边长度, 短边端点 i, 短边端点 j, 顶点 k)
    _short, i, j, k = min(
        ((d[0], 0, 1, 2), (d[1], 1, 2, 0), (d[2], 2, 0, 1)),
        key=lambda item: item[0])
    if previous_points is not None:
        prev = np.asarray(previous_points, np.float64).reshape(-1, 3)
        if len(prev) == 3:
            keep = (np.linalg.norm(p[i] - prev[1])
                    + np.linalg.norm(p[j] - prev[2]))
            swap = (np.linalg.norm(p[i] - prev[2])
                    + np.linalg.norm(p[j] - prev[1]))
            order = [k, i, j] if keep <= swap else [k, j, i]
            return p[order], "previous", order
    # 无历史时给底边两个端点一个确定的临时顺序
    base = sorted((i, j), key=lambda idx: (p[idx][0], p[idx][1]))
    order = [k] + base
    return p[order], "spatial", order


def match_and_triangulate(det0: dict, det1: dict, calib: dict,
                          params: dict, *, previous_points=None,
                          predicted_center=None,
                          max_match_candidates: int = 64) -> dict:
    """极线候选 → 三角化 → 重投影筛选 → 34/79/79 mm 等腰三角形选择。

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
            "diagnostics": {
                "camera_map": calib.get("camera_map"),
                "detected_counts": [len(leds0), len(leds1)],
                "stage": "detection",
                "reason": "每台相机至少需要 3 个候选点",
            },
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

    # 全量诊断：即使最终 0 匹配，也保留每个阶段的输入/结果，供网页和
    # debug JSON 离线复现。大候选集时 pair_records 只保留前 64 条，矩阵完整。
    diag = {
        "camera_map": calib.get("camera_map"),
        "calibration_ids": calib.get("calibration_ids"),
        "detected_counts": [len(leds0), len(leds1)],
        "raw_points_0": [[round(float(v), 3) for v in p] for p in pts0],
        "raw_points_1": [[round(float(v), 3) for v in p] for p in pts1],
        "ideal_points_0": [[round(float(v), 3) for v in p] for p in ip0],
        "ideal_points_1": [[round(float(v), 3) for v in p] for p in ip1],
        "R10": [[round(float(v), 8) for v in row] for row in R10],
        "T10": [round(float(v), 8) for v in T10],
        "F": [[round(float(v), 8) for v in row] for row in F],
        "epipolar_tol_px": epi_tol,
        "reprojection_tol_px": rep_tol,
        "triangle_tol_mm": params["triangle_tol_mm"],
        "expected_edges_mm": [round(float(v * 1000), 3)
                              for v in _EXPECTED_EDGES_M],
        "triangle_sides_m": [float(v) for v in TRIANGLE_SIDES_M],
        "epipolar_distances": [],
        "epipolar_pass": 0,
        "depth_pass": 0,
        "reprojection_pass": 0,
        "candidate_count": 0,
        "triangle_combinations": 0,
        "valid_triangles": 0,
        "best_edges_mm": None,
        "best_edge_errors_mm": None,
        "best_reprojection_px": None,
        "best_epipolar_px": None,
        "best_triangle_indices": None,
        "pairs": [],
    }
    epi_matrix = []
    pair_infos = []
    for i in range(len(leds0)):
        row = []
        for j in range(len(leds1)):
            epi, d0, d1 = _epipolar_distance(ip0[i], ip1[j], F)
            row.append(round(float(epi), 3))
            info = {"i0": i, "i1": j, "epipolar_px": epi,
                    "epipolar_0_px": d0, "epipolar_1_px": d1}
            pair_infos.append(info)
            if epi <= epi_tol:
                diag["epipolar_pass"] += 1
        epi_matrix.append(row)
    diag["epipolar_distances"] = epi_matrix

    candidates = []
    for info in pair_infos:
        i, j, epi = info["i0"], info["i1"], info["epipolar_px"]
        rec = {
            "i0": i, "i1": j,
            "epipolar_px": round(float(epi), 3),
            "epipolar_0_px": round(float(info["epipolar_0_px"]), 3),
            "epipolar_1_px": round(float(info["epipolar_1_px"]), 3),
            "passed_epipolar": bool(epi <= epi_tol),
            "xyz": None, "z0": None, "z1": None,
            "reproj_0_px": None, "reproj_1_px": None,
            "passed_depth": False, "passed_reprojection": False,
        }
        if epi > epi_tol:
            if len(diag["pairs"]) < 64:
                diag["pairs"].append(rec)
            continue
        Xh = cv2.triangulatePoints(P0, P1,
                                   ip0[i].reshape(2, 1),
                                   ip1[j].reshape(2, 1))
        w = float(Xh[3, 0])
        if abs(w) < 1e-12:
            if len(diag["pairs"]) < 64:
                diag["pairs"].append(rec)
            continue
        Xw = (Xh[:3, 0] / w).astype(np.float64)
        if not np.all(np.isfinite(Xw)):
            if len(diag["pairs"]) < 64:
                diag["pairs"].append(rec)
            continue
        z0 = float((R0 @ Xw + t0)[2])
        z1 = float((R1 @ Xw + t1)[2])
        rec.update({"xyz": [round(float(v), 6) for v in Xw],
                    "z0": round(z0, 6), "z1": round(z1, 6)})
        if z0 <= 0.1 or z1 <= 0.1:
            if len(diag["pairs"]) < 64:
                diag["pairs"].append(rec)
            continue
        rec["passed_depth"] = True
        diag["depth_pass"] += 1
        e0 = float(np.linalg.norm(_project_world(Xw, K0, D0, R0, t0) - pts0[i]))
        e1 = float(np.linalg.norm(_project_world(Xw, K1, D1, R1, t1) - pts1[j]))
        rec["reproj_0_px"] = round(e0, 3)
        rec["reproj_1_px"] = round(e1, 3)
        if max(e0, e1) > rep_tol:
            if len(diag["pairs"]) < 64:
                diag["pairs"].append(rec)
            continue
        rec["passed_reprojection"] = True
        diag["reprojection_pass"] += 1
        candidates.append({
            "i0": i, "i1": j,
            "xyz": Xw.tolist(),
            "epipolar_px": round(epi, 4),
            "epipolar_0_px": round(float(info["epipolar_0_px"]), 4),
            "epipolar_1_px": round(float(info["epipolar_1_px"]), 4),
            "reproj_0_px": round(e0, 4),
            "reproj_1_px": round(e1, 4),
            "score": epi + e0 + e1,
        })
        if len(diag["pairs"]) < 64:
            diag["pairs"].append(rec)
    diag["candidate_count"] = len(candidates)

    if len(candidates) < 3:
        return {
            "ok": False, "status": "insufficient_matches",
            "reason": "极线/重投影过滤后只有 %d 个候选匹配（至少需要 3 个）"
                      % len(candidates),
            "matches": [], "leds": [], "raw_center": None,
            "detected_counts": [len(leds0), len(leds1)],
            "candidate_count": len(candidates),
            "diagnostics": diag,
        }

    candidates.sort(key=lambda c: c["score"])
    candidates = candidates[:max_match_candidates]
    points = np.array([c["xyz"] for c in candidates], np.float64)

    best, second, best_bad = None, None, None
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
        sorted_edges = np.sort(edges)
        errs = np.abs(sorted_edges - _EXPECTED_EDGES_M)
        max_err = float(errs.max())
        center = tri.mean(axis=0)
        # 透视投影下二维三角形不一定保持等腰，但两台相机看到的同一三点绕向应一致。
        # 这里只作为软约束（标定误差/相机翻转时不会硬性拒绝）。
        orient0 = _signed_area(pts0[[ca["i0"], cb["i0"], cc["i0"]]])
        orient1 = _signed_area(pts1[[ca["i1"], cb["i1"], cc["i1"]]])
        orient_ok = (abs(orient0) < 1e-9 or abs(orient1) < 1e-9
                     or (orient0 > 0) == (orient1 > 0))
        diag["triangle_combinations"] += 1
        cand = {
            "indices": (a, b, c),
            "points": tri,
            "edges": edges,
            "sorted_edges": sorted_edges,
            "errs": errs,
            "center": center,
            "mean_reproj": float(np.mean([
                ca["reproj_0_px"], ca["reproj_1_px"],
                cb["reproj_0_px"], cb["reproj_1_px"],
                cc["reproj_0_px"], cc["reproj_1_px"]])),
            "mean_epi": float(np.mean([ca["epipolar_px"],
                                       cb["epipolar_px"], cc["epipolar_px"]])),
            "orientation_consistent": bool(orient_ok),
        }
        if max_err > tol_m:
            if best_bad is None or max_err < best_bad[0]:
                best_bad = (max_err, cand)
            continue
        diag["valid_triangles"] += 1
        # 边长残差（排序后对 34/79/79）为主，重投影/极线为辅；
        # 有历史时轻微偏向预测中心。等腰三角形各边天然不等，不再惩罚边长离散度。
        score = (2.0 * float(errs.mean()) / max(tol_m, 1e-9)
                 + cand["mean_reproj"] / max(rep_tol, 1e-9)
                 + 0.5 * cand["mean_epi"] / max(epi_tol, 1e-9)
                 + (0.0 if orient_ok else 2.0))
        if predicted_center is not None:
            pred = np.asarray(predicted_center, np.float64).reshape(3)
            jump = float(np.linalg.norm(center - pred))
            score += 0.15 * jump
        cand["score"] = float(score)
        if best is None or score < best[0]:
            second = best
            best = (score, cand)
        elif second is None or score < second[0]:
            second = (score, cand)

    if best is None:
        bad = best_bad[1] if best_bad else None
        if bad is not None:
            diag.update({
                "best_edges_mm": [round(float(v * 1000), 3)
                                  for v in bad["sorted_edges"]],
                "best_edge_errors_mm": [round(float(v * 1000), 3)
                                        for v in bad["errs"]],
                "best_reprojection_px": round(bad["mean_reproj"], 3),
                "best_epipolar_px": round(bad["mean_epi"], 3),
                "best_triangle_indices": list(bad["indices"]),
            })
        return {
            "ok": False, "status": "geometry_mismatch",
            "reason": "没有三点组合满足 34/79/79 mm 等腰三角形（容差 %.1f mm）"
                      % params["triangle_tol_mm"],
            "matches": [], "leds": [], "raw_center": None,
            "detected_counts": [len(leds0), len(leds1)],
            "best_edges_mm": None if bad is None
                             else [round(float(v * 1000), 3)
                                   for v in bad["sorted_edges"]],
            "best_edge_errors_mm": None if bad is None
                                   else [round(float(v * 1000), 3)
                                         for v in bad["errs"]],
            "best_orientation_consistent": (None if bad is None
                                            else bad["orientation_consistent"]),
            "triangle_sides_m": [float(v) for v in _EXPECTED_EDGES_M],
            "diagnostics": diag,
        }

    second_score = None if second is None else second[0]
    chosen = best[1]
    tri = chosen["points"]
    base_cands = [candidates[i] for i in chosen["indices"]]
    # L0 由几何唯一确定；L1/L2 只是短边端点，用时间连续性避免显示跳号。
    labeled, assignment, order = _triangle_order(tri, previous_points)
    ordered_cands = [base_cands[i] for i in order]
    edges = np.array([
        float(np.linalg.norm(labeled[0] - labeled[1])),
        float(np.linalg.norm(labeled[1] - labeled[2])),
        float(np.linalg.norm(labeled[2] - labeled[0])),
    ], np.float64)
    sorted_edges = np.sort(edges)
    errs = np.abs(sorted_edges - _EXPECTED_EDGES_M)
    center = labeled.mean(axis=0)
    base_mid = (labeled[1] + labeled[2]) / 2.0
    direction = labeled[0] - base_mid
    norm = float(np.linalg.norm(direction))
    board_direction = (direction / norm).tolist() if norm > 1e-9 else None
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
    diag.update({
        "best_edges_mm": [round(float(v * 1000), 3) for v in sorted_edges],
        "best_edge_errors_mm": [round(float(v * 1000), 3) for v in errs],
        "best_reprojection_px": round(chosen["mean_reproj"], 3),
        "best_epipolar_px": round(chosen["mean_epi"], 3),
        "best_triangle_indices": list(chosen["indices"]),
    })
    return {
        "ok": True,
        "status": "ok",
        "reason": "",
        "matches": matches,
        "leds": leds_out,
        "raw_center": [round(float(v), 6) for v in center],
        "board_direction": ([round(float(v), 6) for v in board_direction]
                            if board_direction else None),
        "edges_mm": [round(float(v * 1000), 3) for v in sorted_edges],
        "edge_errors_mm": [round(float(v * 1000), 3) for v in errs],
        "expected_edges_mm": [round(float(v * 1000), 3)
                              for v in _EXPECTED_EDGES_M],
        "mean_reprojection_px": round(chosen["mean_reproj"], 4),
        "mean_epipolar_px": round(chosen["mean_epi"], 4),
        "triangle_score": round(float(chosen.get("score", 0.0)), 4),
        "second_score": (None if second_score is None
                         else round(float(second_score), 4)),
        "orientation_consistent": bool(chosen["orientation_consistent"]),
        "triangle_sides_m": [float(TRIANGLE_SHORT_M),
                             float(TRIANGLE_LONG_M),
                             float(TRIANGLE_LONG_M)],
        "triangle_tol_mm": params["triangle_tol_mm"],
        "labels_are_temporary": True,
        "label_rule": "L0=两条 79 mm 边的公共顶点；L1/L2=34 mm 短边端点",
        "label_assignment": assignment,
        "detected_counts": [len(leds0), len(leds1)],
        "candidate_count": len(candidates),
        "diagnostics": diag,
    }


__all__ = ["skew", "relative_pose", "fundamental_from_pose", "ideal_pixels",
           "match_and_triangulate"]
