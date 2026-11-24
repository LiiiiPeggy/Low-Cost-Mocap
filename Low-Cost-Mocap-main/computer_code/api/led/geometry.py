"""多相机（2..4）极线匹配、联合三角化与 34/79/79 mm 等腰三角形识别。

坐标约定与 calibration 模块完全一致：
    X_cam = R @ X_world + t          （R、t 来自标定文件/结果）
    P     = K @ [R | t]              （世界 → 原始像素，含 K）
匹配/三角化使用 ``ideal_pixels`` 得到的**理想像素**坐标（原始点只去畸变一次），
因此联合三角化的投影矩阵也使用同一个 K；原始检测点只用于重投影误差检查。

匹配流程（与原作者 locate_objects 的思路一致，但修正了缺失观测的下标问题）：
1. 任意相机对的极线距离筛选；
2. 每对候选做双视角 DLT 三角化，检查正深度与重投影；
3. 把其它相机中重投影一致的观测加入同一 3D 假设，再联合 DLT 精化；
4. 按 3D 距离合并重复假设（每个 LED 可以有不同的有效相机组合）；
5. 在 3D 点集合里搜索满足 34/79/79 mm 且同一相机观测不复用的三角形。
"""

from __future__ import annotations

import itertools
import math

import cv2
import numpy as np

from calibration import stereo as st
from led.detector import TRIANGLE_LONG_M, TRIANGLE_SHORT_M, TRIANGLE_SIDES_M

_EXPECTED_EDGES_M = np.sort(np.asarray(TRIANGLE_SIDES_M, np.float64))

MAX_PAIR_HYPOTHESES = 256
MAX_MERGED_POINTS = 24
MAX_TRIANGLE_POINTS = 12      # 三角形组合搜索上限（C(12,3)=220）
MERGE_DIST_M = 0.008        # 同一 LED 的两个 3D 假设合并阈值（板边 34 mm）
MIN_DEPTH_M = 0.1


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


def _project_world_many(Xw, K, D, R, t):
    rvec = cv2.Rodrigues(np.asarray(R, np.float64))[0]
    proj, _ = cv2.projectPoints(np.asarray(Xw, np.float64).reshape(-1, 1, 3),
                                rvec, np.asarray(t, np.float64).reshape(3, 1),
                                np.asarray(K, np.float64),
                                np.asarray(D, np.float64).reshape(-1))
    return proj.reshape(-1, 2)


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


def _epipolar_matrix(A, B, F):
    """成对极线距离矩阵 (Na,Nb)，与 ``_epipolar_distance`` 定义一致（取两向最大）。"""
    Ha = np.c_[np.asarray(A, np.float64), np.ones(len(A))]
    Hb = np.c_[np.asarray(B, np.float64), np.ones(len(B))]
    Lb = Ha @ F.T                      # 每个 A 点在 B 图中的极线
    nb = np.hypot(Lb[:, 0], Lb[:, 1])
    db = (np.abs(Hb @ Lb.T).T
          / np.maximum(nb, 1e-12)[:, None])          # (Na, Nb)
    La = Hb @ F                        # 每个 B 点在 A 图中的极线
    na = np.hypot(La[:, 0], La[:, 1])
    da = (np.abs(Ha @ La.T)
          / np.maximum(na, 1e-12)[None, :])          # (Na, Nb)
    return np.maximum(da, db), da, db


def _signed_area(points) -> float:
    """二维三点有向面积；用于检查不同相机的三角形绕向是否一致。"""
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
    否则 LED 的 label 与各相机检测点会错位。
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


# ---------------------------------------------------------------------------
# 标定输入归一化
# ---------------------------------------------------------------------------
def _camera_table(calib: dict) -> dict[int, dict]:
    """接受新的 ``per_camera`` 结构或旧的 K0/D0/K1/D1 结构。"""
    per = calib.get("per_camera")
    if per:
        cams = [int(c) for c in (calib.get("cameras") or sorted(per))]
        return {c: per[c] for c in cams if c in per}
    out = {}
    for c in (0, 1):
        try:
            out[c] = {
                "K": calib["K%d" % c], "D": calib["D%d" % c],
                "R": calib["R%d" % c], "t": calib["t%d" % c],
                "image_size": calib.get("image_size"),
            }
        except KeyError:
            continue
    return out


def _pose_arrays(cam: dict):
    K = np.asarray(cam["K"], np.float64)
    D = np.asarray(cam["D"], np.float64).reshape(-1)
    R = np.asarray(cam["R"], np.float64)
    t = np.asarray(cam["t"], np.float64).reshape(3)
    return K, D, R, t


def _camera_points(det: dict) -> np.ndarray:
    return np.array([[float(d["x"]), float(d["y"])]
                     for d in ((det or {}).get("leds") or [])], np.float64)


def _depth_ok(X, R, t) -> bool:
    return float((np.asarray(R, np.float64) @ X
                  + np.asarray(t, np.float64).reshape(3))[2]) > MIN_DEPTH_M


def _empty_result(status: str, reason: str, counts: dict, diag: dict) -> dict:
    return {
        "ok": False, "status": status, "reason": reason,
        "matches": [], "leds": [], "raw_center": None,
        "detected_counts": counts,
        "diagnostics": diag,
    }


def match_and_triangulate(dets: dict, calib: dict, params: dict, *,
                          previous_points=None, predicted_center=None,
                          max_match_candidates: int = MAX_PAIR_HYPOTHESES) -> dict:
    """多相机极线候选 → 联合三角化 → 34/79/79 mm 等腰三角形选择。

    ``dets``：``{逻辑相机: detect_leds 结果}``，至少 2 台有候选。
    ``calib``：``{"cameras": [...], "per_camera": {cam: {K,D,R,t,...}}}``，
    也接受旧的 ``K0/D0/K1/D1/R0/t0/R1/t1`` 双相机结构。
    """
    table = _camera_table(calib)
    cams = sorted(table)
    leds = {c: list((dets.get(c) or {}).get("leds") or []) for c in cams}
    counts = {c: len(leds[c]) for c in cams}
    active = [c for c in cams if counts[c] >= 1]

    epi_tol = float(params["epipolar_tol_px"])
    rep_tol = float(params["reprojection_tol_px"])
    tol_m = float(params["triangle_tol_mm"]) / 1000.0
    diag = {
        "cameras": cams,
        "detected_counts": counts,
        "epipolar_tol_px": epi_tol,
        "reprojection_tol_px": rep_tol,
        "triangle_tol_mm": params["triangle_tol_mm"],
        "expected_edges_mm": [round(float(v * 1000), 3)
                              for v in _EXPECTED_EDGES_M],
        "triangle_sides_m": [float(v) for v in TRIANGLE_SIDES_M],
        "pairwise": {},
        "epipolar_distances": [],
        "epipolar_pass": 0,
        "depth_pass": 0,
        "reprojection_pass": 0,
        "pair_hypotheses": 0,
        "candidate_count": 0,
        "triangle_combinations": 0,
        "valid_triangles": 0,
        "best_edges_mm": None,
        "best_edge_errors_mm": None,
        "best_reprojection_px": None,
        "best_epipolar_px": None,
        "best_triangle_indices": None,
        "points": [],
    }
    if len(active) < 2:
        diag["stage"] = "detection"
        diag["reason"] = "至少需要 2 台相机各 1 个候选点"
        return _empty_result(
            "insufficient_detections",
            "至少需要 2 台相机检测到 LED 候选（当前 %s）"
            % "，".join("cam%d=%d" % (c, counts[c]) for c in cams),
            counts, diag)

    K, D, R, t, P, ideal, raw = {}, {}, {}, {}, {}, {}, {}
    for c in active:
        K[c], D[c], R[c], t[c] = _pose_arrays(table[c])
        P[c] = K[c] @ np.c_[R[c], t[c]]
        raw[c] = _camera_points(dets[c])
        ideal[c] = ideal_pixels(raw[c], K[c], D[c])
        diag["raw_points_%d" % c] = [[round(float(v), 3) for v in p]
                                     for p in raw[c]]
        diag["ideal_points_%d" % c] = [[round(float(v), 3) for v in p]
                                       for p in ideal[c]]

    # 任意相机对的极线几何（向量化：候选多时也不做 Python 双重循环）
    Fp, epi_mats = {}, {}
    for a, b in itertools.combinations(active, 2):
        Rab, Tab = relative_pose(R[a], t[a], R[b], t[b])
        Fp[(a, b)] = fundamental_from_pose(K[a], K[b], Rab, Tab)
        epi, da, db = _epipolar_matrix(ideal[a], ideal[b], Fp[(a, b)])
        epi_mats[(a, b)] = (epi, da, db)
        diag["pairwise"]["%d,%d" % (a, b)] = {
            "F": [[round(float(v), 10) for v in row] for row in Fp[(a, b)]],
            "epipolar_distances": np.round(epi, 3).tolist(),
        }
    # 旧字段：第一对相机的矩阵，便于旧页面/脚本继续读取
    if diag["pairwise"]:
        diag["epipolar_distances"] = next(iter(diag["pairwise"].values()))[
            "epipolar_distances"]

    # 1) 任意相机对的候选匹配 → 双视角 3D 假设
    hypotheses = []
    for (a, b), F in Fp.items():
        epi, da, db = epi_mats[(a, b)]
        passed = np.argwhere(epi <= epi_tol)
        diag["epipolar_pass"] += int(len(passed))
        for i, j in passed:
            i, j = int(i), int(j)
            pa, pb = ideal[a][i], ideal[b][j]
            if True:
                # ideal_pixels 输出的是**理想像素**（P=K），投影矩阵用 K[R|t]
                X = st.triangulate_dlt([pa, pb], [P[a], P[b]])
                d0 = float(da[i, j])
                d1 = float(db[i, j])
                if X is None or not (_depth_ok(X, R[a], t[a])
                                     and _depth_ok(X, R[b], t[b])):
                    continue
                diag["depth_pass"] += 1
                e = {}
                ok = True
                for c, idx in ((a, i), (b, j)):
                    proj = _project_world_many(X.reshape(1, 3), K[c], D[c],
                                               R[c], t[c])
                    e[c] = float(np.linalg.norm(proj.reshape(2) - raw[c][idx]))
                    if e[c] > rep_tol:
                        ok = False
                        break
                if not ok:
                    continue
                diag["reprojection_pass"] += 1
                hypotheses.append({
                    "obs": {a: i, b: j}, "xyz": X,
                    "epi": {a: d0, b: d1},
                    "rep": e,
                })
    diag["pair_hypotheses"] = len(hypotheses)
    if not hypotheses:
        diag["stage"] = "matching"
        diag["reason"] = "没有通过极线/深度/重投影的相机对"
        return _empty_result(
            "insufficient_matches",
            "极线/重投影过滤后没有可三角化的相机对（极线 ≤%.1f px，重投影 ≤%.2f px）"
            % (epi_tol, rep_tol), counts, diag)

    hypotheses.sort(key=lambda h: (sum(h["epi"].values())
                                   + sum(h["rep"].values())))
    hypotheses = hypotheses[:max_match_candidates]

    # 2) 用其它相机中重投影一致的观测扩展同一 3D 假设，并联合 DLT 精化
    for h in hypotheses:
        for c in active:
            if c in h["obs"]:
                continue
            proj = _project_world_many(h["xyz"].reshape(1, 3), K[c], D[c],
                                       R[c], t[c])
            d = np.linalg.norm(proj.reshape(1, 2) - raw[c], axis=1)
            j = int(np.argmin(d))
            if float(d[j]) <= rep_tol:
                h["obs"][c] = j
                h["rep"][c] = float(d[j])
        if len(h["obs"]) >= 3:
            X = st.triangulate_dlt(
                [ideal[c][i] for c, i in h["obs"].items()],
                [P[c] for c in h["obs"]])
            if X is not None and all(_depth_ok(X, R[c], t[c])
                                     for c in h["obs"]):
                h["xyz"] = X
                for c, i in h["obs"].items():
                    proj = _project_world_many(X.reshape(1, 3), K[c], D[c],
                                               R[c], t[c])
                    h["rep"][c] = float(np.linalg.norm(
                        proj.reshape(2) - raw[c][i]))

    # 3) 合并同一个 LED 的重复假设（不同相机对会产生同一个 3D 点）
    hypotheses.sort(key=lambda h: (-len(h["obs"]),
                                   sum(h["rep"].values()) / max(1, len(h["rep"]))))
    merged = []
    for h in hypotheses:
        same = None
        for m in merged:
            if float(np.linalg.norm(h["xyz"] - m["xyz"])) < MERGE_DIST_M:
                same = m
                break
        if same is None:
            merged.append(h)
        elif len(h["obs"]) > len(same["obs"]):
            merged[merged.index(same)] = h
        if len(merged) >= MAX_MERGED_POINTS:
            break

    # 组合搜索上限：按观测数/重投影排序后只保留前 N 个 3D 候选，
    # 避免 24 个候选时 C(24,3)=2024 组 × 4 相机的组合爆炸。
    merged.sort(key=lambda h: (-len(h["obs"]),
                               sum(h["rep"].values()) / max(1, len(h["rep"]))))
    if len(merged) > MAX_TRIANGLE_POINTS:
        merged = merged[:MAX_TRIANGLE_POINTS]
    diag["candidate_count"] = len(merged)
    diag["points"] = [{
        "xyz": [round(float(v), 6) for v in h["xyz"]],
        "observations": {str(c): int(i) for c, i in h["obs"].items()},
        "reproj_px": {str(c): round(float(v), 3)
                      for c, v in h["rep"].items()},
        "n_cameras": len(h["obs"]),
    } for h in merged]

    if len(merged) < 3:
        diag["stage"] = "matching"
        diag["reason"] = "有效 3D 候选不足 3 个"
        return _empty_result(
            "insufficient_matches",
            "极线/重投影过滤后只有 %d 个有效 3D 候选（至少需要 3 个）"
            % len(merged), counts, diag)

    # 4) 34/79/79 mm 三角形搜索
    points = np.array([h["xyz"] for h in merged], np.float64)
    best, second, best_bad = None, None, None
    for a, b, c in itertools.combinations(range(len(merged)), 3):
        # 同一相机不能把同一个候选点分配给两个 LED
        conflict = False
        for cam in active:
            idxs = [merged[k]["obs"].get(cam) for k in (a, b, c)]
            idxs = [v for v in idxs if v is not None]
            if len(idxs) != len(set(idxs)):
                conflict = True
                break
        if conflict:
            continue
        tri = points[[a, b, c]]
        edges = np.array([
            float(np.linalg.norm(tri[0] - tri[1])),
            float(np.linalg.norm(tri[1] - tri[2])),
            float(np.linalg.norm(tri[2] - tri[0])),
        ])
        sorted_edges = np.sort(edges)
        errs = np.abs(sorted_edges - _EXPECTED_EDGES_M)
        max_err = float(errs.max())
        center = tri.mean(axis=0)
        # 绕向一致性：同时看到全部三个 LED 的相机对，其二维有向面积符号应一致
        orients = []
        for cam in active:
            ids = [merged[k]["obs"].get(cam) for k in (a, b, c)]
            if all(v is not None for v in ids):
                orients.append(_signed_area(ideal[cam][ids]))
        orient_ok = True
        if len(orients) >= 2:
            signs = [o > 0 for o in orients if abs(o) > 1e-9]
            orient_ok = len(set(signs)) <= 1
        diag["triangle_combinations"] += 1
        cand = {
            "indices": (a, b, c), "points": tri, "edges": edges,
            "sorted_edges": sorted_edges, "errs": errs, "center": center,
            "mean_reproj": float(np.mean([
                merged[k]["rep"][cam] for k in (a, b, c)
                for cam in merged[k]["rep"]])),
            "mean_epi": float(np.mean([
                merged[k]["epi"][cam] for k in (a, b, c)
                for cam in merged[k]["epi"]])),
            "orientation_consistent": bool(orient_ok),
        }
        if max_err > tol_m:
            if best_bad is None or max_err < best_bad[0]:
                best_bad = (max_err, cand)
            continue
        diag["valid_triangles"] += 1
        score = (2.0 * float(errs.mean()) / max(tol_m, 1e-9)
                 + cand["mean_reproj"] / max(rep_tol, 1e-9)
                 + 0.5 * cand["mean_epi"] / max(epi_tol, 1e-9)
                 + (0.0 if orient_ok else 2.0))
        if predicted_center is not None:
            pred = np.asarray(predicted_center, np.float64).reshape(3)
            score += 0.15 * float(np.linalg.norm(center - pred))
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
        diag["stage"] = "geometry"
        return _empty_result(
            "geometry_mismatch",
            "没有三点组合满足 34/79/79 mm 等腰三角形（容差 %.1f mm）"
            % params["triangle_tol_mm"], counts, diag)

    second_score = None if second is None else second[0]
    chosen = best[1]
    tri = chosen["points"]
    base = [merged[i] for i in chosen["indices"]]
    labeled, assignment, order = _triangle_order(tri, previous_points)
    ordered = [base[i] for i in order]
    edges = np.array([
        float(np.linalg.norm(labeled[0] - labeled[1])),
        float(np.linalg.norm(labeled[1] - labeled[2])),
        float(np.linalg.norm(labeled[2] - labeled[0])),
    ])
    sorted_edges = np.sort(edges)
    errs = np.abs(sorted_edges - _EXPECTED_EDGES_M)
    center = labeled.mean(axis=0)
    base_mid = (labeled[1] + labeled[2]) / 2.0
    direction = labeled[0] - base_mid
    norm = float(np.linalg.norm(direction))
    board_direction = (direction / norm).tolist() if norm > 1e-9 else None

    leds_out, matches = [], []
    for k, (pt, cand) in enumerate(zip(labeled, ordered)):
        obs = {str(c): int(i) for c, i in cand["obs"].items()}
        rep = {str(c): round(float(v), 3) for c, v in cand["rep"].items()}
        leds_out.append({
            "label": k,
            "xyz": [round(float(v), 6) for v in pt],
            "reproj_px": rep,
            "observations": [
                {"camera": int(c), "index": int(i),
                 "reproj_px": round(float(cand["rep"].get(c, 0.0)), 3)}
                for c, i in sorted(cand["obs"].items())],
            "n_cameras": len(obs),
            # 兼容旧字段（只有对应相机存在时）
            "cam0_index": obs.get("0"), "cam1_index": obs.get("1"),
            "reproj_0_px": rep.get("0"), "reproj_1_px": rep.get("1"),
            "epipolar_px": round(float(np.mean(
                [cand["epi"][c] for c in cand["epi"]])), 4),
        })
        matches.append({
            "led": k, "obs": obs, "reproj_px": rep,
            "cam0_index": obs.get("0"), "cam1_index": obs.get("1"),
            "reproj_0_px": rep.get("0"), "reproj_1_px": rep.get("1"),
            "epipolar_px": leds_out[-1]["epipolar_px"],
        })
    diag.update({
        "best_edges_mm": [round(float(v * 1000), 3) for v in sorted_edges],
        "best_edge_errors_mm": [round(float(v * 1000), 3) for v in errs],
        "best_reprojection_px": round(chosen["mean_reproj"], 3),
        "best_epipolar_px": round(chosen["mean_epi"], 3),
        "best_triangle_indices": list(chosen["indices"]),
    })
    return {
        "ok": True, "status": "ok", "reason": "",
        "matches": matches, "leds": leds_out,
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
        "detected_counts": counts,
        "candidate_count": len(merged),
        "cameras_used": sorted({int(c) for cand in ordered
                                for c in cand["obs"]}),
        "diagnostics": diag,
    }


__all__ = ["skew", "relative_pose", "fundamental_from_pose", "ideal_pixels",
           "match_and_triangulate"]
