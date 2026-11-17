"""红外 LED 检测 / 三维定位后台服务。

复用边界
--------------------------------------------------------------------------
* 相机帧只从 ``camera_manager.manager()`` 的 worker 读，不重新打开设备。
* 同步帧对只从 ``sync.sync_service().recent_pairs()`` 取，不另建配对线程。
* 标定参数只从 ``calibration.service.service()`` 的当前运行状态读，不另建参数库。
* 后台线程独立于 Flask / 相机采集线程；HTTP 只读取缓存和最新的 JPEG 叠加图。
* 同一个 ``pair_seq`` 只处理一次；参数变化时允许对最新帧重算一次。

二维检测与三维定位是两个独立状态：
* ``detect_enabled``  只做单相机亮点检测和二维可视化；
* ``locate_enabled``  才做极线匹配、三角化、等边三角形和卡尔曼跟踪。
"""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import os
import threading
import time
from datetime import datetime

import cv2
import numpy as np

from camera_manager import manager
from sync import sync_service
from calibration.dataset import atomic_json_write
from calibration.service import DEFAULT_SESSION_ROOT, project_relative, service as calibration_service
from led.detector import (
    DEFAULT_PARAMS, default_params, merge_params, validate_params, detect_leds,
)
from led.geometry import (
    fundamental_from_pose, ideal_pixels, match_and_triangulate, relative_pose,
)
from led.tracker import LedKalmanTracker

DEFAULT_PARAMS_PATH = os.path.join(DEFAULT_SESSION_ROOT, "led_params.json")
DEFAULT_DEBUG_ROOT = os.path.join(os.path.dirname(DEFAULT_SESSION_ROOT),
                                   "led_debug")
MAX_PAIR_AGE_S = 0.5
MAX_PROCESS_FPS = 25.0
MAX_TRAJECTORY = 300
OVERLAY_SCALE = 0.6
ROI_TRACK_PAD_PX = 80
ROI_MAX_MISS = 12
PARAMS_SCHEMA = "lowcost-mocap/led-params"
CAMS = (0, 1)


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _json_sig(obj) -> str:
    raw = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _empty_detection() -> dict:
    return {
        "count": 0, "raw_count": 0, "status": "idle", "reason": "二维检测未启动",
        "leds": [], "rejected": [], "warnings": [],
        "overexposed": False, "unresolved_count": 0,
        "roi": None, "roi_mode": "full", "roi_fallback": False,
    }


def _strip_mask(det: dict | None) -> dict:
    """深拷贝检测结果并去掉 numpy mask，便于 JSON 保存。"""
    out = {}
    for key, value in (det or {}).items():
        if key == "mask":
            continue
        out[key] = value
    return copy.deepcopy(out)


def _slim_detection(det: dict | None) -> dict:
    det = det or {}
    leds = []
    for d in det.get("leds") or []:
        leds.append({
            "x": d.get("x"), "y": d.get("y"), "area": d.get("area"),
            "split": bool(d.get("split")),
            "mean_intensity": d.get("mean_intensity"),
            "max_intensity": d.get("max_intensity"),
            "saturated_ratio": d.get("saturated_ratio"),
        })
    rejected = []
    for r in det.get("rejected") or []:
        rejected.append({
            "bbox": r.get("parent_bbox"), "area": r.get("parent_area"),
            "reason": r.get("reason"),
            "saturated_ratio": r.get("saturated_ratio"),
        })
    out = _empty_detection()
    out.update({
        "count": int(det.get("count", 0)),
        "raw_count": int(det.get("raw_count", 0)),
        "status": det.get("status", "idle"),
        "reason": det.get("reason", ""),
        "leds": leds,
        "rejected": rejected,
        "warnings": list(det.get("warnings") or []),
        "overexposed": bool(det.get("overexposed")),
        "unresolved_count": int(det.get("unresolved_count", 0)),
        "roi": det.get("roi"),
        "roi_mode": det.get("roi_mode", "full"),
        "roi_fallback": bool(det.get("roi_fallback")),
    })
    return out


def _validate_rois(raw) -> dict:
    """校验手动 ROI（归一化 0..1）。None/空表示整幅图。"""
    if raw is None:
        return {"0": None, "1": None}
    if not isinstance(raw, dict):
        raise ValueError("rois 必须是 JSON 对象，例如 {\"0\": [x0,y0,x1,y1]}")
    out = {"0": None, "1": None}
    for c in CAMS:
        item = raw.get(str(c), raw.get(c))
        if item in (None, "", [], {}):
            continue
        if isinstance(item, dict):
            vals = [item.get(k) for k in ("x0", "y0", "x1", "y1")]
        else:
            try:
                vals = list(item)
            except TypeError:
                raise ValueError("相机 %d ROI 必须是 [x0,y0,x1,y1]" % c)
        if len(vals) != 4 or any(v is None for v in vals):
            raise ValueError("相机 %d ROI 必须是 4 个数 [x0,y0,x1,y1]" % c)
        try:
            x0, y0, x1, y1 = [float(v) for v in vals]
        except (TypeError, ValueError):
            raise ValueError("相机 %d ROI 必须是数字" % c)
        if not all(0.0 <= v <= 1.0 for v in (x0, y0, x1, y1)):
            raise ValueError("相机 %d ROI 必须在 0..1 的归一化坐标内" % c)
        if x1 - x0 < 0.02 or y1 - y0 < 0.02:
            raise ValueError("相机 %d ROI 太小（至少占画面 2%%）" % c)
        if x1 <= x0 or y1 <= y0:
            raise ValueError("相机 %d ROI 必须满足 x0<x1、y0<y1" % c)
        out[str(c)] = [round(x0, 6), round(y0, 6),
                       round(x1, 6), round(y1, 6)]
    return out


def _parse_camera_map(raw):
    """``auto`` 或 [live0→calib, live1→calib] 的 0/1 排列。"""
    if raw in (None, "", "auto", "自动"):
        return "auto"
    if isinstance(raw, dict):
        vals = [raw.get("0", raw.get(0)), raw.get("1", raw.get(1))]
    else:
        try:
            vals = list(raw)
        except TypeError:
            raise ValueError("camera_map 必须是 auto 或 [0,1]/[1,0]")
    try:
        out = [int(vals[0]), int(vals[1])]
    except (TypeError, ValueError, IndexError):
        raise ValueError("camera_map 必须是 auto 或 [0,1]/[1,0]")
    if sorted(out) != [0, 1]:
        raise ValueError("camera_map 必须是 auto 或 [0,1]/[1,0]")
    return out


def _compact_diag(diag: dict | None) -> dict | None:
    """状态轮询只带紧凑诊断；完整 3x3/配对明细放在 /api/led/diagnose。"""
    if not diag:
        return None
    keep = ("camera_map", "detected_counts", "epipolar_tol_px",
            "reprojection_tol_px", "triangle_tol_mm", "epipolar_pass",
            "depth_pass", "reprojection_pass", "candidate_count",
            "triangle_combinations", "valid_triangles", "best_edges_mm",
            "best_edge_errors_mm", "best_reprojection_px",
            "best_epipolar_px", "best_triangle_indices")
    out = {k: diag.get(k) for k in keep if k in diag}
    epi = diag.get("epipolar_distances") or []
    if epi and len(epi) <= 8 and all(len(r) <= 8 for r in epi):
        out["epipolar_distances"] = epi
    return out


def _mapping_result_score(result: dict | None) -> float:
    """越小越好：优先真实三角形边长/重投影，其次候选数量。"""
    if not result:
        return float("inf")
    if result.get("ok"):
        err = float(np.mean(result.get("edge_errors_mm") or [999.0]))
        rep = float(result.get("mean_reprojection_px") or 999.0)
        epi = float(result.get("mean_epipolar_px") or 999.0)
        return err + 0.5 * rep + 0.2 * epi
    diag = result.get("diagnostics") or {}
    cand = int(diag.get("candidate_count", result.get("candidate_count", 0)) or 0)
    depth = int(diag.get("depth_pass", 0) or 0)
    best_err = diag.get("best_edge_errors_mm")
    edge = (float(np.mean(best_err)) if best_err else 999.0)
    # 没有合法三角形时，有候选/正深度的一侧优先；仍保持有限分值
    return 10000.0 - 100.0 * cand - 10.0 * depth + edge


def _mapping_strong(result: dict | None) -> bool:
    """有足够正深度/重投影候选；即使三边暂时超容差，也足以锁定映射。"""
    diag = (result or {}).get("diagnostics") or {}
    return (int(diag.get("candidate_count", 0) or 0) >= 3
            and int(diag.get("depth_pass", 0) or 0) >= 3
            and int(diag.get("reprojection_pass", 0) or 0) >= 3)


def _choose_mapping(results: dict) -> tuple[str, dict] | tuple[None, None]:
    """从 {"0,1": result, "1,0": result} 选一个映射。

    没有“合法三角形”或“正深度/重投影候选”的一侧不参与选择，避免两路都
    只有 2 个 LED 时把已确认的映射切回错误编号。
    """
    if not results:
        return None, None
    eligible = {k: r for k, r in results.items()
                if (r or {}).get("ok") or _mapping_strong(r)}
    if not eligible:
        return None, None
    ranked = sorted(eligible.items(), key=lambda kv: _mapping_result_score(kv[1]))
    key, res = ranked[0]
    return key, res


def _roi_norm_to_px(norm, shape) -> list[int] | None:
    if not norm:
        return None
    h, w = int(shape[0]), int(shape[1])
    x0 = max(0, min(w - 1, int(round(float(norm[0]) * w))))
    y0 = max(0, min(h - 1, int(round(float(norm[1]) * h))))
    x1 = max(x0 + 1, min(w, int(round(float(norm[2]) * w))))
    y1 = max(y0 + 1, min(h, int(round(float(norm[3]) * h))))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    return [x0, y0, x1 - x0, y1 - y0]


def _build_calibration(snapshot: dict,
                       camera_map: tuple[int, int] = (0, 1)) -> dict:
    """从 CalibrationService 快照提取 LED 定位需要的矩阵，并做完整性检查。

    ``camera_map`` 把**实时相机索引**映射到**标定结果里的相机编号**：
    ``camera_map[0]`` 是实时 cam0 使用的标定相机，``camera_map[1]`` 同理。
    两台相机硬件完全相同、枚举顺序可能在重启后互换，因此不能假设实时 0/1
    永远对应标定 0/1；LED 服务会同时尝试 (0,1) 和 (1,0) 并自动选择。
    """
    intr = snapshot.get("intrinsics") or {}
    rig = snapshot.get("extrinsic")
    try:
        cal_ids = (int(camera_map[0]), int(camera_map[1]))
    except (TypeError, ValueError):
        raise ValueError("camera_map 必须是两个相机编号")
    if cal_ids[0] == cal_ids[1] or set(cal_ids) != {0, 1}:
        raise ValueError("camera_map 必须是 0/1 的一个排列")

    problems = []
    if not rig:
        problems.append("没有世界坐标系外参；请先解外参或载入 world_extrinsics.json")
    else:
        if not rig.get("orientation_confirmed"):
            problems.append("世界坐标轴方向未确认；请人工确认后重新解外参")
        cams = rig.get("cameras") or {}
        for c in cal_ids:
            if str(c) not in cams:
                problems.append("外参缺少相机 %d" % c)
    for c in cal_ids:
        d = intr.get(str(c))
        if not isinstance(d, dict):
            problems.append("缺少相机 %d 内参；请先载入内参" % c)
            continue
        for key in ("K", "D", "image_size"):
            if key not in d:
                problems.append("相机 %d 内参缺少 %s" % (c, key))
    if problems:
        raise ValueError("；".join(problems))

    def kd(c):
        d = intr[str(c)]
        return (np.asarray(d["K"], np.float64),
                np.asarray(d["D"], np.float64).reshape(-1),
                [int(v) for v in d["image_size"]])

    K0, D0, size0 = kd(cal_ids[0])
    K1, D1, size1 = kd(cal_ids[1])
    if size0 != size1:
        raise ValueError("两台相机内参图像尺寸不一致：%s vs %s" % (size0, size1))
    cams = rig["cameras"]
    try:
        R0 = np.asarray(cams[str(cal_ids[0])]["R"], np.float64)
        t0 = np.asarray(cams[str(cal_ids[0])]["t"], np.float64).reshape(3)
        R1 = np.asarray(cams[str(cal_ids[1])]["R"], np.float64)
        t1 = np.asarray(cams[str(cal_ids[1])]["t"], np.float64).reshape(3)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("外参缺少有效的 R/t：%s" % exc)
    if (R0.shape != (3, 3) or R1.shape != (3, 3)):
        raise ValueError("外参 R 不是 3x3 矩阵")
    R10, T10 = relative_pose(R0, t0, R1, t1)
    F = fundamental_from_pose(K0, K1, R10, T10)
    calib = {
        "K0": K0, "D0": D0, "K1": K1, "D1": D1,
        "R0": R0, "t0": t0, "R1": R1, "t1": t1,
        "R10": R10, "T10": T10, "F": F,
        "image_size": size0,
        "R_10": R10.tolist(), "T_10": T10.tolist(),
        "camera_map": [int(cal_ids[0]), int(cal_ids[1])],
        "calibration_ids": [int(cal_ids[0]), int(cal_ids[1])],
        "board": rig.get("board") or snapshot.get("board"),
        "world_frame": rig.get("world_frame"),
        "source": rig.get("source"),
        "source_session": rig.get("source_session"),
        "loaded_from": rig.get("loaded_from"),
        "created": rig.get("created"),
        "orientation_confirmed": bool(rig.get("orientation_confirmed")),
        "convention": "X_cam = R @ X_world + t; triangulation uses P = K[R|t] in world",
    }
    calib["signature"] = _json_sig({
        "K0": K0.tolist(), "D0": D0.tolist(),
        "K1": K1.tolist(), "D1": D1.tolist(),
        "R0": R0.tolist(), "t0": t0.tolist(),
        "R1": R1.tolist(), "t1": t1.tolist(),
        "image_size": size0,
        "camera_map": calib["camera_map"],
        "orientation_confirmed": calib["orientation_confirmed"],
    })
    return calib


class LedService:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._detect = False
        self._locate = False
        self._last_pair_seq: int | None = None
        self._force_reprocess = False
        self._last_process_mono = 0.0
        self._processing_ms = None
        self._processed_pairs = 0
        self._last_error = ""
        self._params_path = DEFAULT_PARAMS_PATH
        self._global_params = default_params()
        self._overrides = {"0": {}, "1": {}}
        self._rois = {"0": None, "1": None}
        self._roi_tracking = True
        self._roi_track = {"0": None, "1": None}
        self._roi_miss = {"0": 0, "1": 0}
        self._roi_force_full = {"0": False, "1": False}
        # 实时相机 → 标定相机映射。硬件名称相同，USB 枚举顺序可能在重启后
        # 互换；auto 模式下用 33 mm 三角化结果自动判别。
        self._camera_map_mode = "auto"
        self._calib_options: dict[str, dict] = {}
        self._mapping_confirmed = False
        self._last_diag = None
        self._params_updated = None
        self._params_warning = ""
        self._param_version = 0
        self._calib: dict | None = None
        self._calib_warning = ""
        self._last_calib_check = 0.0
        self._tracker = LedKalmanTracker()
        self._tracker_state = "idle"
        self._last_led_points = None
        self._trajectory = []
        self._raw = {0: None, 1: None}
        self._mask = {0: None, 1: None}
        self._det_internal = {0: None, 1: None}
        self._load_params()
        self._state = self._idle_state()

    # ------------------------------------------------------------------
    # 参数
    # ------------------------------------------------------------------
    def _load_params(self) -> None:
        path = self._params_path
        if not os.path.isfile(path):
            return
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
            if not isinstance(data, dict):
                raise ValueError("不是 JSON 对象")
            self._global_params = validate_params(data.get("params") or {})
            overrides = data.get("overrides") or {}
            self._overrides = {
                str(c): validate_params(overrides.get(str(c)) or {}, partial=True)
                for c in CAMS if isinstance(overrides.get(str(c)) or {}, dict)
            }
            self._rois = _validate_rois(data.get("rois"))
            self._roi_tracking = bool(data.get("roi_tracking", True))
            self._camera_map_mode = _parse_camera_map(data.get("camera_map"))
            self._params_updated = data.get("updated")
        except Exception as exc:
            self._params_warning = ("led_params.json 读取失败，使用默认参数：%s"
                                    % str(exc).strip().splitlines()[-1])

    def _save_params(self) -> str:
        payload = {
            "schema": PARAMS_SCHEMA, "version": 1,
            "updated": _now_iso(),
            "params": self._global_params,
            "overrides": self._overrides,
            "rois": self._rois,
            "roi_tracking": self._roi_tracking,
            "camera_map": self._camera_map_mode,
        }
        atomic_json_write(self._params_path, payload)
        self._params_updated = payload["updated"]
        return project_relative(self._params_path)

    def effective_params(self, camera: int) -> dict:
        return merge_params(self._global_params, self._overrides.get(str(int(camera))))

    def update_params(self, payload: dict) -> dict:
        payload = payload or {}
        raw_global = payload.get("params", payload)
        global_params = validate_params({**self._global_params, **(raw_global or {})})
        # 显式传空对象表示“清空所有相机覆盖”；未传字段才保留当前值
        overrides_raw = payload.get("overrides", self._overrides)
        if overrides_raw is None:
            overrides_raw = {}
        overrides = {}
        for c in CAMS:
            item = overrides_raw.get(str(c)) or {}
            if not isinstance(item, dict):
                raise ValueError("相机 %d 覆盖参数必须是 JSON 对象" % c)
            # 空字符串/None 视为“未覆盖”
            clean = {k: v for k, v in item.items() if v not in (None, "")}
            overrides[str(c)] = validate_params(clean, partial=True) if clean else {}
        rois = self._rois
        if "rois" in payload:
            rois = _validate_rois(payload.get("rois"))
        roi_tracking = self._roi_tracking
        if "roi_tracking" in payload:
            roi_tracking = bool(payload.get("roi_tracking"))
        camera_map = self._camera_map_mode
        if "camera_map" in payload:
            camera_map = _parse_camera_map(payload.get("camera_map"))
        with self._lock:
            self._global_params = global_params
            self._overrides = overrides
            self._rois = rois
            self._roi_tracking = roi_tracking
            self._camera_map_mode = camera_map
            self._mapping_confirmed = False
            self._last_diag = None
            self._roi_track = {"0": None, "1": None}
            self._roi_miss = {"0": 0, "1": 0}
            self._roi_force_full = {"0": False, "1": False}
            self._param_version += 1
            self._force_reprocess = True
            self._params_warning = ""
        path = self._save_params()
        if self._calib is not None:
            try:
                self._load_calibration(force=True)
            except Exception:
                pass
        with self._lock:
            self._state["params"] = self._params_state_locked()
        return {"ok": True, "path": path, **self._params_state_locked()}

    def reset_params(self) -> dict:
        with self._lock:
            self._global_params = default_params()
            self._overrides = {"0": {}, "1": {}}
            self._rois = {"0": None, "1": None}
            self._roi_tracking = True
            self._camera_map_mode = "auto"
            self._mapping_confirmed = False
            self._last_diag = None
            self._roi_track = {"0": None, "1": None}
            self._roi_miss = {"0": 0, "1": 0}
            self._roi_force_full = {"0": False, "1": False}
            self._param_version += 1
            self._force_reprocess = True
        path = self._save_params()
        if self._calib is not None:
            try:
                self._load_calibration(force=True)
            except Exception:
                pass
        with self._lock:
            self._state["params"] = self._params_state_locked()
        return {"ok": True, "path": path, **self._params_state_locked()}

    def _params_state_locked(self) -> dict:
        return {
            "defaults": default_params(),
            "params": copy.deepcopy(self._global_params),
            "overrides": copy.deepcopy(self._overrides),
            "effective": {str(c): self.effective_params(c) for c in CAMS},
            "rois": copy.deepcopy(self._rois),
            "roi_tracking": bool(self._roi_tracking),
            "camera_map": copy.deepcopy(self._camera_map_mode),
            "camera_map_effective": (copy.deepcopy(self._calib.get("camera_map"))
                                     if self._calib else None),
            "camera_map_confirmed": bool(self._mapping_confirmed),
            "path": project_relative(self._params_path),
            "exists": os.path.isfile(self._params_path),
            "updated": self._params_updated,
            "version": self._param_version,
            "warning": self._params_warning,
        }

    def params_state(self) -> dict:
        with self._lock:
            return self._params_state_locked()

    # ------------------------------------------------------------------
    # 标定
    # ------------------------------------------------------------------
    def _load_calibration(self, *, force: bool = False) -> tuple[dict | None, bool]:
        """读取并校验当前标定；同时准备两种相机映射。

        返回 ``(active_calib, changed)``。auto 模式下先沿用当前映射；
        真正选择由 ``_process`` 用 33 mm 三角化结果完成。
        """
        try:
            snapshot = calibration_service().localization_snapshot()
            options = {
                "0,1": _build_calibration(snapshot, (0, 1)),
                "1,0": _build_calibration(snapshot, (1, 0)),
            }
        except Exception as exc:
            with self._lock:
                self._calib = None
                self._calib_options = {}
                self._calib_warning = str(exc).strip().splitlines()[-1]
            raise
        with self._lock:
            mode = self._camera_map_mode
            current = self._calib
            if mode == "auto":
                keep = current.get("camera_map") if current else [0, 1]
                key = "%d,%d" % tuple(keep)
                active = options.get(key, options["0,1"])
            else:
                key = "%d,%d" % (mode[0], mode[1])
                active = options[key]
            old_sig = current.get("signature") if current else None
            changed = (force or old_sig != active["signature"]
                       or (current is not None
                           and current.get("camera_map") != active["camera_map"]))
            self._calib_options = options
            self._calib = active
            self._calib_warning = ""
            if mode != "auto":
                self._mapping_confirmed = True
            elif changed or not self._mapping_confirmed:
                self._mapping_confirmed = False
            if changed:
                self._tracker.reset()
                self._tracker_state = "idle"
                self._last_led_points = None
                self._trajectory = []
                self._last_diag = None
        return active, changed

    # ------------------------------------------------------------------
    # ROI：手动归一化 ROI + 匹配后的动态跟踪 ROI
    # ------------------------------------------------------------------
    def _roi_for(self, camera: int, shape) -> tuple[list[int] | None, str]:
        key = str(int(camera))
        h, w = int(shape[0]), int(shape[1])
        if self._roi_force_full.get(key):
            return None, "full_search"
        if self._roi_tracking:
            box = self._roi_track.get(key)
            if box:
                x, y, rw, rh = [int(v) for v in box]
                pad = ROI_TRACK_PAD_PX
                x0 = max(0, x - pad)
                y0 = max(0, y - pad)
                x1 = min(w, x + rw + pad)
                y1 = min(h, y + rh + pad)
                if x1 - x0 >= 8 and y1 - y0 >= 8:
                    return [x0, y0, x1 - x0, y1 - y0], "track"
        manual = _roi_norm_to_px(self._rois.get(key), shape)
        if manual:
            return manual, "manual"
        return None, "full"

    def _detect_camera(self, gray, params: dict, camera: int,
                       need: int) -> tuple[dict, str, bool]:
        """先按 ROI 检测；ROI 结果不足时回退全图搜索一次。"""
        roi, mode = self._roi_for(camera, gray.shape)
        det = detect_leds(gray, params, roi=roi)
        det["roi_mode"] = mode
        det["roi_fallback"] = False
        if mode not in ("full", "full_search") and int(det.get("count") or 0) < int(need):
            full = detect_leds(gray, params)
            full["roi_mode"] = "full_fallback"
            full["roi_fallback"] = True
            if int(full.get("count") or 0) > int(det.get("count") or 0):
                det = full
            else:
                # 全图也没有更多候选：保留 ROI 结果，但标记已经搜索过
                det["roi_fallback"] = True
        # 一次强制全图搜索只持续一帧
        if self._roi_force_full.get(str(int(camera))):
            self._roi_force_full[str(int(camera))] = False
        return det, mode, bool(det.get("roi_fallback"))

    def _clear_track_rois(self) -> None:
        self._roi_track = {"0": None, "1": None}
        self._roi_miss = {"0": 0, "1": 0}
        self._roi_force_full = {"0": False, "1": False}

    def _update_track_rois(self, det0: dict, det1: dict, match: dict) -> None:
        if not self._roi_tracking:
            return
        for cam, det, key in ((0, det0, "cam0_index"),
                              (1, det1, "cam1_index")):
            pts = []
            leds = det.get("leds") or []
            for m in match.get("matches") or []:
                idx = m.get(key)
                if isinstance(idx, int) and 0 <= idx < len(leds):
                    pts.append((float(leds[idx]["x"]), float(leds[idx]["y"])))
            if len(pts) < 3:
                continue
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            pad = ROI_TRACK_PAD_PX
            h, w = det.get("image_size", [0, 0])[1], det.get("image_size", [0, 0])[0]
            if w <= 0 or h <= 0:
                continue
            x0 = max(0, int(round(min(xs))) - pad)
            y0 = max(0, int(round(min(ys))) - pad)
            x1 = min(w, int(round(max(xs))) + pad)
            y1 = min(h, int(round(max(ys))) + pad)
            if x1 - x0 >= 8 and y1 - y0 >= 8:
                self._roi_track[str(cam)] = [x0, y0, x1 - x0, y1 - y0]
                self._roi_miss[str(cam)] = 0
                self._roi_force_full[str(cam)] = False

    def _note_match_failure(self) -> None:
        for c in CAMS:
            key = str(c)
            self._roi_miss[key] = self._roi_miss.get(key, 0) + 1
            if self._roi_miss[key] > ROI_MAX_MISS:
                self._roi_track[key] = None
            self._roi_force_full[key] = True

    def _roi_state_locked(self) -> dict:
        out = {"tracking": bool(self._roi_tracking),
               "manual": copy.deepcopy(self._rois)}
        return out

    def _check_cameras(self) -> None:
        m = manager()
        missing = []
        for c in CAMS:
            w = m.worker(c)
            if w is None or w.state != "streaming":
                missing.append(str(c))
        if missing:
            raise RuntimeError("相机 %s 未处于 streaming；请先启动双相机" % ", ".join(missing))

    def _check_frame_sizes(self, calib: dict) -> None:
        m = manager()
        for c in CAMS:
            w = m.worker(c)
            f = w.latest() if w else None
            if f is not None:
                got = [int(f.gray.shape[1]), int(f.gray.shape[0])]
                if got != list(calib["image_size"]):
                    raise ValueError("相机 %d 当前分辨率 %s 与标定图像尺寸 %s 不一致"
                                     % (c, got, calib["image_size"]))

    # ------------------------------------------------------------------
    # 启停
    # ------------------------------------------------------------------
    def start_detection(self) -> dict:
        self._check_cameras()
        with self._lock:
            self._detect = True
            self._force_reprocess = True
            self._clear_track_rois()
            if not self._locate:
                self._state = self._idle_state()
                self._state.update({"detect_enabled": True, "status": "waiting_sync",
                                    "reason": "等待同步帧对"})
                self._state["params"] = self._params_state_locked()
            self._start_thread_locked()
        return self.state()

    def stop_detection(self) -> dict:
        with self._lock:
            self._detect = False
            self._locate = False
            self._stop.set()
            self._tracker.reset()
            self._tracker_state = "idle"
            self._last_led_points = None
            self._trajectory = []
            self._clear_track_rois()
            self._raw = {0: None, 1: None}
            self._mask = {0: None, 1: None}
            self._det_internal = {0: None, 1: None}
            self._state = self._idle_state()
            self._state["status"] = "stopped"
            self._state["reason"] = "二维检测已停止"
            self._state["params"] = self._params_state_locked()
        return self.state()

    def start_locate(self) -> dict:
        self._check_cameras()
        with self._lock:
            if not self._detect:
                raise ValueError("请先开始二维检测，再启动三维定位")
        calib, _ = self._load_calibration()
        self._check_frame_sizes(calib)
        with self._lock:
            self._locate = True
            self._tracker = LedKalmanTracker()
            self._tracker_state = "idle"
            self._last_led_points = None
            self._trajectory = []
            self._clear_track_rois()
            self._mapping_confirmed = (self._camera_map_mode != "auto")
            self._last_diag = None
            self._last_error = ""
            self._last_calib_check = time.perf_counter()
            self._force_reprocess = True
            self._start_thread_locked()
            self._state.update({
                "locate_enabled": True, "status": "waiting_sync",
                "reason": "等待同步帧对", "valid": False,
                "matches": [], "leds": [], "raw_center": None,
                "filtered_center": None, "velocity": None,
                "edges_mm": None, "edge_errors_mm": None,
                "trajectory": [],
                "calibration": self._calib_summary_locked(),
            })
        return self.state()

    def stop_locate(self) -> dict:
        with self._lock:
            self._locate = False
            self._tracker.reset()
            self._tracker_state = "idle"
            self._last_led_points = None
            self._trajectory = []
            self._clear_track_rois()
            self._state.update({
                "locate_enabled": False, "valid": False,
                "status": "detecting" if self._detect else "stopped",
                "reason": "三维定位已停止",
                "leds": [], "matches": [], "raw_center": None,
                "filtered_center": None, "velocity": None,
                "edges_mm": None, "edge_errors_mm": None,
                "trajectory": [],
            })
        return self.state()

    def _start_thread_locked(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            if not self._stop.is_set():
                return
            self._thread.join(timeout=0.5)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, args=(self._stop,),
                                        name="led-locator", daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------
    # 后台处理
    # ------------------------------------------------------------------
    def _loop(self, stop_event: threading.Event) -> None:
        while not stop_event.is_set():
            with self._lock:
                detect, locate = self._detect, self._locate
            if not detect and not locate:
                stop_event.set()
                break
            try:
                recs = sync_service().recent_pairs(1)
                if not recs:
                    self._set_status("waiting_sync", "还没有配对成功的同步帧对")
                    time.sleep(0.03)
                    continue
                rec = recs[0]
                now = time.perf_counter()
                age = now - rec.created_mono
                with self._lock:
                    same = (self._last_pair_seq == rec.pair_seq)
                    force = self._force_reprocess
                    last_t = self._last_process_mono
                if same and not force:
                    # 没有新帧对：当前结果会随时间变旧，明确标记为过期
                    if age > MAX_PAIR_AGE_S:
                        self._set_status(
                            "sync_stale",
                            "最近同步帧已过期 %.0f ms" % (age * 1000))
                    time.sleep(0.01)
                    continue
                # force 对应“刚启动定位/刚改参数”，允许对同一最新帧重算一次；
                # 除此之外，过期帧不做三角化
                if not force and age > MAX_PAIR_AGE_S:
                    self._set_status("sync_stale",
                                     "最近同步帧已过期 %.0f ms" % (age * 1000))
                    time.sleep(0.03)
                    continue
                if not force and now - last_t < 1.0 / MAX_PROCESS_FPS:
                    time.sleep(0.005)
                    continue
                self._process(rec, now)
                with self._lock:
                    self._last_pair_seq = rec.pair_seq
                    self._force_reprocess = False
                    self._last_process_mono = now
            except Exception as exc:
                self._last_error = str(exc).strip().splitlines()[-1]
                self._set_status("error", self._last_error)
                time.sleep(0.1)

    def _process(self, rec, now: float) -> None:
        fa, fb = rec.frame_a, rec.frame_b
        t0 = time.perf_counter()
        with self._lock:
            p0 = self.effective_params(0)
            p1 = self.effective_params(1)
            locate = self._locate
            calib = self._calib
            param_version = self._param_version
        need = 3 if locate else 1
        det0, roi_mode0, roi_fb0 = self._detect_camera(fa.gray, p0, 0, need)
        det1, roi_mode1, roi_fb1 = self._detect_camera(fb.gray, p1, 1, need)
        raw0 = np.ascontiguousarray(fa.gray).copy()
        raw1 = np.ascontiguousarray(fb.gray).copy()

        if locate:
            # 标定可能在运行期间被重新载入/修改。每秒检查一次签名；变化时
            # 重置跟踪和轨迹，避免旧外参/旧内参与新帧混用。
            with self._lock:
                calib = self._calib
                need_check = (calib is None
                              or time.perf_counter() - self._last_calib_check > 1.0)
            if need_check:
                try:
                    self._load_calibration()
                    with self._lock:
                        calib = self._calib
                        self._last_calib_check = time.perf_counter()
                    if calib is None:
                        raise ValueError(self._calib_warning
                                         or "没有可用的标定参数")
                    if [int(raw0.shape[1]), int(raw0.shape[0])] != list(calib["image_size"]):
                        raise ValueError("相机 0 分辨率与标定不一致")
                    if [int(raw1.shape[1]), int(raw1.shape[0])] != list(calib["image_size"]):
                        raise ValueError("相机 1 分辨率与标定不一致")
                except Exception as exc:
                    with self._lock:
                        self._locate = False
                        self._tracker.reset()
                        self._tracker_state = "idle"
                        self._trajectory = []
                        self._clear_track_rois()
                    self._store_result(rec, det0, det1, raw0, raw1, t0,
                                       status="calibration_invalid",
                                       reason=str(exc).strip().splitlines()[-1],
                                       locate=False, calib=None)
                    return

            with self._lock:
                pred_center = None
                if self._tracker_state == "predicted" and self._state.get("filtered_center"):
                    pred_center = self._state["filtered_center"]
                prev_points = self._last_led_points
                auto_map = (self._camera_map_mode == "auto")
                confirmed = bool(self._mapping_confirmed)
                options = self._calib_options
                active = self._calib

            # auto 模式先同时评估两种“实时相机 ↔ 标定相机”映射；错误的
            # 映射会三角化到相机后方（负深度），正确映射会得到 33 mm 三角。
            if auto_map and not confirmed:
                results = {}
                for key, opt in options.items():
                    results[key] = match_and_triangulate(
                        det0, det1, opt, p0, previous_points=prev_points,
                        predicted_center=pred_center)
                chosen_key, match = _choose_mapping(results)
                if chosen_key is not None:
                    chosen_calib = options[chosen_key]
                    strong = bool(match and match.get("ok")) or _mapping_strong(match)
                    with self._lock:
                        self._calib = chosen_calib
                        calib = chosen_calib
                        self._last_diag = {
                            "active_map": chosen_calib.get("camera_map"),
                            "auto": True,
                            "confirmed": strong,
                            "mappings": {k: r.get("diagnostics")
                                         for k, r in results.items()},
                        }
                        if strong:
                            self._mapping_confirmed = True
                            self._roi_miss = {"0": 0, "1": 0}
                else:
                    # 两路都没有足够证据：保留上一次已确认的映射，不要在两个
                    # 错误编号之间来回切换。
                    active_key = (("%d,%d" % tuple(active.get("camera_map")))
                                  if active else "0,1")
                    match = results.get(active_key)
                    if match is None and active:
                        match = match_and_triangulate(
                            det0, det1, active, p0, previous_points=prev_points,
                            predicted_center=pred_center)
                    if match is None:
                        match = {"ok": False, "status": "insufficient_matches",
                                 "reason": "没有可用的相机映射"}
                    with self._lock:
                        self._last_diag = {
                            "active_map": (active or {}).get("camera_map"),
                            "auto": True,
                            "confirmed": bool(self._mapping_confirmed),
                            "mappings": {k: r.get("diagnostics")
                                         for k, r in results.items()},
                        }
            else:
                match = match_and_triangulate(
                    det0, det1, calib, p0, previous_points=prev_points,
                    predicted_center=pred_center)
                with self._lock:
                    d = match.get("diagnostics") or {}
                    self._last_diag = {
                        "active_map": calib.get("camera_map") if calib else None,
                        "auto": auto_map,
                        "confirmed": bool(self._mapping_confirmed),
                        "mappings": {("%d,%d" % tuple(
                            calib.get("camera_map") or [0, 1])): d},
                    }
            if match.get("ok"):
                stamp = rec.created_mono
                upd = self._tracker.update(match["raw_center"], stamp)
                self._tracker_state = upd["state"]
                with self._lock:
                    self._update_track_rois(det0, det1, match)
                    self._last_led_points = [l["xyz"] for l in match["leds"]]
                    self._trajectory.append(upd["position"])
                    if len(self._trajectory) > MAX_TRAJECTORY:
                        self._trajectory = self._trajectory[-MAX_TRAJECTORY:]
                self._store_result(
                    rec, det0, det1, raw0, raw1, t0,
                    status="tracking", reason="",
                    locate=True, calib=calib,
                    match=match,
                    filtered_center=upd["position"], velocity=upd["velocity"],
                    tracker={"state": upd["state"], "missed": upd["missed"],
                             "reacquired": upd.get("reacquired", False)})
            else:
                with self._lock:
                    self._note_match_failure()
                pred = self._tracker.predict(rec.created_mono)
                self._tracker_state = pred["state"]
                if pred["state"] == "predicted":
                    with self._lock:
                        self._trajectory.append(pred["position"])
                        if len(self._trajectory) > MAX_TRAJECTORY:
                            self._trajectory = self._trajectory[-MAX_TRAJECTORY:]
                    self._store_result(
                        rec, det0, det1, raw0, raw1, t0,
                        status="predicted",
                        reason=match.get("reason") or "当前帧没有有效三点测量",
                        locate=True, calib=calib, match=match,
                        filtered_center=pred["position"], velocity=pred["velocity"],
                        tracker={"state": "predicted", "missed": pred["missed"],
                                 "age_s": pred.get("age_s")})
                else:
                    with self._lock:
                        self._trajectory = []
                        self._last_led_points = None
                        self._clear_track_rois()
                        if self._camera_map_mode == "auto":
                            self._mapping_confirmed = False
                    self._store_result(
                        rec, det0, det1, raw0, raw1, t0,
                        status="lost",
                        reason=match.get("reason") or "跟踪已丢失",
                        locate=True, calib=calib, match=match,
                        filtered_center=None, velocity=None,
                        tracker={"state": "lost"})
        else:
            self._store_result(rec, det0, det1, raw0, raw1, t0,
                               status="detecting", reason="",
                               locate=False, calib=calib)

    def _store_result(self, rec, det0, det1, raw0, raw1, t0, *,
                      status: str, reason: str, locate: bool,
                      calib: dict | None, match: dict | None = None,
                      filtered_center=None, velocity=None, tracker=None):
        # 停止请求发出后，正在处理的旧结果不能再覆盖“已停止”状态
        if locate and not self._locate:
            return
        if not locate and not self._detect:
            return
        duration_ms = (time.perf_counter() - t0) * 1000.0
        with self._lock:
            if self._detect:
                self._raw = {0: raw0, 1: raw1}
                self._mask = {0: det0.get("mask"), 1: det1.get("mask")}
                self._det_internal = {0: det0, 1: det1}
            self._processed_pairs += 1
            self._processing_ms = (duration_ms if self._processing_ms is None
                                   else 0.8 * self._processing_ms + 0.2 * duration_ms)
            state = {
                "ok": True,
                "detect_enabled": bool(self._detect),
                "locate_enabled": bool(self._locate),
                "status": status,
                "valid": status == "tracking",
                "reason": reason,
                "pair": {
                    "pair_seq": int(rec.pair_seq),
                    "dt_ms": round(float(rec.dt_ms), 3),
                    "seq0": int(rec.frame_a.seq),
                    "seq1": int(rec.frame_b.seq),
                    "created_mono": round(float(rec.created_mono), 6),
                    "host_ts0": round(float(rec.frame_a.host_ts), 6),
                    "host_ts1": round(float(rec.frame_b.host_ts), 6),
                },
                "detections": {"0": _slim_detection(det0),
                               "1": _slim_detection(det1)},
                "roi": self._roi_state_locked(),
                "matches": list((match or {}).get("matches") or []),
                "leds": list((match or {}).get("leds") or []) if status in
                        ("tracking", "predicted") else [],
                "raw_center": (match or {}).get("raw_center") if status == "tracking" else None,
                "filtered_center": filtered_center,
                "velocity": velocity,
                "edges_mm": (match or {}).get("edges_mm") if status == "tracking" else None,
                "edge_errors_mm": (match or {}).get("edge_errors_mm") if status == "tracking" else None,
                "mean_reprojection_px": (match or {}).get("mean_reprojection_px"),
                "mean_epipolar_px": (match or {}).get("mean_epipolar_px"),
                "triangle_score": (match or {}).get("triangle_score"),
                "orientation_consistent": (match or {}).get("orientation_consistent"),
                "triangle_tol_mm": (match or {}).get("triangle_tol_mm"),
                "labels_are_temporary": True,
                "label_assignment": (match or {}).get("label_assignment"),
                "tracking": tracker or {"state": self._tracker_state},
                "trajectory": [list(p) for p in self._trajectory],
                "calibration": self._calib_summary_locked(calib),
                "camera_map": (calib or {}).get("camera_map"),
                "mapping": {
                    "mode": (copy.deepcopy(self._camera_map_mode)
                             if self._camera_map_mode != "auto" else "auto"),
                    "effective": (calib or {}).get("camera_map"),
                    "confirmed": bool(self._mapping_confirmed),
                },
                "diagnostics": _compact_diag((match or {}).get("diagnostics")),
                "params": self._params_state_locked(),
                "processed_pairs": self._processed_pairs,
                "processing_ms": round(self._processing_ms or 0.0, 2),
                "timestamp": _now_iso(),
                "last_error": self._last_error,
            }
            self._state = state

    def _calib_summary_locked(self, calib=None) -> dict | None:
        calib = calib or self._calib
        if not calib:
            return {"loaded": False, "reason": self._calib_warning}
        return {
            "loaded": True,
            "source": calib.get("source"),
            "source_session": calib.get("source_session"),
            "loaded_from": calib.get("loaded_from"),
            "created": calib.get("created"),
            "orientation_confirmed": calib.get("orientation_confirmed"),
            "image_size": calib.get("image_size"),
            "camera_map": calib.get("camera_map"),
            "R_10": calib.get("R_10"),
            "T_10": calib.get("T_10"),
            "convention": calib.get("convention"),
            "board": calib.get("board"),
            "warning": self._calib_warning,
        }

    def _set_status(self, status: str, reason: str) -> None:
        with self._lock:
            if self._state.get("status") != status or self._state.get("reason") != reason:
                self._state["status"] = status
                self._state["reason"] = reason
                if status != "tracking":
                    self._state["valid"] = False
                self._state["timestamp"] = _now_iso()
                self._state["params"] = self._params_state_locked()
                self._state["calibration"] = self._calib_summary_locked()

    # ------------------------------------------------------------------
    # 状态 / 参数 / 叠加图
    # ------------------------------------------------------------------
    def _idle_state(self) -> dict:
        return {
            "ok": True, "detect_enabled": False, "locate_enabled": False,
            "status": "idle", "valid": False, "reason": "未启动",
            "pair": None,
            "detections": {"0": _empty_detection(), "1": _empty_detection()},
            "roi": self._roi_state_locked(),
            "matches": [], "leds": [], "raw_center": None,
            "filtered_center": None, "velocity": None,
            "edges_mm": None, "edge_errors_mm": None,
            "mean_reprojection_px": None, "mean_epipolar_px": None,
            "triangle_score": None, "orientation_consistent": None,
            "triangle_tol_mm": None, "labels_are_temporary": True,
            "label_assignment": None,
            "tracking": {"state": "idle"},
            "trajectory": [],
            "camera_map": None,
            "mapping": {"mode": copy.deepcopy(self._camera_map_mode),
                        "effective": None, "confirmed": False},
            "diagnostics": None,
            "calibration": self._calib_summary_locked(),
            "params": self._params_state_locked(),
            "processed_pairs": 0, "processing_ms": 0.0,
            "timestamp": _now_iso(), "last_error": "",
        }

    def state(self) -> dict:
        with self._lock:
            return copy.deepcopy(self._state)

    def calibration_state(self) -> dict:
        with self._lock:
            return copy.deepcopy(self._calib_summary_locked())

    @staticmethod
    def _diagnose_snapshot(snap: dict) -> dict:
        """用同一快照的检测/参数/标定选项重新计算完整诊断。"""
        state = snap.get("state") or {}
        options = snap.get("calib_options") or {}
        calib = snap.get("calib")
        det0 = (snap.get("det") or {}).get(0)
        det1 = (snap.get("det") or {}).get(1)
        params = (snap.get("params") or {}).get(0)
        if not options and calib:
            options = {"%d,%d" % tuple(calib.get("camera_map") or (0, 1)):
                       calib}
        mappings = {}
        chosen_key = None
        if det0 is not None and det1 is not None and params and options:
            results = {}
            for key, opt in options.items():
                results[key] = match_and_triangulate(det0, det1, opt, params)
            if snap.get("camera_map_mode") == "auto":
                chosen_key, _ = _choose_mapping(results)
            mappings = {k: r.get("diagnostics") for k, r in results.items()}
        active = ((options.get(chosen_key) or {}).get("camera_map")
                  if chosen_key else (calib or {}).get("camera_map"))
        return {
            "active_map": active,
            "auto": snap.get("camera_map_mode") == "auto",
            "confirmed": bool(snap.get("mapping_confirmed")),
            "mappings": mappings,
        }

    def diagnose(self) -> dict:
        """返回当前帧的完整分阶段诊断（含两种相机映射）。"""
        try:
            snap = self._overlay_snapshot(CAMS)
        except Exception:
            with self._lock:
                state = copy.deepcopy(self._state)
                calib = self._calib
                diag = copy.deepcopy(self._last_diag)
                mode = copy.deepcopy(self._camera_map_mode)
                confirmed = bool(self._mapping_confirmed)
            return {
                "ok": True, "pair": state.get("pair"),
                "status": state.get("status"), "reason": state.get("reason"),
                "camera_map_mode": mode,
                "camera_map": (calib or {}).get("camera_map"),
                "mapping": state.get("mapping"),
                "diagnostics": diag,
                "params": state.get("params"),
                "calibration": state.get("calibration"),
            }
        with self._lock:
            diag = self._diagnose_snapshot(snap)
            state = copy.deepcopy(snap.get("state") or {})
            mode = copy.deepcopy(self._camera_map_mode)
        return {
            "ok": True,
            "pair": state.get("pair"),
            "status": state.get("status"),
            "reason": state.get("reason"),
            "camera_map_mode": mode,
            "camera_map": state.get("camera_map")
                           or diag.get("active_map"),
            "mapping": state.get("mapping"),
            "diagnostics": diag,
            "params": state.get("params"),
            "calibration": state.get("calibration"),
        }

    # ------------------------------------------------------------------
    # 实机调试保存
    # ------------------------------------------------------------------
    def save_debug(self) -> dict:
        """保存当前同步帧的原始灰度、二值图、检测叠加图和检测 JSON。

        这样可以用实机采集到的图离线复现检测/匹配，不必只依赖合成测试。
        """
        snap = self._overlay_snapshot(CAMS)
        raw = snap.get("raw") or {}
        dets_full = snap.get("det") or {}
        state = snap.get("state") or {}
        if raw.get(0) is None or raw.get(1) is None:
            raise RuntimeError("还没有处理过同步帧对，请等待新帧后再保存")
        masks = {c: (None if (dets_full.get(c) or {}).get("mask") is None
                     else (dets_full[c]["mask"]).copy()) for c in CAMS}
        dets = {c: copy.deepcopy(_strip_mask(dets_full.get(c))) for c in CAMS}
        rois = copy.deepcopy(self._rois)
        roi_tracking = bool(self._roi_tracking)
        diagnosis = self._diagnose_snapshot(snap)
        diagnosis.update({"pair": state.get("pair"),
                          "status": state.get("status"),
                          "reason": state.get("reason"),
                          "camera_map": state.get("camera_map"),
                          "mapping": state.get("mapping")})

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # 跟着当前参数文件走，测试/自定义 MOCAP_CALIB_DIR 时不会写回默认目录
        debug_root = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(self._params_path))),
            "led_debug")
        outdir = os.path.join(debug_root, stamp)
        os.makedirs(outdir, exist_ok=True)
        files = []
        for c in CAMS:
            path = os.path.join(outdir, "cam%d_raw.png" % c)
            if not cv2.imwrite(path, raw[c]):
                raise OSError("写入失败：%s" % path)
            files.append(os.path.basename(path))
            if masks.get(c) is not None:
                path = os.path.join(outdir, "cam%d_binary.png" % c)
                if not cv2.imwrite(path, masks[c]):
                    raise OSError("写入失败：%s" % path)
                files.append(os.path.basename(path))
        # 叠加图在锁外生成，避免长时间持有状态锁
        for c in CAMS:
            try:
                data, _ = self.overlay(c, view="detect", scale=1.0,
                                       quality=92, snapshot=snap)
                path = os.path.join(outdir, "cam%d_detect.jpg" % c)
                with open(path, "wb") as fh:
                    fh.write(data)
                files.append(os.path.basename(path))
            except Exception:
                # 叠加图失败不应该让原始图/JSON 也丢失
                pass
        detections = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "cameras": [int(c) for c in CAMS],
            "detections": {str(c): dets[c] for c in CAMS},
            "params": state.get("params"),
            "rois": rois, "roi_tracking": roi_tracking,
            "pair": state.get("pair"),
            "calibration": state.get("calibration"),
            "camera_map": state.get("camera_map"),
            "mapping": state.get("mapping"),
            "status": state.get("status"),
            "reason": state.get("reason"),
        }
        atomic_json_write(os.path.join(outdir, "detections.json"), detections)
        files.append("detections.json")
        atomic_json_write(os.path.join(outdir, "diagnosis.json"), diagnosis)
        files.append("diagnosis.json")
        atomic_json_write(os.path.join(outdir, "state.json"), state)
        files.append("state.json")
        return {"ok": True, "path": project_relative(outdir),
                "abs_path": os.path.abspath(outdir),
                "files": files, "count": len(files),
                "timestamp": datetime.now().isoformat(timespec="seconds")}

    def _overlay_snapshot(self, cameras) -> dict:
        """在锁内取同一帧对的原始/检测/匹配/状态快照，供两台相机共用。"""
        cams = [int(c) for c in cameras]
        with self._lock:
            if not self._detect:
                raise RuntimeError("二维检测未启动")
            raw = {c: self._raw.get(c) for c in cams}
            # 极线视图需要另一台相机的候选点；即使单路 overlay 也保留两路 det
            det = {c: self._det_internal.get(c) for c in CAMS}
            params = {c: self.effective_params(c) for c in CAMS}
            return {
                "raw": raw,
                "det": det,
                "params": params,
                "calib": self._calib,
                "calib_options": dict(self._calib_options),
                "camera_map_mode": copy.deepcopy(self._camera_map_mode),
                "mapping_confirmed": bool(self._mapping_confirmed),
                "matches": list(self._state.get("matches") or []),
                "state": copy.deepcopy(self._state),
            }

    def overlay(self, camera: int, *, view: str = "detect",
                scale: float = OVERLAY_SCALE, quality: int = 75,
                zoom: float = 1.0, snapshot: dict | None = None
                ) -> tuple[bytes, dict]:
        cam = int(camera)
        if cam not in CAMS:
            raise ValueError("camera 必须是 0 或 1")
        if snapshot is None:
            snapshot = self._overlay_snapshot([cam])
        raw = (snapshot.get("raw") or {}).get(cam)
        det = (snapshot.get("det") or {}).get(cam)
        calib = snapshot.get("calib")
        matches = list(snapshot.get("matches") or [])
        state = snapshot.get("state") or {}
        params = (snapshot.get("params") or {}).get(cam)
        matched_map = {}
        for m in matches:
            idx = m["cam0_index"] if cam == 0 else m["cam1_index"]
            matched_map[int(idx)] = int(m["led"])
        if raw is None or det is None:
            raise RuntimeError("还没有处理过同步帧对，请等待新帧")
        if params is None:
            params = self.effective_params(cam)
        scale = float(np.clip(scale, 0.25, 1.0))
        q = int(np.clip(quality, 40, 95))
        zoom = float(np.clip(zoom, 1.0, 4.0))
        view = str(view or "detect")
        if view not in ("detect", "binary", "raw", "epipolar"):
            view = "detect"
        if view == "epipolar" and not calib:
            # 二维检测本身不要求标定；但极线视图需要 K/D/R/t，按需从当前
            # CalibrationService 状态加载，不修改标定结果本身。
            try:
                self._load_calibration()
                with self._lock:
                    calib = self._calib
            except Exception as exc:
                raise RuntimeError("极线视图需要先载入内外参：%s"
                                   % str(exc).strip().splitlines()[-1])

        if view == "binary":
            mask = det.get("mask")
            if mask is None:
                k = int(params["blur_kernel"])
                proc = cv2.GaussianBlur(raw, (k, k), 0) if k > 0 else raw
                _, mask = cv2.threshold(proc, int(params["threshold"]), 255,
                                        cv2.THRESH_BINARY)
            canvas = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        elif view == "epipolar":
            if not calib:
                raise RuntimeError("极线视图需要先载入内参/外参")
            K = calib["K0"] if cam == 0 else calib["K1"]
            D = calib["D0"] if cam == 0 else calib["D1"]
            canvas = cv2.undistort(raw, K, D)
            canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
        else:
            canvas = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)

        H, W = raw.shape[:2]
        accepted = det.get("leds") or []
        raw_comps = det.get("raw_components") or []
        roi = det.get("roi")
        roi_mode = det.get("roi_mode") or "full"
        epi_info = None

        def pt(p):
            return (int(round(float(p[0]))), int(round(float(p[1]))))

        # ROI 框：手动 ROI 用青色，动态跟踪 ROI 用橙色
        if roi and view != "epipolar":
            x, y, w, h = [int(v) for v in roi]
            color = (0, 180, 255) if roi_mode == "track" else (255, 255, 0)
            cv2.rectangle(canvas, (x, y), (x + w, y + h), color, 1)
            cv2.putText(canvas, "ROI " + ("track" if roi_mode == "track" else "manual"),
                        (x + 3, max(11, y - 3)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.36, color, 1, cv2.LINE_AA)

        if view == "detect":
            # 1) 原始连通域：很小的蓝色十字，不带文字，避免刷屏
            for r in raw_comps:
                cv2.drawMarker(canvas, pt((r.get("x", 0), r.get("y", 0))),
                               (255, 160, 0), cv2.MARKER_CROSS, 3, 1,
                               cv2.LINE_AA)
            # 2) 被拒绝的亮区：细红框，不带文字
            for r in det.get("rejected") or []:
                bbox = r.get("parent_bbox") or r.get("bbox")
                if bbox:
                    x, y, w, h = [int(v) for v in bbox]
                    cv2.rectangle(canvas, (x, y), (x + w, y + h),
                                  (0, 0, 255), 1)
            # 3) 通过筛选的 LED：黄色空心圆；只有 3 个匹配点才写 L 标签
            for idx, d in enumerate(accepted):
                color = (0, 255, 0) if idx in matched_map else (0, 220, 255)
                contour = d.get("contour")
                if contour:
                    arr = np.asarray(contour, np.int32).reshape(-1, 1, 2)
                    cv2.drawContours(canvas, [arr], -1, color, 1,
                                     cv2.LINE_AA)
                p = pt((d["x"], d["y"]))
                cv2.circle(canvas, p, 4, color, 1, cv2.LINE_AA)
            for idx, label in matched_map.items():
                if 0 <= idx < len(accepted):
                    p = pt((accepted[idx]["x"], accepted[idx]["y"]))
                    cv2.circle(canvas, p, 5, (0, 255, 0), -1, cv2.LINE_AA)
                    cv2.putText(canvas, "L%d" % label, (p[0] + 6, p[1] - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 0),
                                1, cv2.LINE_AA)

        # 极线视图：不依赖 matches。只要两台相机有候选点且内外参有效，
        # 就把“另一台相机每个候选点”的极线画到当前图像上，并用同色标记
        # 来源候选点。这样 matched=0 时也能直观看到为什么没配上。
        epi_draw = None
        if view == "epipolar" and calib:
            K = calib["K0"] if cam == 0 else calib["K1"]
            D = calib["D0"] if cam == 0 else calib["D1"]
            F = calib["F"]
            tol = float(params["epipolar_tol_px"])
            other_cam = 1 if cam == 0 else 0
            K_other = calib["K1"] if cam == 0 else calib["K0"]
            D_other = calib["D1"] if cam == 0 else calib["D0"]

            def ideal_of(items):
                if not items:
                    return np.empty((0, 2), np.float64)
                arr = np.array([[d["x"], d["y"]] for d in items],
                               np.float64).reshape(-1, 2)
                return ideal_pixels(arr, K, D)

            here = ideal_of(accepted)
            other_items = ((snapshot.get("det") or {}).get(other_cam) or {}).get(
                "leds") or []
            there = ideal_pixels(
                np.array([[d["x"], d["y"]] for d in other_items],
                         np.float64).reshape(-1, 2),
                K_other, D_other) if other_items else np.empty((0, 2),
                                                               np.float64)

            # 每条极线使用不同颜色；来源候选点在其本相机也用同一颜色。
            lines = []
            for j, op in enumerate(there):
                line = (F.T @ np.array([op[0], op[1], 1.0]) if cam == 0
                        else F @ np.array([op[0], op[1], 1.0]))
                a, b, c = float(line[0]), float(line[1]), float(line[2])
                norm = math_hypot(a, b)
                if norm < 1e-9:
                    continue
                a, b, c = a / norm, b / norm, c / norm
                clipped = _clip_line(a, b, c, W, H)
                if clipped is None:
                    continue
                lines.append((j, a, b, c, _EPI_COLORS[j % len(_EPI_COLORS)],
                              clipped))

            # 当前相机候选点：记录到所有极线的最小距离；绿色=阈值内。
            points = []
            for i, p in enumerate(here):
                color = _EPI_COLORS[i % len(_EPI_COLORS)]
                dmin, jbest = None, None
                for j, a, b, c, _lcolor, _clip in lines:
                    d = _line_distance(a, b, c, float(p[0]), float(p[1]))
                    if dmin is None or d < dmin:
                        dmin, jbest = d, j
                points.append({
                    "i": i, "p": (float(p[0]), float(p[1])),
                    "color": color, "dmin": dmin, "jbest": jbest,
                    "ok": (dmin is not None and dmin <= tol),
                })
            epi_info = {"tol": tol, "here": len(here),
                        "there": len(there), "other": other_cam}
            epi_draw = {"lines": lines, "points": points, "info": epi_info,
                        "cam": cam, "other": other_cam}

        # 局部放大：以匹配点/ROI 为中心裁剪，再缩放到显示尺寸
        focus = None
        if view == "epipolar" and accepted:
            focus = (float(np.mean([d["x"] for d in accepted])),
                     float(np.mean([d["y"] for d in accepted])))
        if focus is None and matched_map and accepted:
            xs, ys = [], []
            for idx in matched_map:
                if 0 <= idx < len(accepted):
                    xs.append(float(accepted[idx]["x"]))
                    ys.append(float(accepted[idx]["y"]))
            if xs:
                focus = (sum(xs) / len(xs), sum(ys) / len(ys))
        if focus is None and roi:
            focus = (roi[0] + roi[2] / 2.0, roi[1] + roi[3] / 2.0)
        if focus is None:
            focus = (W / 2.0, H / 2.0)

        if zoom > 1.0:
            cw = max(8, int(round(W / zoom)))
            ch = max(8, int(round(H / zoom)))
            x0 = int(round(focus[0] - cw / 2.0))
            y0 = int(round(focus[1] - ch / 2.0))
            x0 = max(0, min(W - cw, x0))
            y0 = max(0, min(H - ch, y0))
            crop = canvas[y0:y0 + ch, x0:x0 + cw]
            out = cv2.resize(crop, (max(1, int(W * scale)),
                                    max(1, int(H * scale))),
                             interpolation=cv2.INTER_LINEAR if zoom > 2
                             else cv2.INTER_AREA)
            epi_ox, epi_oy = float(x0), float(y0)
            epi_sx = out.shape[1] / float(cw)
            epi_sy = out.shape[0] / float(ch)
        else:
            out = cv2.resize(canvas, None, fx=scale, fy=scale,
                             interpolation=cv2.INTER_AREA if scale < 1
                             else cv2.INTER_LINEAR)
            epi_ox, epi_oy = 0.0, 0.0
            epi_sx, epi_sy = scale, scale

        if view == "epipolar" and epi_draw:
            _draw_epipolar_annotations(out, epi_draw, epi_ox, epi_oy,
                                       epi_sx, epi_sy)

        # 文字在缩放后绘制，避免局部放大时状态栏被裁掉。
        # 布局：左下角保留颜色图例；右上角信息整体移到右下角并右对齐。
        info_lines = ["cam%d  raw=%d  cand=%d  matched=%d" %
                      (cam, len(raw_comps), len(accepted), len(matched_map))]
        if det.get("unresolved_count"):
            info_lines[0] += "  unresolved=%d" % det["unresolved_count"]
        if det.get("overexposed"):
            info_lines[0] += "  OVEREXPOSED"
        line2 = "status=%s  roi=%s" % (det.get("status", "?"), roi_mode)
        if det.get("roi_fallback"):
            line2 += " (full search used)"
        if zoom > 1.0:
            line2 += "  zoom=%.1fx" % zoom
        info_lines.append(line2)
        if state.get("pair"):
            p = state["pair"]
            info_lines.append("pair %s  dt=%.1f ms" %
                              (p.get("pair_seq"), p.get("dt_ms") or 0.0))
        info_colors = [(0, 255, 0), (220, 220, 220), (220, 220, 220)]

        legend_lines = []
        if view == "detect":
            legend_lines = ["blue=raw yellow=cand green=matched red=rejected"]
        elif view == "epipolar" and epi_info:
            legend_lines = [
                "line=C%d source  point d=min dist  green<=%.1fpx" %
                (epi_info["other"], epi_info["tol"]),
                "C%d cand=%d  C%d cand=%d" %
                (cam, epi_info["here"], epi_info["other"],
                 epi_info["there"]),
            ]
        legend_scale = 0.34
        legend_w = 0
        for k, line in enumerate(legend_lines):
            w, _h = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX,
                                    legend_scale, 1)[0]
            legend_w = max(legend_w, w)
            cv2.putText(out, line, (8, out.shape[0] - 8 - 14 * (len(legend_lines) - 1 - k)),
                        cv2.FONT_HERSHEY_SIMPLEX, legend_scale,
                        (220, 220, 220) if k == 0 else (200, 200, 200),
                        1, cv2.LINE_AA)
        info_scale = max(0.34, min(0.46, out.shape[1] / 1280.0 * 0.55))
        _draw_info_panel(out, info_lines, info_colors,
                         font_scale=info_scale, alpha=0.55,
                         bottom_margin=8,
                         reserve_left=(8 + legend_w) if legend_lines else 0)

        ok, buf = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        if not ok:
            raise RuntimeError("JPEG 编码失败")
        return buf.tobytes(), {
            "camera": cam, "view": view, "scale": scale, "zoom": zoom,
            "count": len(accepted), "raw_count": len(raw_comps),
            "matched": len(matched_map),
            "status": det.get("status"),
            "roi_mode": roi_mode,
            "roi_fallback": bool(det.get("roi_fallback")),
        }

    def overlay_pair(self, *, view: str = "detect",
                     scale: float = OVERLAY_SCALE, quality: int = 75,
                     zoom: float = 1.0) -> dict:
        """两台相机用**同一帧对快照**渲染叠加图，避免前端分别刷新时
        出现 cam0 pair=9464、cam1 pair=9466 的显示错位。"""
        snap = self._overlay_snapshot(CAMS)
        images, infos = {}, {}
        for c in CAMS:
            data, info = self.overlay(c, view=view, scale=scale,
                                      quality=quality, zoom=zoom,
                                      snapshot=snap)
            images[str(c)] = base64.b64encode(data).decode("ascii")
            infos[str(c)] = info
        return {
            "ok": True,
            "pair": snap.get("state", {}).get("pair"),
            "status": snap.get("state", {}).get("status"),
            "mapping": snap.get("state", {}).get("mapping"),
            "images": images,
            "infos": infos,
        }


# 极线视图的颜色：同一条极线与其来源候选点共用一种颜色
_EPI_COLORS = [
    (0, 255, 255), (255, 0, 255), (0, 165, 255), (255, 255, 0),
    (0, 200, 0), (255, 120, 0), (128, 0, 255), (0, 128, 255),
    (255, 128, 0), (128, 255, 0), (0, 0, 255), (255, 255, 255),
]


def _line_distance(a: float, b: float, c: float, x: float, y: float) -> float:
    """归一化直线 ax+by+c=0 到点的距离。"""
    return abs(a * x + b * y + c)


def _clip_line(a: float, b: float, c: float, width: int, height: int):
    """把直线裁剪到图像矩形；正确处理水平、倾斜、接近竖直的线。"""
    pts = []
    if abs(b) > 1e-9:
        for x in (0.0, float(width - 1)):
            y = -(a * x + c) / b
            if -0.5 <= y <= float(height - 1) + 0.5:
                pts.append((x, y))
    if abs(a) > 1e-9:
        for y in (0.0, float(height - 1)):
            x = -(b * y + c) / a
            if -0.5 <= x <= float(width - 1) + 0.5:
                pts.append((x, y))
    if len(pts) < 2:
        return None
    # 选距离最远的两点，避免重复的角点
    best = None
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            d = math_hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1])
            if best is None or d > best[0]:
                best = (d, pts[i], pts[j])
    if best is None or best[0] < 1.0:
        return None
    (x0, y0), (x1, y1) = best[1], best[2]
    return ((int(round(x0)), int(round(y0))),
            (int(round(x1)), int(round(y1))))


def _draw_epipolar_annotations(out, data: dict, ox: float, oy: float,
                                sx: float, sy: float) -> None:
    """在缩放/裁剪后的输出图上画极线、候选点和距离表，保证文字大小稳定。"""
    height, width = out.shape[:2]

    def tr(p):
        return (int(round((float(p[0]) - ox) * sx)),
                int(round((float(p[1]) - oy) * sy)))

    lines = data.get("lines") or []
    points = data.get("points") or []
    cam = data.get("cam", 0)
    other = data.get("other", 1)
    tol = float((data.get("info") or {}).get("tol", 0.0))

    for j, _a, _b, _c, color, clipped in lines:
        p0, p1 = tr(clipped[0]), tr(clipped[1])
        cv2.line(out, p0, p1, color, 1, cv2.LINE_AA)
        if j < 12:
            cv2.putText(out, "C%d#%d" % (other, j),
                        (max(2, p1[0] - 42), max(12, p1[1] - 3)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.34, color, 1,
                        cv2.LINE_AA)

    for rec in points:
        q = tr(rec["p"])
        if not (-8 <= q[0] <= width + 8 and -8 <= q[1] <= height + 8):
            continue
        color = rec["color"]
        cv2.circle(out, q, 4, color, 1, cv2.LINE_AA)
        if rec.get("ok"):
            cv2.circle(out, q, 7, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(out, "C%d#%d" % (cam, rec["i"]),
                    (q[0] + 6, q[1] - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.32, color, 1, cv2.LINE_AA)

    rows = points[:12]
    if rows:
        row_h = 15
        box_h = 20 + row_h * len(rows)
        cv2.rectangle(out, (6, 62), (290, 62 + box_h), (0, 0, 0), -1)
        cv2.putText(out, "distance to epiline (tol %.1fpx)" % tol,
                    (10, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.34,
                    (220, 220, 220), 1, cv2.LINE_AA)
        for k, rec in enumerate(rows):
            y = 76 + row_h * (k + 1)
            src_color = (180, 180, 180)
            if rec.get("jbest") is not None:
                for j, _a, _b, _c, color, _clip in lines:
                    if j == rec["jbest"]:
                        src_color = color
                        break
            cv2.rectangle(out, (12, y - 9), (18, y - 3), src_color, -1)
            d = rec.get("dmin")
            if d is None:
                text = "C%d#%d -> no line" % (cam, rec["i"])
                tc = (200, 200, 200)
            else:
                text = ("C%d#%d -> C%d#%d  d=%.1fpx  %s" %
                        (cam, rec["i"], other, rec["jbest"], d,
                         "OK" if rec.get("ok") else "FAIL"))
                tc = (0, 255, 0) if rec.get("ok") else (0, 165, 255)
            cv2.putText(out, text, (24, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.32, tc, 1, cv2.LINE_AA)


def _draw_info_panel(out, lines, colors, *, font_scale: float = 0.4,
                     alpha: float = 0.55, bottom_margin: int = 8,
                     reserve_left: int = 0) -> None:
    """右下角右对齐信息块；可选半透明深色背景，避免遮挡主体。

    文字尺寸按当前输出图像动态计算。若与左下角图例水平重叠，则整体上移
    一行图例高度，仍保持右对齐并避免越界。
    """
    lines = [str(s) for s in lines if str(s)]
    if not lines:
        return
    height, width = out.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    sizes = [cv2.getTextSize(s, font, font_scale, thickness)[0] for s in lines]
    block_w = max(w for w, _h in sizes)
    # 极端窄图时自动缩小字号，保证右侧/左侧都不越界
    if block_w > width - 14:
        font_scale = max(0.24, font_scale * (width - 14) / float(block_w))
        sizes = [cv2.getTextSize(s, font, font_scale, thickness)[0]
                 for s in lines]
        block_w = max(w for w, _h in sizes)
    line_h = max(12, max(h for _w, h in sizes) + 6)
    right = width - 7
    bottom = height - bottom_margin
    left = right - block_w
    # 与左下角图例冲突时，把信息块上移到图例上方（仍靠右下）
    if left < reserve_left + 6:
        bottom = height - bottom_margin - line_h
        left = right - block_w
    top = bottom - line_h * len(lines)
    x0 = max(0, left - 5)
    y0 = max(0, top - 4)
    x1 = min(width, right + 3)
    y1 = min(height, bottom + 3)
    if x1 > x0 and y1 > y0:
        roi = out[y0:y1, x0:x1]
        dark = roi.copy()
        cv2.rectangle(dark, (0, 0), (x1 - x0, y1 - y0), (0, 0, 0), -1)
        cv2.addWeighted(dark, alpha, roi, 1.0 - alpha, 0, roi)
    for k, (line, (w, h)) in enumerate(zip(lines, sizes)):
        y = top + line_h * k + h
        x = max(2, right - w)
        cv2.putText(out, line, (x, y), font, font_scale,
                    colors[k % len(colors)], thickness, cv2.LINE_AA)


def math_hypot(a: float, b: float) -> float:
    return float(np.hypot(a, b))


_instance: LedService | None = None
_lock = threading.Lock()


def service() -> LedService:
    global _instance
    with _lock:
        if _instance is None:
            _instance = LedService()
        return _instance


__all__ = ["LedService", "service", "DEFAULT_PARAMS_PATH",
           "DEFAULT_DEBUG_ROOT", "MAX_PAIR_AGE_S"]
