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
* ``locate_enabled``  才做极线匹配、三角化、34/79/79 mm 等腰三角形识别和卡尔曼跟踪。
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
    default_params, merge_params, validate_params, detect_leds,
)
from led.geometry import (
    fundamental_from_pose, ideal_pixels, match_and_triangulate, relative_pose,
)
from led.recorder import recorder
from led.tracker import LedKalmanTracker

DEFAULT_PARAMS_PATH = os.path.join(DEFAULT_SESSION_ROOT, "led_params.json")
DEFAULT_DEBUG_ROOT = os.path.join(os.path.dirname(DEFAULT_SESSION_ROOT),
                                   "led_debug")
MAX_PAIR_AGE_S = 0.5
MAX_TRAJECTORY = 300
OVERLAY_SCALE = 0.6
PARAMS_SCHEMA = "lowcost-mocap/led-params"
CAMS = (0, 1, 2, 3)      # 参数容器支持最多 4 台逻辑相机


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
    return {
        "count": int(det.get("count", 0)),
        "raw_count": int(det.get("raw_count", 0)),
        "status": det.get("status", "idle"),
        "reason": det.get("reason", ""),
        "leds": leds,
        "rejected": rejected,
        "warnings": list(det.get("warnings") or []),
        "overexposed": bool(det.get("overexposed")),
        "unresolved_count": int(det.get("unresolved_count", 0)),
    }


def _compact_diag(diag: dict | None) -> dict | None:
    """状态轮询只带紧凑诊断；完整配对明细放在 /api/led/diagnose。"""
    if not diag:
        return None
    keep = ("cameras", "detected_counts", "epipolar_tol_px",
            "reprojection_tol_px", "triangle_tol_mm", "pair_hypotheses",
            "candidate_count", "triangle_combinations", "valid_triangles",
            "best_edges_mm", "best_edge_errors_mm", "best_reprojection_px",
            "best_epipolar_px", "best_triangle_indices",
            "expected_edges_mm", "triangle_sides_m", "stage", "reason")
    out = {k: diag.get(k) for k in keep if k in diag}
    pts = diag.get("points") or []
    if len(pts) <= 12:
        out["points"] = pts
    pairs = diag.get("pairwise") or {}
    if len(pairs) <= 6:
        out["pairwise"] = {k: {"pass": sum(
            1 for row in v.get("epipolar_distances") or []
            for d in row if d <= diag.get("epipolar_tol_px", 5.0))}
            for k, v in pairs.items()}
    return out


def _build_calibration(snapshot: dict, cameras=None) -> dict:
    """从 CalibrationService 快照提取多相机定位矩阵，并做完整性检查。

    逻辑相机编号同时用于采集、标定和 LED 定位（设备身份在 camera_config 里
    固定绑定），因此**不再需要**运行时猜测 live0↔calib0 / live1↔calib1 的
    映射。每台相机可以使用自己的分辨率、K、D 和世界系 R/t。
    """
    intr = snapshot.get("intrinsics") or {}
    rig = snapshot.get("extrinsic")
    cams = [int(c) for c in (cameras or snapshot.get("cameras")
                             or sorted(int(intr_key) for intr_key in intr))]
    if len(cams) < 2:
        raise ValueError("至少需要 2 台相机才能定位（当前 %s）" % cams)

    problems = []
    if not rig:
        problems.append("没有世界坐标系外参；请先解外参或载入 world_extrinsics.json")
    else:
        if not rig.get("orientation_confirmed"):
            problems.append("世界坐标轴方向未确认；请人工确认后重新解外参")
        rig_cams = rig.get("cameras") or {}
        for c in cams:
            if str(c) not in rig_cams:
                problems.append("外参缺少相机 %d" % c)
    for c in cams:
        d = intr.get(str(c))
        if not isinstance(d, dict):
            problems.append("缺少相机 %d 内参；请先载入内参" % c)
            continue
        for key in ("K", "D", "image_size"):
            if key not in d:
                problems.append("相机 %d 内参缺少 %s" % (c, key))
    if problems:
        raise ValueError("；".join(problems))

    per_camera = {}
    for c in cams:
        d = intr[str(c)]
        K = np.asarray(d["K"], np.float64)
        D = np.asarray(d["D"], np.float64).reshape(-1)
        size = [int(v) for v in d["image_size"]]
        entry = rig["cameras"][str(c)]
        try:
            R = np.asarray(entry["R"], np.float64)
            t = np.asarray(entry["t"], np.float64).reshape(3)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("相机 %d 外参缺少有效的 R/t：%s" % (c, exc))
        if R.shape != (3, 3):
            raise ValueError("相机 %d 外参 R 不是 3x3 矩阵" % c)
        if K.shape != (3, 3):
            raise ValueError("相机 %d 内参 K 不是 3x3 矩阵" % c)
        per_camera[c] = {
            "K": K.tolist(), "D": D.tolist(),
            "R": R.tolist(), "t": t.tolist(),
            "P": (K @ np.c_[R, t]).tolist(),
            "image_size": size,
            "device_key": entry.get("device_key"),
        }

    calib = {
        "cameras": cams,
        "per_camera": per_camera,
        "image_sizes": {str(c): per_camera[c]["image_size"] for c in cams},
        # 兼容旧字段：逻辑编号即标定编号
        "calibration_ids": cams,
        "board": rig.get("board") or snapshot.get("board"),
        "world_frame": rig.get("world_frame"),
        "source": rig.get("source"),
        "source_session": rig.get("source_session"),
        "loaded_from": rig.get("loaded_from"),
        "created": rig.get("created"),
        "orientation_confirmed": bool(rig.get("orientation_confirmed")),
        "convention": "X_cam = R @ X_world + t; triangulation uses P = K[R|t] in world",
    }
    first = cams[0]
    second = cams[1]
    R0 = np.asarray(per_camera[first]["R"], np.float64)
    t0 = np.asarray(per_camera[first]["t"], np.float64).reshape(3)
    R1 = np.asarray(per_camera[second]["R"], np.float64)
    t1 = np.asarray(per_camera[second]["t"], np.float64).reshape(3)
    R10, T10 = relative_pose(R0, t0, R1, t1)
    K0 = np.asarray(per_camera[first]["K"], np.float64)
    K1 = np.asarray(per_camera[second]["K"], np.float64)
    calib.update({
        "R0": R0, "t0": t0, "R1": R1, "t1": t1,
        "K0": K0,
        "D0": np.asarray(per_camera[first]["D"], np.float64),
        "K1": K1,
        "D1": np.asarray(per_camera[second]["D"], np.float64),
        "R10": R10, "T10": T10,
        "F": fundamental_from_pose(K0, K1, R10, T10),
        "R_10": R10.tolist(), "T_10": T10.tolist(),
        "image_size": per_camera[first]["image_size"],
    })
    calib["signature"] = _json_sig({
        "cameras": cams,
        "per_camera": {str(c): {
            "K": per_camera[c]["K"], "D": per_camera[c]["D"],
            "R": per_camera[c]["R"], "t": per_camera[c]["t"],
            "image_size": per_camera[c]["image_size"]} for c in cams},
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
        self._processing_stage_ms: dict = {}
        self._processed_pairs = 0
        self._groups_seen = 0
        self._groups_skipped = 0
        self._last_error = ""
        self._params_path = DEFAULT_PARAMS_PATH
        self._global_params = default_params()
        self._overrides = {str(c): {} for c in CAMS}
        # 逻辑编号 ↔ 物理设备由 camera_config.json 固定；不再运行时猜测映射。
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
        self._raw = {c: None for c in CAMS}
        self._det_internal = {c: None for c in CAMS}
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
        }
        atomic_json_write(self._params_path, payload)
        self._params_updated = payload["updated"]
        return project_relative(self._params_path)

    def effective_params(self, camera: int) -> dict:
        return merge_params(self._global_params, self._overrides.get(str(int(camera))))

    def _reset_runtime_after_params_locked(self, *, clear_warning: bool) -> None:
        self._last_diag = None
        self._param_version += 1
        self._force_reprocess = True
        if clear_warning:
            self._params_warning = ""

    def _persist_params(self) -> dict:
        path = self._save_params()
        if self._calib is not None:
            try:
                self._load_calibration(force=True)
            except Exception:
                pass
        with self._lock:
            self._state["params"] = self._params_state_locked()
        return {"ok": True, "path": path, **self._params_state_locked()}

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
        with self._lock:
            self._global_params = global_params
            self._overrides = overrides
            self._reset_runtime_after_params_locked(clear_warning=True)
        return self._persist_params()

    def reset_params(self) -> dict:
        with self._lock:
            self._global_params = default_params()
            self._overrides = {str(c): {} for c in CAMS}
            self._reset_runtime_after_params_locked(clear_warning=False)
        return self._persist_params()

    def _params_state_locked(self) -> dict:
        return {
            "defaults": default_params(),
            "params": copy.deepcopy(self._global_params),
            "overrides": copy.deepcopy(self._overrides),
            "effective": {str(c): self.effective_params(c) for c in CAMS},
            "cameras": (list(self._calib.get("cameras")) if self._calib else None),
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
        """读取并校验当前标定（逻辑编号 = 标定相机编号）。

        返回 ``(calib, changed)``。相机身份由 camera_config 固定，不再做
        live0/live1 的排列猜测。
        """
        try:
            snapshot = calibration_service().localization_snapshot()
            active = _build_calibration(snapshot)
        except Exception as exc:
            with self._lock:
                self._calib = None
                self._calib_warning = str(exc).strip().splitlines()[-1]
            raise
        with self._lock:
            current = self._calib
            old_sig = current.get("signature") if current else None
            changed = force or old_sig != active["signature"]
            self._calib = active
            self._calib_warning = ""
            if changed:
                self._reset_tracking_locked()
                self._last_diag = None
        return active, changed

    def _reset_tracking_locked(self, *, new_tracker: bool = False) -> None:
        if new_tracker:
            self._tracker = LedKalmanTracker()
        else:
            self._tracker.reset()
        self._tracker_state = "idle"
        self._last_led_points = None
        self._trajectory = []

    def _active_cameras(self) -> list[int]:
        """当前配置里实际在采样的逻辑相机编号。"""
        try:
            return [w.index for w in manager().workers()]
        except Exception:
            return [0, 1]

    def _check_cameras(self) -> None:
        m = manager()
        missing = []
        for c in self._active_cameras():
            w = m.worker(c)
            if w is None or w.state != "streaming":
                missing.append(str(c))
        if missing:
            raise RuntimeError("相机 %s 未处于 streaming；请先启动相机"
                               % ", ".join(missing))

    def _check_frame_sizes(self, calib: dict) -> None:
        m = manager()
        per = calib.get("per_camera") or {}
        for c in calib.get("cameras") or []:
            w = m.worker(c)
            f = w.latest() if w else None
            if f is None or c not in per:
                continue
            got = [int(f.gray.shape[1]), int(f.gray.shape[0])]
            want = [int(v) for v in per[c]["image_size"]]
            if got != want:
                raise ValueError("相机 %d 当前分辨率 %s 与内标定尺寸 %s 不一致"
                                 % (c, got, want))

    # ------------------------------------------------------------------
    # 启停
    # ------------------------------------------------------------------
    def start_detection(self) -> dict:
        self._check_cameras()
        with self._lock:
            self._detect = True
            self._force_reprocess = True
            if not self._locate:
                self._state = self._idle_state()
                self._state.update({"detect_enabled": True, "status": "waiting_sync",
                                    "reason": "等待同步帧对"})
                self._state["params"] = self._params_state_locked()
            self._start_thread_locked()
        recorder().set_processing(True)
        return self.state()

    def stop_detection(self) -> dict:
        with self._lock:
            self._detect = False
            self._locate = False
            self._stop.set()
            self._reset_tracking_locked()
            self._raw = {c: None for c in CAMS}
            self._det_internal = {c: None for c in CAMS}
            self._state = self._idle_state()
            self._state["status"] = "stopped"
            self._state["reason"] = "二维检测已停止"
            self._state["params"] = self._params_state_locked()
        recorder().set_processing(False)
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
            self._reset_tracking_locked(new_tracker=True)
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
                "expected_edges_mm": None, "board_direction": None,
                "trajectory": [],
                "calibration": self._calib_summary_locked(),
            })
        recorder().set_processing(True)
        return self.state()

    def stop_locate(self) -> dict:
        with self._lock:
            self._locate = False
            self._reset_tracking_locked()
            self._state.update({
                "locate_enabled": False, "valid": False,
                "status": "detecting" if self._detect else "stopped",
                "reason": "三维定位已停止",
                "leds": [], "matches": [], "raw_center": None,
                "filtered_center": None, "velocity": None,
                "edges_mm": None, "edge_errors_mm": None,
                "expected_edges_mm": None, "board_direction": None,
                "trajectory": [],
            })
        recorder().set_processing(bool(self._detect))
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
        """由新的同步帧组驱动；不做固定 FPS 限频，也不重复处理旧帧。"""
        while not stop_event.is_set():
            with self._lock:
                detect, locate = self._detect, self._locate
            if not detect and not locate:
                stop_event.set()
                break
            try:
                recs = sync_service().recent_groups(1)
                if not recs:
                    self._set_status("waiting_sync", "还没有同步帧组")
                    time.sleep(0.01)
                    continue
                rec = recs[0]
                now = time.perf_counter()
                age = now - rec.created_mono
                with self._lock:
                    same = (self._last_pair_seq == rec.group_seq)
                    force = self._force_reprocess
                    if (not same and self._last_pair_seq is not None
                            and rec.group_seq > self._last_pair_seq + 1):
                        # 处理能力不足时明确统计跳过的帧组，而不是积压历史帧
                        self._groups_skipped += (rec.group_seq
                                                 - self._last_pair_seq - 1)
                    self._groups_seen = max(self._groups_seen, rec.group_seq)
                if same and not force:
                    if age > MAX_PAIR_AGE_S:
                        self._set_status(
                            "sync_stale",
                            "最近同步帧组已过期 %.0f ms" % (age * 1000))
                    time.sleep(0.004)
                    continue
                # force 对应“刚启动定位/刚改参数”，允许对同一最新帧组重算一次
                if not force and age > MAX_PAIR_AGE_S:
                    self._set_status("sync_stale",
                                     "最近同步帧组已过期 %.0f ms" % (age * 1000))
                    time.sleep(0.01)
                    continue
                self._process(rec, now)
                with self._lock:
                    self._last_pair_seq = rec.group_seq
                    self._force_reprocess = False
                    self._last_process_mono = now
            except Exception as exc:
                self._last_error = str(exc).strip().splitlines()[-1]
                self._set_status("error", self._last_error)
                time.sleep(0.05)

    def _process(self, rec, now: float) -> None:
        t_start = time.perf_counter()
        with self._lock:
            locate = self._locate
            calib = self._calib
        cams = [int(c) for c in rec.cameras]
        params = {c: self.effective_params(c) for c in cams}

        dets, raws = {}, {}
        t_det = 0.0
        for c in cams:
            frame = rec.frames[c]
            t0 = time.perf_counter()
            dets[c] = detect_leds(frame.gray, params[c])
            t_det += time.perf_counter() - t0
            raws[c] = np.ascontiguousarray(frame.gray).copy()

        stage_ms = {"detect_ms": round(t_det * 1000.0, 2)}
        if not locate:
            self._store_result(rec, dets, raws, t_start,
                               status="detecting", reason="",
                               locate=False, calib=calib, stage_ms=stage_ms)
            return

        # 标定可能在运行期间被重新载入/修改；每秒检查一次签名。
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
                    raise ValueError(self._calib_warning or "没有可用的标定参数")
            except Exception as exc:
                with self._lock:
                    self._locate = False
                    self._tracker.reset()
                    self._tracker_state = "idle"
                    self._trajectory = []
                self._store_result(
                    rec, dets, raws, t_start,
                    status="calibration_invalid",
                    reason=str(exc).strip().splitlines()[-1],
                    locate=False, calib=None, stage_ms=stage_ms)
                return

        # 每台相机用自己的图像尺寸校验；不匹配的相机不参与定位，
        # 绝不用别的相机的内参/外参顶替。
        usable = []
        size_problems = {}
        for c in cams:
            entry = (calib.get("per_camera") or {}).get(c)
            if not entry:
                size_problems[c] = "标定里没有相机 %d" % c
                continue
            want = [int(v) for v in entry["image_size"]]
            got = [int(raws[c].shape[1]), int(raws[c].shape[0])]
            if got != want:
                size_problems[c] = ("相机 %d 当前分辨率 %s 与内参 %s 不一致"
                                    % (c, got, want))
                continue
            usable.append(c)
        if len(usable) < 2:
            with self._lock:
                self._locate = False
                self._tracker.reset()
                self._tracker_state = "idle"
                self._trajectory = []
            self._store_result(
                rec, dets, raws, t_start, status="calibration_invalid",
                reason="；".join(size_problems.values())
                or "可用于定位的相机不足 2 台",
                locate=False, calib=calib, stage_ms=stage_ms)
            return

        with self._lock:
            pred_center = None
            if (self._tracker_state == "predicted"
                    and self._state.get("filtered_center")):
                pred_center = self._state["filtered_center"]
            prev_points = self._last_led_points

        t0 = time.perf_counter()
        match = match_and_triangulate(
            {c: dets[c] for c in usable}, calib, params[usable[0]],
            previous_points=prev_points, predicted_center=pred_center)
        stage_ms["match_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
        with self._lock:
            self._last_diag = dict(match.get("diagnostics") or {},
                                   cameras=usable)

        if match.get("ok"):
            stamp = rec.created_mono
            t0 = time.perf_counter()
            upd = self._tracker.update(match["raw_center"], stamp)
            self._tracker_state = upd["state"]
            stage_ms["track_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
            with self._lock:
                self._last_led_points = [l["xyz"] for l in match["leds"]]
                self._trajectory.append(upd["position"])
                if len(self._trajectory) > MAX_TRAJECTORY:
                    self._trajectory = self._trajectory[-MAX_TRAJECTORY:]
            self._store_result(
                rec, dets, raws, t_start,
                status="tracking", reason="",
                locate=True, calib=calib, match=match,
                filtered_center=upd["position"], velocity=upd["velocity"],
                tracker={"state": upd["state"], "missed": upd["missed"],
                         "reacquired": upd.get("reacquired", False)},
                stage_ms=stage_ms)
        else:
            pred = self._tracker.predict(rec.created_mono)
            self._tracker_state = pred["state"]
            if pred["state"] == "predicted":
                with self._lock:
                    self._trajectory.append(pred["position"])
                    if len(self._trajectory) > MAX_TRAJECTORY:
                        self._trajectory = self._trajectory[-MAX_TRAJECTORY:]
                self._store_result(
                    rec, dets, raws, t_start,
                    status="predicted",
                    reason=match.get("reason") or "当前帧组没有有效三点测量",
                    locate=True, calib=calib, match=match,
                    filtered_center=pred["position"], velocity=pred["velocity"],
                    tracker={"state": "predicted", "missed": pred["missed"],
                             "age_s": pred.get("age_s")},
                    stage_ms=stage_ms)
            else:
                with self._lock:
                    self._trajectory = []
                    self._last_led_points = None
                self._store_result(
                    rec, dets, raws, t_start,
                    status="lost",
                    reason=match.get("reason") or "跟踪已丢失",
                    locate=True, calib=calib, match=match,
                    filtered_center=None, velocity=None,
                    tracker={"state": "lost"},
                    stage_ms=stage_ms)

    def _store_result(self, rec, dets, raws, t0, *,
                      status: str, reason: str, locate: bool,
                      calib: dict | None, match: dict | None = None,
                      filtered_center=None, velocity=None, tracker=None,
                      stage_ms: dict | None = None):
        # 停止请求发出后，正在处理的旧结果不能再覆盖“已停止”状态
        if locate and not self._locate:
            return
        if not locate and not self._detect:
            return
        duration_ms = (time.perf_counter() - t0) * 1000.0
        with self._lock:
            if self._detect:
                self._raw = {c: raws.get(c) for c in CAMS}
                self._det_internal = {c: dets.get(c) for c in CAMS}
            self._processed_pairs += 1
            self._processing_ms = (duration_ms if self._processing_ms is None
                                   else 0.8 * self._processing_ms + 0.2 * duration_ms)
            self._processing_stage_ms = dict(stage_ms or {})
            m = match or {}
            tracking = status == "tracking"
            shown = status in ("tracking", "predicted")
            cams = [int(c) for c in rec.cameras]
            state = {
                "ok": True,
                "detect_enabled": bool(self._detect),
                "locate_enabled": bool(self._locate),
                "status": status,
                "valid": status == "tracking",
                "reason": reason,
                "pair": {
                    # 兼容旧字段名：pair 现在表示一个 2..N 相机帧组
                    "pair_seq": int(rec.group_seq),
                    "group_seq": int(rec.group_seq),
                    "dt_ms": round(float(rec.span_ms), 3),
                    "span_ms": round(float(rec.span_ms), 3),
                    "cameras": cams,
                    "frames": {
                        str(c): {
                            "seq": int(rec.frames[c].seq),
                            "host_ts": round(float(rec.frames[c].host_ts), 6),
                            "mono_ts": round(float(rec.frames[c].mono_ts), 6),
                            "mf_ts": int(rec.frames[c].mf_ts),
                            "width": int(rec.frames[c].gray.shape[1]),
                            "height": int(rec.frames[c].gray.shape[0]),
                        } for c in cams},
                    "created_mono": round(float(rec.created_mono), 6),
                },
                "detections": {str(c): _slim_detection(dets.get(c))
                               for c in cams},
                "matches": list(m.get("matches") or []),
                "leds": list(m.get("leds") or []) if shown else [],
                "cameras_used": list(m.get("cameras_used") or []),
                "raw_center": m.get("raw_center") if tracking else None,
                "filtered_center": filtered_center,
                "velocity": velocity,
                "edges_mm": m.get("edges_mm") if tracking else None,
                "edge_errors_mm": m.get("edge_errors_mm") if tracking else None,
                "expected_edges_mm": m.get("expected_edges_mm"),
                "board_direction": m.get("board_direction") if tracking else None,
                "label_rule": m.get("label_rule"),
                "mean_reprojection_px": m.get("mean_reprojection_px"),
                "mean_epipolar_px": m.get("mean_epipolar_px"),
                "triangle_score": m.get("triangle_score"),
                "orientation_consistent": m.get("orientation_consistent"),
                "triangle_tol_mm": m.get("triangle_tol_mm"),
                "labels_are_temporary": True,
                "label_assignment": m.get("label_assignment"),
                "tracking": tracker or {"state": self._tracker_state},
                "trajectory": [list(p) for p in self._trajectory],
                "calibration": self._calib_summary_locked(calib),
                "diagnostics": _compact_diag(m.get("diagnostics")),
                "params": self._params_state_locked(),
                "processed_pairs": self._processed_pairs,
                "groups_seen": self._groups_seen,
                "groups_skipped": self._groups_skipped,
                "processing_ms": round(self._processing_ms or 0.0, 2),
                "stage_ms": dict(self._processing_stage_ms),
                "timestamp": _now_iso(),
                "last_error": self._last_error,
            }
            self._state = state
        # 连续录制只登记这一帧组已经完成的检测/定位结果，不重算任何东西；
        # note_result 内部不写盘，实际 PNG/JSONL 由录制写盘线程完成。
        try:
            recorder().note_result(rec.group_seq, state, m.get("diagnostics"))
        except Exception:
            pass

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
            "cameras": calib.get("cameras"),
            "image_sizes": calib.get("image_sizes"),
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
            "matches": [], "leds": [], "raw_center": None,
            "filtered_center": None, "velocity": None,
            "edges_mm": None, "edge_errors_mm": None, "expected_edges_mm": None,
            "board_direction": None,
            "mean_reprojection_px": None, "mean_epipolar_px": None,
            "triangle_score": None, "orientation_consistent": None,
            "triangle_tol_mm": None, "labels_are_temporary": True,
            "label_rule": "L0=两长边公共顶点；L1/L2=短边端点",
            "label_assignment": None,
            "tracking": {"state": "idle"},
            "trajectory": [],
            "cameras_used": [],
            "diagnostics": None,
            "calibration": self._calib_summary_locked(),
            "params": self._params_state_locked(),
            "processed_pairs": 0, "processing_ms": 0.0,
            "groups_seen": 0, "groups_skipped": 0, "stage_ms": {},
            "timestamp": _now_iso(), "last_error": "",
        }

    def state(self) -> dict:
        with self._lock:
            out = copy.deepcopy(self._state)
        try:
            out["recording"] = recorder().status()
        except Exception as exc:
            out["recording"] = {
                "ok": False, "recording": False, "status": "error",
                "error": str(exc).strip().splitlines()[-1],
            }
        return out

    def calibration_state(self) -> dict:
        with self._lock:
            return copy.deepcopy(self._calib_summary_locked())

    @staticmethod
    def _diagnose_snapshot(snap: dict) -> dict:
        """用同一快照的检测/参数/标定重新计算完整诊断。"""
        calib = snap.get("calib")
        dets = snap.get("det") or {}
        params = snap.get("params") or {}
        cams = [c for c in (calib or {}).get("cameras") or [] if dets.get(c)]
        if calib and len(cams) >= 2:
            m = match_and_triangulate(
                {c: dets[c] for c in cams}, calib,
                params.get(cams[0]) or default_params())
            return {"cameras": cams, "diagnostics": m.get("diagnostics")}
        return {"cameras": cams, "diagnostics": None}

    def diagnose(self) -> dict:
        """返回当前帧组的完整分阶段诊断（所有相机对）。"""
        def payload(state, diag):
            return {
                "ok": True,
                "pair": state.get("pair"),
                "status": state.get("status"),
                "reason": state.get("reason"),
                "cameras": (diag or {}).get("cameras")
                if diag else state.get("pair", {}).get("cameras"),
                "diagnostics": (diag or {}).get("diagnostics") if diag else None,
                "params": state.get("params"),
                "calibration": state.get("calibration"),
            }

        try:
            snap = self._overlay_snapshot(CAMS)
        except Exception:
            with self._lock:
                state = copy.deepcopy(self._state)
                diag = copy.deepcopy(self._last_diag)
            return payload(state, {"cameras": (diag or {}).get("cameras"),
                                   "diagnostics": diag})
        with self._lock:
            diag = self._diagnose_snapshot(snap)
            state = copy.deepcopy(snap.get("state") or {})
        return payload(state, diag)

    # ------------------------------------------------------------------
    # 连续调试录制
    # ------------------------------------------------------------------
    def start_recording(self) -> dict:
        """开始复用同步帧对的连续录制；不新开相机、不重跑 LED 定位。"""
        self._check_cameras()
        with self._lock:
            session = {
                "created": _now_iso(),
                "processing": bool(self._detect or self._locate),
                "led": {
                    "detect_enabled": bool(self._detect),
                    "locate_enabled": bool(self._locate),
                    "status": self._state.get("status"),
                },
                "params": self._params_state_locked(),
                "calibration": self._calib_summary_locked(),
                "cameras": self._active_cameras(),
                "camera_count": len(self._active_cameras()),
            }
            try:
                session["sync_max_dt_ms"] = sync_service().max_dt_ms
            except Exception:
                pass
        try:
            session["cameras"] = manager().status().get("cameras") or []
        except Exception:
            session["cameras"] = []
        root = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(self._params_path))),
            "led_recordings")
        return recorder().start(session, root)

    def stop_recording(self) -> dict:
        return recorder().stop()

    def recording_state(self) -> dict:
        return recorder().status()

    # ------------------------------------------------------------------
    # 实机调试保存
    # ------------------------------------------------------------------
    def save_debug(self) -> dict:
        """保存当前同步帧组每台相机的原始灰度、二值图、叠加图和 JSON。"""
        cams = self._active_cameras()
        snap = self._overlay_snapshot(cams)
        raw = snap.get("raw") or {}
        dets_full = snap.get("det") or {}
        state = snap.get("state") or {}
        present = [c for c in cams if raw.get(c) is not None]
        if len(present) < 2:
            raise RuntimeError("还没有处理过同步帧组，请等待新帧后再保存")
        masks = {c: (None if (dets_full.get(c) or {}).get("mask") is None
                     else (dets_full[c]["mask"]).copy()) for c in present}
        dets = {c: copy.deepcopy(_strip_mask(dets_full.get(c)))
                for c in present}
        diagnosis = self._diagnose_snapshot(snap)
        diagnosis.update({"pair": state.get("pair"),
                          "status": state.get("status"),
                          "reason": state.get("reason"),
                          "cameras": present})

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # 跟着当前参数文件走，测试/自定义 MOCAP_CALIB_DIR 时不会写回默认目录
        debug_root = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(self._params_path))),
            "led_debug")
        outdir = os.path.join(debug_root, stamp)
        os.makedirs(outdir, exist_ok=True)
        files = []
        for c in present:
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
        for c in present:
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
            "cameras": present,
            "detections": {str(c): dets[c] for c in present},
            "params": state.get("params"),
            "pair": state.get("pair"),
            "calibration": state.get("calibration"),
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
                "matches": list(self._state.get("matches") or []),
                "state": copy.deepcopy(self._state),
            }

    def overlay(self, camera: int, *, view: str = "detect",
                scale: float = OVERLAY_SCALE, quality: int = 75,
                zoom: float = 1.0, snapshot: dict | None = None,
                pair=None) -> tuple[bytes, dict]:
        cam = int(camera)
        if cam not in CAMS:
            raise ValueError("camera 必须是 0..3")
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
            idx = (m.get("obs") or {}).get(str(cam))
            if isinstance(idx, int):
                matched_map[int(idx)] = int(m["led"])
        if raw is None or det is None:
            raise RuntimeError("还没有处理过同步帧对，请等待新帧")
        if params is None:
            params = self.effective_params(cam)
        scale = float(np.clip(scale, 0.25, 1.0))
        q = int(np.clip(quality, 40, 95))
        zoom = float(np.clip(zoom, 1.0, 4.0))
        view = str(view or "detect")
        if view not in ("detect", "binary", "raw", "epipolar", "undistort"):
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
        elif view in ("epipolar", "undistort"):
            if not calib:
                raise RuntimeError("去畸变/极线视图需要先载入内参")
            entry = (calib.get("per_camera") or {}).get(cam) or {}
            K = np.asarray(entry.get("K"), np.float64)
            D = np.asarray(entry.get("D"), np.float64).reshape(-1)
            if K.shape != (3, 3):
                raise RuntimeError("相机 %d 没有可用内参" % cam)
            canvas = cv2.undistort(raw, K, D)
            canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
        else:
            canvas = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)

        H, W = raw.shape[:2]
        accepted = det.get("leds") or []
        raw_comps = det.get("raw_components") or []
        epi_info = None

        def pt(p):
            return (int(round(float(p[0]))), int(round(float(p[1]))))

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
            per = calib.get("per_camera") or {}
            cams = [int(c) for c in calib.get("cameras") or []]
            other_cam = None
            if pair and int(pair[0]) == cam:
                other_cam = int(pair[1])
            elif pair and int(pair[1]) == cam:
                other_cam = int(pair[0])
            if other_cam is None or other_cam not in per:
                other_cam = next((c for c in cams if c != cam), None)
            if other_cam is None:
                raise RuntimeError("极线视图需要至少两台已标定相机")
            Rab, Tab = relative_pose(
                np.asarray(per[other_cam]["R"], np.float64),
                np.asarray(per[other_cam]["t"], np.float64).reshape(3),
                np.asarray(per[cam]["R"], np.float64),
                np.asarray(per[cam]["t"], np.float64).reshape(3))
            F = fundamental_from_pose(
                np.asarray(per[other_cam]["K"], np.float64),
                np.asarray(per[cam]["K"], np.float64), Rab, Tab)
            tol = float(params["epipolar_tol_px"])
            K_other = np.asarray(per[other_cam]["K"], np.float64)
            D_other = np.asarray(per[other_cam]["D"], np.float64)

            def ideal_of(items, k, d):
                if not items:
                    return np.empty((0, 2), np.float64)
                arr = np.array([[item["x"], item["y"]] for item in items],
                               np.float64).reshape(-1, 2)
                return ideal_pixels(arr, k, d)

            here = ideal_of(accepted, K, D)
            other_items = ((snapshot.get("det") or {}).get(other_cam) or {}).get(
                "leds") or []
            there = ideal_of(other_items, K_other, D_other)

            # 每条极线使用不同颜色；来源候选点在其本相机也用同一颜色。
            lines = []
            for j, op in enumerate(there):
                line = (F.T @ np.array([op[0], op[1], 1.0]) if cam == 0
                        else F @ np.array([op[0], op[1], 1.0]))
                a, b, c = float(line[0]), float(line[1]), float(line[2])
                norm = float(np.hypot(a, b))
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

        # 局部放大：以匹配点为中心裁剪，再缩放到显示尺寸
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
        line2 = "status=%s" % det.get("status", "?")
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
        }

    def overlay_group(self, *, view: str = "detect",
                      scale: float = OVERLAY_SCALE, quality: int = 75,
                      zoom: float = 1.0, pair=None,
                      cameras=None) -> dict:
        """所有相机用**同一帧组快照**渲染叠加图，避免前端分别刷新时
        出现不同帧组混在一起的显示错位。"""
        with self._lock:
            cams = [int(c) for c in (cameras or self._active_cameras())]
        snap = self._overlay_snapshot(cams)
        images, infos, errors = {}, {}, {}
        for c in cams:
            try:
                data, info = self.overlay(c, view=view, scale=scale,
                                          quality=quality, zoom=zoom,
                                          snapshot=snap, pair=pair)
                images[str(c)] = base64.b64encode(data).decode("ascii")
                infos[str(c)] = info
            except Exception as exc:
                errors[str(c)] = str(exc).strip().splitlines()[-1]
        return {
            "ok": True,
            "pair": snap.get("state", {}).get("pair"),
            "status": snap.get("state", {}).get("status"),
            "cameras": cams,
            "images": images,
            "infos": infos,
            "errors": errors,
        }

    def overlay_pair(self, *, view: str = "detect",
                     scale: float = OVERLAY_SCALE, quality: int = 75,
                     zoom: float = 1.0) -> dict:
        """兼容入口：只渲染 0/1 两台相机。"""
        return self.overlay_group(view=view, scale=scale, quality=quality,
                                  zoom=zoom, cameras=[0, 1])


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
            d = float(np.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1]))
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
        region = out[y0:y1, x0:x1]
        dark = region.copy()
        cv2.rectangle(dark, (0, 0), (x1 - x0, y1 - y0), (0, 0, 0), -1)
        cv2.addWeighted(dark, alpha, region, 1.0 - alpha, 0, region)
    for k, (line, (w, h)) in enumerate(zip(lines, sizes)):
        y = top + line_h * k + h
        x = max(2, right - w)
        cv2.putText(out, line, (x, y), font, font_scale,
                    colors[k % len(colors)], thickness, cv2.LINE_AA)


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
