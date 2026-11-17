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
MAX_PAIR_AGE_S = 0.5
MAX_PROCESS_FPS = 25.0
MAX_TRAJECTORY = 300
OVERLAY_SCALE = 0.6
PARAMS_SCHEMA = "lowcost-mocap/led-params"
CAMS = (0, 1)


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _json_sig(obj) -> str:
    raw = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _empty_detection() -> dict:
    return {
        "count": 0, "status": "idle", "reason": "二维检测未启动",
        "leds": [], "rejected": [], "warnings": [],
        "overexposed": False, "unresolved_count": 0,
    }


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
        "status": det.get("status", "idle"),
        "reason": det.get("reason", ""),
        "leds": leds,
        "rejected": rejected,
        "warnings": list(det.get("warnings") or []),
        "overexposed": bool(det.get("overexposed")),
        "unresolved_count": int(det.get("unresolved_count", 0)),
    })
    return out


def _build_calibration(snapshot: dict) -> dict:
    """从 CalibrationService 快照提取 LED 定位需要的矩阵，并做完整性检查。"""
    intr = snapshot.get("intrinsics") or {}
    rig = snapshot.get("extrinsic")
    problems = []
    if not rig:
        problems.append("没有世界坐标系外参；请先解外参或载入 world_extrinsics.json")
    else:
        if not rig.get("orientation_confirmed"):
            problems.append("世界坐标轴方向未确认；请人工确认后重新解外参")
        cams = rig.get("cameras") or {}
        for c in CAMS:
            if str(c) not in cams:
                problems.append("外参缺少相机 %d" % c)
    for c in CAMS:
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

    K0, D0, size0 = kd(0)
    K1, D1, size1 = kd(1)
    if size0 != size1:
        raise ValueError("两台相机内参图像尺寸不一致：%s vs %s" % (size0, size1))
    cams = rig["cameras"]
    try:
        R0 = np.asarray(cams["0"]["R"], np.float64)
        t0 = np.asarray(cams["0"]["t"], np.float64).reshape(3)
        R1 = np.asarray(cams["1"]["R"], np.float64)
        t1 = np.asarray(cams["1"]["t"], np.float64).reshape(3)
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

    def update_params(self, payload: dict) -> dict:
        payload = payload or {}
        raw_global = payload.get("params", payload)
        global_params = validate_params({**self._global_params, **(raw_global or {})})
        overrides_raw = payload.get("overrides") or self._overrides
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
            self._param_version += 1
            self._force_reprocess = True
            self._params_warning = ""
        path = self._save_params()
        with self._lock:
            self._state["params"] = self._params_state_locked()
        return {"ok": True, "path": path, **self._params_state_locked()}

    def reset_params(self) -> dict:
        with self._lock:
            self._global_params = default_params()
            self._overrides = {"0": {}, "1": {}}
            self._param_version += 1
            self._force_reprocess = True
        path = self._save_params()
        with self._lock:
            self._state["params"] = self._params_state_locked()
        return {"ok": True, "path": path, **self._params_state_locked()}

    def _params_state_locked(self) -> dict:
        return {
            "defaults": default_params(),
            "params": copy.deepcopy(self._global_params),
            "overrides": copy.deepcopy(self._overrides),
            "effective": {str(c): self.effective_params(c) for c in CAMS},
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
        """读取并校验当前标定；返回 (calib, changed)。"""
        try:
            snapshot = calibration_service().localization_snapshot()
            calib = _build_calibration(snapshot)
        except Exception as exc:
            with self._lock:
                self._calib = None
                self._calib_warning = str(exc).strip().splitlines()[-1]
            raise
        old_sig = self._calib.get("signature") if self._calib else None
        changed = force or old_sig != calib["signature"]
        if changed:
            with self._lock:
                self._calib = calib
                self._calib_warning = ""
                self._tracker.reset()
                self._tracker_state = "idle"
                self._last_led_points = None
                self._trajectory = []
        return calib, changed

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
        det0 = detect_leds(fa.gray, p0)
        det1 = detect_leds(fb.gray, p1)
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
            match = match_and_triangulate(
                det0, det1, calib, p0, previous_points=prev_points,
                predicted_center=pred_center)
            if match.get("ok"):
                stamp = rec.created_mono
                upd = self._tracker.update(match["raw_center"], stamp)
                self._tracker_state = upd["state"]
                with self._lock:
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
                "triangle_tol_mm": (match or {}).get("triangle_tol_mm"),
                "labels_are_temporary": True,
                "label_assignment": (match or {}).get("label_assignment"),
                "tracking": tracker or {"state": self._tracker_state},
                "trajectory": [list(p) for p in self._trajectory],
                "calibration": self._calib_summary_locked(calib),
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
            "edges_mm": None, "edge_errors_mm": None,
            "mean_reprojection_px": None, "mean_epipolar_px": None,
            "triangle_tol_mm": None, "labels_are_temporary": True,
            "label_assignment": None,
            "tracking": {"state": "idle"},
            "trajectory": [],
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

    def overlay(self, camera: int, *, view: str = "detect",
                scale: float = OVERLAY_SCALE, quality: int = 75) -> tuple[bytes, dict]:
        cam = int(camera)
        if cam not in CAMS:
            raise ValueError("camera 必须是 0 或 1")
        with self._lock:
            if not self._detect:
                raise RuntimeError("二维检测未启动")
            raw = self._raw.get(cam)
            det = self._det_internal.get(cam)
            mask = self._mask.get(cam)
            calib = self._calib
            matches = list(self._state.get("matches") or [])
            led_labels = {m["led"]: m for m in matches}
            matched_map = {}
            for m in matches:
                idx = m["cam0_index"] if cam == 0 else m["cam1_index"]
                matched_map[int(idx)] = int(m["led"])
            params = self.effective_params(cam)
            state = copy.deepcopy(self._state)
        if raw is None or det is None:
            raise RuntimeError("还没有处理过同步帧对，请等待新帧")
        scale = float(np.clip(scale, 0.25, 1.0))
        q = int(np.clip(quality, 40, 95))
        view = str(view or "detect")
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
            k = int(params["blur_kernel"])
            proc = cv2.GaussianBlur(raw, (k, k), 0) if k > 0 else raw
            _, m = cv2.threshold(proc, int(params["threshold"]), 255,
                                 cv2.THRESH_BINARY)
            canvas = cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)
        elif view == "epipolar":
            if not calib:
                raise RuntimeError("极线视图需要先载入内参/外参")
            K = calib["K0"] if cam == 0 else calib["K1"]
            D = calib["D0"] if cam == 0 else calib["D1"]
            canvas = cv2.undistort(raw, K, D)
            canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
        else:
            canvas = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)

        # 非叠加原图仍显示中心；二值图也显示中心
        draw_contours = (view != "raw")
        out = cv2.resize(canvas, None, fx=scale, fy=scale,
                         interpolation=cv2.INTER_AREA if scale < 1
                         else cv2.INTER_LINEAR)

        def sp(pt):
            return (int(round(pt[0] * scale)), int(round(pt[1] * scale)))

        if draw_contours:
            for idx, d in enumerate(det.get("leds") or []):
                color = (0, 255, 0) if idx in matched_map else (0, 220, 255)
                if d.get("split"):
                    color = (255, 200, 0) if idx not in matched_map else color
                contour = d.get("contour")
                if contour:
                    arr = (np.asarray(contour, np.float32) * scale)
                    cv2.drawContours(out, [arr.astype(np.int32)], -1, color, 1)
            for r in det.get("rejected") or []:
                bbox = r.get("parent_bbox")
                if bbox:
                    x, y, w, h = [int(v * scale) for v in bbox]
                    cv2.rectangle(out, (x, y), (x + w, y + h), (0, 0, 255), 1)
            # 已匹配的 LED 用绿色，未匹配候选用黄色
            for idx, d in enumerate(det.get("leds") or []):
                color = (0, 255, 0) if idx in matched_map else (0, 220, 255)
                p = sp((d["x"], d["y"]))
                cv2.circle(out, p, 4, color, -1, cv2.LINE_AA)
                if idx in matched_map:
                    label = "L%d" % matched_map[idx]
                else:
                    label = "C%d" % idx
                cv2.putText(out, label, (p[0] + 5, p[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1,
                            cv2.LINE_AA)
        else:
            for idx, d in enumerate(det.get("leds") or []):
                p = sp((d["x"], d["y"]))
                color = (0, 255, 0) if idx in matched_map else (0, 220, 255)
                cv2.circle(out, p, 3, color, -1, cv2.LINE_AA)

        # 极线：在去畸变理想像素上绘制，避免把直线画在畸变图像上造成误导
        if view == "epipolar" and calib:
            K = calib["K0"] if cam == 0 else calib["K1"]
            D = calib["D0"] if cam == 0 else calib["D1"]
            F = calib["F"]
            pts = np.array([[d["x"], d["y"]] for d in det.get("leds") or []],
                           np.float64).reshape(-1, 2)
            ideal = (ideal_pixels(pts, K, D) if len(pts)
                     else np.empty((0, 2)))
            for idx, p in enumerate(ideal):
                color = (0, 255, 0) if idx in matched_map else (0, 220, 255)
                q = (int(round(p[0] * scale)), int(round(p[1] * scale)))
                cv2.circle(out, q, 4, color, -1, cv2.LINE_AA)
            for m in matches:
                other = (m["cam1_index"] if cam == 0 else m["cam0_index"])
                od = det.get("leds") or []
                # 用另一台相机的检测点构造极线
                other_pts = (state.get("detections", {}).get("1" if cam == 0 else "0", {})
                             .get("leds") or [])
                if other >= len(other_pts):
                    continue
                op = np.array([[other_pts[other]["x"], other_pts[other]["y"]]], float)
                K_other = calib["K1"] if cam == 0 else calib["K0"]
                D_other = calib["D1"] if cam == 0 else calib["D0"]
                oi = ideal_pixels(op, K_other, D_other)[0]
                line = (F.T @ np.array([oi[0], oi[1], 1.0]) if cam == 0
                        else F @ np.array([oi[0], oi[1], 1.0]))
                a, b, c = float(line[0]), float(line[1]), float(line[2])
                norm = math_hypot(a, b)
                if norm < 1e-9:
                    continue
                c /= norm
                a /= norm
                b /= norm
                h, w = out.shape[:2]
                pts_line = []
                for x in (0, w - 1):
                    y = -(a * x / scale + c) / b if abs(b) > 1e-9 else 0
                    pts_line.append((int(x), int(round(y * scale))))
                if abs(b) > 1e-9:
                    cv2.line(out, pts_line[0], pts_line[1], (200, 200, 200), 1,
                             cv2.LINE_AA)

        text = ("cam%d  candidates=%d  matched=%d" %
                (cam, len(det.get("leds") or []), len(matched_map)))
        if det.get("unresolved_count"):
            text += "  unresolved=%d" % det["unresolved_count"]
        if det.get("overexposed"):
            text += "  OVEREXPOSED"
        cv2.putText(out, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 1, cv2.LINE_AA)
        if state.get("pair"):
            p = state["pair"]
            cv2.putText(out, "pair %s  dt=%.1f ms" %
                        (p.get("pair_seq"), p.get("dt_ms") or 0.0),
                        (8, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (220, 220, 220), 1, cv2.LINE_AA)
        if det.get("reason"):
            cv2.putText(out, det["reason"][:72], (8, out.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1,
                        cv2.LINE_AA)
        ok, buf = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        if not ok:
            raise RuntimeError("JPEG 编码失败")
        return buf.tobytes(), {
            "camera": cam, "view": view, "scale": scale,
            "count": len(det.get("leds") or []),
            "matched": len(matched_map),
            "status": det.get("status"),
        }


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


__all__ = ["LedService", "service", "DEFAULT_PARAMS_PATH", "MAX_PAIR_AGE_S"]
