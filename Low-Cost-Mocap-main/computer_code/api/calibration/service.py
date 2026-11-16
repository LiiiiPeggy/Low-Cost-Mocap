"""标定流程编排：从已有采集链路取帧 → 检测 → 落盘 → 求解 → 保存/校验。

与采集链路的边界
--------------------------------------------------------------------------
* **不打开、不关闭任何相机**。所有帧都从 ``camera_manager.manager()`` 已有的
  Media Foundation worker 的 latest-frame / 历史缓存里读，COM STA 线程模型
  完全不动。因此标定全程不会出现 "第二次打开同一设备 → 0xC00D3704"。
* 双相机帧对直接复用 ``sync.SyncService`` 的配对结果（主机单调时钟最近邻），
  不再自己写一套配对逻辑。
* 采集是"点一次取一帧"的同步调用，检测耗时（1280x720 上 SB 约 100–300ms）
  会占用当前 Flask 请求线程，但**不阻塞**采集线程与推流。

坐标一致性
--------------------------------------------------------------------------
存盘的 PNG 就是 ``frame.gray`` 原样（未旋转、未缩放、未去畸变、未补边），
角点坐标是在这张 PNG 上检测出来的像素坐标，一一对应。任何显示层的缩放只
发生在 ``draw_overlay`` 的输出图上，不回流到计算。
"""

from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime

import cv2
import numpy as np

from camera_manager import manager
from sync import sync_service
from calibration import board as board_mod
from calibration import extrinsic as ext_mod
from calibration import intrinsic as intr_mod
from calibration import stereo as stereo_mod
from calibration.board import BoardSpec, to_jsonable
from calibration.dataset import Session

# 超过这个"年龄"的帧视为过期：用户点了采集，但相机可能已经停流/断线
MAX_FRAME_AGE_S = 1.5
MAX_PAIR_AGE_S = 2.0

MIN_SHARPNESS = 8.0        # Laplacian 方差，低于此基本可以断定是糊的
MIN_COVERAGE = 0.008       # 内参图门限：角点凸包占画面比例
# 固定板外参/验证的板通常放在地面、离相机更远，不能用内参的 0.8% 门限。
# 按现场要求放宽到 0.1%（仅要求能检出 88 个角点并做 PnP）；低于内参门限
# 0.8% 时仍会写覆盖率偏低的告警，提示位姿/深度精度会下降。
MIN_COVERAGE_EXTRINSIC = 0.001


def coverage_limit_for(phase: str) -> float:
    """该阶段允许的最低角点覆盖率（凸包面积 / 画面面积）。"""
    return MIN_COVERAGE if phase == "intrinsic" else MIN_COVERAGE_EXTRINSIC


def low_coverage_warning(coverage: float) -> str:
    """覆盖率低于内参门限、但达到固定板下限时的告警文本。"""
    return ("角点覆盖率 %.2f%% 偏低（内参建议 ≥%.1f%%）；单张固定板仍可求解，"
            "但位姿/深度精度会下降" % (100 * coverage, 100 * MIN_COVERAGE))

CALIB_DIR = os.path.dirname(os.path.abspath(__file__))
# computer_code 是项目数据基准目录：api/calibration/service.py 往上两级。
PROJECT_DIR = os.path.dirname(os.path.dirname(CALIB_DIR))
DEFAULT_RESULT_PATH = os.path.join(CALIB_DIR, "calibration.json")


def resolve_path(path: str) -> str:
    """把项目内相对路径解析成绝对路径。

    **不依赖进程当前工作目录**：相对路径一律以 ``computer_code/`` 为基准。
    绝对路径原样返回。这样从任何目录启动 preview_server / CLI 都指向同一份数据。
    """
    p = os.path.expanduser(str(path))
    if os.path.isabs(p):
        return os.path.normpath(p)
    return os.path.normpath(os.path.join(PROJECT_DIR, p))


def project_relative(path: str) -> str:
    """绝对路径 → 项目内相对路径（统一用 ``/``）；项目外路径保持绝对。"""
    try:
        rel = os.path.relpath(os.path.abspath(path), PROJECT_DIR)
    except ValueError:                      # Windows 不同盘符
        return os.path.abspath(path)
    if rel == ".." or rel.startswith(".." + os.sep):
        return os.path.abspath(path)
    return rel.replace(os.sep, "/")


_calib_env = os.environ.get("MOCAP_CALIB_DIR")
_capture_env = os.environ.get("MOCAP_CAPTURE_DIR")
if _calib_env:
    DEFAULT_SESSION_ROOT = resolve_path(_calib_env)
elif _capture_env:
    DEFAULT_SESSION_ROOT = resolve_path(os.path.join(_capture_env, "calibration"))
else:
    DEFAULT_SESSION_ROOT = os.path.join(PROJECT_DIR, "captures", "calibration")


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _normalize_validation_units(val: dict | None) -> dict | None:
    """旧验证结果里的汇总间距误差单位是米，读回时换算成毫米一次。

    逐位置字段（``spacing_mean_abs_error_mm`` 等）一直是正确的毫米值；只有
    ``aggregate.spacing_abs_error_mm`` 在旧版本里漏乘了 1000。新版结果带
    ``unit: "mm"`` 标记，所以不会重复换算。
    """
    if not isinstance(val, dict):
        return val
    agg = val.get("aggregate") or {}
    sp = agg.get("spacing_abs_error_mm")
    if isinstance(sp, dict) and sp.get("unit") != "mm":
        for key in ("mean", "p95", "max"):
            v = sp.get(key)
            if isinstance(v, (int, float)):
                sp[key] = round(float(v) * 1000.0, 4)
        sp["unit"] = "mm"
        val["units_migrated_from_m"] = True
    return val


def _detections_for(samples, camera: int) -> list:
    """把已落盘样本还原成 ``board.Detection``，供 coverage_map 复用。"""
    key = str(int(camera))
    out = []
    for s in samples:
        d = (s.detections or {}).get(key) or {}
        raw = d.get("corners") or []
        if not d.get("ok") or not raw:
            continue
        corners = np.asarray(raw, dtype=np.float32).reshape(-1, 1, 2)
        out.append(board_mod.Detection(
            ok=True, corners=corners, method=d.get("method", ""),
            polarity=d.get("polarity", "normal"), reason="",
            quality=d.get("quality") or {}))
    return out


class CalibrationService:
    """进程级单例：持有当前会话与最近一次标定结果。"""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._session: Session | None = None
        self._intrinsics: dict[str, dict] = {}
        self._stereo: dict | None = None
        self._extrinsic: dict | None = None
        self._validation: dict | None = None
        self._result_path: str | None = None

    # ------------------------------------------------------------------
    # 会话
    # ------------------------------------------------------------------
    @property
    def session(self) -> Session | None:
        return self._session

    @property
    def result_path(self) -> str:
        """最近一次保存的路径（项目相对，供界面展示）；没保存过则给默认路径。"""
        return project_relative(self.result_path_abs)

    @property
    def result_path_abs(self) -> str:
        """内部读写用的绝对路径。"""
        return self._result_path or DEFAULT_RESULT_PATH

    def open_session(self, *, cols: int, rows: int, square_mm: float,
                     cameras=None, session_id: str | None = None,
                     pixel_format: str = "", note: str = "",
                     surface_height_m: float = 0.005,
                     root: str | None = None) -> dict:
        spec = BoardSpec(cols=cols, rows=rows, square_mm=square_mm,
                         surface_height_m=surface_height_m)
        cams = [int(c) for c in (cameras or [0, 1])]

        size = self._probe_image_size(cams)
        sid = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        path = resolve_path(root) if root else os.path.join(DEFAULT_SESSION_ROOT, sid)

        with self._lock:
            sess = Session.create_or_load(path, spec, cams, size, pixel_format)
            self._session = sess
            # 切换会话时清掉上一轮求解结果，避免张冠李戴
            self._intrinsics, self._stereo, self._validation = {}, None, None
            self._extrinsic = None
        # 如果这个目录里已有求解结果（再次打开旧会话），直接恢复
        self._restore_results(sess)
        return self.status()

    def reload_session(self, root: str) -> dict:
        """载入会话目录，并恢复该目录里已保存的求解结果（如果有）。"""
        sess = Session.load(resolve_path(root))
        with self._lock:
            self._session = sess
            self._intrinsics, self._stereo, self._validation = {}, None, None
            self._extrinsic = None
        self._restore_results(sess)
        return self.status()

    def _restore_results(self, sess: Session) -> None:
        """把会话目录里的 result_*.json 读回内存，避免"重载后结果丢失"。"""
        def _load(name):
            path = os.path.join(sess.root, name)
            if not os.path.isfile(path):
                return None
            try:
                with open(path, encoding="utf-8") as fh:
                    return json.load(fh)
            except (OSError, ValueError):
                return None

        data = _load("result_intrinsic.json")
        if data:
            with self._lock:
                self._intrinsics = dict(data.get("intrinsics") or {})
        ext = _load("result_extrinsic.json")
        if ext:
            with self._lock:
                self._extrinsic = ext
        val = _normalize_validation_units(_load("result_validation.json"))
        if val:
            with self._lock:
                self._validation = val
        stereo = _load("result_stereo.json")
        if stereo:
            with self._lock:
                self._stereo = stereo

    def _invalidate(self, phase: str) -> None:
        """新增样本后，作废依赖它的求解结果，避免把旧结果当新结果保存。"""
        with self._lock:
            if phase == "intrinsic":
                self._intrinsics = {}
                self._extrinsic = None
                self._validation = None
            elif phase == "extrinsic":
                self._extrinsic = None
                self._validation = None
            else:
                self._validation = None

    def close_session(self) -> dict:
        with self._lock:
            self._session = None
            self._intrinsics, self._stereo, self._validation = {}, None, None
            self._extrinsic = None
        return {"session": None}

    def _probe_image_size(self, cameras) -> tuple[int, int]:
        """从未停流的相机上读一帧，确定图像尺寸；读不到就用请求参数兜底。"""
        for c in cameras:
            w = manager().worker(c)
            if w is None:
                continue
            f = w.latest()
            if f is not None:
                return int(f.gray.shape[1]), int(f.gray.shape[0])
            if w.requested:
                return int(w.requested["width"]), int(w.requested["height"])
        w = manager().worker(cameras[0]) if cameras else None
        if w is not None and w.requested:
            return int(w.requested["width"]), int(w.requested["height"])
        return (1280, 720)

    # ------------------------------------------------------------------
    # 采集
    # ------------------------------------------------------------------
    def _require_session(self) -> Session:
        if self._session is None:
            raise RuntimeError("还没有打开标定会话，请先 POST /api/calib/session")
        return self._session

    def _fresh_frame(self, camera: int, after_seq: int | None = None,
                     timeout: float = 1.5):
        """取一帧新的原始灰度图。

        给了 ``after_seq`` 就等 **seq 更新** 的帧（``CameraWorker.wait_frame``），
        这样连点两次「采集」也不会把同一帧当成两次独立观测；超时直接报错，
        而不是悄悄复用旧帧。
        """
        worker = manager().worker(camera)
        if worker is None:
            raise RuntimeError("相机 %d 未启动，无法采集（本模块不会自己打开设备）" % camera)
        if worker.state != "streaming":
            raise RuntimeError("相机 %d 当前状态是 %s，不是 streaming" % (camera, worker.state))

        if after_seq is None:
            frame = worker.latest()
        else:
            frame = worker.wait_frame(after_seq, timeout=timeout)
            if frame is None:
                raise RuntimeError(
                    "相机 %d 在 %.1f s 内没有新帧（上一张 seq=%d），"
                    "请等采集刷新后再按" % (camera, timeout, after_seq))
        if frame is None:
            raise RuntimeError("相机 %d 还没有出帧" % camera)
        age = time.perf_counter() - frame.mono_ts
        if age > MAX_FRAME_AGE_S:
            raise RuntimeError("相机 %d 的帧已过期 %.0f ms（可能已停流），拒绝采集"
                               % (camera, age * 1000))
        return frame, age

    @staticmethod
    def _last_seq(sess: Session, camera: int, phase: str) -> int | None:
        """该相机在该阶段已存样本里的最大 seq；没有则 None。"""
        last = None
        for s in sess.samples:
            if s.kind != "single" or s.phase != phase:
                continue
            meta = s.images.get(str(camera)) or {}
            if meta.get("seq") is not None:
                last = int(meta["seq"])
        return last

    @staticmethod
    def _last_pair_key(sess: Session, phase: str):
        """最近一条同步帧对的 (seq_a, seq_b)，用于判断是否真的换了新帧对。"""
        for s in reversed(sess.samples):
            if s.kind != "stereo" or s.phase != phase:
                continue
            try:
                return int(s.images["0"]["seq"]), int(s.images["1"]["seq"])
            except (KeyError, TypeError, ValueError):
                return None
        return None

    def capture_single(self, camera: int, *, force: bool = False,
                       phase: str = "intrinsic") -> dict:
        sess = self._require_session()
        spec = sess.spec
        camera = int(camera)

        frame, age = self._fresh_frame(camera, after_seq=self._last_seq(sess, camera, phase))
        gray = frame.gray
        det = board_mod.detect(gray, spec)
        overlay = board_mod.draw_overlay(gray, det, spec, label_step=4)

        meta = {
            "seq": int(frame.seq),
            "mf_ts": int(frame.mf_ts),
            "host_ts": float(frame.host_ts),
            "mono_ts": float(frame.mono_ts),
            "device": (manager().worker(camera).device_name
                       if manager().worker(camera) else None),
            "format": (manager().worker(camera).format_info
                       if manager().worker(camera) else None),
            "frame_age_ms": round(age * 1000.0, 1),
        }

        status, reason, warning = "valid", "", ""
        if not det.ok:
            status, reason = "invalid", "角点检测失败：%s" % det.reason
        else:
            q = det.quality
            sig = board_mod.pose_signature(det, spec)
            # 外参阶段板是固定的，多拍几张本来就应该几乎一样，不做判重
            dup = (None if phase == "extrinsic"
                   else board_mod.is_duplicate(sig, sess.signatures(camera, phase=phase)))
            if q["sharpness"] < MIN_SHARPNESS:
                status, reason = "invalid", ("图像偏模糊（Laplacian 方差 %.0f < %.0f）"
                                             % (q["sharpness"], MIN_SHARPNESS))
            elif q["coverage"] < coverage_limit_for(phase):
                status, reason = "invalid", ("标定板在画面中太小（占画面 %.1f%%，"
                                             "该阶段最低 %.1f%%）"
                                             % (100 * q["coverage"],
                                                100 * coverage_limit_for(phase)))
            elif dup and not force:
                status, reason = "invalid", ("与已有样本 %s 姿态几乎相同（位移 %.3f/"
                                             "尺度 %.2f/转角 %.1f°），换一个角度再采"
                                             % (dup["sample"], dup["dpos"],
                                                dup["dscale"], dup["dangle"]))
            else:
                meta["signature"] = sig
                if q["coverage"] < MIN_COVERAGE:
                    warning = low_coverage_warning(q["coverage"])

        sample = sess.add_single(camera, gray, det, meta, overlay=overlay,
                                 status=status, reason=reason, note=warning,
                                 signature=meta.get("signature", {}), phase=phase)
        self._invalidate(phase)

        return {
            "ok": status == "valid",
            "sample": sample.id,
            "camera": camera,
            "phase": phase,
            "status": status,
            "reason": reason,
            "warning": warning,
            "detection": det.to_dict(),
            "quality": det.quality,
            "frame_age_ms": meta["frame_age_ms"],
            "stats": sess.stats(),
            "message": ("已采集 %s（相机 %d，%s）" % (sample.id, camera, phase)
                        if status == "valid" else "未计入有效样本：%s" % reason),
        }

    @staticmethod
    def _last_single_pair_seq(sess: Session, phase: str) -> int | None:
        """外参样本里记录的最大同步帧对序号（双相机同帧采集时写入）。"""
        last = None
        for s in sess.samples:
            if s.kind != "single" or s.phase != phase:
                continue
            for meta in (s.images or {}).values():
                v = meta.get("pair_seq")
                if v is not None:
                    last = int(v) if last is None else max(last, int(v))
        return last

    def _acquire_sync_pair(self, sess: Session, *, last_key=None,
                           last_pair_seq: int | None = None,
                           force: bool = False, timeout: float = 1.5):
        """从运行中的同步服务取一对**新**帧；重复帧对会被拒绝。"""
        svc = sync_service()
        cams = sess.cameras
        ca, cb = int(cams[0]), int(cams[1])
        deadline = time.perf_counter() + timeout
        while True:
            recents = svc.recent_pairs(1)
            if not recents:
                raise RuntimeError("还没有配对成功的帧对：确认两台相机都在 streaming，"
                                   "且 /api/sync/status 的 pairs_total 在增长")
            rec = recents[0]
            same = ((last_key is not None
                     and last_key == (rec.frame_a.seq, rec.frame_b.seq))
                    or (last_pair_seq is not None and last_pair_seq == rec.pair_seq))
            if force or not same:
                break
            if time.perf_counter() > deadline:
                raise RuntimeError(
                    "最近一对同步帧（seq=%d/%d）已经采过，%.1f s 内没有新帧对，"
                    "请稍后再按" % (rec.frame_a.seq, rec.frame_b.seq, timeout))
            time.sleep(0.05)

        age = time.perf_counter() - rec.created_mono
        if age > MAX_PAIR_AGE_S:
            raise RuntimeError("最近一对同步帧已过期 %.0f ms，请重新触发（相机是否停流？）"
                               % (age * 1000))
        if abs(rec.dt_ms) > svc.max_dt_ms + 1e-6:
            raise RuntimeError("帧对时间差 |Δt|=%.1f ms 超过阈值 %.0f ms"
                               % (abs(rec.dt_ms), svc.max_dt_ms))

        fa, fb = rec.frame_a, rec.frame_b
        for cam, frame in ((ca, fa), (cb, fb)):
            if (frame.gray.shape[1], frame.gray.shape[0]) != tuple(sess.image_size):
                raise RuntimeError("相机 %d 当前分辨率与会话登记的不一致（%s vs %s）"
                                   % (cam, frame.gray.shape[:2][::-1],
                                      tuple(sess.image_size)))
        return rec, age

    def capture_stereo(self, *, force: bool = False, phase: str = "validation") -> dict:
        """采一对同步帧。默认写入 validation 阶段；外参用 capture_extrinsic_pair。"""
        sess = self._require_session()
        spec = sess.spec
        cams = sess.cameras
        if len(cams) < 2:
            raise RuntimeError("会话只登记了 %d 台相机，无法做双相机采集" % len(cams))
        ca, cb = int(cams[0]), int(cams[1])

        rec, age = self._acquire_sync_pair(
            sess, last_key=self._last_pair_key(sess, phase), force=force)
        fa, fb = rec.frame_a, rec.frame_b

        deta, detb, info = board_mod.detect_pair(fa.gray, fb.gray, spec)
        overlay = {str(ca): board_mod.draw_overlay(fa.gray, deta, spec, label_step=3),
                   str(cb): board_mod.draw_overlay(fb.gray, detb, spec, label_step=3)}

        status, reason, warning = "valid", "", ""
        signature = {}
        if not (deta.ok and detb.ok):
            status = "invalid"
            reason = "双相机未能同时检测到完整棋盘格：%s / %s" % (
                deta.reason or "OK", detb.reason or "OK")
        elif not info.get("ordering_consistent", True):
            status, reason = "invalid", "两视角角点手性相反，拒绝入库（请检查是否拍到了板子背面）"
        else:
            signature = board_mod.pose_signature(deta, spec)
            dup = board_mod.is_duplicate(signature, sess.signatures(ca, kind="stereo"))
            qa, qb = deta.quality, detb.quality
            min_cov = min(qa["coverage"], qb["coverage"])
            if min(qa["sharpness"], qb["sharpness"]) < MIN_SHARPNESS:
                status, reason = "invalid", "至少一路图像偏模糊"
            elif min_cov < MIN_COVERAGE_EXTRINSIC:
                status, reason = "invalid", ("至少一路标定板太小（最低 %.1f%%，"
                                             "外参/验证最低 %.1f%%）"
                                             % (100 * min_cov,
                                                100 * MIN_COVERAGE_EXTRINSIC))
            elif dup and not force:
                status, reason = "invalid", ("与已有帧对 %s 姿态几乎相同，换一个角度再采"
                                             % dup["sample"])
            elif min_cov < MIN_COVERAGE:
                warning = low_coverage_warning(min_cov)

        pair_meta = {
            str(ca): self._side_meta(fa, ca),
            str(cb): self._side_meta(fb, cb),
        }
        sample = sess.add_stereo(
            pair_meta, {ca: fa.gray, cb: fb.gray}, {ca: deta, cb: detb},
            dt_ms=rec.dt_ms, overlay=overlay, status=status, reason=reason,
            signature=signature, phase=phase, note=warning)
        self._invalidate(phase)

        return {
            "ok": status == "valid",
            "sample": sample.id,
            "status": status,
            "reason": reason,
            "warning": warning,
            "dt_ms": round(rec.dt_ms, 3),
            "pair_seq": rec.pair_seq,
            "detections": {str(ca): deta.to_dict(), str(cb): detb.to_dict()},
            "ordering": info,
            "frame_age_ms": round(age * 1000.0, 1),
            "stats": sess.stats(),
            "message": ("已采集帧对 %s（Δt=%.1f ms）" % (sample.id, rec.dt_ms)
                        if status == "valid" else "未计入有效帧对：%s" % reason),
            "caveat": "帧对来自软件级时间戳配对，非硬件曝光同步；仅适用于静止标定板",
        }

    # ------------------------------------------------------------------
    # 世界系外参：固定棋盘格 + solvePnP
    # ------------------------------------------------------------------
    def capture_extrinsic(self, camera: int) -> dict:
        """外参阶段单相机采集（CLI/旧接口保留；网页已改为双相机同帧采集）。"""
        return self.capture_single(camera, force=True, phase="extrinsic")

    def capture_extrinsic_pair(self, *, force: bool = False) -> dict:
        """一次采集两台相机的**同一对同步帧**，分别存进各自的 extrinsic 阶段。

        两台相机的检测质量、有效状态分别判断；一张有效图就足够 ``solvePnP``。
        复用 ``SyncService``，不会重新打开相机；同一帧对被拒绝重复采集。
        """
        sess = self._require_session()
        spec = sess.spec
        cams = sess.cameras
        if len(cams) < 2:
            raise RuntimeError("会话只登记了 %d 台相机，无法同时采集两台" % len(cams))
        ca, cb = int(cams[0]), int(cams[1])

        rec, age = self._acquire_sync_pair(
            sess, last_pair_seq=self._last_single_pair_seq(sess, "extrinsic"),
            force=force)
        frames = {ca: rec.frame_a, cb: rec.frame_b}
        results = {}

        for cam in (ca, cb):
            frame = frames[cam]
            det = board_mod.detect(frame.gray, spec)
            overlay = board_mod.draw_overlay(frame.gray, det, spec, label_step=3)
            status, reason, warning = "valid", "", ""
            if not det.ok:
                status, reason = "invalid", "角点检测失败：%s" % det.reason
            elif det.quality["sharpness"] < MIN_SHARPNESS:
                status, reason = "invalid", ("图像偏模糊（Laplacian 方差 %.0f < %.0f）"
                                             % (det.quality["sharpness"], MIN_SHARPNESS))
            elif det.quality["coverage"] < MIN_COVERAGE_EXTRINSIC:
                status, reason = "invalid", ("标定板在画面中太小（占画面 %.1f%%，"
                                             "外参最低 %.1f%%）"
                                             % (100 * det.quality["coverage"],
                                                100 * MIN_COVERAGE_EXTRINSIC))
            elif det.quality["coverage"] < MIN_COVERAGE:
                warning = low_coverage_warning(det.quality["coverage"])

            meta = self._side_meta(frame, cam)
            meta["pair_seq"] = int(rec.pair_seq)
            meta["dt_ms"] = round(float(rec.dt_ms), 3)
            sample = sess.add_single(
                cam, frame.gray, det, meta, overlay=overlay,
                status=status, reason=reason, note=warning,
                signature=(board_mod.pose_signature(det, spec) if det.ok else {}),
                phase="extrinsic")
            results[str(cam)] = {
                "ok": status == "valid",
                "status": status,
                "reason": reason,
                "warning": warning,
                "sample": sample.id,
                "detection": det.to_dict(),
                "quality": det.quality,
                "frames": sess.stats()["by_phase"]["extrinsic"].get(str(cam), 0),
            }
        self._invalidate("extrinsic")

        ok = all(v["ok"] for v in results.values())
        summary = "，".join("相机 %s %s" % (k, "有效" if v["ok"] else "无效")
                           for k, v in results.items())
        return {
            "ok": ok,
            "message": "已采集双相机图片（%s，Δt=%.1f ms）" % (summary, rec.dt_ms),
            "pair_seq": rec.pair_seq,
            "dt_ms": round(rec.dt_ms, 3),
            "frame_age_ms": round(age * 1000.0, 1),
            "cameras": results,
            "stats": sess.stats(),
            "caveat": "帧对来自软件级时间戳配对，非硬件曝光同步；仅适用于静止标定板",
        }

    def solve_extrinsic(self, *, flip180: bool = False, flip_cam=None,
                        orientation_confirmed: bool = False, exclude=()) -> dict:
        sess = self._require_session()
        cams = sess.cameras
        if len(cams) < 2:
            raise ValueError("会话只登记了 %d 台相机" % len(cams))
        ca, cb = str(int(cams[0])), str(int(cams[1]))
        missing = [c for c in (ca, cb) if c not in self._intrinsics]
        if missing:
            raise ValueError("相机 %s 还没有内参结果，请先执行内参标定" % ", ".join(missing))

        frames, ids, low_cov = {}, {}, []
        for cam in (int(cams[0]), int(cams[1])):
            samples = sess.single_samples(cam, phase="extrinsic", exclude=exclude)
            if len(samples) < ext_mod.MIN_FRAMES:
                raise ValueError(
                    "相机 %d 的外参图像只有 %d 张，至少需要 %d 张（建议 %d 张）"
                    % (cam, len(samples), ext_mod.MIN_FRAMES, ext_mod.RECOMMENDED_FRAMES))
            frames[cam] = [s.corners(cam) for s in samples]
            ids[cam] = [s.id for s in samples]
            low_cov += [s.id for s in samples if s.note]
        rig = ext_mod.solve_rig(frames, sess.spec,
                                intr_mod.K_from(self._intrinsics[ca]),
                                intr_mod.D_from(self._intrinsics[ca]),
                                intr_mod.K_from(self._intrinsics[cb]),
                                intr_mod.D_from(self._intrinsics[cb]),
                                flip180=flip180, flip_cam=flip_cam,
                                orientation_confirmed=orientation_confirmed)
        rig["created"] = _now_iso()
        rig["camera_pair"] = [int(cams[0]), int(cams[1])]
        rig["frames"] = {str(k): v for k, v in ids.items()}
        rig["intrinsic_rms_px"] = {ca: self._intrinsics[ca].get("rms_px"),
                                   cb: self._intrinsics[cb].get("rms_px")}
        if low_cov:
            rig.setdefault("warnings", []).append(
                "外参图覆盖率偏低：%s。单张固定板仍可求解，但位姿/深度精度会下降；"
                "条件允许时请把板移近或更正对相机后重采"
                % "、".join(sorted(set(low_cov))))

        with self._lock:
            self._extrinsic = rig
        sess.save_result("result_extrinsic.json", rig)
        return rig

    def extrinsic_overlay(self, camera: int, *, flip180: bool = False,
                          scale: float = 0.6, quality: int = 72) -> tuple[bytes, dict]:
        """把世界系原点与 X/Y/Z 轴投影到当前画面，供人工确认角点对应关系。

        不需要先解外参：现场实时解一张图的位姿即可，改 ``flip180`` 后轴方向会一起
        转 180°，直接看 X 轴是否沿 12 格长边就能定。
        """
        sess = self._require_session()
        spec = sess.spec
        cam = int(camera)
        key = str(cam)
        if key not in self._intrinsics:
            raise RuntimeError("相机 %d 还没有内参结果" % cam)
        frame, age = self._fresh_frame(cam)
        det = board_mod.detect(frame.gray, spec)
        if not det.ok:
            raise RuntimeError("相机 %d 未检测到完整棋盘格：%s" % (cam, det.reason))

        K = intr_mod.K_from(self._intrinsics[key])
        D = intr_mod.D_from(self._intrinsics[key])
        pose = ext_mod.solve_pose(det.corners, spec, K, D, flip180=flip180)
        corners_used = ext_mod._apply_correspondence(det.corners, spec,
                                                    pose["correspondence"])
        img = board_mod.draw_world_overlay(frame.gray, spec, K, D,
                                           pose["R"], pose["t"],
                                           corners=corners_used, scale=scale,
                                           flipped=flip180)
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        if not ok:
            raise RuntimeError("JPEG 编码失败")
        return buf.tobytes(), {
            "camera": cam,
            "ok": True,
            "correspondence": pose["correspondence"],
            "flip180": bool(flip180),
            "camera_centre_world": pose["camera_centre_world"],
            "height_above_floor_m": pose["height_above_floor_m"],
            "reprojection_rms_px": pose["reprojection"]["rms_px"],
            "frame_age_ms": round(age * 1000.0, 1),
        }

    def validate_rig(self, *, holdout: bool = False, expected_centres=None) -> dict:
        """独立验证：用 validation 阶段的同步帧对（板挪到多个位置/高度）。"""
        sess = self._require_session()
        rig = self._extrinsic
        if not rig:
            raise ValueError("还没有外参结果，请先执行 solvePnP 外参标定")
        cams = sess.cameras
        ca, cb = str(int(cams[0])), str(int(cams[1]))
        pairs = sess.stereo_samples(phase="validation")
        if not pairs:
            raise ValueError("没有验证帧对：请把板挪到其它位置/高度，用"
                             "「采集并验证三维精度」逐点采集，且不要把验证用的图混进外参标定")

        if holdout and len(pairs) >= 6:
            pairs_sorted = sorted(pairs, key=lambda s: s.id)
            val = [s for i, s in enumerate(pairs_sorted) if i % 3 == 2]
            pairs = val or pairs_sorted

        placements = [{"id": s.id,
                       "corners0": s.corners(0),
                       "corners1": s.corners(1),
                       "dt_ms": s.dt_ms} for s in pairs]
        out = ext_mod.validate_rig(
            placements, sess.spec, rig,
            intr_mod.K_from(self._intrinsics[ca]), intr_mod.D_from(self._intrinsics[ca]),
            intr_mod.K_from(self._intrinsics[cb]), intr_mod.D_from(self._intrinsics[cb]),
            expected_centres=expected_centres)
        out["created"] = _now_iso()
        out["sample_ids"] = [p["id"] for p in placements]
        low_cov = [s.id for s in pairs if s.note]
        if low_cov:
            out.setdefault("notes", []).append(
                "验证帧对覆盖率偏低：%s；间距/平面度指标可能被噪声放大"
                % "、".join(sorted(set(low_cov))))
        with self._lock:
            self._validation = out
        sess.save_result("result_validation.json", out)
        return out

    def capture_and_validate(self, *, force: bool = False) -> dict:
        """「采集并验证三维精度」：取一对同步帧 → 入库 → 立刻用已有内外参验证。

        只有内外参都已求解、且本次采集通过质检时才自动验证。失败样本以
        ``status=invalid`` 落盘（原图保留供离线检查），不会计入有效结果。
        """
        sess = self._require_session()
        cams = sess.cameras
        if len(cams) < 2:
            raise ValueError("会话只登记了 %d 台相机，无法采集同步帧对" % len(cams))
        missing = [str(c) for c in cams[:2] if str(int(c)) not in self._intrinsics]
        if missing:
            raise ValueError("相机 %s 还没有内参结果，请先完成内参标定"
                             % ", ".join(missing))
        if not self._extrinsic:
            raise ValueError("还没有外参结果，请先采集双相机图片并执行 solvePnP 外参标定")

        capture = self.capture_stereo(force=force, phase="validation")
        out = {"ok": bool(capture.get("ok")), "capture": capture,
               "validation": None, "validation_error": None}
        if not capture.get("ok"):
            out["validation_error"] = ("本次采集未通过质检，未计入有效验证样本：%s"
                                       % (capture.get("reason") or "未知原因"))
            return out
        try:
            out["validation"] = self.validate_rig()
        except Exception as exc:
            out["ok"] = False
            out["validation_error"] = str(exc).strip().splitlines()[-1]
        return out

    def _side_meta(self, frame, camera: int) -> dict:
        w = manager().worker(camera)
        return {
            "seq": int(frame.seq),
            "mf_ts": int(frame.mf_ts),
            "host_ts": float(frame.host_ts),
            "mono_ts": float(frame.mono_ts),
            "device": w.device_name if w else None,
            "format": w.format_info if w else None,
        }

    # ------------------------------------------------------------------
    # 实况检测（只画图，不入库）
    # ------------------------------------------------------------------
    def detect_live(self, camera: int, *, scale: float = 0.6,
                    quality: int = 70) -> tuple[bytes, dict]:
        frame, age = self._fresh_frame(camera)
        sess = self._session
        spec = sess.spec if sess else BoardSpec()
        t0 = time.perf_counter()
        # 实时叠加只做显示，用快速检测（SB 默认标志）；真正入库的采集走
        # EXHAUSTIVE|ACCURACY，两者不共用同一个质量等级是有意为之。
        det = board_mod.detect(frame.gray, spec, fast=True)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        img = board_mod.draw_overlay(frame.gray, det, spec, scale=scale, label_step=3)
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        if not ok:
            raise RuntimeError("JPEG 编码失败")
        return buf.tobytes(), {
            "ok": det.ok,
            "method": det.method,
            "quality": det.quality,
            "reason": det.reason,
            "detect_ms": round(dt_ms, 1),
            "frame_age_ms": round(age * 1000.0, 1),
            "seq": int(frame.seq),
        }

    # ------------------------------------------------------------------
    # 求解
    # ------------------------------------------------------------------
    def run_intrinsic(self, cameras=None, *, exclude=()) -> dict:
        sess = self._require_session()
        spec = sess.spec
        size = tuple(sess.image_size)
        targets = [int(c) for c in (cameras if isinstance(cameras, (list, tuple)) else
                                    ([cameras] if cameras is not None else sess.cameras))]

        results, errors = {}, {}
        for cam in targets:
            samples = sess.single_samples(cam, exclude=exclude, phase="intrinsic")
            try:
                res = intr_mod.calibrate(samples, spec, size, camera=cam)
                res["camera"] = cam
                res["created"] = _now_iso()
                # 区域覆盖率：复用 board.coverage_map 的 6×6 网格统计，只统计
                # 本相机 phase="intrinsic" 的有效样本（samples 已按该阶段筛选）。
                cov = board_mod.coverage_map(_detections_for(samples, cam), size, spec)
                res["coverage"] = {k: cov[k] for k in
                                   ("cells", "covered_ratio", "corner_hits", "suggestion")}
                results[str(cam)] = res
            except (ValueError, cv2.error) as exc:
                errors[str(cam)] = str(exc).strip().splitlines()[-1]

        if not results:
            raise ValueError("内参标定失败：" + "；".join(
                "%s: %s" % (k, v) for k, v in errors.items()))

        with self._lock:
            self._intrinsics.update(results)
        sess.save_result("result_intrinsic.json", {"intrinsics": self._intrinsics,
                                                  "errors": errors,
                                                  "created": _now_iso()})
        return {"intrinsics": results, "errors": errors,
                "saved_to": os.path.join(sess.root, "result_intrinsic.json")}

    def run_stereo(self, *, exclude=(), auto_flip: bool = True,
                   flip_b: bool | None = None) -> dict:
        sess = self._require_session()
        cams = sess.cameras
        if len(cams) < 2:
            raise ValueError("会话只登记了 %d 台相机" % len(cams))
        ca, cb = str(int(cams[0])), str(int(cams[1]))
        missing = [c for c in (ca, cb) if c not in self._intrinsics]
        if missing:
            raise ValueError("相机 %s 还没有内参结果，请先执行内参标定"
                             % ", ".join(missing))

        samples = sess.stereo_samples(exclude=exclude)
        res = stereo_mod.calibrate(
            samples, sess.spec, tuple(sess.image_size),
            self._intrinsics[ca], self._intrinsics[cb],
            auto_flip=auto_flip, flip_b=flip_b)
        res["created"] = _now_iso()
        res["camera_pair"] = [int(cams[0]), int(cams[1])]
        res["intrinsic_source"] = {"camera_0": self._intrinsics[ca].get("rms_px"),
                                   "camera_1": self._intrinsics[cb].get("rms_px")}

        with self._lock:
            self._stereo = res
        sess.save_result("result_stereo.json", res)
        return res

    # ------------------------------------------------------------------
    # 结果保存 / 加载
    # ------------------------------------------------------------------
    def build_result(self) -> dict:
        sess = self._require_session()
        cams = sess.cameras
        ca, cb = str(int(cams[0])), str(int(cams[1]))
        if ca not in self._intrinsics or cb not in self._intrinsics:
            raise ValueError("内参不完整：缺少相机 %s 的结果"
                             % ", ".join([c for c in (ca, cb) if c not in self._intrinsics]))
        if not self._extrinsic and not self._stereo:
            raise ValueError("还没有外参结果，请先执行「解外参（solvePnP）」")

        workers = {c: manager().worker(c) for c in cams}
        spec = sess.spec
        ext = None
        if self._extrinsic:
            rig = self._extrinsic
            cams_out = {}
            for key in ("0", "1"):
                p = rig["cameras"][key]
                cw = p["cam_to_world"]
                cams_out[key] = {
                    "index": int(key),
                    # 世界 → 相机：X_cam = R @ X_world + t
                    "R_world_to_cam": p["R"],
                    "t_world_to_cam_m": p["t"],
                    # 相机 → 世界：X_world = R_wc @ X_cam + t_wc
                    "R_cam_to_world": cw["R_wc"],
                    "t_cam_to_world_m": cw["t_wc"],
                    "camera_centre_world_m": p["camera_centre_world"],
                    "height_above_pattern_m": p["height_above_pattern_m"],
                    "height_above_floor_m": p["height_above_floor_m"],
                    "distance_to_pattern_centre_m": p["distance_to_pattern_centre_m"],
                    "optical_axis_world": p["optical_axis_world"],
                    "reprojection_rms_px": p["reprojection"]["rms_px"],
                    "reprojection_max_px": p["reprojection"]["max_px"],
                    "points_in_front": p["points_in_front"],
                    "correspondence": p["correspondence"],
                    "rotation": p["rotation"],
                    "frames": rig.get("frames", {}).get(key),
                }
            rel = rig["relative"]
            ext = {
                "method": rig["method"],
                "camera_pair": rig.get("camera_pair", [int(c) for c in cams]),
                "world_frame": spec.world_frame(),
                "flip_cam": rig.get("flip_cam"),
                "orientation_confirmed": bool(rig.get("orientation_confirmed")),
                "orientation_ok": bool(rig.get("orientation_ok")),
                "cameras": cams_out,
                # 相对位姿：X_cam1 = R_10 @ X_cam0 + T_10
                "convention": "X_cam1 = R_10 @ X_cam0 + T_10",
                "R_10": rel["R_10"],
                "T_10": rel["T_10"],
                "T_unit": "m",
                "baseline_m": rel["baseline_m"],
                "baseline_vector_c0_to_c1_m": rel["baseline_vector_c0_to_c1_m"],
                "rotation_angle_deg": rel["rotation_angle_deg"],
                "E": rel["E"],
                "F": rel.get("F"),
                "relative_check": rel["rotation"],
                "consistency": rig["consistency"],
                "stability": rig["stability"],
                "corner_correspondence": rig["corner_correspondence"],
                "warnings": rig.get("warnings", []),
                "caveat": rig.get("caveat"),
            }

        result = {
            "schema": "lowcost-mocap/calibration",
            "version": 2,
            "created": _now_iso(),
            "generated_by": "computer_code/api/calibration",
            "session": {
                # 项目内相对路径（可迁移）；读回时 service 会按 computer_code 解析
                "root": project_relative(sess.root),
                "session_id": sess.meta.get("session_id"),
                "created": sess.meta.get("created"),
            },
            "cameras": {
                str(c): {
                    "index": int(c),
                    "device": workers[c].device_name if workers[c] else None,
                    "format": workers[c].format_info if workers[c] else None,
                    "width": int(sess.image_size[0]),
                    "height": int(sess.image_size[1]),
                } for c in cams},
            "board": spec.to_dict(),
            "world_frame": spec.world_frame(),
            "image_convention": sess.meta.get("image_convention"),
            "intrinsics": {
                ca: _slim_intrinsic(self._intrinsics[ca]),
                cb: _slim_intrinsic(self._intrinsics[cb]),
            },
            "extrinsics": ext,
            "validation": self._validation,
            "data": sess.stats(),
            "caveats": [
                "帧同步为软件级（主机单调时间戳最近邻配对），非硬件曝光同步",
                "标定精度只在标定距离段、静止目标条件下成立，不能外推为动捕整体精度",
                "图像坐标基于未旋转未缩放的原始灰度图；若后续改变分辨率/裁剪，"
                "K 必须按比例换算",
                "已去畸变的点不得再次去畸变（例如同时用了 cv2.undistort 和 "
                "undistortPoints）",
            ],
        }
        return result

    def save_result(self, path: str | None = None) -> dict:
        result = self.build_result()
        path = resolve_path(path or DEFAULT_RESULT_PATH)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(to_jsonable(result), fh, ensure_ascii=False, indent=2)
        os.replace(tmp, path)

        sess = self._require_session()
        sess.save_result("calibration.json", result)
        self._result_path = path
        ext = result["extrinsics"] or {}
        # 两台相机**各自**的重投影 RMS 取大者；与"三角化自洽残差（毫米）"不是一回事
        rms = None
        if ext.get("cameras"):
            vals = [c.get("reprojection_rms_px") for c in ext["cameras"].values()]
            vals = [v for v in vals if v is not None]
            rms = round(max(vals), 4) if vals else None
        return {"path": project_relative(path),
                "baseline_m": ext.get("baseline_m"),
                "max_reprojection_rms_px": rms,
                "consistency_max_mm": (round(ext.get("consistency", {})
                                             .get("max_m", 0.0) * 1000, 4)
                                       if ext.get("consistency") else None),
                "orientation_confirmed": ext.get("orientation_confirmed"),
                "method": ext.get("method"),
                "session_copy": project_relative(
                    os.path.join(sess.root, "calibration.json"))}

    # ------------------------------------------------------------------
    # 状态
    # ------------------------------------------------------------------
    def status(self) -> dict:
        sess = self._session
        sync_st = None
        try:
            sync_st = sync_service().status()
        except Exception:
            pass

        sess_summary = sess.summary() if sess else None
        if sess_summary:
            sess_summary["root"] = project_relative(sess_summary["root"])
        out = {
            "session": sess_summary,
            "intrinsics": {
                k: {"rms_px": v.get("rms_px"), "num_views": v.get("num_views"),
                    "focal_px": v.get("focal_px"),
                    "coverage": v.get("coverage")}
                for k, v in self._intrinsics.items()},
            "stereo": None,
            "extrinsic": None,
            "validation": None,
            "result_path": self.result_path,
            "result_exists": os.path.isfile(self.result_path_abs),
            "sync": {k: sync_st.get(k) for k in
                     ("pairs_total", "pair_rate", "max_dt_ms", "last_dt_ms")}
                    if sync_st else None,
            "cameras": [{"index": c.index, "name": c.device_name, "state": c.state,
                         "format": c.format_info}
                        for c in (manager().worker(i) for i in (sess.cameras if sess else []))
                        if c is not None],
        }
        if self._extrinsic:
            rig = self._extrinsic
            out["extrinsic"] = {
                "method": rig["method"],
                "flip_cam": rig.get("flip_cam"),
                "orientation_confirmed": bool(rig.get("orientation_confirmed")),
                "orientation_ok": bool(rig.get("orientation_ok")),
                "baseline_m": rig["relative"]["baseline_m"],
                "relative_rotation_deg": rig["relative"]["rotation_angle_deg"],
                "consistency_max_mm": round(rig["consistency"]["max_m"] * 1000, 4),
                "cameras": {
                    k: {
                        "centre_m": v["camera_centre_world"],
                        "height_above_floor_m": v["height_above_floor_m"],
                        "distance_to_pattern_m": v["distance_to_pattern_centre_m"],
                        "optical_axis": v["optical_axis_world"],
                        # 相机系 → 世界系的完整旋转，供三维视图画真实视锥体
                        "R_cam_to_world": v["cam_to_world"]["R_wc"],
                        "t_cam_to_world_m": v["cam_to_world"]["t_wc"],
                        "reprojection_rms_px": v["reprojection"]["rms_px"],
                        "correspondence": v["correspondence"],
                        "frames": len((rig.get("frames", {}).get(k) or [])),
                    } for k, v in rig["cameras"].items()},
                "stability": rig["stability"],
                "warnings": rig.get("warnings", []),
            }
        if self._stereo:
            out["stereo"] = {
                "rms_px": self._stereo["rms_px"],
                "baseline_m": self._stereo["baseline_m"],
                "num_pairs": self._stereo["num_pairs"],
                "flip_b": self._stereo["flip_b"],
                "rotation_angle_deg": self._stereo["checks"]["rotation_angle_deg"],
                "epipolar_median_px": self._stereo["epipolar"]["median_px"],
                "role": "cross-check only (stereoCalibrate)",
            }
        if self._validation:
            val = self._validation or {}
            agg = val.get("aggregate") or {}
            out["validation"] = {
                "placements": agg.get("placements"),
                "spacing_abs_error_mm": agg.get("spacing_abs_error_mm"),
                "reprojection_rms_px": agg.get("reprojection_rms_px"),
                "epipolar_median_px": agg.get("epipolar_median_px"),
                "centre_z_range_m": agg.get("centre_z_range_m"),
                # 供前端显示最新一组 / 历史列表
                "aggregate": agg,
                "per_placement": val.get("per_placement") or [],
                "notes": val.get("notes") or [],
                "created": val.get("created"),
            }
        return out


def _slim_intrinsic(d: dict) -> dict:
    """落盘时保留必要字段，砍掉逐张明细以外的冗余。"""
    keep = ("K", "D", "rms_px", "reprojection", "num_views", "sample_ids",
            "image_size", "focal_px", "principal_point_px", "distortion_model",
            "warnings", "created", "camera", "per_view", "outlier_suggestion",
            "coverage")
    return {k: d[k] for k in keep if k in d}


# ---------------------------------------------------------------------------
# 供后续三维定位用的加载接口（采集模块不需要它）
# ---------------------------------------------------------------------------
def load_calibration(path: str | None = None) -> dict:
    """读取并做基本一致性检查。**不修改** calibration.json。"""
    path = resolve_path(path or DEFAULT_RESULT_PATH)
    if not os.path.isfile(path):
        raise FileNotFoundError("没有标定结果文件：%s（请先完成标定并保存）" % path)
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    _normalize_validation_units(data.get("validation"))

    problems = []
    if data.get("schema") != "lowcost-mocap/calibration":
        problems.append("schema 不匹配")
    intr = data.get("intrinsics") or {}
    for key in ("0", "1"):
        ok, why = intr_mod.validate_result(intr.get(key, {}))
        if not ok:
            problems.append("相机 %s 内参：%s" % (key, "；".join(why)))
    ext = data.get("extrinsics") or {}
    for key in ("R_10", "T_10", "baseline_m"):
        if key not in ext:
            problems.append("外参缺少 %s" % key)
    if problems:
        raise ValueError("标定文件不合法：\n  - " + "\n  - ".join(problems))
    return data


def stereo_matrices(data: dict, *, scaled_size: tuple[int, int] | None = None):
    """取出 (K0, D0, K1, D1, R, T)，可选按新分辨率缩放 K。

    ``scaled_size`` 用于"标定时 1280x720，运行时改成别的分辨率"的场景：
    按 fx*sx, fy*sy, cx*sx, cy*sy 缩放。畸变系数不变。
    """
    intr = data["intrinsics"]
    K0 = np.asarray(intr["0"]["K"], np.float64)
    D0 = np.asarray(intr["0"]["D"], np.float64).reshape(-1)
    K1 = np.asarray(intr["1"]["K"], np.float64)
    D1 = np.asarray(intr["1"]["D"], np.float64).reshape(-1)

    if scaled_size is not None:
        w0, h0 = intr["0"]["image_size"]
        sx, sy = scaled_size[0] / float(w0), scaled_size[1] / float(h0)
        if abs(sx - 1) > 1e-9 or abs(sy - 1) > 1e-9:
            S = np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]], np.float64)
            K0, K1 = S @ K0, S @ K1

    ext = data["extrinsics"]
    R = np.asarray(ext["R_10"], np.float64)
    T = np.asarray(ext["T_10"], np.float64).reshape(3)
    return K0, D0, K1, D1, R, T


_instance: CalibrationService | None = None
_lock = threading.Lock()


def service() -> CalibrationService:
    global _instance
    with _lock:
        if _instance is None:
            _instance = CalibrationService()
        return _instance
