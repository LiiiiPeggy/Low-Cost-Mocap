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
MIN_COVERAGE = 0.008       # 角点凸包占画面比例，太小说明板子太远

CALIB_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RESULT_PATH = os.path.join(CALIB_DIR, "calibration.json")
DEFAULT_SESSION_ROOT = os.environ.get(
    "MOCAP_CALIB_DIR",
    os.path.join(os.environ.get("MOCAP_CAPTURE_DIR", os.path.abspath("captures")),
                 "calibration"))


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


class CalibrationService:
    """进程级单例：持有当前会话与最近一次标定结果。"""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._session: Session | None = None
        self._intrinsics: dict[str, dict] = {}
        self._stereo: dict | None = None
        self._extrinsic: dict | None = None
        self._validation: dict | None = None
        self._stereo_validation: dict | None = None
        self._result_path: str | None = None

    # ------------------------------------------------------------------
    # 会话
    # ------------------------------------------------------------------
    @property
    def session(self) -> Session | None:
        return self._session

    @property
    def result_path(self) -> str:
        """最近一次保存的路径；没保存过则给默认路径。"""
        return self._result_path or DEFAULT_RESULT_PATH

    def open_session(self, *, cols: int, rows: int, square_mm: float,
                     cameras=None, session_id: str | None = None,
                     pixel_format: str = "", note: str = "",
                     surface_height_m: float = 0.0,
                     root: str | None = None) -> dict:
        spec = BoardSpec(cols=cols, rows=rows, square_mm=square_mm,
                         surface_height_m=surface_height_m)
        cams = [int(c) for c in (cameras or [0, 1])]

        size = self._probe_image_size(cams)
        sid = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        path = root or os.path.join(DEFAULT_SESSION_ROOT, sid)

        with self._lock:
            sess = Session.create_or_load(path, spec, cams, size, pixel_format)
            self._session = sess
            # 切换会话时清掉上一轮求解结果，避免张冠李戴
            self._intrinsics, self._stereo, self._validation = {}, None, None
            self._extrinsic = None
        return self.status()

    def reload_session(self, root: str) -> dict:
        """离线重算：直接载入已采集的会话目录。"""
        sess = Session.load(root)
        with self._lock:
            self._session = sess
            self._intrinsics, self._stereo, self._validation = {}, None, None
            self._extrinsic = None
        return self.status()

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

    def _fresh_frame(self, camera: int):
        worker = manager().worker(camera)
        if worker is None:
            raise RuntimeError("相机 %d 未启动，无法采集（本模块不会自己打开设备）" % camera)
        if worker.state != "streaming":
            raise RuntimeError("相机 %d 当前状态是 %s，不是 streaming" % (camera, worker.state))
        frame = worker.latest()
        if frame is None:
            raise RuntimeError("相机 %d 还没有出帧" % camera)
        age = time.perf_counter() - frame.mono_ts
        if age > MAX_FRAME_AGE_S:
            raise RuntimeError("相机 %d 的帧已过期 %.0f ms（可能已停流），拒绝采集"
                               % (camera, age * 1000))
        return frame, age

    def capture_single(self, camera: int, *, force: bool = False,
                       phase: str = "intrinsic") -> dict:
        sess = self._require_session()
        spec = sess.spec
        camera = int(camera)

        frame, age = self._fresh_frame(camera)
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

        status, reason = "valid", ""
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
            elif q["coverage"] < MIN_COVERAGE:
                status, reason = "invalid", ("标定板在画面中太小（占画面 %.1f%%）"
                                             % (100 * q["coverage"]))
            elif dup and not force:
                status, reason = "invalid", ("与已有样本 %s 姿态几乎相同（位移 %.3f/"
                                             "尺度 %.2f/转角 %.1f°），换一个角度再采"
                                             % (dup["sample"], dup["dpos"],
                                                dup["dscale"], dup["dangle"]))
            else:
                meta["signature"] = sig

        sample = sess.add_single(camera, gray, det, meta, overlay=overlay,
                                 status=status, reason=reason,
                                 signature=meta.get("signature", {}), phase=phase)

        return {
            "ok": status == "valid",
            "sample": sample.id,
            "camera": camera,
            "phase": phase,
            "status": status,
            "reason": reason,
            "detection": det.to_dict(),
            "quality": det.quality,
            "frame_age_ms": meta["frame_age_ms"],
            "stats": sess.stats(),
            "message": ("已采集 %s（相机 %d，%s）" % (sample.id, camera, phase)
                        if status == "valid" else "未计入有效样本：%s" % reason),
        }

    def capture_stereo(self, *, force: bool = False) -> dict:
        sess = self._require_session()
        spec = sess.spec
        cams = sess.cameras
        if len(cams) < 2:
            raise RuntimeError("会话只登记了 %d 台相机，无法做双相机采集" % len(cams))
        ca, cb = int(cams[0]), int(cams[1])

        svc = sync_service()
        recents = svc.recent_pairs(1)
        if not recents:
            raise RuntimeError("还没有配对成功的帧对：确认两台相机都在 streaming，"
                               "且 /api/sync/status 的 pairs_total 在增长")
        rec = recents[0]
        age = time.perf_counter() - rec.created_mono
        if age > MAX_PAIR_AGE_S:
            raise RuntimeError("最近一对同步帧已过期 %.0f ms，请重新触发（相机是否停流？）"
                               % (age * 1000))
        if abs(rec.dt_ms) > svc.max_dt_ms + 1e-6:
            raise RuntimeError("帧对时间差 |Δt|=%.1f ms 超过阈值 %.0f ms"
                               % (abs(rec.dt_ms), svc.max_dt_ms))

        fa, fb = rec.frame_a, rec.frame_b
        if (fa.gray.shape[1], fa.gray.shape[0]) != tuple(sess.image_size):
            raise RuntimeError("相机 %d 当前分辨率与会话登记的不一致（%s vs %s）"
                               % (ca, fa.gray.shape[:2][::-1], tuple(sess.image_size)))

        deta, detb, info = board_mod.detect_pair(fa.gray, fb.gray, spec)
        overlay = {str(ca): board_mod.draw_overlay(fa.gray, deta, spec, label_step=3),
                   str(cb): board_mod.draw_overlay(fb.gray, detb, spec, label_step=3)}

        status, reason = "valid", ""
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
            if min(qa["sharpness"], qb["sharpness"]) < MIN_SHARPNESS:
                status, reason = "invalid", "至少一路图像偏模糊"
            elif min(qa["coverage"], qb["coverage"]) < MIN_COVERAGE:
                status, reason = "invalid", "至少一路标定板太小"
            elif dup and not force:
                status, reason = "invalid", ("与已有帧对 %s 姿态几乎相同，换一个角度再采"
                                             % dup["sample"])

        pair_meta = {
            str(ca): self._side_meta(fa, ca),
            str(cb): self._side_meta(fb, cb),
        }
        sample = sess.add_stereo(
            pair_meta, {ca: fa.gray, cb: fb.gray}, {ca: deta, cb: detb},
            dt_ms=rec.dt_ms, overlay=overlay, status=status, reason=reason,
            signature=signature)

        return {
            "ok": status == "valid",
            "sample": sample.id,
            "status": status,
            "reason": reason,
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
        """外参阶段：棋盘格固定在中央地面，按一次存一张（不做姿态判重）。"""
        return self.capture_single(camera, force=True, phase="extrinsic")

    def solve_extrinsic(self, *, flip180: bool = False, exclude=()) -> dict:
        sess = self._require_session()
        cams = sess.cameras
        if len(cams) < 2:
            raise ValueError("会话只登记了 %d 台相机" % len(cams))
        ca, cb = str(int(cams[0])), str(int(cams[1]))
        missing = [c for c in (ca, cb) if c not in self._intrinsics]
        if missing:
            raise ValueError("相机 %s 还没有内参结果，请先执行内参标定" % ", ".join(missing))

        frames, ids = {}, {}
        for cam in (int(cams[0]), int(cams[1])):
            samples = sess.single_samples(cam, phase="extrinsic", exclude=exclude)
            if len(samples) < ext_mod.MIN_FRAMES:
                raise ValueError(
                    "相机 %d 的外参图像只有 %d 张，至少需要 %d 张（建议 %d 张）"
                    % (cam, len(samples), ext_mod.MIN_FRAMES, ext_mod.RECOMMENDED_FRAMES))
            frames[cam] = [s.corners(cam) for s in samples]
            ids[cam] = [s.id for s in samples]

        rig = ext_mod.solve_rig(frames, sess.spec,
                                intr_mod.K_from(self._intrinsics[ca]),
                                intr_mod.D_from(self._intrinsics[ca]),
                                intr_mod.K_from(self._intrinsics[cb]),
                                intr_mod.D_from(self._intrinsics[cb]),
                                flip180=flip180)
        rig["created"] = _now_iso()
        rig["camera_pair"] = [int(cams[0]), int(cams[1])]
        rig["frames"] = {str(k): v for k, v in ids.items()}
        rig["intrinsic_rms_px"] = {ca: self._intrinsics[ca].get("rms_px"),
                                   cb: self._intrinsics[cb].get("rms_px")}

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
                             "「采集同步帧对」逐点采集，且不要把验证用的图混进外参标定")

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
        with self._lock:
            self._validation = out
        sess.save_result("result_validation.json", out)
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
    # 独立验证（留出集 / 已知尺寸）
    # ------------------------------------------------------------------
    def validate(self, *, exclude=(), holdout: float = 0.3,
                 auto_flip: bool = None, expected_mm: float | None = None,
                 corner_pair: tuple[int, int] | None = None) -> dict:
        """独立验证：把帧对按比例切成训练/验证，分别评估。

        ``holdout`` 为验证集比例（0 表示全部用于训练，即"自洽评估"）。
        默认的已知尺寸检查用板面角点间距；``expected_mm`` + ``corner_pair``
        可以换成你自己量的任意两点间物理距离。
        """
        sess = self._require_session()
        spec = sess.spec
        pairs = sess.stereo_samples(exclude=exclude)
        if len(pairs) < 6:
            raise ValueError("有效帧对只有 %d 对，做留出验证至少需要 6 对"
                             "（建议 15 对以上）" % len(pairs))

        expected_pairs = None
        if expected_mm is not None and corner_pair is not None:
            i, j = int(corner_pair[0]), int(corner_pair[1])
            expected_pairs = [{"i": i, "j": j, "expected_mm": float(expected_mm),
                               "label": "自定义 %d-%d" % (i, j)}]

        pairs_sorted = sorted(pairs, key=lambda s: s.id)
        if holdout and holdout > 0:
            k = max(1, int(round(1.0 / holdout)))
            val = [s for idx, s in enumerate(pairs_sorted) if idx % k == k - 1]
            train = [s for s in pairs_sorted if s not in val]
            if len(train) < stereo_mod.MIN_PAIRS or not val:
                raise ValueError("样本太少，无法按 %.0f%% 切分训练/验证集" % (100 * holdout))
        else:
            train, val = pairs_sorted, []

        cams = sess.cameras
        ca, cb = str(int(cams[0])), str(int(cams[1]))
        if ca not in self._intrinsics or cb not in self._intrinsics:
            raise ValueError("缺少内参结果，请先做内参标定")

        res = stereo_mod.calibrate(
            train, spec, tuple(sess.image_size),
            self._intrinsics[ca], self._intrinsics[cb],
            auto_flip=True if auto_flip is None else auto_flip)
        R = np.asarray(res["R_10"], np.float64)
        T = np.asarray(res["T_10"], np.float64)
        F = np.asarray(res["F"], np.float64)

        on_train = stereo_mod.evaluate(train, spec, tuple(sess.image_size),
                                       self._intrinsics[ca], self._intrinsics[cb],
                                       R, T, F, flip_b=res["flip_b"],
                                       expected_pairs=expected_pairs)
        on_val = (stereo_mod.evaluate(val, spec, tuple(sess.image_size),
                                      self._intrinsics[ca], self._intrinsics[cb],
                                      R, T, F, flip_b=res["flip_b"],
                                      expected_pairs=expected_pairs) if val else None)

        out = {
            "created": _now_iso(),
            "split": {"train": len(train), "validation": len(val),
                      "strategy": "按样本编号排序后每 1/holdout 个留一个作验证集"
                                  if val else "未切分（全部用于训练，指标为自洽评估）"},
            "train_ids": [s.id for s in train],
            "validation_ids": [s.id for s in val],
            "flip_b": res["flip_b"],
            "baseline_m": res["baseline_m"],
            "fit": {"rms_px": res["rms_px"],
                    "epipolar": res["epipolar"], "triangulation": res["triangulation"]},
            "on_train": on_train,
            "on_validation": on_val,
            "interpretation": [
                "on_validation 才是独立精度参考；on_train 必然偏乐观",
                "已知尺寸误差反映三角化的米制尺度是否可信，与重投影 RMS 不是一回事",
                "即使全部指标很好，也只说明该距离段的静止目标几何一致，"
                "不能据此宣称动捕达到毫米级",
            ],
        }

        # 用留出集结果作为最终验证结论；没有留出集时退化为自洽评估
        self._stereo_validation = out
        sess.save_result("result_stereo_validation.json", out)
        return out

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
                "flip180_global": rig["flip180"],
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
                "root": sess.root,
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
        path = os.path.abspath(path or DEFAULT_RESULT_PATH)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(to_jsonable(result), fh, ensure_ascii=False, indent=2)
        os.replace(tmp, path)

        sess = self._require_session()
        sess.save_result("calibration.json", result)
        self._result_path = path
        ext = result["extrinsics"] or {}
        rms = None
        if ext.get("cameras"):
            vals = [c.get("reprojection_rms_px") for c in ext["cameras"].values()]
            rms = round(max(v for v in vals if v is not None), 4) if any(vals) else None
        return {"path": path,
                "baseline_m": ext.get("baseline_m"),
                "stereo_rms_px": rms,
                "method": ext.get("method"),
                "session_copy": os.path.join(sess.root, "calibration.json")}

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

        out = {
            "session": sess.summary() if sess else None,
            "intrinsics": {
                k: {"rms_px": v.get("rms_px"), "num_views": v.get("num_views"),
                    "focal_px": v.get("focal_px")}
                for k, v in self._intrinsics.items()},
            "stereo": None,
            "extrinsic": None,
            "validation": None,
            "result_path": self.result_path,
            "result_exists": os.path.isfile(self.result_path),
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
                "flip180_global": rig["flip180"],
                "baseline_m": rig["relative"]["baseline_m"],
                "relative_rotation_deg": rig["relative"]["rotation_angle_deg"],
                "consistency_max_mm": round(rig["consistency"]["max_m"] * 1000, 4),
                "cameras": {
                    k: {
                        "centre_m": v["camera_centre_world"],
                        "height_above_floor_m": v["height_above_floor_m"],
                        "distance_to_pattern_m": v["distance_to_pattern_centre_m"],
                        "optical_axis": v["optical_axis_world"],
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
            agg = (self._validation or {}).get("aggregate") or {}
            out["validation"] = {
                "placements": agg.get("placements"),
                "spacing_abs_error_mm": agg.get("spacing_abs_error_mm"),
                "reprojection_rms_px": agg.get("reprojection_rms_px"),
                "epipolar_median_px": agg.get("epipolar_median_px"),
                "centre_z_range_m": agg.get("centre_z_range_m"),
            }
        return out


def _slim_intrinsic(d: dict) -> dict:
    """落盘时保留必要字段，砍掉逐张明细以外的冗余。"""
    keep = ("K", "D", "rms_px", "reprojection", "num_views", "sample_ids",
            "image_size", "focal_px", "principal_point_px", "distortion_model",
            "warnings", "created", "camera", "per_view", "outlier_suggestion")
    return {k: d[k] for k in keep if k in d}


# ---------------------------------------------------------------------------
# 供后续三维定位用的加载接口（采集模块不需要它）
# ---------------------------------------------------------------------------
def load_calibration(path: str | None = None) -> dict:
    """读取并做基本一致性检查。**不修改** calibration.json。"""
    path = os.path.abspath(path or DEFAULT_RESULT_PATH)
    if not os.path.isfile(path):
        raise FileNotFoundError("没有标定结果文件：%s（请先完成标定并保存）" % path)
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)

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
