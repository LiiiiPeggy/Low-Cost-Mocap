"""红外 LED 连续调试录制。

只做两件事，尽量不碰现有定位链路：

* **采集线程**轮询 ``sync_service().recent_pairs()``，把新同步帧对放进有界队列；
* **写盘线程**保存两台相机的原始灰度 PNG，并在诊断数据可用后写
  ``frames.jsonl``。

LED 定位线程处理完一帧后调用 :meth:`LedRecorder.note_result` 登记现有状态；
录制模块不重新检测、匹配或三角化。未进入 LED 处理的同步帧对仍保存图像，
在 JSON 中标记为 ``not_processed``。
"""

from __future__ import annotations

import copy
import json
import os
import queue
import threading
import time
from collections import OrderedDict, deque
from datetime import datetime

import cv2

from calibration.board import to_jsonable
from calibration.dataset import atomic_json_write
from calibration.service import project_relative
from sync import sync_service

SCHEMA = "lowcost-mocap/led-recording"
FRAME_SCHEMA = "lowcost-mocap/led-recording-frame"
QUEUE_SIZE = 32
POLL_S = 0.006
PNG_COMPRESSION = 1
DIAG_STALE_S = 2.0
DIAG_SETTLE_S = 0.3
DIAG_OFF_GRACE_S = 0.3
DIAG_CACHE_SIZE = 512

# 每帧只保存现有定位状态中的诊断字段；轨迹/参数等长期字段放在 session.json，
# 避免 30 fps 下把同一份轨迹重复写进每一行。
_FRAME_STATE_KEYS = (
    "status", "valid", "reason", "detections", "matches", "leds",
    "raw_center", "filtered_center", "velocity",
    "edges_mm", "edge_errors_mm", "expected_edges_mm", "board_direction",
    "label_rule", "label_assignment", "mean_reprojection_px",
    "mean_epipolar_px", "triangle_score", "orientation_consistent",
    "triangle_tol_mm", "tracking", "calibration", "camera_map", "mapping",
    "diagnostics", "processing_ms", "processed_pairs",
)


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="milliseconds")


def _frame_state(state: dict, diagnostics: dict | None = None) -> dict:
    """从 LedService 当前状态提取单帧诊断，保留原字段名/结构。

    ``diagnostics`` 是本次匹配的完整诊断（极线矩阵、候选对、重投影等）。
    LED 网页状态为了 4 Hz 轮询只保存紧凑版，录制时单独取完整版。
    """
    out = {k: copy.deepcopy(state[k]) for k in _FRAME_STATE_KEYS
           if k in state and not (k == "diagnostics"
                                  and diagnostics is not None)}
    if diagnostics is not None:
        out["diagnostics"] = copy.deepcopy(diagnostics)
    out["schema"] = FRAME_SCHEMA
    return out


def _status_name(state: dict | None) -> str:
    return str((state or {}).get("status") or "not_processed")


class LedRecorder:
    def __init__(self, *, poll_s: float = POLL_S,
                 queue_size: int = QUEUE_SIZE) -> None:
        self._poll_s = max(0.002, float(poll_s))
        self._queue_size = max(1, int(queue_size))
        self._lock = threading.RLock()
        self._write_lock = threading.Lock()
        self._stop = threading.Event()
        self._capture_thread: threading.Thread | None = None
        self._writer_thread: threading.Thread | None = None
        self._queue: queue.Queue | None = None

        self._status = "idle"
        self._error = ""
        self._directory = ""
        self._abs_path = ""
        self._frames_path = ""
        self._session_path = ""
        self._session: dict = {}
        self._frames_fh = None
        self._finalized = False

        self._started_mono = 0.0
        self._ended_mono = 0.0
        self._started_iso = ""
        self._ended_iso = ""

        self._last_seq = 0            # 已观察到的最后一个 pair_seq
        self._last_processed_seq = 0  # LED 服务已处理到哪个 pair_seq
        self._last_result_mono = 0.0
        self._processing = False
        self._processing_changed_mono = 0.0
        self._pairs_seen = 0
        self._saved_pairs = 0
        self._saved_processed = 0
        self._saved_not_processed = 0
        self._dropped_queue = 0
        self._dropped_gap = 0
        self._led_results = 0
        self._next_index = 1
        self._diagnostics: OrderedDict[int, dict] = OrderedDict()
        self._records: deque[dict] = deque()

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    def start(self, session: dict, root: str) -> dict:
        with self._lock:
            if self._status == "recording":
                raise ValueError("已有连续录制进行中")
            if (self._capture_thread and self._capture_thread.is_alive()) or \
                    (self._writer_thread and self._writer_thread.is_alive()):
                raise RuntimeError("上一次录制的线程尚未结束，请稍后重试")
            if not root:
                raise ValueError("录制目录不能为空")

            os.makedirs(root, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            outdir = os.path.join(root, stamp)
            suffix = 1
            while True:
                try:
                    os.makedirs(outdir)
                    break
                except FileExistsError:
                    outdir = os.path.join(root, "%s_%02d" % (stamp, suffix))
                    suffix += 1
            frames_dir = os.path.join(outdir, "frames")
            os.makedirs(frames_dir, exist_ok=True)

            self._stop = threading.Event()
            self._queue = queue.Queue(maxsize=self._queue_size)
            self._diagnostics = OrderedDict()
            self._records = deque()
            self._finalized = False
            self._error = ""
            self._directory = project_relative(outdir)
            self._abs_path = os.path.abspath(outdir)
            self._frames_path = os.path.join(outdir, "frames.jsonl")
            self._session_path = os.path.join(outdir, "session.json")
            self._session = copy.deepcopy(session or {})
            self._session.update({
                "schema": SCHEMA,
                "created": _now_iso(),
                "status": "recording",
                "directory": self._directory,
                "abs_path": self._abs_path,
                "frames": "frames.jsonl",
                "recording": {
                    "source": "sync_service.recent_pairs",
                    "queue_size": self._queue_size,
                    "png_compression": PNG_COMPRESSION,
                },
            })
            try:
                self._frames_fh = open(self._frames_path, "a",
                                       encoding="utf-8", buffering=1)
                atomic_json_write(self._session_path, self._session)
            except BaseException as exc:
                if self._frames_fh is not None:
                    try:
                        self._frames_fh.close()
                    except OSError:
                        pass
                    self._frames_fh = None
                self._error = str(exc).strip().splitlines()[-1]
                self._status = "error"
                raise RuntimeError("创建录制文件失败：%s" % self._error) from exc

            self._status = "recording"
            self._started_mono = time.perf_counter()
            self._ended_mono = 0.0
            self._started_iso = self._session["created"]
            self._ended_iso = ""
            self._last_seq = 0
            self._last_processed_seq = 0
            self._last_result_mono = self._started_mono
            self._processing = bool(self._session.get("processing"))
            self._processing_changed_mono = self._started_mono
            self._pairs_seen = 0
            self._saved_pairs = 0
            self._saved_processed = 0
            self._saved_not_processed = 0
            self._dropped_queue = 0
            self._dropped_gap = 0
            self._led_results = 0
            self._next_index = 1

            try:
                recent = sync_service().recent_pairs(1)
                self._last_seq = int(recent[-1].pair_seq) if recent else 0
            except Exception:
                self._last_seq = 0

            self._capture_thread = threading.Thread(
                target=self._capture_loop, name="led-record-capture",
                daemon=True)
            self._writer_thread = threading.Thread(
                target=self._write_loop, name="led-record-writer",
                daemon=True)

        self._capture_thread.start()
        self._writer_thread.start()
        return self.status()

    def stop(self) -> dict:
        with self._lock:
            threads = (self._capture_thread, self._writer_thread)
            active = self._status == "recording"
            self._stop.set()
        for thread in threads:
            if thread and thread.is_alive():
                thread.join(timeout=15.0)
        with self._lock:
            active = (self._status == "recording")
        writer = self._writer_thread
        if active and not (writer and writer.is_alive()):
            # 写盘线程异常退出时由这里收尾，保证 frames.jsonl/session.json
            self._finalize()
        return self.status()

    def status(self) -> dict:
        with self._lock:
            active = self._status == "recording"
            end = time.perf_counter() if active else (self._ended_mono
                                                      or self._started_mono)
            elapsed = max(0.0, end - self._started_mono) \
                if self._started_mono else 0.0
            queue_depth = self._queue.qsize() if self._queue is not None else 0
            return {
                "ok": True,
                "status": self._status,
                "recording": active,
                "error": self._error,
                "directory": self._directory,
                "abs_path": self._abs_path,
                "started": self._started_iso,
                "ended": self._ended_iso,
                "elapsed_s": round(elapsed, 2),
                "pairs_seen": self._pairs_seen,
                "saved_pairs": self._saved_pairs,
                "saved_files": self._saved_pairs * 2,
                "saved_led_pairs": self._saved_processed,
                "saved_not_processed_pairs": self._saved_not_processed,
                "pending_pairs": len(self._records),
                "dropped_pairs": self._dropped_queue + self._dropped_gap,
                "dropped_queue_full": self._dropped_queue,
                "dropped_sync_gap": self._dropped_gap,
                "led_processed_pairs": self._led_results,
                "queue_depth": queue_depth,
                "queue_size": self._queue_size,
                "sync_fps": round(self._pairs_seen / elapsed, 2)
                if elapsed > 0 else None,
                "led_fps": round(self._led_results / elapsed, 2)
                if elapsed > 0 else None,
                "saved_fps": round(self._saved_pairs / elapsed, 2)
                if elapsed > 0 else None,
                "png_compression": PNG_COMPRESSION,
            }

    def set_processing(self, processing: bool) -> None:
        """LED 检测/定位启停时更新；停止后新帧可立即标为 not_processed。"""
        with self._lock:
            value = bool(processing)
            if value != self._processing:
                self._processing_changed_mono = time.perf_counter()
            self._processing = value
        if self._status == "recording":
            try:
                self._flush_ready()
            except Exception as exc:
                # 录制写盘失败只结束录制，不能影响 LED 检测/定位启停
                self._fail(exc)

    def note_result(self, pair_seq: int, state: dict,
                    diagnostics: dict | None = None) -> bool:
        """登记 LED 服务已经处理完的同一 pair_seq 的状态快照。"""
        with self._lock:
            if self._status != "recording" or self._finalized:
                return False
        diag = _frame_state(state, diagnostics)
        now = time.perf_counter()
        with self._lock:
            if self._status != "recording" or self._finalized:
                return False
            seq = int(pair_seq)
            if seq < self._last_processed_seq:
                # 相机/同步服务重启后 pair_seq 倒退
                self._last_processed_seq = seq
            new_result = seq > self._last_processed_seq
            self._diagnostics[seq] = diag
            self._diagnostics.move_to_end(seq)
            while len(self._diagnostics) > DIAG_CACHE_SIZE:
                self._diagnostics.popitem(last=False)
            self._last_processed_seq = max(self._last_processed_seq, seq)
            self._last_result_mono = now
            if new_result:
                self._led_results += 1
        # 不在 LED 定位线程里写盘；写盘线程的空闲轮询会完成 frames.jsonl。
        return True

    # ------------------------------------------------------------------
    # 采集线程 / 写盘线程
    # ------------------------------------------------------------------
    def _stopped_and_captured(self) -> bool:
        """停止后必须等采集线程退出，避免最后一对入队后写盘线程已结束。"""
        capture = self._capture_thread
        return (self._stop.is_set()
                and not (capture and capture.is_alive()))

    def _capture_loop(self) -> None:
        while not self._stop.is_set():
            try:
                recs = sync_service().recent_pairs(16)
            except Exception:
                time.sleep(0.05)
                continue
            if not recs:
                self._stop.wait(self._poll_s)
                continue
            with self._lock:
                # 相机/同步服务重启后 pair_seq 从 1 重新计数。只有“最新的
                # 一对”仍旧于已观察值时才重置，避免同一批里先看到旧帧就
                # 把指针清零、随后重复录制整批历史帧。
                if int(recs[-1].pair_seq) < self._last_seq:
                    self._last_seq = 0
            advanced = False
            for rec in recs:
                with self._lock:
                    seq = int(rec.pair_seq)
                    if seq <= self._last_seq:
                        continue
                    gap = max(0, seq - self._last_seq - 1)
                    self._last_seq = seq
                    self._pairs_seen += 1
                    advanced = True
                    if gap > 0:
                        self._pairs_seen += gap
                        self._dropped_gap += gap
                    target = self._queue
                try:
                    target.put_nowait(rec)
                except queue.Full:
                    with self._lock:
                        self._dropped_queue += 1
            if not advanced:
                self._stop.wait(self._poll_s)

    def _write_loop(self) -> None:
        stopping = False
        try:
            while True:
                item = None
                try:
                    item = self._queue.get(timeout=0.05)
                except queue.Empty:
                    self._flush_ready()
                    if self._stopped_and_captured():
                        stopping = True
                        break
                    continue
                self._write_record(item)
                self._flush_ready()
                if self._stopped_and_captured() and self._queue.empty():
                    stopping = True
                    break
            if stopping and self._records:
                # 给 LED 线程最后一点时间登记已处理帧的诊断
                deadline = time.perf_counter() + DIAG_SETTLE_S
                while time.perf_counter() < deadline:
                    self._flush_ready()
                    with self._lock:
                        if not self._records:
                            break
                    time.sleep(0.01)
        except BaseException as exc:
            self._fail(exc)
        finally:
            self._finalize()

    def _write_record(self, rec) -> None:
        with self._lock:
            index = self._next_index
            frames_dir = os.path.join(self._abs_path, "frames")
            fname0 = "%06d_cam0.png" % index
            fname1 = "%06d_cam1.png" % index
            path0 = os.path.join(frames_dir, fname0)
            path1 = os.path.join(frames_dir, fname1)
        params = [int(cv2.IMWRITE_PNG_COMPRESSION), PNG_COMPRESSION]
        ok0 = cv2.imwrite(path0, rec.frame_a.gray, params)
        ok1 = cv2.imwrite(path1, rec.frame_b.gray, params)
        if not ok0 or not ok1:
            for path in (path0, path1):
                try:
                    os.remove(path)
                except OSError:
                    pass
            raise OSError("PNG 写盘失败（磁盘空间或目录权限）：%s" % frames_dir)

        meta = rec.meta()
        try:
            host_ts = max(float(rec.frame_a.host_ts), float(rec.frame_b.host_ts))
            captured = datetime.fromtimestamp(host_ts).isoformat(
                timespec="milliseconds")
        except (TypeError, ValueError, OSError):
            captured = ""
        record = {
            "schema": FRAME_SCHEMA,
            "index": index,
            "images": {
                "0": "frames/" + fname0,
                "1": "frames/" + fname1,
            },
            "pair": meta,
            "captured": captured,
            "recorded_at": _now_iso(),
            "led_processed": False,
            "led_status": "not_processed",
            "led_reason": "该同步帧对未经过 LED 定位处理",
            "led": None,
        }
        with self._lock:
            self._records.append(record)
            self._saved_pairs += 1
            self._next_index += 1

    def _flush_ready(self, *, force: bool = False) -> None:
        ready = []
        now = time.perf_counter()
        with self._lock:
            while self._records:
                record = self._records[0]
                seq = int((record.get("pair") or {}).get("pair_seq", 0))
                diag = self._diagnostics.get(seq)
                if diag is not None:
                    record["led_processed"] = True
                    record["led_status"] = _status_name(diag)
                    record["led_reason"] = diag.get("reason") or ""
                    record["led"] = diag
                    self._diagnostics.pop(seq, None)
                    self._saved_processed += 1
                else:
                    stale = now - max(self._last_result_mono,
                                      self._started_mono) > DIAG_STALE_S
                    # 检测刚停止时给在途结果一点登记时间，避免把已处理帧
                    # 误标为 not_processed。
                    processing_off = (not self._processing
                                      and now - self._processing_changed_mono
                                      > DIAG_OFF_GRACE_S)
                    skipped = (seq <= self._last_processed_seq
                               or processing_off or stale)
                    if not force and not skipped:
                        break
                    record["led_processed"] = False
                    record["led_status"] = "not_processed"
                    record["led_reason"] = \
                        "该同步帧对未经过 LED 定位处理"
                    record["led"] = None
                    self._saved_not_processed += 1
                self._records.popleft()
                ready.append(record)
        if not ready:
            return
        with self._write_lock:
            fh = self._frames_fh
            if fh is None:
                raise OSError("frames.jsonl 已关闭")
            for record in ready:
                fh.write(json.dumps(to_jsonable(record), ensure_ascii=False,
                                    separators=(",", ":")) + "\n")

    def _fail(self, exc: BaseException) -> None:
        message = str(exc).strip().splitlines()
        with self._lock:
            if not self._error:
                self._error = message[-1] if message else type(exc).__name__
            self._status = "error"
            self._stop.set()

    def _finalize(self) -> None:
        try:
            self._flush_ready(force=True)
        except BaseException as exc:
            self._fail(exc)
        with self._lock:
            if self._finalized:
                return
            self._finalized = True
            self._ended_mono = time.perf_counter()
            self._ended_iso = _now_iso()
            if self._status != "error":
                self._status = "stopped"
            session = copy.deepcopy(self._session)
            session.update({
                "ended": self._ended_iso,
                "status": self._status,
                "error": self._error,
                "stats": {
                    "elapsed_s": round(max(
                        0.0, self._ended_mono - self._started_mono), 2),
                    "pairs_seen": self._pairs_seen,
                    "saved_pairs": self._saved_pairs,
                    "saved_files": self._saved_pairs * 2,
                    "saved_led_pairs": self._saved_processed,
                    "saved_not_processed_pairs": self._saved_not_processed,
                    "dropped_pairs": (self._dropped_queue
                                      + self._dropped_gap),
                    "dropped_queue_full": self._dropped_queue,
                    "dropped_sync_gap": self._dropped_gap,
                    "led_processed_pairs": self._led_results,
                    "sync_fps": round(self._pairs_seen / max(
                        1e-9, self._ended_mono - self._started_mono), 2),
                    "led_fps": round(self._led_results / max(
                        1e-9, self._ended_mono - self._started_mono), 2),
                    "saved_fps": round(self._saved_pairs / max(
                        1e-9, self._ended_mono - self._started_mono), 2),
                },
            })
            session_path = self._session_path
            self._session = session
        with self._write_lock:
            fh = self._frames_fh
            self._frames_fh = None
            if fh is not None:
                try:
                    fh.close()
                except OSError:
                    pass
        try:
            atomic_json_write(session_path, session)
        except BaseException as exc:
            self._fail(exc)


_instance: LedRecorder | None = None
_instance_lock = threading.Lock()


def recorder() -> LedRecorder:
    global _instance
    with _instance_lock:
        if _instance is None:
            _instance = LedRecorder()
        return _instance


__all__ = ["LedRecorder", "recorder", "SCHEMA", "FRAME_SCHEMA"]
