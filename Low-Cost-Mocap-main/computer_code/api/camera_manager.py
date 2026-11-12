"""双 / 多 USB 红外相机并发采集管理器（Windows + Media Foundation）。

设计要点
--------------------------------------------------------------------------
* 复用仓库根目录的 ``ir_camera.py``（设备枚举 + 单相机取帧 + COM/MF 资源管理），
  本模块只负责“多路调度”，不改动单相机 CLI 的行为。
* Media Foundation 的采集对象按 COM STA 规则“谁创建谁释放”，所以**每台相机
  一个线程，open() / grab() / close() 全部在该线程内完成**，主线程只读缓存。
* 只保留**最新一帧**（latest-frame cache）：读端永远拿到最新数据，网页卡顿或
  刷新都不会造成延迟累积。
* 一个 worker 只打开一次设备；多个浏览器请求同一个 MJPEG 源时只是多读几份
  缓存 + 复用同一份已编码 JPEG，不会重复打开摄像头。

帧数据
--------------------------------------------------------------------------
``Frame`` 同时保留：
  gray       np.uint8[H, W]      原始灰度，未做任何拉伸/伪彩色
  seq        递增帧号
  mf_ts      Media Foundation 的流时间戳（100ns 为单位，原始值）
  host_ts    取到该帧时的主机墙钟时间 time.time()
  mono_ts    time.perf_counter()，用于本机单调计时
"""

from __future__ import annotations

import os
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# 定位仓库根目录下的 ir_camera.py
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
# api/ -> computer_code/ -> Low-Cost-Mocap-main/ -> <repo root>
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))

_IR_CAMERA_DIR = os.environ.get("MOCAP_IR_CAMERA_DIR", _REPO_ROOT)
if _IR_CAMERA_DIR not in sys.path:
    sys.path.insert(0, _IR_CAMERA_DIR)

try:
    import ir_camera as irc
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "找不到 ir_camera.py。请把仓库根目录加入 PYTHONPATH，"
        "或设置环境变量 MOCAP_IR_CAMERA_DIR 指向它。当前尝试的目录：%s" % _IR_CAMERA_DIR
    ) from exc


MF_TICK_SECONDS = 1e-7  # Media Foundation 时间单位是 100ns


@dataclass
class Frame:
    """一帧红外图 + 元数据。gray 为未处理的原始灰度。"""

    gray: np.ndarray
    seq: int
    mf_ts: int
    host_ts: float
    mono_ts: float

    @property
    def shape(self) -> tuple[int, int]:
        return int(self.gray.shape[1]), int(self.gray.shape[0])

    def meta(self) -> dict:
        return {
            "seq": self.seq,
            "width": int(self.gray.shape[1]),
            "height": int(self.gray.shape[0]),
            "mf_timestamp": int(self.mf_ts),
            "mf_timestamp_seconds": self.mf_ts * MF_TICK_SECONDS,
            "host_timestamp": self.host_ts,
            "monotonic_timestamp": self.mono_ts,
            "dtype": str(self.gray.dtype),
            "mean": float(self.gray.mean()) if self.gray.size else 0.0,
        }


class CameraWorker:
    """一台相机的采集线程。start() 之后用 latest()/wait_frame() 读最新帧。"""

    def __init__(self, index: int, width: int = 1280, height: int = 720,
                 fps: float = 30.0, pixel_format: str = "NV12",
                 name: str | None = None, reconnect: bool = True,
                 jpeg_quality: int = 80):
        self.index = int(index)
        self.requested = {
            "width": int(width), "height": int(height),
            "fps": float(fps), "pixel_format": pixel_format.upper(),
        }
        self.name_filter = name
        self.reconnect = reconnect
        self.jpeg_quality = int(jpeg_quality)

        self._cond = threading.Condition()
        self._frame: Frame | None = None
        self._jpeg: bytes | None = None
        self._jpeg_seq = -1
        self._seq = 0
        self._thread: threading.Thread | None = None
        self._stop_evt = threading.Event()

        self._fps = 0.0
        self._fps_count = 0
        self._fps_t0 = time.perf_counter()
        self._frames_total = 0

        self.state = "idle"          # idle / opening / streaming / error / stopped
        self.error: str | None = None
        self.format_info: str | None = None
        self.device_name: str | None = None
        self._opened_at: float | None = None

    # -- 生命周期 ---------------------------------------------------------
    def start(self) -> "CameraWorker":
        if self._thread and self._thread.is_alive():
            return self
        self._stop_evt.clear()
        self.state = "opening"
        self.error = None
        self._thread = threading.Thread(target=self._run, name="cam-%d" % self.index,
                                        daemon=True)
        self._thread.start()
        return self

    def stop(self, timeout: float = 3.0) -> None:
        self._stop_evt.set()
        with self._cond:
            self._cond.notify_all()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)
        self.state = "stopped"

    @property
    def alive(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    # -- 采集线程 ---------------------------------------------------------
    def _run(self) -> None:
        backoff = 0.5
        while not self._stop_evt.is_set():
            cam = None
            try:
                # 注意：必须在**本线程**内 open/close，COM STA 要求同线程创建与释放
                cam = irc.IRCamera(index=self.index, name=self.name_filter,
                                   verbose=False, **{
                                       "width": self.requested["width"],
                                       "height": self.requested["height"],
                                       "fps": self.requested["fps"],
                                       "pixel_format": self.requested["pixel_format"],
                                   })
                cam.open()
                self.state = "streaming"
                self.error = None
                self.format_info = cam.info
                self.device_name = self._device_name()
                self._opened_at = time.time()
                backoff = 0.5

                while not self._stop_evt.is_set():
                    gray, mf_ts = cam.grab()
                    if gray is None:
                        # 刚开流时可能还没有帧，稍等避免空转
                        time.sleep(0.002)
                        continue
                    self._publish(gray, mf_ts)

            except BaseException as exc:  # 断线 / 格式不支持 / UVC 被占用
                if self._stop_evt.is_set():
                    break
                self.state = "error"
                self.error = "%s: %s" % (type(exc).__name__, exc)
                self.format_info = None
                print("[cam %d] 采集异常：%s" % (self.index, self.error), file=sys.stderr)
                traceback.print_exc()
            finally:
                if cam is not None:
                    try:
                        cam.close()
                    except BaseException:
                        pass

            if self._stop_evt.is_set() or not self.reconnect:
                break
            # 断线重连（拔插 / 被其它程序占用后释放）
            self.state = "reconnecting"
            slept = 0.0
            while slept < backoff and not self._stop_evt.is_set():
                time.sleep(0.1)
                slept += 0.1
            backoff = min(backoff * 2, 5.0)

        self.state = "stopped"

    def _device_name(self) -> str | None:
        try:
            for i, friendly, _link in irc.list_devices():
                if i == self.index:
                    return friendly
        except BaseException:
            pass
        return None

    def _publish(self, gray: np.ndarray, mf_ts: int) -> None:
        now_wall = time.time()
        now_mono = time.perf_counter()
        with self._cond:
            self._seq += 1
            self._frame = Frame(gray=gray, seq=self._seq, mf_ts=int(mf_ts),
                                host_ts=now_wall, mono_ts=now_mono)
            self._cond.notify_all()

        self._frames_total += 1
        self._fps_count += 1
        dt = now_mono - self._fps_t0
        if dt >= 0.5:
            self._fps = self._fps_count / dt
            self._fps_count = 0
            self._fps_t0 = now_mono

    # -- 读端 -------------------------------------------------------------
    def latest(self) -> Frame | None:
        with self._cond:
            return self._frame

    def wait_frame(self, after_seq: int = -1, timeout: float = 1.0) -> Frame | None:
        """等一帧比 after_seq 更新的帧；超时返回 None（由调用方决定是否继续）。"""
        with self._cond:
            if self._frame is None or self._frame.seq <= after_seq:
                self._cond.wait(timeout)
            frame = self._frame
        if frame is None or frame.seq <= after_seq:
            return None
        return frame

    def jpeg(self) -> bytes | None:
        """把最新帧编码成 JPEG，同一帧只编码一次（多浏览器共享）。"""
        frame = self.latest()
        if frame is None:
            return None
        with self._cond:
            if self._jpeg_seq == frame.seq and self._jpeg is not None:
                return self._jpeg
        ok, buf = cv2.imencode(".jpg", frame.gray,
                               [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
        if not ok:
            return None
        data = buf.tobytes()
        with self._cond:
            self._jpeg, self._jpeg_seq = data, frame.seq
        return data

    def status(self) -> dict:
        frame = self.latest()
        return {
            "index": self.index,
            "name": self.device_name,
            "state": self.state,
            "error": self.error,
            "format": self.format_info,
            "requested": dict(self.requested),
            "fps": round(self._fps, 1),
            "frames_total": self._frames_total,
            "frame": frame.meta() if frame else None,
            "opened_at": self._opened_at,
        }


class CameraManager:
    """按设备索引管理多路相机。索引全局唯一，同一设备不会被打开两次。"""

    def __init__(self, jpeg_quality: int = 80):
        self._lock = threading.RLock()
        self._workers: dict[int, CameraWorker] = {}
        self._jpeg_quality = jpeg_quality
        self._shutdown = threading.Event()

    # -- 启停 -------------------------------------------------------------
    @property
    def stopping(self) -> bool:
        """进程是否正在退出（MJPEG 生成器据此收尾）。"""
        return self._shutdown.is_set()

    def start(self, indices, width=1280, height=720, fps=30.0,
              pixel_format="NV12", names=None) -> dict:
        indices = [int(i) for i in indices]
        if len(set(indices)) != len(indices):
            raise ValueError("设备索引重复：%s" % indices)

        self._shutdown.clear()
        with self._lock:
            running = {i for i, w in self._workers.items() if w.alive}
            dup = running & set(indices)
            if dup:
                raise ValueError("设备 %s 已在采集中，不能重复打开" % sorted(dup))

            for idx in indices:
                old = self._workers.pop(idx, None)
                if old is not None:
                    old.stop()
                worker = CameraWorker(
                    index=idx, width=width, height=height, fps=fps,
                    pixel_format=pixel_format,
                    name=(names or {}).get(str(idx)) or (names or {}).get(idx),
                    jpeg_quality=self._jpeg_quality,
                )
                self._workers[idx] = worker
                worker.start()
        return self.status()

    def stop(self, indices=None) -> dict:
        with self._lock:
            targets = list(self._workers) if indices is None else [int(i) for i in indices]
            for idx in targets:
                worker = self._workers.pop(idx, None)
                if worker is not None:
                    worker.stop()
        return self.status()

    def stop_all(self) -> None:
        self._shutdown.set()
        self.stop()

    # -- 读取 -------------------------------------------------------------
    def worker(self, index: int) -> CameraWorker | None:
        with self._lock:
            return self._workers.get(int(index))

    def status(self) -> dict:
        with self._lock:
            workers = list(self._workers.values())
        return {
            "cameras": [w.status() for w in workers],
            "count": len(workers),
        }

    def devices(self) -> list[dict]:
        """枚举当前系统里的视频采集设备（不打开它们）。"""
        out = []
        for i, friendly, link in irc.list_devices():
            busy = False
            with self._lock:
                w = self._workers.get(i)
                busy = bool(w and w.alive)
            out.append({"index": i, "name": friendly, "symbolic_link": link, "in_use": busy})
        return out

    # -- 保存原始帧 -------------------------------------------------------
    def save_raw(self, out_dir: str, indices=None) -> list[dict]:
        """把各路最新帧按**原始灰度**存 PNG（不拉伸、不伪彩色），带时间戳元数据。"""
        os.makedirs(out_dir, exist_ok=True)
        with self._lock:
            targets = list(self._workers) if indices is None else [int(i) for i in indices]
        saved = []
        for idx in targets:
            worker = self.worker(idx)
            frame = worker.latest() if worker else None
            if frame is None:
                continue
            stamp = datetime.fromtimestamp(frame.host_ts).strftime("%Y%m%d_%H%M%S_%f")[:-3]
            path = os.path.abspath(os.path.join(out_dir, "ir_cam%d_%s.png" % (idx, stamp)))
            if not cv2.imwrite(path, frame.gray):  # 8bit 灰度，原样写入
                raise OSError("写入失败：%s" % path)
            meta = frame.meta()
            meta.update({"index": idx, "path": path,
                         "host_time_iso": datetime.fromtimestamp(frame.host_ts).isoformat(),
                         "format": worker.format_info, "device": worker.device_name})
            saved.append(meta)
        return saved

    def shutdown(self) -> None:
        self.stop_all()


# 进程级单例：Flask 的多个请求共享同一份采集线程
_instance: CameraManager | None = None
_instance_lock = threading.Lock()


def manager() -> CameraManager:
    global _instance
    with _instance_lock:
        if _instance is None:
            # 带宽/CPU 紧张时可用 MOCAP_JPEG_QUALITY=60 降低单帧体积
            _instance = CameraManager(jpeg_quality=int(os.environ.get("MOCAP_JPEG_QUALITY", "80")))
        return _instance
