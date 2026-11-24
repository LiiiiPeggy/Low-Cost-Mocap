"""双 / 多 USB 红外相机并发采集管理器（Windows + Media Foundation）。

设计要点
--------------------------------------------------------------------------
* 复用同目录的 ``ir_camera.py``（设备枚举 + 单相机取帧 + COM/MF 资源管理），
  本模块只负责“多路调度”，不改动单相机 CLI 的行为。
* **逻辑编号**（0..3）是采集、同步、标定、LED、录制共用的身份；每台相机按
  ``camera_config.json`` 里的设备实例 key 打开，不依赖枚举顺序。
* Media Foundation 的采集对象按 COM STA 规则“谁创建谁释放”，所以**每台相机
  一个线程，open() / grab() / close() 全部在该线程内完成**，主线程只读缓存。
* 只保留**最新一帧**（latest-frame cache）：读端永远拿到最新数据，网页卡顿或
  刷新都不会造成延迟累积。
* 另外维护一份**定长历史缓存**（默认 45 帧，约 1.5s@30fps）供帧同步使用；
  两份缓存共享同一份 ndarray，不额外占内存（见 ``_publish``）。
* 一台相机离线时保留逻辑编号并持续重连，其他相机不补位。

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
from collections import deque
from dataclasses import dataclass
from datetime import datetime

import cv2
import numpy as np

import camera_controls
import ir_camera as irc


MF_TICK_SECONDS = 1e-7  # Media Foundation 时间单位是 100ns

# 历史缓存深度：45 帧 ≈ 1.5s@30fps。1280x720 uint8 单帧 0.92MB，
# 四路合计约 166MB；内存紧张可用 MOCAP_SYNC_BUFFER=30 调小。
DEFAULT_HISTORY = int(os.environ.get("MOCAP_SYNC_BUFFER", "45"))


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


def _resolve_device(device_key: str | None):
    """按设备实例 key 一次性取得 (枚举索引, 名称, symbolic link)。

    每次打开只枚举一次：``ir_camera.list_devices`` 会 MFStartup/MFShutdown，
    在 4 路并发时反复枚举会增加启动抖动。
    """
    if not device_key:
        return None
    try:
        for index, friendly, link in irc.list_devices():
            if camera_controls.device_key(link) == device_key:
                return int(index), friendly, link
    except Exception:
        return None
    return None


def _resolve_mf_index(device_key: str | None) -> int | None:
    resolved = _resolve_device(device_key)
    return resolved[0] if resolved else None


def _error_hint(message: str) -> str:
    """把常见 HRESULT 翻译成可操作提示；日志只打印一次/分钟，提示要够清楚。"""
    if "0xC00D3704" in message:
        return "；设备被其他程序占用或无法启动取流，请关闭其它相机程序/预览实例"
    return ""


class CameraWorker:
    """一台相机的采集线程。start() 之后用 latest()/wait_frame() 读最新帧。"""

    def __init__(self, index: int, width: int = 1280, height: int = 720,
                 fps: float = 30.0, pixel_format: str = "NV12",
                 name: str | None = None, reconnect: bool = True,
                 jpeg_quality: int = 80, history_len: int = DEFAULT_HISTORY,
                 device_key: str | None = None, mf_index: int | None = None):
        self.index = int(index)                     # 逻辑编号
        self.device_key = camera_controls.device_key(device_key) or None
        self.mf_index = mf_index                    # 最近一次解析到的枚举索引
        self.requested = {
            "width": int(width), "height": int(height),
            "fps": float(fps), "pixel_format": pixel_format.upper(),
        }
        self.name_filter = name
        self.reconnect = reconnect
        self.jpeg_quality = int(jpeg_quality)
        # 供帧同步使用；为 0 时不保留历史（省内存）
        self.history_len = max(0, int(history_len))
        self._history: deque[Frame] = deque(maxlen=self.history_len or None) \
            if self.history_len else deque()

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
        self._last_frame_mono: float | None = None
        self._intervals_ms: deque[float] = deque(maxlen=120)

        # idle / offline / opening / streaming / error / stopped
        self.state = "idle"
        self.error: str | None = None
        self.format_info: str | None = None
        self.device_name: str | None = None
        self.device_link: str | None = None
        self.control_restore: dict | None = None
        self._opened_at: float | None = None
        self._offline_since: float | None = None
        self._last_logged_error: str | None = None
        self._last_logged_at = 0.0

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

    @property
    def streaming(self) -> bool:
        return self.state == "streaming"

    # -- 采集线程 ---------------------------------------------------------
    def _run(self) -> None:
        backoff = 0.5
        while not self._stop_evt.is_set():
            cam = None
            try:
                friendly = link = None
                if self.device_key:
                    resolved = _resolve_device(self.device_key)
                    if resolved is None:
                        self._mark_offline("设备未连接（按设备实例标识未找到）")
                        if not self.reconnect:
                            break
                        self._sleep_backoff(backoff)
                        backoff = min(backoff * 2, 5.0)
                        continue
                    mf_index, friendly, link = resolved
                else:
                    mf_index = self.mf_index
                if mf_index is None:
                    self._mark_offline("设备未连接（没有枚举索引）")
                    if not self.reconnect:
                        break
                    self._sleep_backoff(backoff)
                    backoff = min(backoff * 2, 5.0)
                    continue
                self.mf_index = mf_index
                # 注意：必须在**本线程**内 open/close，COM STA 要求同线程创建与释放
                cam = irc.IRCamera(index=mf_index, name=self.name_filter,
                                   verbose=False, **{
                                       "width": self.requested["width"],
                                       "height": self.requested["height"],
                                       "fps": self.requested["fps"],
                                       "pixel_format": self.requested["pixel_format"],
                                   })
                cam.open()
                self.state = "streaming"
                self.error = None
                self._offline_since = None
                self.format_info = cam.info
                # 名称/link 来自打开前的同一次枚举，避免 open 之后再
                # MFStartup/MFShutdown 两次（4 路并发时会造成启动抖动）。
                self.device_name = friendly
                self.device_link = link
                if self.device_key is None:
                    self.device_name = self._device_name(mf_index)
                    self.device_link = self._device_link(mf_index)
                    if self.device_link:
                        self.device_key = camera_controls.device_key(
                            self.device_link)
                self._opened_at = time.time()
                # 按设备实例路径恢复上一次成功设置的曝光/增益；即使 USB
                # 枚举顺序变化，也不会把 cam0 的参数写到 cam1。
                try:
                    import camera_controls as _controls
                    self.control_restore = _controls.restore_controls(
                        self.index, self.device_link)
                except BaseException as exc:
                    self.control_restore = {
                        "ok": False,
                        "error": "%s: %s" % (type(exc).__name__, exc)}
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
                # 同一个错误最多每分钟打印一次：设备被其它程序占用时，
                # 重连循环会反复“打开成功 → ReadSample 失败”，只在错误
                # 变化时打印仍会每秒刷屏。
                now = time.time()
                if (self.error != self._last_logged_error
                        or now - self._last_logged_at > 60.0):
                    print("[cam %d] 采集异常：%s%s（将自动重连）"
                          % (self.index, self.error,
                             _error_hint(self.error)), file=sys.stderr)
                    self._last_logged_error = self.error
                    self._last_logged_at = now
                    if os.environ.get("MOCAP_DEBUG"):
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
            self._sleep_backoff(backoff)
            backoff = min(backoff * 2, 5.0)

        self.state = "stopped"

    def _mark_offline(self, reason: str) -> None:
        if self.state != "offline":
            print("[cam %d] %s" % (self.index, reason), file=sys.stderr)
        self.state = "offline"
        self.error = reason
        if self._offline_since is None:
            self._offline_since = time.time()

    def _sleep_backoff(self, seconds: float) -> None:
        slept = 0.0
        while slept < seconds and not self._stop_evt.is_set():
            time.sleep(0.1)
            slept += 0.1

    def _device_name(self, mf_index: int) -> str | None:
        try:
            for i, friendly, _link in irc.list_devices():
                if i == mf_index:
                    return friendly
        except BaseException:
            pass
        return None

    def _device_link(self, mf_index: int) -> str | None:
        try:
            for i, _friendly, link in irc.list_devices():
                if i == mf_index:
                    return link
        except BaseException:
            pass
        return None

    def _publish(self, gray: np.ndarray, mf_ts: int) -> None:
        now_wall = time.time()
        now_mono = time.perf_counter()
        with self._cond:
            if self._last_frame_mono is not None:
                dt_ms = (now_mono - self._last_frame_mono) * 1000.0
                if 0.0 < dt_ms < 2000.0:
                    self._intervals_ms.append(dt_ms)
            self._last_frame_mono = now_mono
        # NV12/YUY2 的 Y 平面是跨步视图（[:, ::2]），会连带持有整块 1.38MB 的
        # NV12 buffer。转成连续数组后：缓存 45 帧只占 45*0.92MB，
        # 且 cv2 编码/写盘不用再做隐式转换（对已连续的 MJPG 则是零拷贝）。
        if not gray.flags.c_contiguous:
            gray = np.ascontiguousarray(gray)
        frame = Frame(gray=gray, seq=self._seq + 1, mf_ts=int(mf_ts),
                      host_ts=now_wall, mono_ts=now_mono)
        with self._cond:
            self._seq += 1
            self._frame = frame
            if self.history_len:
                self._history.append(frame)   # 与 latest 共享同一个 ndarray
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

    def history_after(self, after_seq: int) -> list[Frame]:
        """返回历史缓存中 seq > after_seq 的帧（按时间升序，最老在前）。

        供帧同步配对使用。在锁内复制一份列表，避免调用方遍历时 deque 被
        采集线程改动（deque 迭代期间被修改会抛 RuntimeError）。
        """
        if not self.history_len:
            return []
        with self._cond:
            return [f for f in self._history if f.seq > after_seq]

    @property
    def buffered(self) -> int:
        """当前历史缓存里有多少帧。"""
        with self._cond:
            return len(self._history)

    @property
    def frames_total(self) -> int:
        return self._frames_total

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
        with self._cond:
            intervals = list(self._intervals_ms)
        if intervals:
            arr = np.asarray(intervals, np.float64)
            interval = {
                "mean_ms": round(float(arr.mean()), 2),
                "p95_ms": round(float(np.percentile(arr, 95)), 2),
                "max_ms": round(float(arr.max()), 2),
                "min_ms": round(float(arr.min()), 2),
                "samples": int(arr.size),
                "delivered_fps": round(1000.0 / float(arr.mean()), 2)
                if arr.mean() > 0 else None,
            }
        else:
            interval = None
        return {
            "index": self.index,                 # 逻辑编号
            "logical_index": self.index,
            "mf_index": self.mf_index,
            "device_key": self.device_key,
            "name": self.device_name,
            "device_link": self.device_link,
            "control_restore": self.control_restore,
            "state": self.state,
            "error": self.error,
            "format": self.format_info,
            "requested": dict(self.requested),
            "fps": round(self._fps, 1),
            "frame_interval_ms": interval,
            "frames_total": self._frames_total,
            "buffered": self.buffered,
            "frame": frame.meta() if frame else None,
            "opened_at": self._opened_at,
        }


class CameraManager:
    """按**逻辑编号**管理多路相机。逻辑编号全局唯一，同一设备不会被打开两次。"""

    def __init__(self, jpeg_quality: int = 80):
        self._lock = threading.RLock()
        self._workers: dict[int, CameraWorker] = {}
        self._jpeg_quality = jpeg_quality
        self._shutdown = threading.Event()

    # -- 启停 -------------------------------------------------------------
    @property
    def stopping(self) -> bool:
        return self._shutdown.is_set()

    def start(self, configs, width=None, height=None, fps=None,
              pixel_format=None, names=None, history_len=None) -> dict:
        """启动相机。

        ``configs`` 可以是 ``[{index, device_key, width, height, fps, format}]``
        （推荐），也可以是旧的整数索引列表（此时用 width/height/fps/format）。
        """
        if configs and isinstance(configs[0], dict):
            entries = []
            for item in configs:
                entries.append({
                    "index": int(item["index"]),
                    "device_key": camera_controls.device_key(
                        item.get("device_key")) or None,
                    "width": int(item.get("width", width or 1280)),
                    "height": int(item.get("height", height or 720)),
                    "fps": float(item.get("fps", fps or 30.0)),
                    "format": str(item.get("format")
                                  or pixel_format or "NV12").upper(),
                })
        else:
            entries = [{"index": int(i), "device_key": None,
                        "width": int(width or 1280),
                        "height": int(height or 720),
                        "fps": float(fps or 30.0),
                        "format": str(pixel_format or "NV12").upper()}
                       for i in configs]
        idxs = [e["index"] for e in entries]
        if len(set(idxs)) != len(idxs):
            raise ValueError("逻辑编号重复：%s" % idxs)
        keys = [e["device_key"] for e in entries if e["device_key"]]
        if len(set(keys)) != len(keys):
            raise ValueError("同一台物理设备不能绑定两个逻辑编号")

        hist = DEFAULT_HISTORY if history_len is None else int(history_len)

        self._shutdown.clear()
        with self._lock:
            running = {i for i, w in self._workers.items() if w.alive}
            dup = running & set(idxs)
            if dup:
                raise ValueError("相机 %s 已在采集中，不能重复打开" % sorted(dup))

            for entry in entries:
                idx = entry["index"]
                old = self._workers.pop(idx, None)
                if old is not None:
                    old.stop()
                worker = CameraWorker(
                    index=idx, width=entry["width"], height=entry["height"],
                    fps=entry["fps"], pixel_format=entry["format"],
                    name=(names or {}).get(str(idx)) or (names or {}).get(idx),
                    jpeg_quality=self._jpeg_quality,
                    history_len=hist, device_key=entry["device_key"],
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

    def workers(self) -> list[CameraWorker]:
        with self._lock:
            return [self._workers[k] for k in sorted(self._workers)]

    def status(self) -> dict:
        with self._lock:
            workers = [self._workers[k] for k in sorted(self._workers)]
        return {
            "cameras": [w.status() for w in workers],
            "count": len(workers),
            "indices": [w.index for w in workers],
        }

    def devices(self) -> list[dict]:
        """枚举当前系统里的视频采集设备（不打开它们），并标出逻辑绑定。"""
        with self._lock:
            bound = {w.device_key: w.index for w in self._workers.values()
                     if w.device_key}
        out = []
        for i, friendly, link in irc.list_devices():
            key = camera_controls.device_key(link)
            out.append({"mf_index": i, "name": friendly, "symbolic_link": link,
                        "device_key": key, "logical_index": bound.get(key),
                        "in_use": key in bound})
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
                         "format": worker.format_info, "device": worker.device_name,
                         "device_key": worker.device_key})
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
