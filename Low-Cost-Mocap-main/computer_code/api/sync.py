"""双相机软件帧同步：基于**主机单调时间戳**的流式最近邻配对。

==========================================================================
时间戳的边界（这一节决定了本模块能声称什么、不能声称什么）
==========================================================================
1. 配对只用 ``time.perf_counter()``（monotonic）——同一进程、同一时钟、同一
   时间基准。**不用** Wall clock，避免系统对时导致时间轴跳变。

2. ``mf_timestamp`` 是每台相机**各自**的 Media Foundation 流时间轴，起点由
   驱动/会话决定（本机实测两台从 67289s 这种"开机以来"的量级起步，但起点
   是否一致没有保证）。因此**跨设备直接相减是没有意义的**，它只用于：
     - 单机内检测丢帧 / 间隔抖动（同设备相邻 mf_ts 之差 ≈ 帧周期）；
     - 作为溯源信息随图保存，便于事后回放。
   本次配对**完全不使用** mf_ts。

3. 时间戳记录的是**帧交付给用户态的时刻**（``ReadSample`` 返回、解码完成的
   时刻），**不是曝光中心时刻**：
       交付时刻 = 曝光中心 + 曝光时长 + 传感器读出 + USB 传输 + 驱动缓冲
   注意 USB 2.0/3.0 的传输抖动、MJPEG 解码耗时都会进入这个差值，而且它随
   曝光参数、帧率、负载变化，**不是常数偏移**，所以无法用一个固定量补偿。

4. 因此这里实现的是**软件级近似同步**，不是硬件同步：
     - 本机 30fps 下，帧周期 33.3ms，时间量化误差本身就有 ±16.7ms 量级；
     - 加上第 3 条的交付延迟差，单对帧的"真实曝光时间差"可能比报告的
       Δt 更大，且方向未知；
     - 结论：可满足"同一时刻的两视角大致对应"（标定板静止、低速运动），
       **不能**用于快速运动目标的亚毫秒三角化。要更高精度需要硬件触发
       （共享时钟/闪光同步）或 LED 检测做偏差标定，本模块不涉及。

配对策略
--------------------------------------------------------------------------
* 每台相机维护定长环形缓存（默认 45 帧，约 1.5s@30fps）；
* 两路各自有一个"已消费指针"（按 seq 单调推进），保证**帧不重复使用**、
  **配对时间戳单调不减**；
* 双指针贪心：比较两路各自最老的未消费帧，
    |Δt| ≤ max_dt        → 配对，双双前进；
    Δt > 0（A 更老）     → A 落单，丢弃 A 并计入 dropped；
    Δt < 0（B 更老）     → B 落单，丢弃 B 并计入 dropped。
  因为始终丢掉"更老的那一帧"，所以留下的最新帧对必然是时间上最接近的，
  这就是流式条件下的最近邻；
* 一路停流超过 ``stall_s`` 时，把另一路积压的帧全部计为丢弃，避免统计失真。
"""

from __future__ import annotations

import json
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

from camera_manager import Frame, manager


DEFAULT_MAX_DT_MS = float(os.environ.get("MOCAP_SYNC_MAX_DT_MS", "20"))
DEFAULT_BUFFER = int(os.environ.get("MOCAP_SYNC_BUFFER", "45"))
DEFAULT_WINDOW = int(os.environ.get("MOCAP_SYNC_WINDOW", "300"))
DEFAULT_RECENT = int(os.environ.get("MOCAP_SYNC_RECENT", "16"))
DEFAULT_OUT_DIR = os.environ.get("MOCAP_SYNC_DIR", os.path.join(
    os.environ.get("MOCAP_CAPTURE_DIR", os.path.abspath("captures")), "sync"))


@dataclass
class PairRecord:
    """一对已配对的帧。持有 Frame 引用，所以只保留最近若干对（默认 16）。"""

    pair_seq: int
    dt_ms: float                 # b.mono - a.mono，有符号，正数表示 B 更晚
    frame_a: Frame
    frame_b: Frame
    created_mono: float

    def meta(self) -> dict:
        return {
            "pair_seq": self.pair_seq,
            "dt_ms": round(self.dt_ms, 3),
            "dt_definition": "b.monotonic_timestamp - a.monotonic_timestamp",
            "a": dict(self.frame_a.meta(), index=None),
            "b": dict(self.frame_b.meta(), index=None),
        }


class SyncService:
    """后台常驻的配对线程 + 统计。线程安全：统计只在自身线程写，读端拿快照。"""

    def __init__(self, idx_a: int = 0, idx_b: int = 1,
                 max_dt_ms: float = DEFAULT_MAX_DT_MS,
                 window: int = DEFAULT_WINDOW,
                 recent: int = DEFAULT_RECENT,
                 poll_interval: float = 0.01,
                 stall_s: float = 0.5):
        self.idx_a = int(idx_a)
        self.idx_b = int(idx_b)
        self.max_dt_ms = float(max_dt_ms)
        self.window = int(window)
        self.stall_s = float(stall_s)
        self.poll_interval = float(poll_interval)

        self._lock = threading.Lock()
        self._stop = threading.Event()

        # 已消费指针（seq，单调递增）
        self._last_a = 0
        self._last_b = 0

        # 统计
        self._pairs_total = 0
        self._dt_ms: deque[float] = deque(maxlen=self.window)        # |Δt|
        self._dt_signed: deque[float] = deque(maxlen=self.window)    # 有符号 Δt
        self._dropped_a = 0
        self._dropped_b = 0
        self._recent: deque[PairRecord] = deque(maxlen=int(recent))

        # 配对率基线：同步启动那一刻两路各已产出多少帧
        self._base_a: int | None = None
        self._base_b: int | None = None

        self._last_pair_mono: float | None = None
        self._error: str | None = None

        self._thread = threading.Thread(target=self._loop, name="sync-pairer", daemon=True)
        self._thread.start()

    # -- 生命周期 ---------------------------------------------------------
    def stop(self) -> None:
        self._stop.set()

    @property
    def alive(self) -> bool:
        return self._thread.is_alive()

    def reset_pointers(self) -> None:
        """相机重启后调用：清空指针与统计（历史缓存也不会再增长）。"""
        with self._lock:
            self._last_a = 0
            self._last_b = 0
            self._pairs_total = 0
            self._dt_ms.clear()
            self._dt_signed.clear()
            self._dropped_a = 0
            self._dropped_b = 0
            self._recent.clear()
            self._base_a = None
            self._base_b = None

    def set_max_dt(self, max_dt_ms: float) -> float:
        v = max(1.0, min(500.0, float(max_dt_ms)))
        with self._lock:
            self.max_dt_ms = v
        return v

    # -- 配对线程 ---------------------------------------------------------
    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._tick()
                self._error = None
            except BaseException as exc:  # 单次异常不应该让同步线程死掉
                self._error = "%s: %s" % (type(exc).__name__, exc)
                traceback.print_exc()
            time.sleep(self.poll_interval)

    def _tick(self) -> None:
        m = manager()
        wa = m.worker(self.idx_a)
        wb = m.worker(self.idx_b)
        if wa is None or wb is None:
            return
        if wa.state != "streaming" or wb.state != "streaming":
            return

        la = wa.latest()
        lb = wb.latest()
        if la is None or lb is None:
            return

        # 配对率基线取"缓存里最老的帧之前"，否则缓存中早于首次 tick 的帧会被配对
        # 却不计入分母，导致配对数 > 产生数（>100% 的假值）。
        if self._base_a is None or self._base_b is None:
            ha = wa.history_after(0)
            hb = wb.history_after(0)
            with self._lock:
                if self._base_a is None:
                    self._base_a = (ha[0].seq - 1) if ha else wa.frames_total
                if self._base_b is None:
                    self._base_b = (hb[0].seq - 1) if hb else wb.frames_total

        with self._lock:
            last_a, last_b = self._last_a, self._last_b
            max_dt = self.max_dt_ms / 1000.0

        # 相机重启后 seq 从 1 重新计数，若指针还在旧的大值上，配对会永久停摆。
        # 检测到倒退就就地重置（不依赖调用方显式通知）。
        if la.seq < last_a:
            with self._lock:
                self._last_a = 0
                self._base_a = wa.frames_total
            last_a = 0
        if lb.seq < last_b:
            with self._lock:
                self._last_b = 0
                self._base_b = wb.frames_total
            last_b = 0

        A = wa.history_after(last_a)
        B = wb.history_after(last_b)
        if not A and not B:
            return

        now = time.perf_counter()
        i = j = 0
        while i < len(A) and j < len(B):
            a, b = A[i], B[j]
            dt = b.mono_ts - a.mono_ts          # 有符号：正数表示 B 更晚
            if abs(dt) <= max_dt:
                self._emit(a, b, dt)
                i += 1
                j += 1
            elif dt > 0:
                self._drop("a")
                i += 1                            # A 太老，丢掉
            else:
                self._drop("b")
                j += 1                            # B 太老，丢掉

        # 一路停流：另一路的积压帧不可能再配上，按丢弃计，避免配对率虚高
        if i < len(A) and (now - lb.mono_ts) > self.stall_s:
            with self._lock:
                self._dropped_a += len(A) - i
                self._last_a = A[-1].seq
            i = len(A)
        elif i:
            with self._lock:
                self._last_a = A[i - 1].seq

        if j < len(B) and (now - la.mono_ts) > self.stall_s:
            with self._lock:
                self._dropped_b += len(B) - j
                self._last_b = B[-1].seq
            j = len(B)
        elif j:
            with self._lock:
                self._last_b = B[j - 1].seq

    def _emit(self, a: Frame, b: Frame, dt: float) -> None:
        dt_ms = dt * 1000.0
        with self._lock:
            self._pairs_total += 1
            self._dt_ms.append(abs(dt_ms))
            self._dt_signed.append(dt_ms)
            self._last_pair_mono = time.perf_counter()
            self._recent.append(PairRecord(
                pair_seq=self._pairs_total, dt_ms=dt_ms,
                frame_a=a, frame_b=b, created_mono=self._last_pair_mono,
            ))

    def _drop(self, which: str) -> None:
        with self._lock:
            if which == "a":
                self._dropped_a += 1
            else:
                self._dropped_b += 1

    # -- 读取 -------------------------------------------------------------
    def status(self) -> dict:
        m = manager()
        wa = m.worker(self.idx_a)
        wb = m.worker(self.idx_b)

        with self._lock:
            dt = np.asarray(self._dt_ms, dtype=np.float64)
            signed = np.asarray(self._dt_signed, dtype=np.float64)
            pairs_total = self._pairs_total
            dropped_a, dropped_b = self._dropped_a, self._dropped_b
            max_dt = self.max_dt_ms
            base_a, base_b = self._base_a, self._base_b
            last_pair = self._last_pair_mono
            recent = list(self._recent)

        produced_a = (wa.frames_total - base_a) if (wa and base_a is not None) else 0
        produced_b = (wb.frames_total - base_b) if (wb and base_b is not None) else 0

        # 配对率 = 配对数 / 已消费帧数。每帧要么配对要么被丢弃，所以这个分母
        # 与分子同源，不会出现 >100% 的假值（produced 仅作参考展示）。
        consumed_a = pairs_total + dropped_a
        consumed_b = pairs_total + dropped_b
        denom = max(consumed_a, consumed_b)

        stats = {
            "mean": float(dt.mean()) if dt.size else None,
            "p95": float(np.percentile(dt, 95)) if dt.size else None,
            "max": float(dt.max()) if dt.size else None,
            "min": float(dt.min()) if dt.size else None,
            "signed_mean": float(signed.mean()) if signed.size else None,
        }
        now = time.perf_counter()
        return {
            "running": self.alive,
            "error": self._error,
            "cameras": [self.idx_a, self.idx_b],
            "max_dt_ms": round(max_dt, 2),
            "buffer_frames": DEFAULT_BUFFER,
            "window_pairs": self.window,
            "pairs_total": pairs_total,
            "pairs_in_window": int(dt.size),
            "dt_ms": {k: (round(v, 3) if v is not None else None) for k, v in stats.items()},
            "pair_rate": round(pairs_total / denom, 4) if denom else None,
            "pair_rate_definition": "配对成功数 / 已消费帧数（配对+丢弃），两路取较大者",
            "produced": {"a": produced_a, "b": produced_b},
            "consumed": {"a": consumed_a, "b": consumed_b},
            "dropped": {"a": dropped_a, "b": dropped_b},
            "buffered": {
                "a": wa.buffered if wa else 0,
                "b": wb.buffered if wb else 0,
            },
            "last_dt_ms": round(recent[-1].dt_ms, 3) if recent else None,
            "last_pair_age_ms": (round((now - last_pair) * 1000.0, 1)
                                 if last_pair is not None else None),
            "savable_pairs": len(recent),
            "notes": [
                "Δt = b.monotonic_timestamp - a.monotonic_timestamp（同一主机同一时钟）",
                "mf_timestamp 为各设备自身的 MF 流时间轴，跨设备不可相减，未参与配对",
                "时间戳是帧交付时刻，非曝光中心时刻；本功能为软件级近似同步，非硬件同步",
            ],
        }

    # -- 保存 -------------------------------------------------------------
    def save_pairs(self, out_dir: str | None = None, count: int = 8) -> dict:
        """保存最近若干对同步帧：原图 PNG（未拉伸/未伪彩色）+ 时间戳 JSON。"""
        out_dir = out_dir or DEFAULT_OUT_DIR
        count = max(1, int(count))
        with self._lock:
            recs = list(self._recent)[-count:]
            max_dt = self.max_dt_ms

        if not recs:
            return {"saved": [], "count": 0, "out_dir": os.path.abspath(out_dir),
                    "message": "还没有配对成功的帧（相机是否在采集？）"}

        os.makedirs(out_dir, exist_ok=True)
        saved = []
        for k, rec in enumerate(recs, 1):
            tag = "pair_%03d" % k
            name_a = "%s_A_cam%d.png" % (tag, self.idx_a)
            name_b = "%s_B_cam%d.png" % (tag, self.idx_b)
            path_a = os.path.abspath(os.path.join(out_dir, name_a))
            path_b = os.path.abspath(os.path.join(out_dir, name_b))
            # 8bit 单通道原样写入
            if not cv2.imwrite(path_a, rec.frame_a.gray):
                raise OSError("写入失败：%s" % path_a)
            if not cv2.imwrite(path_b, rec.frame_b.gray):
                raise OSError("写入失败：%s" % path_b)

            entry = {
                "pair_seq": rec.pair_seq,
                "dt_ms": round(rec.dt_ms, 3),
                "max_dt_ms": round(max_dt, 2),
                "a": self._side_meta(rec.frame_a, self.idx_a, name_a),
                "b": self._side_meta(rec.frame_b, self.idx_b, name_b),
            }
            with open(os.path.abspath(os.path.join(out_dir, tag + ".json")), "w",
                      encoding="utf-8") as fh:
                json.dump(entry, fh, ensure_ascii=False, indent=2)
            saved.append(entry)

        manifest = {
            "created": datetime.now().isoformat(),
            "pairs": len(saved),
            "out_dir": os.path.abspath(out_dir),
            "max_dt_ms": round(max_dt, 2),
            "time_base": "host monotonic (time.perf_counter), same process",
            "dt_definition": "b.monotonic_timestamp - a.monotonic_timestamp",
            "caveats": [
                "时间戳为帧交付时刻（ReadSample 返回），不是曝光中心时刻",
                "真实曝光时刻差 = 交付时刻差 + 两路(曝光+读出+传输+缓冲)延迟之差，该差值非常数",
                "mf_timestamp 属各设备自身时间轴，跨设备不可相减",
                "本数据为软件级近似同步，未使用硬件触发/共享时钟",
            ],
            "entries": saved,
        }
        with open(os.path.abspath(os.path.join(out_dir, "manifest.json")), "w",
                  encoding="utf-8") as fh:
            json.dump(manifest, fh, ensure_ascii=False, indent=2)

        return {"saved": saved, "count": len(saved),
                "out_dir": os.path.abspath(out_dir)}

    def _side_meta(self, frame: Frame, index: int, filename: str) -> dict:
        worker = manager().worker(index)
        meta = frame.meta()
        meta.update({
            "index": index,
            "file": filename,
            "host_time_iso": datetime.fromtimestamp(frame.host_ts).isoformat(),
            "device": worker.device_name if worker else None,
            "format": worker.format_info if worker else None,
        })
        return meta


# 进程级单例（Flask 多请求共享同一条配对线程）
_instance: SyncService | None = None
_instance_lock = threading.Lock()


def sync_service() -> SyncService:
    global _instance
    with _instance_lock:
        if _instance is None:
            _instance = SyncService()
        return _instance
