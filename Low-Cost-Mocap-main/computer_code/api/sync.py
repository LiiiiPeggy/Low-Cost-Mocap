"""多相机软件帧同步：基于**主机单调时间戳**的流式最近邻分组。

==========================================================================
时间戳的边界（这一节决定了本模块能声称什么、不能声称什么）
==========================================================================
1. 分组只用 ``time.perf_counter()``（monotonic）——同一进程、同一时钟、同一
   时间基准。**不用** Wall clock，避免系统对时导致时间轴跳变。

2. ``mf_timestamp`` 是每台相机**各自**的 Media Foundation 流时间轴，起点由
   驱动/会话决定，**跨设备直接相减没有意义**，只用于单机内检测丢帧和溯源。
   分组**完全不使用** mf_ts。

3. 时间戳记录的是**帧交付给用户态**的时刻，**不是曝光中心时刻**：
       交付时刻 = 曝光中心 + 曝光时长 + 传感器读出 + USB 传输 + 驱动缓冲
   该差值随曝光参数、帧率、负载变化，不是常数偏移，无法固定补偿。

4. 因此这里是**软件级近似同步**，不是硬件同步：30fps 下半帧量化误差就有
   ±16.7ms。适用于静止标定板 / 低速目标，**不能**用于快速运动目标的亚毫秒
   三角化。

分组策略
--------------------------------------------------------------------------
* 每台相机维护定长环形缓存（默认 45 帧，约 1.5s@30fps）；
* 每台相机一个"已消费指针"（按 seq 单调推进），帧不重复使用；
* 每次取**最老的未消费帧**作为锚点，其他相机各取与锚点时间差 ≤ ``max_dt``
  的最近一帧；组内至少 2 台相机才输出；
* 组内跨度 = 最大时间戳 − 最小时间戳；
* 某台相机离线/丢帧时，剩余相机照常组成三相机或双相机帧组，不长期等待；
* 一路停流超过 ``stall_s`` 时，积压帧按丢弃计，避免统计失真。
"""

from __future__ import annotations

import json
import os
import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass, field
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
MAX_GROUP_CAMERAS = 4


@dataclass
class GroupRecord:
    """一组已同步的帧（2..N 台相机）。持有 Frame 引用，只保留最近若干组。"""

    group_seq: int
    frames: dict[int, Frame]          # 逻辑相机编号 -> 帧
    span_ms: float                    # max(mono) - min(mono)
    created_mono: float
    dropped_before: dict[int, int] = field(default_factory=dict)

    @property
    def pair_seq(self) -> int:
        """兼容旧字段名。"""
        return self.group_seq

    @property
    def dt_ms(self) -> float:
        return self.span_ms

    @property
    def cameras(self) -> list[int]:
        return sorted(self.frames)

    @property
    def frame_a(self) -> Frame:
        return self.frames[self.cameras[0]]

    @property
    def frame_b(self) -> Frame:
        return self.frames[self.cameras[1]]

    def frame(self, camera: int) -> Frame | None:
        return self.frames.get(int(camera))

    def meta(self) -> dict:
        cams = self.cameras
        return {
            "group_seq": self.group_seq,
            "pair_seq": self.group_seq,      # 兼容旧字段名
            "cameras": cams,
            "span_ms": round(self.span_ms, 3),
            "span_definition": "max(monotonic) - min(monotonic) within the group",
            "frames": {str(c): dict(self.frames[c].meta(), index=c)
                       for c in cams},
            "dropped_before": {str(k): int(v)
                               for k, v in self.dropped_before.items()},
        }


class SyncService:
    """后台常驻的分组线程 + 统计。线程安全：统计只在自身线程写，读端拿快照。"""

    def __init__(self, indices=None, max_dt_ms: float = DEFAULT_MAX_DT_MS,
                 window: int = DEFAULT_WINDOW,
                 recent: int = DEFAULT_RECENT,
                 poll_interval: float = 0.01,
                 stall_s: float = 0.5):
        self.indices: list[int] = []
        self.set_indices(indices or [0, 1])
        self.max_dt_ms = float(max_dt_ms)
        self.window = int(window)
        self.stall_s = float(stall_s)
        self.poll_interval = float(poll_interval)

        self._lock = threading.Lock()
        self._stop = threading.Event()

        # 已消费指针（每台相机一个 seq，单调递增）
        self._last: dict[int, int] = {i: 0 for i in self.indices}

        # 统计
        self._groups_total = 0
        self._span_ms: deque[float] = deque(maxlen=self.window)
        self._signed_span: deque[float] = deque(maxlen=self.window)
        self._dropped: dict[int, int] = {i: 0 for i in self.indices}
        self._recent: deque[GroupRecord] = deque(maxlen=int(recent))
        self._recent_events: deque[str] = deque(maxlen=300)
        self._group_sizes: dict[str, int] = {}
        # 实测分组/帧速率（最近 2 s），区分“组/秒”和“相机帧/秒”
        self._group_times: deque[float] = deque(maxlen=self.window)
        self._group_frames: deque[int] = deque(maxlen=self.window)

        # 基线：同步启动那一刻各相机已产出多少帧
        self._base: dict[int, int | None] = {i: None for i in self.indices}

        self._last_group_mono: float | None = None
        self._error: str | None = None

        self._thread = threading.Thread(target=self._loop, name="sync-grouper",
                                        daemon=True)
        self._thread.start()

    # -- 生命周期 ---------------------------------------------------------
    def set_indices(self, indices) -> list[int]:
        idxs = [int(i) for i in indices]
        with getattr(self, "_lock", threading.Lock()):
            self.indices = idxs
            if hasattr(self, "_last"):
                self._last = {i: self._last.get(i, 0) for i in idxs}
                self._dropped = {i: self._dropped.get(i, 0) for i in idxs}
                self._base = {i: self._base.get(i) for i in idxs}
        return idxs

    def stop(self) -> None:
        self._stop.set()

    @property
    def alive(self) -> bool:
        return self._thread.is_alive()

    def reset_pointers(self) -> None:
        """相机重启后调用：清空指针与统计（历史缓存也不会再增长）。"""
        with self._lock:
            self._last = {i: 0 for i in self.indices}
            self._groups_total = 0
            self._span_ms.clear()
            self._signed_span.clear()
            self._dropped = {i: 0 for i in self.indices}
            self._recent.clear()
            self._recent_events.clear()
            self._base = {i: None for i in self.indices}
            self._group_sizes = {}
            self._group_times.clear()
            self._group_frames.clear()

    def set_max_dt(self, max_dt_ms: float) -> float:
        v = max(1.0, min(500.0, float(max_dt_ms)))
        with self._lock:
            self.max_dt_ms = v
        return v

    # -- 分组线程 ---------------------------------------------------------
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
        workers = {}
        for i in self.indices:
            w = m.worker(i)
            if w is not None and w.state == "streaming":
                workers[i] = w
        if len(workers) < 2:
            return

        # 基线取"缓存里最老的帧之前"，否则缓存中早于首次 tick 的帧会被配对
        # 却不计入分母，导致配对数 > 产生数（>100% 的假值）。
        with self._lock:
            base_missing = [i for i in workers if self._base.get(i) is None]
        for i in base_missing:
            w = workers[i]
            h = w.history_after(0)
            with self._lock:
                if self._base.get(i) is None:
                    self._base[i] = (h[0].seq - 1) if h else w.frames_total

        # 每台相机：取未消费帧；检测 seq 倒退（相机重启）
        buffers: dict[int, list[Frame]] = {}
        with self._lock:
            last_map = dict(self._last)
            max_dt = self.max_dt_ms / 1000.0
        for i, w in workers.items():
            last = last_map.get(i, 0)
            latest = w.latest()
            if latest is not None and latest.seq < last:
                with self._lock:
                    self._last[i] = 0
                    self._base[i] = w.frames_total
                last = 0
            buffers[i] = w.history_after(last)
        if not any(buffers.values()):
            return

        now = time.perf_counter()
        # 停流相机：积压帧不可能再配上，直接计入丢弃
        for i, w in workers.items():
            latest = w.latest()
            if latest is not None and (now - latest.mono_ts) > self.stall_s:
                frames = buffers.get(i) or []
                if frames:
                    self._drop_many(i, len(frames))
                    with self._lock:
                        self._last[i] = frames[-1].seq
                    buffers[i] = []

        while True:
            heads = [(buf[0].mono_ts, i) for i, buf in buffers.items() if buf]
            if not heads:
                break
            # 以每台相机的最老未消费帧作为候选窗口起点，选**参与相机最多、
            # 跨度最小**的一组；这样不会因为固定以某台相机为锚点而漏掉
            # 其它相机（4 台相机相位不同时尤其明显）。
            best = None            # (n_cameras, -span, start, group{ cam: (j, frame) })
            for start, _cam in heads:
                group = {}
                for i, buf in buffers.items():
                    j = 0
                    while j < len(buf) and buf[j].mono_ts < start - 1e-9:
                        j += 1
                    if j < len(buf) and buf[j].mono_ts <= start + max_dt:
                        group[i] = (j, buf[j])
                if len(group) < 2:
                    continue
                times = [f.mono_ts for _j, f in group.values()]
                span = max(times) - min(times)
                key = (len(group), -span)
                if best is None or key > best[0]:
                    best = (key, start, group)
            if best is None:
                # 最老的帧无法与任何相机配对：丢掉一帧后继续
                _, cam = min(heads)
                self._drop_many(cam, 1)
                with self._lock:
                    self._last[cam] = buffers[cam][0].seq
                del buffers[cam][:1]
                continue
            _key, _start, group = best
            self._emit({c: f for c, (_j, f) in group.items()})
            for i, (j, frame) in group.items():
                with self._lock:
                    self._last[i] = frame.seq
                del buffers[i][:j + 1]
            # 未被选中的相机：丢掉已经比窗口起点还老、且不会再被任何窗口
            # 选中的帧（它们比当前窗口早超过 max_dt）。
            horizon = _start + max_dt
            for i, buf in buffers.items():
                if i in group:
                    continue
                k = 0
                while k < len(buf) and buf[k].mono_ts < _start:
                    k += 1
                if k:
                    self._drop_many(i, k)
                    with self._lock:
                        self._last[i] = buf[k - 1].seq
                    del buf[:k]
                _ = horizon

    def _emit(self, group: dict[int, Frame]) -> None:
        times = [f.mono_ts for f in group.values()]
        span_ms = (max(times) - min(times)) * 1000.0
        signed = (max(times) - min(times)) * 1000.0
        with self._lock:
            self._groups_total += 1
            self._span_ms.append(abs(span_ms))
            self._signed_span.append(signed)
            self._recent_events.append("group")
            self._last_group_mono = time.perf_counter()
            key = ",".join(str(c) for c in sorted(group))
            self._group_sizes[key] = self._group_sizes.get(key, 0) + 1
            self._group_times.append(time.perf_counter())
            self._group_frames.append(len(group))
            self._recent.append(GroupRecord(
                group_seq=self._groups_total, frames=dict(group),
                span_ms=span_ms, created_mono=self._last_group_mono,
            ))

    def _drop_many(self, camera: int, count: int) -> None:
        if count <= 0:
            return
        with self._lock:
            self._dropped[camera] = self._dropped.get(camera, 0) + int(count)
            for _ in range(min(int(count), 300)):
                self._recent_events.append("drop_%d" % int(camera))

    # -- 读取 -------------------------------------------------------------
    def status(self) -> dict:
        m = manager()
        workers = {i: m.worker(i) for i in self.indices}

        with self._lock:
            span = np.asarray(self._span_ms, dtype=np.float64)
            signed = np.asarray(self._signed_span, dtype=np.float64)
            groups_total = self._groups_total
            dropped = dict(self._dropped)
            max_dt = self.max_dt_ms
            base = dict(self._base)
            last_group = self._last_group_mono
            recent = list(self._recent)
            recent_events = list(self._recent_events)
            group_sizes = dict(self._group_sizes)
            group_times = list(self._group_times)
            group_frames = list(self._group_frames)

        stats = {
            "mean": float(span.mean()) if span.size else None,
            "p95": float(np.percentile(span, 95)) if span.size else None,
            "max": float(span.max()) if span.size else None,
            "min": float(span.min()) if span.size else None,
            "signed_mean": float(signed.mean()) if signed.size else None,
        }
        now = time.perf_counter()
        cut = now - 2.0
        n_recent = sum(1 for t in group_times if t >= cut)
        frames_recent = sum(n for t, n in zip(group_times, group_frames)
                            if t >= cut)
        measured = {
            "window_s": 2.0,
            "groups_per_s": round(n_recent / 2.0, 2),
            "frames_per_s": round(frames_recent / 2.0, 2),
            "mean_group_size": (round(frames_recent / n_recent, 2)
                                if n_recent else None),
            "definition": ("groups_per_s = 每秒生成的同步帧组数；"
                           "frames_per_s = 这些组包含的相机帧总数/秒，"
                           "约等于各相机采集帧率之和"),
        }
        consumed = {i: groups_total + dropped.get(i, 0) for i in self.indices}
        denom = max(consumed.values()) if consumed else 0

        rp = recent_events.count("group")
        rd = {i: recent_events.count("drop_%d" % i) for i in self.indices}
        rden = max([rp + v for v in rd.values()] or [0])
        recent_stats = {
            "events": len(recent_events),
            "groups": rp,
            "dropped": {str(k): v for k, v in rd.items()},
            "group_rate": round(rp / rden, 4) if rden else None,
            "definition": "最近 300 个分组/丢弃事件；与累计值分开显示",
            "span_ms": {k: (round(v, 3) if v is not None else None)
                        for k, v in stats.items()},
        }
        camera_stats = {}
        produced = {}
        buffered = {}
        for i in self.indices:
            worker = workers.get(i)
            if worker is None:
                camera_stats[str(i)] = None
                produced[i] = 0
                buffered[i] = 0
                continue
            st = worker.status()
            camera_stats[str(i)] = {
                "state": st.get("state"),
                "fps": st.get("fps"),
                "frame_interval_ms": st.get("frame_interval_ms"),
                "frames_total": st.get("frames_total"),
                "buffered": st.get("buffered"),
                "device_key": st.get("device_key"),
            }
            b = base.get(i)
            produced[i] = (worker.frames_total - b) if b is not None else 0
            buffered[i] = worker.buffered
        return {
            "running": self.alive,
            "error": self._error,
            "cameras": list(self.indices),
            "max_dt_ms": round(max_dt, 2),
            "buffer_frames": DEFAULT_BUFFER,
            "window_groups": self.window,
            "groups_total": groups_total,
            "groups_in_window": int(span.size),
            "measured": measured,
            "group_sizes": group_sizes,
            "span_ms": {k: (round(v, 3) if v is not None else None)
                        for k, v in stats.items()},
            # 兼容旧字段名（页面/工具仍可能读取）
            "pairs_total": groups_total,
            "dt_ms": {k: (round(v, 3) if v is not None else None)
                      for k, v in stats.items()},
            "group_rate": round(groups_total / denom, 4) if denom else None,
            "pair_rate": round(groups_total / denom, 4) if denom else None,
            "group_rate_definition": "成功帧组数 / 已消费帧数（分组+丢弃），各相机取较大者",
            "recent": recent_stats,
            "camera_stats": camera_stats,
            "produced": {str(k): v for k, v in produced.items()},
            "consumed": {str(k): v for k, v in consumed.items()},
            "dropped": {str(k): v for k, v in dropped.items()},
            "buffered": {str(k): v for k, v in buffered.items()},
            "last_span_ms": round(recent[-1].span_ms, 3) if recent else None,
            "last_group_age_ms": (round((now - last_group) * 1000.0, 1)
                                  if last_group is not None else None),
            "savable_groups": len(recent),
            "notes": [
                "组内时间跨度 = max(monotonic) - min(monotonic)（同一主机同一时钟）",
                "mf_timestamp 为各设备自身的 MF 流时间轴，跨设备不可相减，未参与分组",
                "时间戳是帧交付时刻，非曝光中心时刻；本功能为软件级近似同步，非硬件同步",
            ],
        }

    # -- 保存 -------------------------------------------------------------
    def recent_groups(self, count: int = 1) -> list[GroupRecord]:
        """取最近若干组同步帧（供标定/LED 取帧用，只读，不改指针）。"""
        count = max(1, int(count))
        with self._lock:
            return list(self._recent)[-count:]

    # 兼容旧调用
    def recent_pairs(self, count: int = 1) -> list[GroupRecord]:
        return self.recent_groups(count)

    def save_groups(self, out_dir: str | None = None, count: int = 8) -> dict:
        """保存最近若干组同步帧：每台相机原始灰度 PNG + 时间戳 JSON。"""
        out_dir = out_dir or DEFAULT_OUT_DIR
        count = max(1, int(count))
        with self._lock:
            recs = list(self._recent)[-count:]
            max_dt = self.max_dt_ms

        if not recs:
            return {"saved": [], "count": 0, "out_dir": os.path.abspath(out_dir),
                    "message": "还没有同步帧组（相机是否在采集？）"}

        os.makedirs(out_dir, exist_ok=True)
        saved = []
        for k, rec in enumerate(recs, 1):
            tag = "group_%03d" % k
            entry = {"group_seq": rec.group_seq,
                     "span_ms": round(rec.span_ms, 3),
                     "max_dt_ms": round(max_dt, 2),
                     "cameras": rec.cameras,
                     "frames": {}}
            for cam in rec.cameras:
                name = "%s_cam%d.png" % (tag, cam)
                path = os.path.abspath(os.path.join(out_dir, name))
                if not cv2.imwrite(path, rec.frames[cam].gray):
                    raise OSError("写入失败：%s" % path)
                entry["frames"][str(cam)] = self._side_meta(
                    rec.frames[cam], cam, name)
            with open(os.path.abspath(os.path.join(out_dir, tag + ".json")), "w",
                      encoding="utf-8") as fh:
                json.dump(entry, fh, ensure_ascii=False, indent=2)
            saved.append(entry)

        manifest = {
            "created": datetime.now().isoformat(),
            "groups": len(saved),
            "out_dir": os.path.abspath(out_dir),
            "max_dt_ms": round(max_dt, 2),
            "time_base": "host monotonic (time.perf_counter), same process",
            "span_definition": "max(monotonic) - min(monotonic) within the group",
            "caveats": [
                "时间戳为帧交付时刻（ReadSample 返回），不是曝光中心时刻",
                "真实曝光时刻差 = 交付时刻差 + 各路(曝光+读出+传输+缓冲)延迟之差，该差值非常数",
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

    def save_pairs(self, out_dir: str | None = None, count: int = 8) -> dict:
        return self.save_groups(out_dir=out_dir, count=count)

    def _side_meta(self, frame: Frame, index: int, filename: str) -> dict:
        worker = manager().worker(index)
        meta = frame.meta()
        meta.update({
            "index": index,
            "file": filename,
            "host_time_iso": datetime.fromtimestamp(frame.host_ts).isoformat(),
            "device": worker.device_name if worker else None,
            "device_key": worker.device_key if worker else None,
            "format": worker.format_info if worker else None,
        })
        return meta


# 进程级单例（Flask 多请求共享同一条分组线程）
_instance: SyncService | None = None
_instance_lock = threading.Lock()


def sync_service() -> SyncService:
    global _instance
    with _instance_lock:
        if _instance is None:
            indices = os.environ.get("MOCAP_CAM_INDICES", "0,1")
            _instance = SyncService([int(x) for x in indices.split(",") if x != ""])
        return _instance


def set_sync_cameras(indices) -> None:
    """相机集合变化时更新同步线程的逻辑编号。"""
    sync_service().set_indices(indices)
