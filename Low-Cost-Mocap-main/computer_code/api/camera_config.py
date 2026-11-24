"""逻辑相机编号 ↔ 物理设备 的固定绑定。

逻辑编号（0..3）是采集、同步、标定、LED 定位、录制共用的唯一身份。物理设备
用 USB 设备实例 key 标识（``camera_controls.device_key`` 归一化后的
``\\\\?\\usb#vid_...#6&5332b2&0&0000``），它绑定到固定 USB 接口，不随枚举顺序、
设备名称或 VID/PID 变化。

配置文件：``captures/calibration/camera_config.json``（可用
``MOCAP_CALIB_DIR`` 重定向）。相机离线时保留原条目和逻辑编号，不会重新编号。

``confirmed`` 表示“人已确认这台物理设备就是逻辑相机 N”。未确认的相机可以
采集和预览，但不会套用历史标定结果（避免把别的相机的内参/外参用错）。
"""

from __future__ import annotations

import copy
import json
import os
import threading
from datetime import datetime

import camera_controls
import ir_camera

from calibration.dataset import atomic_json_write
from calibration.service import DEFAULT_SESSION_ROOT, project_relative

CONFIG_PATH = os.path.join(DEFAULT_SESSION_ROOT, "camera_config.json")
SCHEMA = "lowcost-mocap/camera-config"
MAX_CAMERAS = 4
DEFAULT_CAPTURE = {"width": 1280, "height": 720, "fps": 30.0,
                   "format": "NV12"}

_lock = threading.RLock()


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def device_key(path: str | None) -> str:
    return camera_controls.device_key(path)


_enum_cache: dict = {"t": 0.0, "devices": []}
ENUM_CACHE_S = 2.0


def enumerate_devices() -> list[dict]:
    """当前系统里的视频采集设备（不打开）。key 是稳定的设备实例标识。

    网页每秒轮询状态，MF 枚举有毫秒级开销，所以做 2 s 缓存；用户点“刷新设备
    状态”或改绑定时最多等 2 s 就能看到新设备。
    """
    import time
    now = time.monotonic()
    if _enum_cache["devices"] and now - _enum_cache["t"] < ENUM_CACHE_S:
        return [dict(d) for d in _enum_cache["devices"]]
    out = []
    for index, friendly, link in ir_camera.list_devices():
        out.append({"mf_index": int(index), "name": friendly,
                    "link": link, "device_key": device_key(link)})
    _enum_cache["t"] = now
    _enum_cache["devices"] = out
    return [dict(d) for d in out]


def default_config(devices: list[dict] | None = None) -> dict:
    """首次运行时的**建议**配置：按当前枚举顺序给出编号，全部待人工确认。"""
    devices = enumerate_devices() if devices is None else devices
    cameras = []
    for i, dev in enumerate(devices[:MAX_CAMERAS]):
        cameras.append({
            "index": i,
            "device_key": dev["device_key"],
            "device_name": dev["name"],
            "confirmed": False,
            **DEFAULT_CAPTURE,
        })
    return {"schema": SCHEMA, "version": 1, "updated": now_iso(),
            "cameras": cameras,
            "note": "逻辑编号 ↔ 物理设备绑定；confirmed=false 的相机不会套用历史标定"}


def _clean_entry(item: dict, fallback_index: int) -> dict:
    entry = {
        "index": int(item.get("index", fallback_index)),
        "device_key": device_key(item.get("device_key")),
        "device_name": str(item.get("device_name") or ""),
        "confirmed": bool(item.get("confirmed", False)),
        "width": int(item.get("width", DEFAULT_CAPTURE["width"])),
        "height": int(item.get("height", DEFAULT_CAPTURE["height"])),
        "fps": float(item.get("fps", DEFAULT_CAPTURE["fps"])),
        "format": str(item.get("format", DEFAULT_CAPTURE["format"])).upper(),
    }
    if not entry["device_key"]:
        raise ValueError("相机 %d 缺少 device_key" % entry["index"])
    return entry


def load(*, create: bool = True) -> dict:
    """读取配置；不存在时按当前枚举生成建议配置并落盘（全部未确认）。"""
    with _lock:
        path = CONFIG_PATH
        if os.path.isfile(path):
            try:
                with open(path, encoding="utf-8") as fh:
                    data = json.load(fh)
            except (OSError, ValueError) as exc:
                raise ValueError("camera_config.json 读取失败：%s" % exc)
            cameras = [_clean_entry(item, i)
                       for i, item in enumerate(data.get("cameras") or [])]
            if not cameras:
                raise ValueError("camera_config.json 没有相机条目")
            idx = sorted(c["index"] for c in cameras)
            if len(set(idx)) != len(idx):
                raise ValueError("camera_config.json 的逻辑编号重复：%s" % idx)
            return {"schema": SCHEMA, "version": 1,
                    "updated": data.get("updated") or "",
                    "cameras": sorted(cameras, key=lambda c: c["index"]),
                    "path": project_relative(path)}
        cfg = default_config()
        cfg["path"] = project_relative(path)
        if create:
            save(cfg)
        return cfg


def save(cfg: dict) -> dict:
    with _lock:
        cameras = [_clean_entry(item, i)
                   for i, item in enumerate(cfg.get("cameras") or [])]
        payload = {"schema": SCHEMA, "version": 1, "updated": now_iso(),
                   "cameras": sorted(cameras, key=lambda c: c["index"])}
        atomic_json_write(CONFIG_PATH, payload)
        payload["path"] = project_relative(CONFIG_PATH)
        return payload


def update(index: int, patch: dict) -> dict:
    """修改单台相机的绑定或采集配置（供网页/CLI 人工确认）。"""
    with _lock:
        cfg = load()
        target = None
        for item in cfg["cameras"]:
            if item["index"] == int(index):
                target = item
                break
        if target is None:
            raise ValueError("配置里没有逻辑相机 %d" % int(index))
        allowed = ("device_key", "device_name", "confirmed",
                   "width", "height", "fps", "format")
        for key in allowed:
            if key in patch and patch[key] is not None:
                target[key] = patch[key]
        if "device_key" in patch and "confirmed" not in patch:
            # 换了设备身份，原确认自动失效
            target["confirmed"] = False
        # 同一个 device_key 不能绑定两个逻辑编号
        keys = [c["device_key"] for c in cfg["cameras"] if c["device_key"]]
        if len(set(keys)) != len(keys):
            raise ValueError("同一台物理设备不能绑定两个逻辑编号")
        return save(cfg)


def logical_indices(cfg: dict | None = None) -> list[int]:
    cfg = cfg or load()
    return [int(c["index"]) for c in cfg["cameras"]]


def entry(index: int, cfg: dict | None = None) -> dict | None:
    cfg = cfg or load()
    for item in cfg["cameras"]:
        if int(item["index"]) == int(index):
            return copy.deepcopy(item)
    return None


def resolve(cfg: dict | None = None) -> list[dict]:
    """把逻辑编号解析成当前 MF 枚举索引；离线相机 ``mf_index=None``。"""
    cfg = cfg or load()
    present = {d["device_key"]: d for d in enumerate_devices()}
    out = []
    for item in cfg["cameras"]:
        dev = present.get(item["device_key"])
        out.append({
            **item,
            "mf_index": dev["mf_index"] if dev else None,
            "online": dev is not None,
            "name": dev["name"] if dev else item.get("device_name") or "",
            "link": dev["link"] if dev else None,
        })
    return out


def capture_configs(cfg: dict | None = None) -> list[dict]:
    """CameraManager.start 用的配置列表（逻辑编号 + 设备 key + 采集格式）。"""
    return [{"index": int(c["index"]), "device_key": c["device_key"],
             "width": int(c["width"]), "height": int(c["height"]),
             "fps": float(c["fps"]), "format": str(c["format"])}
            for c in (cfg or load())["cameras"]]


def calibration_check(index: int, result_entry: dict | None,
                      cfg: dict | None = None) -> tuple[bool, str]:
    """判断某台相机能否套用一份标定结果。

    * 结果里记录了 ``device_key`` 且与当前绑定不同 → 拒绝；
    * 结果没有设备信息（旧文件）且当前绑定未确认 → 拒绝（需人工确认）；
    * 其他情况允许。
    """
    item = entry(index, cfg)
    if item is None:
        return False, "配置里没有逻辑相机 %d" % int(index)
    saved = device_key((result_entry or {}).get("device_key"))
    if saved:
        if saved != item["device_key"]:
            return False, ("相机 %d 的标定结果属于设备 %s，当前绑定是 %s；"
                           "请重新标定或修正绑定"
                           % (int(index), saved, item["device_key"]))
        return True, ""
    if not item.get("confirmed"):
        return False, ("相机 %d 的旧标定文件没有设备信息，当前绑定尚未人工确认；"
                       "确认这台设备就是标定时的相机后再使用" % int(index))
    return True, ""


__all__ = ["CONFIG_PATH", "SCHEMA", "MAX_CAMERAS", "DEFAULT_CAPTURE",
           "device_key", "enumerate_devices", "default_config", "load",
           "save", "update", "logical_indices", "entry", "resolve",
           "capture_configs", "calibration_check", "now_iso"]
