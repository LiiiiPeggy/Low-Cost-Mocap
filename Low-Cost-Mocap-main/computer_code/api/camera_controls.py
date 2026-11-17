"""UVC 相机曝光 / 增益控制（DirectShow IAMCameraControl + IAMVideoProcAmp）。

为什么不能直接用 Media Foundation？
--------------------------------------------------------------------------
``ir_camera.py`` 的采集链路是 MF ``IMFSourceReader``，只负责取流，没有暴露
UVC 的 CameraControl/VideoProcAmp 属性。实测本机 LRCP G720P 的 UVC 控制
可以从 DirectShow 的捕获 filter 上拿到：

* ``IAMCameraControl``：曝光（``CameraControl_Exposure``）
* ``IAMVideoProcAmp``：增益（``VideoProcAmp_Gain``）

关键是**不需要启动 DirectShow 图 / 不需要再取流**：创建 filter、QueryInterface
后直接 GetRange/Get/Set，MF 的 SourceReader 仍在正常推流。这样不会新增采集
线程、不会重复打开设备、也不会打断同步。filter 在函数返回前释放。

单位
--------------------------------------------------------------------------
DirectShow 的曝光值是对数时间：``exposure_ms = 1000 * 2 ** value``。
本机实测范围 ``-13 .. -1``（约 0.12 ms .. 500 ms），步长 1；增益为
``0 .. 63`` 的整数索引。范围/当前值每次从设备读取，不做硬编码。

线程安全
--------------------------------------------------------------------------
每次调用在调用线程内 CoInitialize、创建/释放 COM 对象；不跨线程缓存接口
指针。Flask 的 threaded 请求模式可以安全调用。只读范围可以重复查询。
"""

from __future__ import annotations

import ctypes
import json
import os
import threading
import time
import uuid
from datetime import datetime

LP = ctypes.c_void_p
COINIT_APARTMENTTHREADED = 0x2
CLSCTX_INPROC_SERVER = 1

CLSID_SystemDeviceEnum = uuid.UUID("62BE5D10-60EB-11d0-BD3B-00A0C911CE86")
IID_ICreateDevEnum = uuid.UUID("29840822-5B84-11D0-BD3B-00A0C911CE86")
CLSID_VideoInputDeviceCategory = uuid.UUID("860BB310-5D01-11d0-BD3B-00A0C911CE86")
IID_IBaseFilter = uuid.UUID("56a86895-0ad4-11ce-b03a-0020af0ba770")
IID_IAMCameraControl = uuid.UUID("C6E13370-30AC-11D0-A18C-00A0C9118956")
IID_IAMVideoProcAmp = uuid.UUID("C6E13360-30AC-11D0-A18C-00A0C9118956")

# IAMCameraControl / IAMVideoProcAmp 的 vtable 索引
_VT_GET_RANGE = 3
_VT_SET = 4
_VT_GET = 5

CAMERA_CONTROL_EXPOSURE = 4
VIDEOPROCAMP_GAIN = 9
VIDEOPROCAMP_BRIGHTNESS = 0

CAMERA_CONTROL_FLAGS_AUTO = 0x0001
CAMERA_CONTROL_FLAGS_MANUAL = 0x0002
CAMERA_CONTROL_FLAGS_BOTH = 0x0003

IID_IPropertyBag = uuid.UUID("55272A00-42CB-11CE-8135-00AA004BB851")

# 每台设备独立保存：按 USB 设备实例路径（去掉 DirectShow/MF 不同的 GUID
# 后缀）做 key，重启/换 USB 口后仍能恢复各自曝光，而不是按 0/1 索引。
STORE_SCHEMA = "lowcost-mocap/camera-controls/1"
DEFAULT_STORE_PATH = os.path.join(
    os.environ.get("MOCAP_CALIB_DIR",
                   os.path.join(os.path.dirname(os.path.dirname(
                       os.path.abspath(__file__))), "captures", "calibration")),
    "camera_controls.json")


class _VARIANT_UNION(ctypes.Union):
    _fields_ = [("llVal", ctypes.c_longlong), ("bstrVal", ctypes.c_void_p),
                ("pwszVal", ctypes.c_wchar_p), ("punkVal", ctypes.c_void_p),
                ("p", ctypes.c_void_p), ("padding", ctypes.c_byte * 16)]


class VARIANT(ctypes.Structure):
    _anonymous_ = ("u",)
    _fields_ = [("vt", ctypes.c_ushort), ("r1", ctypes.c_ushort),
                ("r2", ctypes.c_ushort), ("r3", ctypes.c_ushort),
                ("u", _VARIANT_UNION)]


class _GUID(ctypes.Structure):
    _fields_ = [("Data1", ctypes.c_uint32), ("Data2", ctypes.c_uint16),
                ("Data3", ctypes.c_uint16), ("Data4", ctypes.c_ubyte * 8)]

    @classmethod
    def from_uuid(cls, value: uuid.UUID) -> "_GUID":
        g = cls()
        g.Data1 = value.time_low
        g.Data2 = value.time_mid
        g.Data3 = value.time_hi_version
        for i, b in enumerate(value.bytes[8:]):
            g.Data4[i] = b
        return g


def _method(ptr, index, *argtypes, restype=ctypes.c_long):
    table = ctypes.cast(
        ptr, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
    return ctypes.WINFUNCTYPE(restype, ctypes.c_void_p, *argtypes)(table[index])


def _hr(value) -> int:
    return int(value) & 0xFFFFFFFF


def _release(ptr) -> None:
    if ptr:
        try:
            _method(ptr, 2)(ptr)
        except Exception:
            pass


def _co_init() -> None:
    try:
        ctypes.OleDLL("ole32").CoInitializeEx(None, COINIT_APARTMENTTHREADED)
    except Exception:
        # RPC_E_CHANGED_MODE 等由调用方后续的 CoCreateInstance 决定是否真的失败
        pass


def _enum_monikers() -> list:
    """枚举 DirectShow 视频输入设备（返回 IMoniker 指针列表）。"""
    ole32 = ctypes.OleDLL("ole32")
    enum = LP()
    clsid = _GUID.from_uuid(CLSID_SystemDeviceEnum)
    iid = _GUID.from_uuid(IID_ICreateDevEnum)
    hr = _hr(ole32.CoCreateInstance(
        ctypes.byref(clsid), None, CLSCTX_INPROC_SERVER,
        ctypes.byref(iid), ctypes.byref(enum)))
    if hr:
        raise RuntimeError("CoCreateInstance(SystemDeviceEnum) 0x%08X" % hr)
    try:
        cat = _GUID.from_uuid(CLSID_VideoInputDeviceCategory)
        mon_enum = LP()
        hr = _hr(_method(
            enum, 3, ctypes.POINTER(_GUID), ctypes.POINTER(LP),
            ctypes.c_ulong)(enum, ctypes.byref(cat), ctypes.byref(mon_enum), 0))
        if hr:
            raise RuntimeError("CreateClassEnumerator 0x%08X" % hr)
    finally:
        _release(enum)

    out = []
    try:
        while True:
            mon = LP()
            got = ctypes.c_ulong(0)
            hr = _hr(_method(
                mon_enum, 3, ctypes.c_ulong, ctypes.POINTER(LP),
                ctypes.POINTER(ctypes.c_ulong))(mon_enum, 1,
                                               ctypes.byref(mon),
                                               ctypes.byref(got)))
            if hr or not mon.value:
                break
            out.append(mon)
    finally:
        _release(mon_enum)
    return out


def _bind_filter(moniker):
    filt = LP()
    iid = _GUID.from_uuid(IID_IBaseFilter)
    hr = _hr(_method(
        moniker, 8, LP, LP, ctypes.POINTER(_GUID), ctypes.POINTER(LP))(
            moniker, None, None, ctypes.byref(iid), ctypes.byref(filt)))
    if hr:
        raise RuntimeError("IMoniker::BindToObject(IBaseFilter) 0x%08X" % hr)
    return filt


def _query(obj, iid_uuid):
    iid = _GUID.from_uuid(iid_uuid)
    ptr = LP()
    hr = _hr(_method(
        obj, 0, ctypes.POINTER(_GUID), ctypes.POINTER(LP))(
            obj, ctypes.byref(iid), ctypes.byref(ptr)))
    return (ptr if not hr else None), hr


def _range(obj, prop: int) -> dict:
    lo = ctypes.c_long()
    hi = ctypes.c_long()
    step = ctypes.c_long()
    default = ctypes.c_long()
    caps = ctypes.c_long()
    hr = _hr(_method(
        obj, _VT_GET_RANGE, ctypes.c_long,
        ctypes.POINTER(ctypes.c_long), ctypes.POINTER(ctypes.c_long),
        ctypes.POINTER(ctypes.c_long), ctypes.POINTER(ctypes.c_long),
        ctypes.POINTER(ctypes.c_long))(
            obj, prop, ctypes.byref(lo), ctypes.byref(hi), ctypes.byref(step),
            ctypes.byref(default), ctypes.byref(caps)))
    if hr:
        raise RuntimeError("GetRange(property=%d) 0x%08X" % (prop, hr))
    return {"min": int(lo.value), "max": int(hi.value),
            "step": max(1, int(step.value)), "default": int(default.value),
            "caps": int(caps.value)}


def _get(obj, prop: int) -> dict:
    value = ctypes.c_long()
    flags = ctypes.c_long()
    hr = _hr(_method(
        obj, _VT_GET, ctypes.c_long, ctypes.POINTER(ctypes.c_long),
        ctypes.POINTER(ctypes.c_long))(
            obj, prop, ctypes.byref(value), ctypes.byref(flags)))
    if hr:
        raise RuntimeError("Get(property=%d) 0x%08X" % (prop, hr))
    return {"value": int(value.value), "flags": int(flags.value)}


def _set(obj, prop: int, value: int, flags: int) -> None:
    hr = _hr(_method(
        obj, _VT_SET, ctypes.c_long, ctypes.c_long, ctypes.c_long)(
            obj, prop, int(value), int(flags)))
    if hr:
        raise RuntimeError("Set(property=%d, value=%d, flags=%d) 0x%08X"
                           % (prop, value, flags, hr))


def _clamp_step(value: int, rng: dict) -> int:
    lo, hi, step = rng["min"], rng["max"], max(1, rng["step"])
    v = max(lo, min(hi, int(round(float(value)))))
    if step > 1:
        v = lo + int(round((v - lo) / float(step))) * step
        v = max(lo, min(hi, v))
    return int(v)


def _exposure_ms(value: int) -> float:
    return float(1000.0 * (2.0 ** int(value)))


def device_key(path: str | None) -> str:
    """DirectShow DevicePath / MF symbolic link → 稳定的设备实例 key。

    两者只在末尾接口 GUID 不同，例如：
    ``\\\\?\\usb#vid...#6&5332b2&0&0000#{65e8773d...}\\global`` (DS)
    ``\\\\?\\usb#vid...#6&5332b2&0&0000#{e5323777...}\\global`` (MF)
    去掉 ``#{...}`` 之后完全一致。
    """
    s = str(path or "").strip().lower()
    s = s.split("#{", 1)[0]
    return s.rstrip("\\/ ")


def _moniker_property(moniker, name: str) -> str:
    bag, _ = _query(moniker, IID_IPropertyBag)
    if not bag:
        return ""
    oleaut = ctypes.OleDLL("oleaut32")
    var = None
    try:
        var = VARIANT()
        oleaut.VariantInit(ctypes.byref(var))
        hr = _hr(_method(
            bag, 3, ctypes.c_wchar_p, ctypes.POINTER(VARIANT),
            LP)(bag, name, ctypes.byref(var), None))
        if hr:
            return ""
        if var.vt == 8 and var.u.bstrVal:        # VT_BSTR
            return ctypes.wstring_at(var.u.bstrVal)
        if var.vt == 31 and var.u.pwszVal:       # VT_LPWSTR
            return var.u.pwszVal
        return ""
    finally:
        try:
            if var is not None:
                oleaut.VariantClear(ctypes.byref(var))
        except Exception:
            pass
        _release(bag)


def _store_path() -> str:
    # 测试可通过 MOCAP_CALIB_DIR 重定向；DEFAULT_STORE_PATH 在 import 时
    # 固定，这里再读一次环境变量以支持运行期切换。
    root = os.environ.get("MOCAP_CALIB_DIR")
    if root:
        return os.path.join(root, "camera_controls.json")
    return DEFAULT_STORE_PATH


def _load_store() -> dict:
    path = _store_path()
    if not os.path.isfile(path):
        return {"schema": STORE_SCHEMA, "devices": {}}
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, dict):
            raise ValueError("不是 JSON 对象")
        devices = data.get("devices")
        if not isinstance(devices, dict):
            devices = {}
        return {"schema": STORE_SCHEMA, "devices": devices,
                "updated": data.get("updated")}
    except Exception:
        return {"schema": STORE_SCHEMA, "devices": {}}


def _save_store(data: dict) -> None:
    path = _store_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = dict(data)
    data["schema"] = STORE_SCHEMA
    data["updated"] = datetime.now().isoformat(timespec="seconds")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _persist_snapshot(snapshot: dict) -> dict | None:
    key = snapshot.get("device_key")
    if not key:
        return None
    exp = snapshot.get("exposure") or {}
    gain = snapshot.get("gain") or {}
    entry = {
        "friendly": snapshot.get("friendly") or "",
        "device_path": (snapshot.get("device_path")
                        or snapshot.get("device_link") or ""),
        "exposure": exp.get("value"),
        "auto_exposure": bool(exp.get("auto")),
        "gain": gain.get("value"),
        "updated": datetime.now().isoformat(timespec="seconds"),
    }
    store = _load_store()
    store.setdefault("devices", {})[key] = entry
    _save_store(store)
    return entry


class _Controls:
    """一次 COM 会话：filter + IAMCameraControl + IAMVideoProcAmp。"""

    def __init__(self, index: int, device_link: str | None = None):
        self.index = int(index)
        self.want_key = device_key(device_link)
        self.monikers = []
        self.filter = LP()
        self.camera = LP()
        self.proc = LP()
        self.device_key = ""
        self.friendly = ""
        self.device_path = ""
        self.matched_by_key = None

    def __enter__(self):
        _co_init()
        self.monikers = _enum_monikers()
        chosen = None
        if self.want_key:
            for mon in self.monikers:
                path = _moniker_property(mon, "DevicePath")
                if device_key(path) == self.want_key:
                    chosen = mon
                    self.device_path = path
                    break
        if chosen is None:
            if self.index < 0 or self.index >= len(self.monikers):
                raise RuntimeError("DirectShow 设备索引 %d 超范围（共 %d 个）"
                                   % (self.index, len(self.monikers)))
            chosen = self.monikers[self.index]
            self.device_path = _moniker_property(chosen, "DevicePath")
        self.friendly = _moniker_property(chosen, "FriendlyName")
        self.device_key = device_key(self.device_path)
        self.matched_by_key = (bool(self.want_key)
                               and self.device_key == self.want_key)
        self.filter = _bind_filter(chosen)
        self.camera, hr_cam = _query(self.filter, IID_IAMCameraControl)
        self.proc, hr_proc = _query(self.filter, IID_IAMVideoProcAmp)
        if not self.camera and not self.proc:
            raise RuntimeError("设备没有暴露 IAMCameraControl/IAMVideoProcAmp"
                               "（0x%08X / 0x%08X）" % (hr_cam, hr_proc))
        return self

    def __exit__(self, *exc):
        _release(self.camera)
        _release(self.proc)
        _release(self.filter)
        for mon in self.monikers:
            _release(mon)
        self.camera = LP()
        self.proc = LP()
        self.filter = LP()
        self.monikers = []

    # -- 读取 -------------------------------------------------------------
    def snapshot(self) -> dict:
        out = {"ok": True, "index": self.index, "source": "DirectShow/UVC",
               "device_key": self.device_key,
               "device_path": self.device_path,
               "friendly": self.friendly,
               "matched_by_device_key": self.matched_by_key}
        if self.camera:
            rng = _range(self.camera, CAMERA_CONTROL_EXPOSURE)
            cur = _get(self.camera, CAMERA_CONTROL_EXPOSURE)
            auto = bool(cur["flags"] & CAMERA_CONTROL_FLAGS_AUTO)
            out["exposure"] = {
                "supported": True,
                "min": rng["min"], "max": rng["max"], "step": rng["step"],
                "default": rng["default"], "caps": rng["caps"],
                "auto_supported": bool(rng["caps"] & CAMERA_CONTROL_FLAGS_AUTO),
                "manual_supported": bool(rng["caps"] & CAMERA_CONTROL_FLAGS_MANUAL),
                "value": cur["value"], "auto": auto,
                "value_ms": round(_exposure_ms(cur["value"]), 4),
                "unit": "log2(s)",
            }
        else:
            out["exposure"] = {"supported": False,
                               "reason": "没有 IAMCameraControl"}
        if self.proc:
            rng = _range(self.proc, VIDEOPROCAMP_GAIN)
            cur = _get(self.proc, VIDEOPROCAMP_GAIN)
            out["gain"] = {
                "supported": True,
                "min": rng["min"], "max": rng["max"], "step": rng["step"],
                "default": rng["default"], "caps": rng["caps"],
                "auto_supported": bool(rng["caps"] & CAMERA_CONTROL_FLAGS_AUTO),
                "manual_supported": bool(rng["caps"] & CAMERA_CONTROL_FLAGS_MANUAL),
                "value": cur["value"],
                "auto": bool(cur["flags"] & CAMERA_CONTROL_FLAGS_AUTO),
                "unit": "index",
                "note": ("该设备增益只支持手动" if not (rng["caps"] &
                         CAMERA_CONTROL_FLAGS_AUTO) else ""),
            }
        else:
            out["gain"] = {"supported": False,
                           "reason": "没有 IAMVideoProcAmp"}
        return out

    # -- 写入 -------------------------------------------------------------
    def apply(self, *, exposure=None, auto_exposure=None, gain=None) -> dict:
        if exposure is not None and not self.camera:
            raise ValueError("当前设备没有曝光控制接口")
        if gain is not None and not self.proc:
            raise ValueError("当前设备没有增益控制接口")

        requested = {"auto_exposure": auto_exposure,
                     "exposure": None, "gain": None, "warnings": []}
        # 曝光：先按目标模式写值，再单独切换自动/手动，避免先切手动后
        # 读回旧值。DirectShow 的 flags 是“本次操作使用自动/手动”。
        if self.camera:
            rng = _range(self.camera, CAMERA_CONTROL_EXPOSURE)
            cur = _get(self.camera, CAMERA_CONTROL_EXPOSURE)
            if auto_exposure is True:
                if not (rng["caps"] & CAMERA_CONTROL_FLAGS_AUTO):
                    raise ValueError("当前设备不支持自动曝光")
                val = _clamp_step(cur["value"], rng)
                _set(self.camera, CAMERA_CONTROL_EXPOSURE, val,
                     CAMERA_CONTROL_FLAGS_AUTO)
                requested["exposure"] = val
            elif exposure is not None or auto_exposure is False:
                val = _clamp_step(cur["value"] if exposure is None else exposure,
                                  rng)
                _set(self.camera, CAMERA_CONTROL_EXPOSURE, val,
                     CAMERA_CONTROL_FLAGS_MANUAL)
                requested["exposure"] = val

        if self.proc:
            rng = _range(self.proc, VIDEOPROCAMP_GAIN)
            if gain is not None:
                if not (rng["caps"] & CAMERA_CONTROL_FLAGS_MANUAL):
                    raise ValueError("当前设备不支持手动增益")
                val = _clamp_step(gain, rng)
                _set(self.proc, VIDEOPROCAMP_GAIN, val,
                     CAMERA_CONTROL_FLAGS_MANUAL)
                requested["gain"] = val

        # 稍等一下再读回，部分 UVC 固件是异步生效
        time.sleep(0.08)
        snap = self.snapshot()
        exp = snap.get("exposure") or {}
        gain_st = snap.get("gain") or {}
        verified = True
        if auto_exposure is True and not exp.get("auto"):
            verified = False
            requested["warnings"].append("设备没有保持自动曝光")
        if auto_exposure is False and exp.get("auto"):
            verified = False
            requested["warnings"].append("设备没有切到手动曝光")
        if requested["exposure"] is not None and auto_exposure is not True:
            if exp.get("value") != requested["exposure"]:
                verified = False
                requested["warnings"].append(
                    "曝光读回 %s，期望 %s" % (exp.get("value"),
                                              requested["exposure"]))
        if requested["gain"] is not None:
            if gain_st.get("value") != requested["gain"]:
                verified = False
                requested["warnings"].append(
                    "增益读回 %s，期望 %s" % (gain_st.get("value"),
                                              requested["gain"]))
        snap["requested"] = requested
        snap["verified"] = bool(verified)
        if not verified:
            snap["ok"] = False
            snap["error"] = "设置未完全生效：" + "；".join(requested["warnings"])
        return snap


# 全局串行化：COM 对象本身不跨线程缓存，但多个同时的“读/写”请求没有意义，
# 且 DirectShow filter 的创建/释放并发时可能互相影响。
_LOCK = threading.RLock()


def get_controls(index: int, device_link: str | None = None) -> dict:
    """读取指定相机的曝光/增益范围与当前值，并附上保存的上次设置。"""
    with _LOCK:
        try:
            with _Controls(index, device_link) as ctl:
                snap = ctl.snapshot()
                snap["saved"] = _load_store().get("devices", {}).get(
                    ctl.device_key)
                return snap
        except Exception as exc:
            return {"ok": False, "index": int(index),
                    "supported": False,
                    "error": "%s: %s" % (type(exc).__name__, exc)}


def set_controls(index: int, *, exposure=None, auto_exposure=None,
                 gain=None, device_link: str | None = None) -> dict:
    """设置曝光/自动曝光/增益，返回设备读回值。

    ``exposure`` 是 DirectShow 对数曝光值；``auto_exposure`` 为 True/False/None；
    ``gain`` 是整数索引。None 表示本次不改该项。设置成功后会按设备实例
    路径保存，供重启后恢复。
    """
    with _LOCK:
        try:
            with _Controls(index, device_link) as ctl:
                result = ctl.apply(exposure=exposure,
                                   auto_exposure=auto_exposure, gain=gain)
                if result.get("ok"):
                    result["saved"] = _persist_snapshot(result)
                return result
        except ValueError:
            raise
        except Exception as exc:
            return {"ok": False, "index": int(index),
                    "supported": False,
                    "error": "%s: %s" % (type(exc).__name__, exc)}


def restore_controls(index: int, device_link: str | None = None) -> dict:
    """把该设备上次成功保存的曝光/自动曝光/增益写回并验证。"""
    with _LOCK:
        try:
            with _Controls(index, device_link) as ctl:
                saved = _load_store().get("devices", {}).get(ctl.device_key)
                if not saved:
                    snap = ctl.snapshot()
                    snap["restored"] = False
                    snap["reason"] = "没有该设备的保存记录"
                    return snap
                result = ctl.apply(
                    exposure=saved.get("exposure"),
                    auto_exposure=saved.get("auto_exposure"),
                    gain=saved.get("gain"))
                result["restored"] = bool(result.get("ok"))
                result["saved"] = saved
                return result
        except Exception as exc:
            return {"ok": False, "index": int(index),
                    "supported": False, "restored": False,
                    "error": "%s: %s" % (type(exc).__name__, exc)}


def restore_all(indices, links: dict | None = None) -> list[dict]:
    """启动/重启相机后恢复多台设备；``links`` 是 index → MF symbolic link。"""
    links = links or {}
    return [restore_controls(i, links.get(str(i)) or links.get(i))
            for i in indices]


def range_summary(index: int, device_link: str | None = None) -> dict:
    """给 UI 的稳定摘要：只保留范围/当前值，不含 COM 细节。"""
    st = get_controls(index, device_link)
    if not st.get("ok"):
        return st
    out = {"ok": True, "index": st["index"], "source": st.get("source"),
           "device_key": st.get("device_key"),
           "friendly": st.get("friendly"),
           "matched_by_device_key": st.get("matched_by_device_key"),
           "saved": st.get("saved")}
    for key in ("exposure", "gain"):
        item = st.get(key) or {}
        out[key] = {k: item.get(k) for k in (
            "supported", "min", "max", "step", "default", "caps",
            "auto_supported", "manual_supported", "value", "auto",
            "value_ms", "unit", "note") if k in item}
    return out


__all__ = ["get_controls", "set_controls", "restore_controls",
           "restore_all", "range_summary", "device_key",
           "CAMERA_CONTROL_EXPOSURE", "VIDEOPROCAMP_GAIN"]
