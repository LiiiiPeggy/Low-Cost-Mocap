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
import threading
import time
import uuid
from ctypes import wintypes as wt

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


class _Controls:
    """一次 COM 会话：filter + IAMCameraControl + IAMVideoProcAmp。"""

    def __init__(self, index: int):
        self.index = int(index)
        self.monikers = []
        self.filter = LP()
        self.camera = LP()
        self.proc = LP()

    def __enter__(self):
        _co_init()
        self.monikers = _enum_monikers()
        if self.index < 0 or self.index >= len(self.monikers):
            raise RuntimeError("DirectShow 设备索引 %d 超范围（共 %d 个）"
                               % (self.index, len(self.monikers)))
        self.filter = _bind_filter(self.monikers[self.index])
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
        out = {"ok": True, "index": self.index, "source": "DirectShow/UVC"}
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

        # 曝光：先按目标模式写值，再单独切换自动/手动，避免先切手动后
        # 读回旧值。DirectShow 的 flags 是“本次操作使用自动/手动”。
        if self.camera:
            rng = _range(self.camera, CAMERA_CONTROL_EXPOSURE)
            cur = _get(self.camera, CAMERA_CONTROL_EXPOSURE)
            if auto_exposure is True:
                if not (rng["caps"] & CAMERA_CONTROL_FLAGS_AUTO):
                    raise ValueError("当前设备不支持自动曝光")
                _set(self.camera, CAMERA_CONTROL_EXPOSURE,
                     _clamp_step(cur["value"], rng), CAMERA_CONTROL_FLAGS_AUTO)
            elif exposure is not None or auto_exposure is False:
                val = _clamp_step(cur["value"] if exposure is None else exposure,
                                  rng)
                _set(self.camera, CAMERA_CONTROL_EXPOSURE, val,
                     CAMERA_CONTROL_FLAGS_MANUAL)

        if self.proc:
            rng = _range(self.proc, VIDEOPROCAMP_GAIN)
            if gain is not None:
                if not (rng["caps"] & CAMERA_CONTROL_FLAGS_MANUAL):
                    raise ValueError("当前设备不支持手动增益")
                _set(self.proc, VIDEOPROCAMP_GAIN,
                     _clamp_step(gain, rng), CAMERA_CONTROL_FLAGS_MANUAL)

        # 稍等一下再读回，部分 UVC 固件是异步生效
        time.sleep(0.05)
        return self.snapshot()


# 全局串行化：COM 对象本身不跨线程缓存，但多个同时的“读/写”请求没有意义，
# 且 DirectShow filter 的创建/释放并发时可能互相影响。
_LOCK = threading.RLock()


def get_controls(index: int) -> dict:
    """读取指定 DirectShow 索引的曝光/增益范围与当前值。"""
    with _LOCK:
        try:
            with _Controls(index) as ctl:
                return ctl.snapshot()
        except Exception as exc:
            return {"ok": False, "index": int(index),
                    "supported": False,
                    "error": "%s: %s" % (type(exc).__name__, exc)}


def set_controls(index: int, *, exposure=None, auto_exposure=None,
                 gain=None) -> dict:
    """设置曝光/自动曝光/增益，返回设备读回值。

    ``exposure`` 是 DirectShow 对数曝光值；``auto_exposure`` 为 True/False/None；
    ``gain`` 是整数索引。None 表示本次不改该项。
    """
    with _LOCK:
        try:
            with _Controls(index) as ctl:
                return ctl.apply(exposure=exposure,
                                 auto_exposure=auto_exposure, gain=gain)
        except ValueError:
            raise
        except Exception as exc:
            return {"ok": False, "index": int(index),
                    "supported": False,
                    "error": "%s: %s" % (type(exc).__name__, exc)}


def range_summary(index: int) -> dict:
    """给 UI 的稳定摘要：只保留范围/当前值，不含 COM 细节。"""
    st = get_controls(index)
    if not st.get("ok"):
        return st
    out = {"ok": True, "index": st["index"], "source": st.get("source")}
    for key in ("exposure", "gain"):
        item = st.get(key) or {}
        out[key] = {k: item.get(k) for k in (
            "supported", "min", "max", "step", "default", "caps",
            "auto_supported", "manual_supported", "value", "auto",
            "value_ms", "unit", "note") if k in item}
    return out


__all__ = ["get_controls", "set_controls", "range_summary",
           "CAMERA_CONTROL_EXPOSURE", "VIDEOPROCAMP_GAIN"]
