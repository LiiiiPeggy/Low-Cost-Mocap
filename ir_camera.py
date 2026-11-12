#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ir_camera.py —— USB 红外相机取帧与可视化（Windows / Media Foundation，纯 ctypes 实现）

为什么不用随相机附带的 SDK？
--------------------------------------------------------------------------
`image_driver/camera windows sdk/` 里的文档（TXY_Win-SDK开发文档.docx、
USB_UVC协议_SDK开发文档.pdf）其实是同一份 **Microsoft DirectShow / AMCap 通用教程**，
`sdk/USBCamera_SDK/` 只是官方 DirectShow SDK（Include/Lib/Samples）。
文档里没有任何厂商私有协议，因为这台相机本身就是一个标准 **UVC** 设备
（设备管理器里叫 "LRCP G720P"，USB\\VID_BC15&PID_2E1B&MI_00）。
所以不需要编译 DirectShow BaseClasses，直接用系统自带的取流 API 即可。

实测该相机通过 Media Foundation 暴露的格式为 NV12 / MJPG / YUY2，
最高 1280x960@30fps（1280x720@30 亦可用），**没有 Y16 等 16bit 原始格式**，
因此拿到的是 8bit 亮度图（NV12 的 Y 平面、YUY2 的 Y 分量、或 MJPG 解码后的灰度），
它是红外/近红外相机的成像亮度，而不是可换算成摄氏度的辐射数据。

依赖
--------------------------------------------------------------------------
  numpy            必需（帧数据容器）
  opencv-python    必需（显示、伪彩色、MJPG 解码、存图全部走 cv2）

运行环境（实测）
--------------------------------------------------------------------------
  conda activate lowcost_mocap      # Python 3.10 + numpy 1.26 + opencv 4.10
  python ir_camera.py

用法
--------------------------------------------------------------------------
  python ir_camera.py                          # 默认 1280x720@30 NV12 实时显示
  python ir_camera.py --list                   # 列出相机
  python ir_camera.py --formats                # 列出本机该相机支持的所有格式
  python ir_camera.py --width 640 --height 480 --fps 30
  python ir_camera.py --format MJPG            # 换压缩格式取流
  python ir_camera.py --snapshot shot.png      # 抓一帧存盘后退出
  python ir_camera.py --cmap inferno           # 启动即用伪彩色
  python ir_camera.py --no-display --frames 60 # 无窗口连续取 60 帧（压测/采数据）

显示窗口内的快捷键
--------------------------------------------------------------------------
  g / j / t / h / i / v   切换 colormap（gray / jet / turbo / hot / inferno / viridis）
  a                       开关自动对比度拉伸（按 1%~99% 分位）
  s                       保存当前帧 PNG
  q 或 Esc                退出
"""

from __future__ import annotations

import argparse
import ctypes
import os
import sys
import threading
import time
import uuid
from ctypes import wintypes as wt
from datetime import datetime

import numpy as np

try:
    import cv2
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "需要 opencv-python（当前环境没有）：\n"
        "  conda install -c conda-forge opencv\n"
        "  或 pip install opencv-python"
    ) from exc

# ---------------------------------------------------------------------------
# COM / Media Foundation 基础设施
# ---------------------------------------------------------------------------

LP = ctypes.c_void_p
COINIT_APARTMENTTHREADED = 0x2
MF_VERSION = 0x00020070  # MF_SDK_VERSION 2, MF_API_VERSION 0x70
MFSTARTUP_FULL = 0

MF_SOURCE_READER_FIRST_VIDEO_STREAM = 0xFFFFFFFC
MF_SOURCE_READERF_ERROR = 0x00000001
MF_SOURCE_READERF_ENDOFSTREAM = 0x00000002
MF_SOURCE_READERF_NEWSTREAM = 0x00000004
MF_SOURCE_READERF_CURRENTMEDIATYPECHANGED = 0x00000020

try:
    _mfplat = ctypes.WinDLL("mfplat")
    _mfdev = ctypes.WinDLL("mf")
    _mfrw = ctypes.WinDLL("mfreadwrite")
except OSError as exc:  # pragma: no cover - 非 Windows 环境
    raise SystemExit("本脚本仅支持 Windows（需要 Media Foundation）") from exc


class GUID(ctypes.Structure):
    _fields_ = [
        ("Data1", ctypes.c_uint32),
        ("Data2", ctypes.c_uint16),
        ("Data3", ctypes.c_uint16),
        ("Data4", ctypes.c_ubyte * 8),
    ]

    def __str__(self) -> str:
        d = self.Data4
        return "%08X-%04X-%04X-%02X%02X-%02X%02X%02X%02X%02X%02X" % (
            self.Data1, self.Data2, self.Data3, d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7])


def _guid(text: str) -> GUID:
    u = uuid.UUID(text)
    g = GUID()
    g.Data1 = u.time_low
    g.Data2 = u.time_mid
    g.Data3 = u.time_hi_version
    for i, b in enumerate(u.bytes[8:]):
        g.Data4[i] = b
    return g


def _fourcc(g: GUID) -> str:
    """把 Media Foundation 的 subtype GUID 还原成可读的 fourcc（如 NV12 / MJPG）。"""
    raw = ctypes.string_at(ctypes.byref(g), 16)
    try:
        return raw[:4].decode("ascii")
    except UnicodeDecodeError:
        return str(g)


def _method(ptr, index: int, *argtypes, restype=ctypes.c_long):
    """取 COM 接口 vtable[index] 的方法并返回可调用对象（第一个参数固定为 this）。"""
    table = ctypes.cast(ptr, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
    return ctypes.WINFUNCTYPE(restype, ctypes.c_void_p, *argtypes)(table[index])


def _release(ptr) -> None:
    if ptr:
        _method(ptr, 2)(ptr)


def _hr(hr: int) -> int:
    """把 ctypes 返回的 HRESULT 规范化成 32 位无符号，方便和常量比较。"""
    return hr & 0xFFFFFFFF


def _hr_text(hr: int) -> str:
    return "0x%08X" % _hr(hr)


# --- 属性 / 媒体类型 GUID ------------------------------------------------
GUID_SOURCE_TYPE = _guid("C60AC5FE-252A-478F-A0EF-BC8FA5F7CAD3")
GUID_SOURCE_TYPE_VIDCAP = _guid("8AC3587A-4AE7-42D8-99E0-0A6013EEF90F")
GUID_FRIENDLY_NAME = _guid("60D0E559-52F8-4FA2-BBCE-ACDB34A8EC01")
GUID_SYMBOLIC_LINK = _guid("58F0AAD8-22BF-4F8A-BB3D-D2C4978C6E2F")
GUID_MT_SUBTYPE = _guid("F7E34C9A-42E8-4714-B74B-CB29D72C35E5")
GUID_MT_FRAME_SIZE = _guid("1652C33D-D6B2-4012-B834-72030849A37D")
GUID_MT_FRAME_RATE = _guid("C459A2E8-3D2C-4E44-B132-FEE5156C7BB0")
GUID_MT_DEFAULT_STRIDE = _guid("644B4E48-1E02-4516-B0EB-C01CA9D49AC6")

# --- 函数原型 ------------------------------------------------------------
_mfplat.MFStartup.argtypes = [ctypes.c_uint32, ctypes.c_uint32]
_mfplat.MFShutdown.argtypes = []
_mfplat.MFCreateAttributes.argtypes = [ctypes.POINTER(LP), ctypes.c_uint32]
_mfdev.MFEnumDeviceSources.argtypes = [LP, ctypes.POINTER(ctypes.POINTER(LP)),
                                       ctypes.POINTER(ctypes.c_uint32)]
_mfdev.MFCreateDeviceSource.argtypes = [LP, ctypes.POINTER(LP)]
_mfdev.MFCreateDeviceSource.restype = ctypes.c_long
_mfrw.MFCreateSourceReaderFromMediaSource.argtypes = [LP, LP, ctypes.POINTER(LP)]


def _co_initialize() -> None:
    hr = ctypes.windll.ole32.CoInitializeEx(None, COINIT_APARTMENTTHREADED)
    if _hr(hr) not in (0x00000000, 0x00000001, 0x80010106):  # S_OK / S_FALSE / RPC_E_CHANGED_MODE
        raise OSError("CoInitializeEx 失败 " + _hr_text(hr))


# ---------------------------------------------------------------------------
# 相机
# ---------------------------------------------------------------------------

class CameraFormat:
    """一路可用的取流格式。ptr 指向 IMFMediaType，仅内部使用。"""

    __slots__ = ("index", "fourcc", "width", "height", "fps_num", "fps_den", "ptr")

    def __init__(self, index, fourcc, width, height, fps_num, fps_den, ptr):
        self.index = index
        self.fourcc = fourcc
        self.width = width
        self.height = height
        self.fps_num = fps_num
        self.fps_den = fps_den
        self.ptr = ptr

    @property
    def fps(self) -> float:
        return self.fps_num / self.fps_den if self.fps_den else 0.0

    def __str__(self) -> str:
        return "%s %dx%d @ %.0ffps" % (self.fourcc, self.width, self.height, self.fps)


def list_devices() -> list[tuple[int, str, str]]:
    """返回 [(索引, 友好名, 符号链接)]。"""
    _co_initialize()
    _mfplat.MFStartup(MF_VERSION, MFSTARTUP_FULL)

    attrs = LP()
    if _hr(_mfplat.MFCreateAttributes(ctypes.byref(attrs), 1)):
        raise OSError("MFCreateAttributes 失败")
    _method(attrs, 24, ctypes.POINTER(GUID), ctypes.POINTER(GUID))(
        attrs, ctypes.byref(GUID_SOURCE_TYPE), ctypes.byref(GUID_SOURCE_TYPE_VIDCAP))

    array = ctypes.POINTER(LP)()
    count = ctypes.c_uint32(0)
    hr = _hr(_mfdev.MFEnumDeviceSources(attrs, ctypes.byref(array), ctypes.byref(count)))
    _release(attrs)
    if hr:
        raise OSError("MFEnumDeviceSources 失败 " + _hr_text(hr))

    devices = []
    for i in range(count.value):
        activate = LP(array[i])
        devices.append((i, _attr_string(activate, GUID_FRIENDLY_NAME),
                        _attr_string(activate, GUID_SYMBOLIC_LINK)))
        _release(activate)
    _mfplat.MFShutdown()
    return devices


def _attr_string(obj, key: GUID) -> str:
    buf = ctypes.c_wchar_p()
    size = ctypes.c_uint32(0)
    hr = _method(obj, 13, ctypes.POINTER(GUID), ctypes.POINTER(ctypes.c_wchar_p),
                 ctypes.POINTER(ctypes.c_uint32))(
        obj, ctypes.byref(key), ctypes.byref(buf), ctypes.byref(size))
    if _hr(hr) == 0 and buf.value:
        ctypes.windll.ole32.CoTaskMemFree(buf)
        return buf.value
    return ""


class IRCamera:
    """UVC 红外相机。open() 之后用 grab() 取帧，close() 释放。"""

    def __init__(self, index: int = 0, width: int = 1280, height: int = 720,
                 fps: float = 30.0, pixel_format: str = "NV12", name: str | None = None,
                 verbose: bool = True):
        self.index = index
        self.name = name
        self.want = (int(width), int(height), float(fps), pixel_format.upper())
        self.verbose = verbose

        self._source = LP()
        self._reader = LP()
        self._activates: list[LP] = []
        self._formats: list[CameraFormat] = []
        self._current: CameraFormat | None = None
        self._stride = 0
        self._started = False

    # -- 生命周期 ---------------------------------------------------------
    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *exc):
        self.close()

    def open(self) -> "IRCamera":
        _co_initialize()
        hr = _hr(_mfplat.MFStartup(MF_VERSION, MFSTARTUP_FULL))
        if hr:
            raise OSError("MFStartup 失败 " + _hr_text(hr))

        attrs = LP()
        _mfplat.MFCreateAttributes(ctypes.byref(attrs), 1)
        _method(attrs, 24, ctypes.POINTER(GUID), ctypes.POINTER(GUID))(
            attrs, ctypes.byref(GUID_SOURCE_TYPE), ctypes.byref(GUID_SOURCE_TYPE_VIDCAP))

        array = ctypes.POINTER(LP)()
        count = ctypes.c_uint32(0)
        hr = _hr(_mfdev.MFEnumDeviceSources(attrs, ctypes.byref(array), ctypes.byref(count)))
        _release(attrs)
        if hr or not count.value:
            raise OSError("没有找到视频采集设备（%s）" % _hr_text(hr))

        for i in range(count.value):
            self._activates.append(LP(array[i]))

        activate = self._pick_activate()
        friendly = _attr_string(activate, GUID_FRIENDLY_NAME)

        hr = _hr(_mfdev.MFCreateDeviceSource(activate, ctypes.byref(self._source)))
        if hr:
            raise OSError("MFCreateDeviceSource 失败 " + _hr_text(hr))

        hr = _hr(_mfrw.MFCreateSourceReaderFromMediaSource(self._source, None,
                                                          ctypes.byref(self._reader)))
        if hr:
            raise OSError("MFCreateSourceReaderFromMediaSource 失败 " + _hr_text(hr))

        stream = MF_SOURCE_READER_FIRST_VIDEO_STREAM
        hr = _hr(_method(self._reader, 4, ctypes.c_uint32, wt.BOOL)(self._reader, stream, True))
        if hr:
            raise OSError("SetStreamSelection 失败 " + _hr_text(hr))

        self._formats = self._enumerate_formats()
        if not self._formats:
            raise OSError("相机没有报告任何可用格式")

        chosen = self._choose(self.want[0], self.want[1], self.want[2], self.want[3])
        hr = _hr(_method(self._reader, 7, ctypes.c_uint32, LP, LP)(
            self._reader, stream, None, chosen.ptr))
        if hr:
            raise OSError("SetCurrentMediaType 失败 " + _hr_text(hr))

        self._current = chosen
        self._read_stride()
        self._started = True

        if self.verbose:
            print("已打开相机：%s" % (friendly or "设备 %d" % self.index))
            print("取流格式：%s" % chosen)
        return self

    def _pick_activate(self):
        if not self._activates:
            raise OSError("设备列表为空")
        if self.name:
            key = self.name.lower()
            for activate in self._activates:
                if key in _attr_string(activate, GUID_FRIENDLY_NAME).lower():
                    return activate
            raise OSError("找不到名称包含 %r 的相机" % self.name)
        if self.index >= len(self._activates):
            raise OSError("索引 %d 超范围（共 %d 个设备）" % (self.index, len(self._activates)))
        return self._activates[self.index]

    def close(self) -> None:
        self._started = False
        self._formats.clear()
        self._current = None
        _release(self._reader)
        self._reader = LP()
        _release(self._source)
        self._source = LP()
        for activate in self._activates:
            _release(activate)
        self._activates.clear()
        _mfplat.MFShutdown()

    # -- 格式 -------------------------------------------------------------
    def _enumerate_formats(self) -> list[CameraFormat]:
        out = []
        i = 0
        while True:
            ptr = LP()
            hr = _hr(_method(self._reader, 5, ctypes.c_uint32, ctypes.c_uint32,
                             ctypes.POINTER(LP))(
                self._reader, MF_SOURCE_READER_FIRST_VIDEO_STREAM, i, ctypes.byref(ptr)))
            if hr or not ptr.value:
                break
            subtype = GUID()
            _method(ptr, 10, ctypes.POINTER(GUID), ctypes.POINTER(GUID))(
                ptr, ctypes.byref(GUID_MT_SUBTYPE), ctypes.byref(subtype))
            size = ctypes.c_uint64(0)
            _method(ptr, 8, ctypes.POINTER(GUID), ctypes.POINTER(ctypes.c_uint64))(
                ptr, ctypes.byref(GUID_MT_FRAME_SIZE), ctypes.byref(size))
            rate = ctypes.c_uint64(0)
            _method(ptr, 8, ctypes.POINTER(GUID), ctypes.POINTER(ctypes.c_uint64))(
                ptr, ctypes.byref(GUID_MT_FRAME_RATE), ctypes.byref(rate))
            out.append(CameraFormat(
                index=i,
                fourcc=_fourcc(subtype),
                width=size.value >> 32,
                height=size.value & 0xFFFFFFFF,
                fps_num=rate.value >> 32,
                fps_den=rate.value & 0xFFFFFFFF,
                ptr=ptr,
            ))
            i += 1
        return out

    def formats(self) -> list[CameraFormat]:
        return list(self._formats)

    def _choose(self, width, height, fps, fmt) -> CameraFormat:
        """按“格式 → 分辨率 → 帧率”的优先级做最优匹配，找不到就退到最接近的一项。"""
        candidates = self._formats
        if fmt and fmt != "AUTO":
            exact = [f for f in candidates if f.fourcc == fmt]
            if exact:
                candidates = exact
        exact = [f for f in candidates if f.width == width and f.height == height]
        if exact:
            candidates = exact
        else:
            same_ratio = [f for f in candidates if f.width * height == f.height * width]
            if same_ratio:
                candidates = same_ratio
        best = min(candidates, key=lambda f: (abs(f.fps - fps), -f.width * f.height))
        return best

    def _read_stride(self) -> None:
        ptr = LP()
        hr = _hr(_method(self._reader, 6, ctypes.c_uint32, ctypes.POINTER(LP))(
            self._reader, MF_SOURCE_READER_FIRST_VIDEO_STREAM, ctypes.byref(ptr)))
        self._stride = 0
        if hr or not ptr.value:
            return
        stride = ctypes.c_uint32(0)
        if _hr(_method(ptr, 7, ctypes.POINTER(GUID), ctypes.POINTER(ctypes.c_uint32))(
                ptr, ctypes.byref(GUID_MT_DEFAULT_STRIDE), ctypes.byref(stride))) == 0:
            self._stride = abs(int(stride.value))
        _release(ptr)

    # -- 取帧 -------------------------------------------------------------
    @property
    def info(self) -> str:
        return str(self._current) if self._current else "<未打开>"

    def grab(self, max_skip: int = 8):
        """取一帧，返回 (灰度图 ndarray[H,W] uint8 或 None, 时间戳 100ns)。"""
        if not self._started:
            raise RuntimeError("调用 grab() 之前需要先 open()")

        stream = MF_SOURCE_READER_FIRST_VIDEO_STREAM
        for _ in range(max_skip):
            idx = ctypes.c_uint32(0)
            flags = ctypes.c_uint32(0)
            stamp = ctypes.c_int64(0)
            sample = LP()
            # 注意参数顺序：ReadSample(stream, ctrlFlags, *actualIndex, *flags, *timestamp, **sample)
            hr = _hr(_method(self._reader, 9, ctypes.c_uint32, ctypes.c_uint32,
                             ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32),
                             ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(LP))(
                self._reader, stream, 0, ctypes.byref(idx), ctypes.byref(flags),
                ctypes.byref(stamp), ctypes.byref(sample)))
            if hr:
                raise OSError("ReadSample 失败 " + _hr_text(hr))
            if flags.value & MF_SOURCE_READERF_CURRENTMEDIATYPECHANGED:
                self._read_stride()
            if flags.value & MF_SOURCE_READERF_ENDOFSTREAM:
                raise OSError("视频流意外结束")
            if not sample.value:
                continue  # 首帧之前会返回空 sample

            buffer = LP()
            if _hr(_method(sample, 41, ctypes.POINTER(LP))(sample, ctypes.byref(buffer))) or not buffer.value:
                _release(sample)
                continue
            ptr = LP()
            cap = ctypes.c_uint32(0)
            length = ctypes.c_uint32(0)
            hr = _hr(_method(buffer, 3, ctypes.POINTER(LP), ctypes.POINTER(ctypes.c_uint32),
                             ctypes.POINTER(ctypes.c_uint32))(
                buffer, ctypes.byref(ptr), ctypes.byref(cap), ctypes.byref(length)))
            if hr:
                _release(buffer)
                _release(sample)
                continue
            raw = ctypes.string_at(ptr, length.value)
            _method(buffer, 4)(buffer)  # Unlock
            _release(buffer)
            _release(sample)
            return self._decode(raw, length.value), stamp.value
        return None, 0

    def _decode(self, raw: bytes, length: int):
        fmt = self._current
        if fmt is None:
            return None
        w, h = fmt.width, fmt.height

        if fmt.fourcc == "MJPG":
            img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise OSError("MJPG 解码失败")
            return img

        if fmt.fourcc in ("NV12", "YUY2"):
            # 两者都是平面/半平面 YUV，Y 分量即红外亮度
            stride = self._stride or (w * 2 if fmt.fourcc == "YUY2" else w)
            kind = np.uint8
            if fmt.fourcc == "YUY2":
                plane = np.frombuffer(raw, kind, count=min(len(raw), stride * h))
                plane = plane.reshape(h, stride)[:, 0:w * 2:2]
            else:
                plane = np.frombuffer(raw, kind, count=min(len(raw), stride * h))
                plane = plane.reshape(h, stride)[:, :w]
            return np.ascontiguousarray(plane)

        if fmt.fourcc in ("Y16 ", "Y16"):
            stride = self._stride or w * 2
            plane = np.frombuffer(raw, np.uint16, count=min(len(raw) // 2, (stride // 2) * h))
            return np.ascontiguousarray(plane.reshape(h, stride // 2)[:, :w])

        # 兜底：当作 8bit 单通道
        side = int(np.sqrt(length))
        if side * side == length:
            return np.frombuffer(raw, np.uint8).reshape(side, side)
        raise OSError("暂不支持的像素格式：%s" % fmt.fourcc)


# ---------------------------------------------------------------------------
# 取帧线程
# ---------------------------------------------------------------------------

class Grabber(threading.Thread):
    """后台取帧，只保留最新一帧，避免界面慢时延迟累积。"""

    def __init__(self, camera: IRCamera):
        super().__init__(daemon=True, name="ir-grabber")
        self.camera = camera
        self._lock = threading.Lock()
        self._frame = None
        self._stamp = 0
        self._seq = 0
        self._stop_evt = threading.Event()
        self._error: BaseException | None = None
        self.frames = 0

    def run(self):
        _co_initialize()
        while not self._stop_evt.is_set():
            try:
                frame, stamp = self.camera.grab()
            except BaseException as exc:  # 交给主线程报告
                self._error = exc
                return
            if frame is None:
                continue
            with self._lock:
                self._frame = frame
                self._stamp = stamp
                self._seq += 1
                self.frames += 1

    def latest(self):
        with self._lock:
            return self._frame, self._seq

    def stop(self):
        self._stop_evt.set()

    @property
    def error(self):
        return self._error


# ---------------------------------------------------------------------------
# 可视化
# ---------------------------------------------------------------------------

COLORMAPS = ["gray", "jet", "turbo", "hot", "inferno", "viridis"]
KEY_TO_CMAP = {"g": "gray", "j": "jet", "t": "turbo", "h": "hot", "i": "inferno", "v": "viridis"}
# OpenCV colormap 常量（4.x 才有 turbo/magma/inferno/plasma/viridis）
CMAP_CODES = {"jet": "COLORMAP_JET", "turbo": "COLORMAP_TURBO", "hot": "COLORMAP_HOT",
              "inferno": "COLORMAP_INFERNO", "viridis": "COLORMAP_VIRIDIS"}


def _normalize(frame, stretch: bool = True):
    """把红外亮度图线性映射到 uint8（可选按 1%~99% 分位自动拉伸对比度）。"""
    if frame is None or not frame.size:
        return frame

    if frame.dtype == np.uint8:
        if not stretch:
            return frame
        lo, hi = np.percentile(frame, (1, 99))
        if hi - lo < 4:  # 画面几乎均匀时退回全局极值
            lo, hi = float(frame.min()), float(frame.max())
        if hi - lo < 1:
            return frame
    else:
        info = np.iinfo(frame.dtype)
        if stretch:
            lo, hi = np.percentile(frame, (1, 99))
            if hi - lo < 4:
                lo, hi = float(frame.min()), float(frame.max())
        else:
            lo, hi = float(info.min), float(info.max)
        if hi - lo < 1:
            hi = lo + 1

    out = (frame.astype(np.float32) - lo) * (255.0 / (hi - lo))
    return np.clip(out, 0, 255).astype(np.uint8)


def colorize(frame, cmap: str = "gray", stretch: bool = True):
    """返回可直接交给 cv2.imshow / imwrite 的 BGR 图。"""
    gray = _normalize(frame, stretch)
    if cmap == "gray" or gray.ndim == 3:
        return gray if gray.ndim == 3 else cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return cv2.applyColorMap(gray, getattr(cv2, CMAP_CODES.get(cmap, "COLORMAP_JET")))


def save_image(path: str, frame, cmap: str = "gray", stretch: bool = True) -> None:
    """存图（PNG 无压缩损失，PGM/TIFF 亦可）。"""
    if not cv2.imwrite(path, colorize(frame, cmap, stretch)):
        raise OSError("写入失败：%s" % path)


class CvViewer:
    """基于 OpenCV HighGUI 的实时窗口。"""

    WINDOW = "IR Camera - lowcost_mocap"

    def __init__(self, grabber: Grabber, cmap: str = "gray", stretch: bool = True,
                 window_size=(960, 540)):
        self.grabber = grabber
        self.cmap = cmap if cmap in COLORMAPS else "gray"
        self.stretch = stretch
        self.window_size = window_size
        self._last_ts = time.perf_counter()
        self._fps = 0.0
        self._seq = -1
        self._count = 0
        self._written = []

    def run(self):
        cv2.namedWindow(self.WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(self.WINDOW, *self.window_size)
        try:
            while True:
                if self.grabber.error:
                    raise self.grabber.error

                frame, seq = self.grabber.latest()
                if frame is not None and seq != self._seq:
                    self._seq = seq
                    self._count += 1
                    now = time.perf_counter()
                    if now - self._last_ts >= 0.5:
                        self._fps = self._count / (now - self._last_ts)
                        self._count = 0
                        self._last_ts = now

                    canvas = colorize(frame, self.cmap, self.stretch)
                    self._draw_hud(canvas, frame)
                    cv2.imshow(self.WINDOW, canvas)

                key = cv2.waitKey(1) & 0xFF
                if key == 255:  # 无按键
                    pass
                elif key in (ord("q"), 27):
                    break
                elif chr(key).lower() in KEY_TO_CMAP:
                    self.cmap = KEY_TO_CMAP[chr(key).lower()]
                elif chr(key).lower() == "a":
                    self.stretch = not self.stretch
                elif chr(key).lower() == "s":
                    self._save(frame)

                # 用户点了窗口右上角关闭
                try:
                    if cv2.getWindowProperty(self.WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                        break
                except cv2.error:
                    break
        finally:
            self.grabber.stop()
            # 用户点右上角关闭时窗口已被销毁，再 destroy 会抛 NULL window
            try:
                cv2.destroyWindow(self.WINDOW)
            except cv2.error:
                pass
            cv2.waitKey(1)
        return self._written

    def _draw_hud(self, canvas, frame):
        cam = self.grabber.camera
        lines = [
            "%s  %dx%d  %s  %.1f fps" % (cam.info.split(" ")[0] if cam._current else "?",
                                         frame.shape[1], frame.shape[0], self.cmap, self._fps),
            "mean %.1f  min %d  max %d  stretch %s"
            % (float(frame.mean()), int(frame.min()), int(frame.max()),
               "on" if self.stretch else "off"),
        ]
        for i, text in enumerate(lines):
            cv2.putText(canvas, text, (10, 24 + i * 22), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(canvas, text, (10, 24 + i * 22), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 255, 0), 1, cv2.LINE_AA)
        hint = "g/j/t/h/i/v colormap   a stretch   s save   q quit"
        h = canvas.shape[0]
        cv2.putText(canvas, hint, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, hint, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (200, 200, 200), 1, cv2.LINE_AA)

    def _save(self, frame):
        path = os.path.abspath("ir_%s.png" % datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3])
        save_image(path, frame, self.cmap, self.stretch)
        self._written.append(path)
        print("已保存：%s" % path)



# ---------------------------------------------------------------------------
# 命令行
# ---------------------------------------------------------------------------

def _cmd_list():
    devices = list_devices()
    if not devices:
        print("没有检测到任何视频采集设备")
        return 1
    print("检测到 %d 个视频采集设备：" % len(devices))
    for i, name, link in devices:
        print("  [%d] %s" % (i, name or "<未命名>"))
        print("      %s" % link)
    return 0


def _cmd_formats(args):
    cam = IRCamera(index=args.index, name=args.name, verbose=False)
    cam.open()
    try:
        print("相机：%s" % cam.info)
        print("支持 %d 种格式：" % len(cam.formats()))
        for f in cam.formats():
            print("  [%2d] %s" % (f.index, f))
    finally:
        cam.close()
    return 0


def _cmd_snapshot(args):
    cam = IRCamera(index=args.index, name=args.name, width=args.width, height=args.height,
                   fps=args.fps, pixel_format=args.format)
    cam.open()
    try:
        frame = None
        for _ in range(30):  # 丢弃前几帧，等自动曝光稳定
            frame, _ = cam.grab()
            if frame is not None:
                break
        if frame is None:
            print("没有取到图像", file=sys.stderr)
            return 1
        path = os.path.abspath(args.snapshot)
        save_image(path, frame, args.cmap, not args.no_stretch)
        print("已保存：%s   %dx%d  min=%d max=%d mean=%.1f"
              % (path, frame.shape[1], frame.shape[0], frame.min(), frame.max(), frame.mean()))
    finally:
        cam.close()
    return 0


def _cmd_stream(args):
    cam = IRCamera(index=args.index, name=args.name, width=args.width, height=args.height,
                   fps=args.fps, pixel_format=args.format)
    cam.open()
    try:
        if args.no_display:
            t0 = time.perf_counter()
            n = args.frames or 60
            last = 0
            for _ in range(n):
                frame, _ = cam.grab()
                if frame is not None:
                    last += 1
            dt = time.perf_counter() - t0
            print("连续取帧 %d/%d，用时 %.2fs，平均 %.1f fps" % (last, n, dt, last / dt))
            return 0

        grabber = Grabber(cam)
        grabber.start()
        viewer = CvViewer(grabber, cmap=args.cmap, stretch=not args.no_stretch)
        try:
            saved = viewer.run()
        finally:
            grabber.stop()
            grabber.join(timeout=2.0)
        if grabber.error:
            raise grabber.error
        if saved:
            print("本次会话保存了 %d 张图片" % len(saved))
        return 0
    finally:
        cam.close()


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="读取并可视化 USB 红外相机（UVC / Media Foundation）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("显示窗口内的快捷键")[-1])
    parser.add_argument("--list", action="store_true", help="列出相机后退出")
    parser.add_argument("--formats", action="store_true", help="列出该相机支持的所有格式后退出")
    parser.add_argument("--index", type=int, default=0, help="相机索引，默认 0")
    parser.add_argument("--name", default=None, help="按名称子串选择相机，例如 LRCP")
    parser.add_argument("--width", type=int, default=1280, help="期望宽度，默认 1280")
    parser.add_argument("--height", type=int, default=720, help="期望高度，默认 720")
    parser.add_argument("--fps", type=float, default=30.0, help="期望帧率，默认 30")
    parser.add_argument("--format", default="NV12",
                        help="像素格式 NV12 / MJPG / YUY2 / AUTO，默认 NV12")
    parser.add_argument("--cmap", default="gray", help="初始配色，默认 gray")
    parser.add_argument("--no-stretch", action="store_true", help="关闭自动对比度拉伸")
    parser.add_argument("--snapshot", metavar="PATH", help="抓一帧保存为图片后退出")
    parser.add_argument("--frames", type=int, default=0, help="配合 --no-display 指定取帧数量")
    parser.add_argument("--no-display", action="store_true", help="不弹窗口，只跑取帧")
    args = parser.parse_args(argv)

    if args.list:
        return _cmd_list()
    if args.formats:
        return _cmd_formats(args)
    if args.snapshot:
        return _cmd_snapshot(args)
    return _cmd_stream(args)


if __name__ == "__main__":
    sys.exit(main())
