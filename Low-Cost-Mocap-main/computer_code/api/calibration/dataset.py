"""标定数据集的落盘与读取。

目录结构（默认 ``captures/calibration/<session_id>/``）::

    <session>/
      session.json          板规格、相机身份、图像尺寸、采集格式、备注
      samples.json          样本清单（角点、质量、有效状态、失败原因）
      img/<id>_cam0.png     原始灰度图（未旋转/未缩放/未去畸变）
      img/<id>_cam1.png
      overlay/<id>_cam0.png 角点叠加图（仅供查看，不参与计算）
      result_intrinsic.json 内参中间结果
      result_stereo.json    外参中间结果

设计原则：**采集即落盘**，所以标定可以完全离线重算，不需要相机在线。
所有图像一律按采集到的原始灰度写 PNG，保证角点像素坐标与图片一一对应。
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable

import cv2
import numpy as np

from calibration.board import BoardSpec, Detection, to_jsonable


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _atomic_json(path: str, data) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(to_jsonable(data), fh, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


@dataclass
class Sample:
    """一条样本：单相机一张，或双相机一对。

    ``phase`` 区分用途：
      * ``intrinsic``  —— 板在移动，用于内参标定
      * ``extrinsic``  —— 板固定在中央地面，用于 solvePnP 求世界系外参
      * ``validation`` —— 板在其它位置/高度，用于独立精度验证
    """

    id: str
    kind: str                       # "single" | "stereo"
    created: str
    images: dict                    # {cam_key: {file, seq, mf_ts, host_ts, mono_ts, width, height}}
    detections: dict                # {cam_key: Detection.to_dict()}
    status: str = "valid"           # valid | invalid
    reason: str = ""
    dt_ms: float | None = None      # 双相机配对时间差
    flip_b: bool = False            # 兼容旧字段（stereoCalibrate 路线用）
    signature: dict = field(default_factory=dict)   # 姿态签名（判重用）
    note: str = ""
    phase: str = "intrinsic"

    def corners(self, cam) -> np.ndarray | None:
        d = self.detections.get(str(cam)) or {}
        if not d.get("ok"):
            return None
        c = d.get("corners") or []
        if not c:
            return None
        return np.asarray(c, dtype=np.float32).reshape(-1, 1, 2)

    def to_dict(self) -> dict:
        return {
            "id": self.id, "kind": self.kind, "created": self.created,
            "images": self.images, "detections": self.detections,
            "status": self.status, "reason": self.reason,
            "dt_ms": self.dt_ms, "flip_b": bool(self.flip_b),
            "signature": self.signature, "note": self.note,
            "phase": self.phase,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Sample":
        return cls(
            id=d["id"], kind=d.get("kind", "single"), created=d.get("created", ""),
            images=d.get("images", {}), detections=d.get("detections", {}),
            status=d.get("status", "valid"), reason=d.get("reason", ""),
            dt_ms=d.get("dt_ms"), flip_b=bool(d.get("flip_b", False)),
            signature=d.get("signature", {}), note=d.get("note", ""),
            phase=d.get("phase", "intrinsic"),
        )


class Session:
    """一个标定会话：数据采集 + 清单维护。"""

    def __init__(self, root: str):
        self.root = os.path.abspath(root)
        self.img_dir = os.path.join(self.root, "img")
        self.overlay_dir = os.path.join(self.root, "overlay")
        self.meta: dict = {}
        self.samples: list[Sample] = []
        self._dirty = False

    # -- 创建 / 载入 ------------------------------------------------------
    @classmethod
    def create(cls, root: str, spec: BoardSpec, cameras: Iterable[int],
               image_size: tuple[int, int], pixel_format: str = "",
               note: str = "") -> "Session":
        s = cls(root)
        os.makedirs(s.img_dir, exist_ok=True)
        os.makedirs(s.overlay_dir, exist_ok=True)
        idxs = [int(c) for c in cameras]
        s.meta = {
            "schema": "lowcost-mocap/calibration-session",
            "version": 1,
            "session_id": os.path.basename(s.root),
            "created": _now(),
            "updated": _now(),
            "board": spec.to_dict(),
            "cameras": idxs,
            "image_size": [int(image_size[0]), int(image_size[1])],
            "pixel_format": pixel_format,
            "image_convention": {
                "source": "raw grayscale frame from camera_manager (NV12 Y plane or MJPG decode)",
                "rotation": 0,
                "crop": None,
                "resize": None,
                "undistort": False,
                "note": "角点坐标 = 该原始图像像素坐标；显示层的旋转/缩放不影响本目录数据",
            },
            "note": note,
        }
        s.save()
        return s

    @classmethod
    def load(cls, root: str) -> "Session":
        s = cls(root)
        meta_path = os.path.join(s.root, "session.json")
        if not os.path.isfile(meta_path):
            raise FileNotFoundError("不是有效的标定会话目录（缺少 session.json）：%s" % s.root)
        with open(meta_path, encoding="utf-8") as fh:
            s.meta = json.load(fh)
        samples_path = os.path.join(s.root, "samples.json")
        if os.path.isfile(samples_path):
            with open(samples_path, encoding="utf-8") as fh:
                s.samples = [Sample.from_dict(d) for d in json.load(fh)]
        os.makedirs(s.img_dir, exist_ok=True)
        os.makedirs(s.overlay_dir, exist_ok=True)
        return s

    @classmethod
    def create_or_load(cls, root: str, spec: BoardSpec, cameras: Iterable[int],
                       image_size: tuple[int, int], pixel_format: str = "") -> "Session":
        if os.path.isfile(os.path.join(root, "session.json")):
            s = cls.load(root)
            old = s.spec
            if (old.cols, old.rows) != (spec.cols, spec.rows):
                raise ValueError(
                    "已有会话的棋盘格规格是 %dx%d，与当前的 %dx%d 不一致；"
                    "请新建会话或改回原规格" % (old.cols, old.rows, spec.cols, spec.rows))
            if abs(old.square_mm - spec.square_mm) > 1e-6:
                raise ValueError(
                    "已有会话的方格边长是 %g mm，与当前的 %g mm 不一致"
                    % (old.square_mm, spec.square_mm))
            return s
        return cls.create(root, spec, cameras, image_size, pixel_format)

    # -- 属性 -------------------------------------------------------------
    @property
    def spec(self) -> BoardSpec:
        return BoardSpec.from_dict(self.meta.get("board", {}))

    @property
    def image_size(self) -> tuple[int, int]:
        return tuple(self.meta.get("image_size", [0, 0]))

    @property
    def cameras(self) -> list[int]:
        return list(self.meta.get("cameras", []))

    def save(self) -> None:
        os.makedirs(self.root, exist_ok=True)
        self.meta["updated"] = _now()
        _atomic_json(os.path.join(self.root, "session.json"), self.meta)
        _atomic_json(os.path.join(self.root, "samples.json"),
                     [s.to_dict() for s in self.samples])
        self._dirty = False

    # -- 编号 -------------------------------------------------------------
    def _next_id(self, prefix: str) -> str:
        n = 0
        for s in self.samples:
            if not s.id.startswith(prefix):
                continue
            try:
                n = max(n, int(s.id[len(prefix):]))
            except ValueError:
                pass
        return "%s%04d" % (prefix, n + 1)

    # -- 写入样本 ---------------------------------------------------------
    def _write_image(self, path: str, gray: np.ndarray) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not cv2.imwrite(path, np.ascontiguousarray(gray)):
            raise OSError("写入失败：%s" % path)

    def add_single(self, camera: int, gray: np.ndarray, det: Detection,
                   frame_meta: dict, *, overlay: np.ndarray | None = None,
                   status: str = "valid", reason: str = "",
                   signature: dict | None = None,
                   phase: str = "intrinsic") -> Sample:
        sid = self._next_id("s")
        key = str(int(camera))
        fname = "%s_cam%s.png" % (sid, key)
        path = os.path.join(self.img_dir, fname)
        self._write_image(path, gray)
        if overlay is not None:
            self._write_image(os.path.join(self.overlay_dir, fname), overlay)

        sample = Sample(
            id=sid, kind="single", created=_now(),
            images={key: dict(frame_meta, file=fname,
                              width=int(gray.shape[1]), height=int(gray.shape[0]))},
            detections={key: det.to_dict()},
            status=status, reason=reason,
            signature=signature or {}, phase=phase,
        )
        self.samples.append(sample)
        self.save()
        return sample

    def add_stereo(self, pair_meta: dict, grays: dict, dets: dict,
                   *, dt_ms: float | None, flip_b: bool = False,
                   overlay: dict | None = None, status: str = "valid",
                   reason: str = "", signature: dict | None = None,
                   phase: str = "validation") -> Sample:
        sid = self._next_id("p")
        images, detections = {}, {}
        for cam, gray in grays.items():
            key = str(int(cam))
            fname = "%s_%s_cam%s.png" % (sid, "AB"[list(grays).index(cam)], key)
            self._write_image(os.path.join(self.img_dir, fname), gray)
            if overlay and key in overlay:
                self._write_image(os.path.join(self.overlay_dir, fname), overlay[key])
            images[key] = dict((pair_meta.get(key) or {}), file=fname,
                               width=int(gray.shape[1]), height=int(gray.shape[0]))
            detections[key] = dets[cam].to_dict()

        sample = Sample(
            id=sid, kind="stereo", created=_now(), images=images,
            detections=detections, status=status, reason=reason,
            dt_ms=None if dt_ms is None else round(float(dt_ms), 3),
            flip_b=bool(flip_b), signature=signature or {}, phase=phase,
        )
        self.samples.append(sample)
        self.save()
        return sample

    def remove(self, sample_id: str) -> bool:
        keep, drop = [], None
        for s in self.samples:
            if s.id == sample_id and drop is None:
                drop = s
            else:
                keep.append(s)
        if drop is None:
            return False
        self.samples = keep
        for key, meta in (drop.images or {}).items():
            for base in (self.img_dir, self.overlay_dir):
                p = os.path.join(base, meta.get("file", ""))
                if meta.get("file") and os.path.isfile(p):
                    try:
                        os.remove(p)
                    except OSError:
                        pass
        self.save()
        return True

    def clear(self) -> int:
        n = len(self.samples)
        for d in (self.img_dir, self.overlay_dir):
            if os.path.isdir(d):
                shutil.rmtree(d, ignore_errors=True)
            os.makedirs(d, exist_ok=True)
        self.samples = []
        self.save()
        return n

    def set_status(self, sample_id: str, status: str, reason: str = "") -> bool:
        for s in self.samples:
            if s.id == sample_id:
                s.status = status
                s.reason = reason
                self.save()
                return True
        return False

    # -- 查询 -------------------------------------------------------------
    def single_samples(self, camera: int, *, include_invalid: bool = False,
                       exclude: Iterable[str] = (),
                       phase: str | None = None) -> list[Sample]:
        ex = set(exclude)
        return [s for s in self.samples
                if s.kind == "single" and s.id not in ex
                and (include_invalid or s.status == "valid")
                and (phase is None or s.phase == phase)
                and s.corners(camera) is not None]

    def stereo_samples(self, *, include_invalid: bool = False,
                       exclude: Iterable[str] = (),
                       phase: str | None = None) -> list[Sample]:
        ex = set(exclude)
        return [s for s in self.samples
                if s.kind == "stereo" and s.id not in ex
                and (include_invalid or s.status == "valid")
                and (phase is None or s.phase == phase)
                and s.corners(0) is not None and s.corners(1) is not None]

    def signatures(self, camera: int, kind: str = "single",
                   phase: str | None = None) -> list[dict]:
        out = []
        for s in self.samples:
            if s.kind != kind or s.status != "valid":
                continue
            if phase is not None and s.phase != phase:
                continue
            if kind == "single" and str(camera) not in s.detections:
                continue
            out.append({"id": s.id, "signature": s.signature})
        return out

    def stats(self) -> dict:
        def count(kind, status):
            return sum(1 for s in self.samples if s.kind == kind and s.status == status)

        return {
            "total": len(self.samples),
            "single_valid": {str(c): len(self.single_samples(c)) for c in self.cameras},
            "single_invalid": sum(1 for s in self.samples
                                  if s.kind == "single" and s.status != "valid"),
            "stereo_valid": len(self.stereo_samples()),
            "stereo_invalid": sum(1 for s in self.samples
                                  if s.kind == "stereo" and s.status != "valid"),
            "by_phase": {
                "intrinsic": {str(c): len(self.single_samples(c, phase="intrinsic"))
                              for c in self.cameras},
                "extrinsic": {str(c): len(self.single_samples(c, phase="extrinsic"))
                              for c in self.cameras},
                "validation_pairs": len(self.stereo_samples(phase="validation")),
            },
            "ids": [s.id for s in self.samples],
        }

    def summary(self) -> dict:
        return {
            "root": self.root,
            "session_id": self.meta.get("session_id"),
            "created": self.meta.get("created"),
            "updated": self.meta.get("updated"),
            "board": self.spec.to_dict(),
            "cameras": self.cameras,
            "image_size": list(self.image_size),
            "pixel_format": self.meta.get("pixel_format"),
            "image_convention": self.meta.get("image_convention"),
            "note": self.meta.get("note", ""),
            "stats": self.stats(),
        }

    def save_result(self, name: str, data: dict) -> str:
        path = os.path.join(self.root, name)
        _atomic_json(path, data)
        return path
