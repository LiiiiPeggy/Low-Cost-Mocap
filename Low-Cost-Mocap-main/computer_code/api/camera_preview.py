"""Flask 蓝图：多路 USB 红外相机的状态查询、MJPEG 推流与原始帧保存。

挂载方式（两种入口都支持）::

    # 1) 独立的最小服务（只要摄像头，不需要 serial / ruckig）
    python api/preview_server.py --indices 0 1

    # 2) 复用原项目的 Flask app（index.py 里已注册）
    python api/index.py

所有路由都以 ``/api`` 开头，前端通过 Vite 代理访问。
推流只读 CameraManager 的最新帧缓存，**一个设备只打开一次**，
浏览器刷新 / 多标签页不会重复打开摄像头。
"""

from __future__ import annotations

import os
import time
from datetime import datetime

from flask import Blueprint, Response, jsonify, request, send_from_directory

from camera_manager import manager

bp = Blueprint("camera_preview", __name__)

_STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

DEFAULT_INDICES = os.environ.get("MOCAP_CAM_INDICES", "0,1")
DEFAULT_WIDTH = int(os.environ.get("MOCAP_CAM_WIDTH", "1280"))
DEFAULT_HEIGHT = int(os.environ.get("MOCAP_CAM_HEIGHT", "720"))
DEFAULT_FPS = float(os.environ.get("MOCAP_CAM_FPS", "30"))
DEFAULT_FORMAT = os.environ.get("MOCAP_CAM_FORMAT", "NV12")
DEFAULT_OUT_DIR = os.environ.get("MOCAP_CAPTURE_DIR", os.path.abspath("captures"))


def _parse_indices(raw) -> list[int]:
    if raw is None or raw == "":
        raw = DEFAULT_INDICES
    if isinstance(raw, (list, tuple)):
        return [int(x) for x in raw]
    return [int(x) for x in str(raw).replace(" ", "").split(",") if x != ""]


# ---------------------------------------------------------------------------
# 状态 / 设备
# ---------------------------------------------------------------------------
@bp.get("/api/cameras")
def cameras_status():
    st = manager().status()
    st["server_time"] = time.time()
    st["defaults"] = {
        "indices": _parse_indices(None),
        "width": DEFAULT_WIDTH, "height": DEFAULT_HEIGHT,
        "fps": DEFAULT_FPS, "format": DEFAULT_FORMAT,
    }
    return jsonify(st)


@bp.get("/api/cameras/devices")
def cameras_devices():
    try:
        return jsonify({"devices": manager().devices()})
    except Exception as exc:
        return jsonify({"error": "%s: %s" % (type(exc).__name__, exc)}), 500


# ---------------------------------------------------------------------------
# 启动 / 停止
# ---------------------------------------------------------------------------
@bp.post("/api/cameras/start")
def cameras_start():
    body = request.get_json(silent=True) or {}
    indices = _parse_indices(body.get("indices"))
    if not indices:
        return jsonify({"error": "没有指定相机索引"}), 400
    try:
        st = manager().start(
            indices,
            width=int(body.get("width", DEFAULT_WIDTH)),
            height=int(body.get("height", DEFAULT_HEIGHT)),
            fps=float(body.get("fps", DEFAULT_FPS)),
            pixel_format=str(body.get("format", body.get("pixel_format", DEFAULT_FORMAT))),
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 409
    except Exception as exc:
        return jsonify({"error": "%s: %s" % (type(exc).__name__, exc)}), 500
    return jsonify(st)


@bp.post("/api/cameras/stop")
def cameras_stop():
    body = request.get_json(silent=True) or {}
    indices = body.get("indices")
    try:
        st = manager().stop(None if indices is None else _parse_indices(indices))
    except Exception as exc:
        return jsonify({"error": "%s: %s" % (type(exc).__name__, exc)}), 500
    return jsonify(st)


# ---------------------------------------------------------------------------
# 取帧
# ---------------------------------------------------------------------------
@bp.get("/api/camera/<int:index>/meta")
def camera_meta(index: int):
    worker = manager().worker(index)
    if worker is None:
        return jsonify({"error": "相机 %d 未启动" % index}), 404
    return jsonify(worker.status())


@bp.get("/api/camera/<int:index>/snapshot")
def camera_snapshot(index: int):
    """单帧快照。默认 JPEG（网页预览用），?raw=1 返回 8bit 原始灰度 PNG。"""
    worker = manager().worker(index)
    if worker is None:
        return jsonify({"error": "相机 %d 未启动" % index}), 404
    frame = worker.latest()
    if frame is None:
        return jsonify({"error": "相机 %d 还没有帧" % index}), 503

    if request.args.get("raw") in ("1", "true", "yes"):
        import cv2

        ok, buf = cv2.imencode(".png", frame.gray)  # 原始灰度，不拉伸
        mime = "image/png"
    else:
        data = worker.jpeg()
        if data is None:
            return jsonify({"error": "编码失败"}), 500
        buf, mime = data, "image/jpeg"

    resp = Response(buf.tobytes(), mimetype=mime)
    resp.headers["Cache-Control"] = "no-store"
    resp.headers["X-Camera-Index"] = str(index)
    resp.headers["X-Frame-Seq"] = str(frame.seq)
    resp.headers["X-MF-Timestamp"] = str(frame.mf_ts)
    resp.headers["X-Host-Timestamp"] = "%.6f" % frame.host_ts
    return resp


@bp.get("/api/camera/<int:index>/stream")
def camera_stream(index: int):
    """MJPEG 实时流。只读共享缓存，不会重复打开设备。"""
    if manager().worker(index) is None:
        return jsonify({"error": "相机 %d 未启动" % index}), 404

    return Response(_mjpeg_generator(index),
                    mimetype="multipart/x-mixed-replace; boundary=frame",
                    direct_passthrough=True,
                    headers={"Cache-Control": "no-store",
                             "X-Accel-Buffering": "no"})


def _mjpeg_generator(index: int):
    m = manager()
    last_seq = -1
    idle_since = time.time()

    while not m.stopping:
        worker = m.worker(index)
        if worker is None:  # 被 stop() 掉，结束这个连接
            break

        frame = worker.wait_frame(last_seq, timeout=0.5)
        if frame is None:
            # 还没新帧：短超时后继续等，让“断线重连”期间连接不中断
            if not worker.alive and not worker.reconnect:
                break
            # 空闲太久且已经不在推流（例如设备永久失效）就收尾
            if worker.state in ("error", "stopped") and time.time() - idle_since > 15:
                break
            continue

        last_seq = frame.seq
        idle_since = time.time()
        data = worker.jpeg()
        if not data:
            continue
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n"
               b"Content-Length: " + str(len(data)).encode() + b"\r\n"
               b"X-Seq: " + str(frame.seq).encode() + b"\r\n\r\n" + data + b"\r\n")


# ---------------------------------------------------------------------------
# 保存原始灰度
# ---------------------------------------------------------------------------
@bp.post("/api/cameras/save")
def cameras_save():
    body = request.get_json(silent=True) or {}
    out_dir = body.get("out_dir") or DEFAULT_OUT_DIR
    indices = body.get("indices")
    try:
        saved = manager().save_raw(out_dir, None if indices is None else _parse_indices(indices))
    except Exception as exc:
        return jsonify({"error": "%s: %s" % (type(exc).__name__, exc)}), 500
    return jsonify({"saved": saved, "out_dir": os.path.abspath(out_dir),
                    "count": len(saved),
                    "time": datetime.now().isoformat()})


# ---------------------------------------------------------------------------
# 免构建的兜底预览页（本机没有 node_modules 时可直接看）
# ---------------------------------------------------------------------------
@bp.get("/preview")
def preview_page():
    return send_from_directory(_STATIC_DIR, "preview.html")


def register(app) -> None:
    """把蓝图挂到任意 Flask app 上。"""
    app.register_blueprint(bp)
