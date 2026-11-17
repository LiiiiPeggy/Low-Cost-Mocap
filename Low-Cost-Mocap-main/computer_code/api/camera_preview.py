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
import subprocess
import time
from datetime import datetime

from flask import Blueprint, Response, jsonify, request, send_from_directory

import camera_controls
from camera_manager import manager
from sync import sync_service
from sync import DEFAULT_OUT_DIR as SYNC_OUT_DIR

bp = Blueprint("camera_preview", __name__)

_STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

DEFAULT_INDICES = os.environ.get("MOCAP_CAM_INDICES", "0,1")
DEFAULT_WIDTH = int(os.environ.get("MOCAP_CAM_WIDTH", "1280"))
DEFAULT_HEIGHT = int(os.environ.get("MOCAP_CAM_HEIGHT", "720"))
DEFAULT_FPS = float(os.environ.get("MOCAP_CAM_FPS", "30"))
DEFAULT_FORMAT = os.environ.get("MOCAP_CAM_FORMAT", "NV12")
DEFAULT_OUT_DIR = os.environ.get("MOCAP_CAPTURE_DIR", os.path.abspath("captures"))

_SERVER_STARTED = datetime.now().isoformat(timespec="seconds")
_GIT_INFO: dict | None = None


def _git_info() -> dict:
    """进程启动版本：git 短 SHA / 分支 / 是否有未提交改动。失败时返回 unknown。"""
    global _GIT_INFO
    if _GIT_INFO is not None:
        return _GIT_INFO
    api_dir = os.path.dirname(os.path.abspath(__file__))

    def run(*args) -> str:
        return subprocess.check_output(
            ["git", *args], cwd=api_dir, text=True, encoding="utf-8",
            errors="replace", stderr=subprocess.DEVNULL, timeout=2).strip()

    try:
        sha = run("rev-parse", "--short", "HEAD")
        branch = run("rev-parse", "--abbrev-ref", "HEAD")
        dirty = bool(run("status", "--porcelain"))
    except Exception:
        sha, branch, dirty = "unknown", "unknown", False
    _GIT_INFO = {"git_sha": sha, "branch": branch, "dirty": dirty}
    return _GIT_INFO


@bp.get("/api/version")
def api_version():
    info = dict(_git_info())
    info.update({"started_at": _SERVER_STARTED, "pid": os.getpid(),
                 "server": "preview_server"})
    resp = jsonify(info)
    resp.headers["Cache-Control"] = "no-store"
    return resp


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
# 曝光 / 增益（DirectShow UVC 控制，不打断 MF 取流）
# ---------------------------------------------------------------------------
@bp.get("/api/cameras/controls")
def cameras_controls_get():
    raw = request.args.get("camera")
    try:
        indices = (_parse_indices(raw) if raw not in (None, "")
                   else _parse_indices(None))
    except (TypeError, ValueError):
        return jsonify({"error": "camera 必须是逗号分隔的整数索引"}), 400
    controls = []
    for i in indices:
        worker = manager().worker(i)
        link = worker.device_link if worker else None
        item = camera_controls.get_controls(i, link)
        item["mf_device_link"] = link
        if link and item.get("device_key"):
            item["device_match"] = (camera_controls.device_key(link)
                                    == camera_controls.device_key(
                                        item.get("device_key")))
        else:
            item["device_match"] = None
        controls.append(item)
    resp = jsonify({"controls": controls, "indices": indices,
                    "note": "曝光为 DirectShow 对数时间 log2(s)；增益为设备索引"})
    resp.headers["Cache-Control"] = "no-store"
    return resp


def _runtime_snapshot() -> dict:
    """曝光设置前后用于对比的轻量运行状态。"""
    try:
        sync = sync_service().status()
    except Exception:
        sync = {}
    st = manager().status()
    return {
        "cameras": [{
            "index": c.get("index"),
            "state": c.get("state"),
            "fps": c.get("fps"),
            "frame_interval_ms": c.get("frame_interval_ms"),
        } for c in st.get("cameras", [])],
        "sync": {
            "pair_rate": sync.get("pair_rate"),
            "recent": sync.get("recent"),
            "dt_ms": sync.get("dt_ms"),
            "dropped": sync.get("dropped"),
            "produced": sync.get("produced"),
        },
    }


@bp.post("/api/cameras/controls")
def cameras_controls_set():
    body = request.get_json(silent=True) or {}
    try:
        index = int(body.get("camera", 0))
    except (TypeError, ValueError):
        return jsonify({"error": "camera 必须是整数索引"}), 400
    exposure = body.get("exposure", None)
    auto_exposure = body.get("auto_exposure", None)
    gain = body.get("gain", None)
    try:
        if exposure is not None:
            exposure = int(round(float(exposure)))
        if auto_exposure is not None:
            if not isinstance(auto_exposure, bool):
                auto_exposure = str(auto_exposure).strip().lower() in (
                    "1", "true", "yes", "on", "开")
        if gain is not None:
            gain = int(round(float(gain)))
    except (TypeError, ValueError):
        return jsonify({"error": "exposure/gain 必须是数字，auto_exposure 必须是布尔值"}), 400
    if exposure is None and auto_exposure is None and gain is None:
        return jsonify({"error": "至少提供 exposure、auto_exposure、gain 之一"}), 400
    worker = manager().worker(index)
    link = worker.device_link if worker else None
    before = _runtime_snapshot()
    try:
        result = camera_controls.set_controls(
            index, exposure=exposure, auto_exposure=auto_exposure, gain=gain,
            device_link=link)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": "%s: %s" % (type(exc).__name__, exc)}), 500
    # 给同步/帧间隔统计一点时间反映设置后的状态（不修改任何统计）
    time.sleep(0.25)
    after = _runtime_snapshot()
    result["before"] = before
    result["after"] = after
    result["mf_device_link"] = link
    if link and result.get("device_key"):
        result["device_match"] = (camera_controls.device_key(link)
                                  == camera_controls.device_key(
                                      result.get("device_key")))
    if not result.get("ok"):
        return jsonify(result), 409
    resp = jsonify(result)
    resp.headers["Cache-Control"] = "no-store"
    return resp


@bp.post("/api/cameras/controls/restore")
def cameras_controls_restore():
    """手动恢复两台相机上次保存的曝光/增益（启动时也会自动恢复）。"""
    body = request.get_json(silent=True) or {}
    raw = body.get("indices")
    try:
        indices = (_parse_indices(raw) if raw not in (None, "")
                   else _parse_indices(None))
    except (TypeError, ValueError):
        return jsonify({"error": "indices 必须是整数数组"}), 400
    links = {}
    for i in indices:
        w = manager().worker(i)
        if w and w.device_link:
            links[str(i)] = w.device_link
    results = camera_controls.restore_all(indices, links)
    resp = jsonify({"ok": True, "results": results, "indices": indices})
    resp.headers["Cache-Control"] = "no-store"
    return resp


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
    # 新会话：清空配对指针与统计（相机 seq 从 1 重新计数）
    try:
        sync_service().reset_pointers()
    except Exception:
        pass
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
# 帧同步（软件级，基于主机单调时间戳）
# ---------------------------------------------------------------------------
@bp.get("/api/sync/status")
def sync_status():
    """配对统计：Δt 的均值/P95/最大、配对率、丢弃数、缓存深度。"""
    try:
        return jsonify(sync_service().status())
    except Exception as exc:
        return jsonify({"error": "%s: %s" % (type(exc).__name__, exc)}), 500


@bp.post("/api/sync/config")
def sync_config():
    """运行时调整最大允许时间差（毫秒）。"""
    body = request.get_json(silent=True) or {}
    if "max_dt_ms" not in body:
        return jsonify({"error": "缺少 max_dt_ms"}), 400
    try:
        v = sync_service().set_max_dt(float(body["max_dt_ms"]))
    except (TypeError, ValueError):
        return jsonify({"error": "max_dt_ms 必须是数字"}), 400
    return jsonify({"max_dt_ms": v, "status": sync_service().status()})


@bp.post("/api/sync/save")
def sync_save():
    """保存最近若干对同步帧：两张原始灰度 PNG + 每对时间戳 JSON + manifest。"""
    body = request.get_json(silent=True) or {}
    count = int(body.get("count", 8))
    out_dir = body.get("out_dir")
    if not out_dir:
        out_dir = os.path.join(SYNC_OUT_DIR,
                               "sync_%s" % datetime.now().strftime("%Y%m%d_%H%M%S"))
    try:
        res = sync_service().save_pairs(out_dir, count)
    except Exception as exc:
        return jsonify({"error": "%s: %s" % (type(exc).__name__, exc)}), 500
    return jsonify(res)


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
    # 标定页更新后不能让浏览器继续用旧缓存里的 HTML/JS
    resp = send_from_directory(_STATIC_DIR, "preview.html", max_age=0)
    resp.headers["Cache-Control"] = "no-store, must-revalidate"
    return resp


def register(app) -> None:
    """把蓝图挂到任意 Flask app 上。"""
    app.register_blueprint(bp)
