"""Flask 蓝图：红外 LED 检测、三维定位、参数与叠加图（/api/led/*）。"""

from __future__ import annotations

from flask import Blueprint, Response, jsonify, request

from led.service import service

bp = Blueprint("led", __name__)


def _body() -> dict:
    return request.get_json(silent=True) or {}


def _fail(exc: Exception, code: int = 400):
    msg = str(exc).strip().splitlines()
    return jsonify({"error": msg[-1] if msg else type(exc).__name__,
                    "type": type(exc).__name__}), code


@bp.get("/api/led/state")
def led_state():
    try:
        resp = jsonify(service().state())
        resp.headers["Cache-Control"] = "no-store"
        return resp
    except Exception as exc:
        return _fail(exc, 500)


@bp.get("/api/led/calibration")
def led_calibration():
    try:
        return jsonify(service().calibration_state())
    except Exception as exc:
        return _fail(exc, 500)


@bp.get("/api/led/params")
def led_params_get():
    try:
        return jsonify(service().params_state())
    except Exception as exc:
        return _fail(exc, 500)


@bp.post("/api/led/params")
def led_params_set():
    try:
        return jsonify(service().update_params(_body()))
    except ValueError as exc:
        return _fail(exc, 400)
    except Exception as exc:
        return _fail(exc, 500)


@bp.post("/api/led/params/reset")
def led_params_reset():
    try:
        return jsonify(service().reset_params())
    except Exception as exc:
        return _fail(exc, 500)


@bp.post("/api/led/detect/start")
def led_detect_start():
    try:
        return jsonify(service().start_detection())
    except RuntimeError as exc:
        return _fail(exc, 409)
    except ValueError as exc:
        return _fail(exc, 400)
    except Exception as exc:
        return _fail(exc, 500)


@bp.post("/api/led/detect/stop")
def led_detect_stop():
    try:
        return jsonify(service().stop_detection())
    except Exception as exc:
        return _fail(exc, 500)


@bp.post("/api/led/locate/start")
def led_locate_start():
    try:
        return jsonify(service().start_locate())
    except RuntimeError as exc:
        return _fail(exc, 409)
    except ValueError as exc:
        return _fail(exc, 400)
    except Exception as exc:
        return _fail(exc, 500)


@bp.post("/api/led/locate/stop")
def led_locate_stop():
    try:
        return jsonify(service().stop_locate())
    except Exception as exc:
        return _fail(exc, 500)


@bp.get("/api/led/overlay")
def led_overlay():
    try:
        cam = int(request.args.get("camera", 0))
        view = request.args.get("view", "detect")
        scale = float(request.args.get("scale", 0.6))
        quality = int(request.args.get("quality", 75))
        zoom = float(request.args.get("zoom", 1.0))
        data, info = service().overlay(cam, view=view, scale=scale,
                                       quality=quality, zoom=zoom)
        resp = Response(data, mimetype="image/jpeg")
        resp.headers["Cache-Control"] = "no-store"
        resp.headers["X-LED-Count"] = str(info.get("count"))
        resp.headers["X-LED-Matched"] = str(info.get("matched"))
        resp.headers["X-LED-Raw"] = str(info.get("raw_count"))
        resp.headers["X-LED-Zoom"] = str(info.get("zoom"))
        return resp
    except RuntimeError as exc:
        return _fail(exc, 409)
    except ValueError as exc:
        return _fail(exc, 400)
    except Exception as exc:
        return _fail(exc, 500)


@bp.post("/api/led/debug/save")
def led_debug_save():
    try:
        return jsonify(service().save_debug())
    except RuntimeError as exc:
        return _fail(exc, 409)
    except Exception as exc:
        return _fail(exc, 500)


def register(app) -> None:
    app.register_blueprint(bp)
