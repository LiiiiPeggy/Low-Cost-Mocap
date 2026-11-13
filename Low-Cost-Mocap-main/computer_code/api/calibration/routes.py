"""Flask 蓝图：标定采集、求解、保存与查看（挂载在 /api/calib 下）。

与 ``camera_preview`` 蓝图并存，互不依赖：标定页可以只开标定接口，预览页照常工作。
所有接口都返回 JSON（图片接口除外），错误统一是 ``{"error": "..."}`` + 4xx/5xx。
"""

from __future__ import annotations

import os

from flask import Blueprint, Response, jsonify, request, send_file

from calibration import board as board_mod
from calibration import intrinsic as intr_mod
from calibration.service import (
    DEFAULT_RESULT_PATH, load_calibration, service, stereo_matrices,
)

bp = Blueprint("calibration", __name__)


def _body() -> dict:
    return request.get_json(silent=True) or {}


def _fail(exc: Exception, code: int = 400):
    msg = str(exc).strip().splitlines()
    return jsonify({"error": msg[-1] if msg else type(exc).__name__,
                    "type": type(exc).__name__}), code


# ---------------------------------------------------------------------------
# 状态 / 会话
# ---------------------------------------------------------------------------
@bp.get("/api/calib/status")
def calib_status():
    try:
        return jsonify(service().status())
    except Exception as exc:
        return _fail(exc, 500)


@bp.post("/api/calib/session")
def calib_session_create():
    b = _body()
    try:
        return jsonify(service().open_session(
            cols=int(b.get("cols", board_mod.DEFAULT_COLS)),
            rows=int(b.get("rows", board_mod.DEFAULT_ROWS)),
            square_mm=float(b.get("square_mm", board_mod.DEFAULT_SQUARE_MM)),
            cameras=b.get("cameras") or [0, 1],
            session_id=b.get("session_id"),
            pixel_format=str(b.get("format", "")),
            note=str(b.get("note", "")),
            surface_height_m=float(b.get("surface_height_mm", 0.0)) / 1000.0,
            root=b.get("root") or None,
        ))
    except Exception as exc:
        return _fail(exc)


@bp.post("/api/calib/session/load")
def calib_session_load():
    """离线重算：把已采集的会话目录重新载入（不需要相机在线）。"""
    b = _body()
    root = b.get("root")
    if not root:
        return jsonify({"error": "缺少 root（会话目录）"}), 400
    try:
        return jsonify(service().reload_session(root))
    except FileNotFoundError as exc:
        return _fail(exc, 404)
    except Exception as exc:
        return _fail(exc)


@bp.post("/api/calib/session/close")
def calib_session_close():
    try:
        return jsonify(service().close_session())
    except Exception as exc:
        return _fail(exc, 500)


@bp.get("/api/calib/samples")
def calib_samples():
    try:
        sess = service().session
        if sess is None:
            return jsonify({"samples": [], "stats": None})
        return jsonify({
            "root": sess.root,
            "board": sess.spec.to_dict(),
            "image_size": list(sess.image_size),
            "stats": sess.stats(),
            "samples": [{
                "id": s.id, "kind": s.kind, "status": s.status, "reason": s.reason,
                "created": s.created, "dt_ms": s.dt_ms, "flip_b": s.flip_b,
                "cams": list(s.detections.keys()),
                "quality": {k: (v.get("quality") or {}) for k, v in s.detections.items()},
                "method": {k: v.get("method") for k, v in s.detections.items()},
            } for s in sess.samples],
        })
    except Exception as exc:
        return _fail(exc, 500)


@bp.post("/api/calib/sample/<sample_id>/delete")
def calib_sample_delete(sample_id: str):
    try:
        sess = service().session
        if sess is None:
            return jsonify({"error": "没有打开的会话"}), 400
        ok = sess.remove(sample_id)
        return (jsonify({"deleted": sample_id, "stats": sess.stats()}) if ok
                else (jsonify({"error": "没有样本 %s" % sample_id}), 404))
    except Exception as exc:
        return _fail(exc)


@bp.post("/api/calib/samples/clear")
def calib_samples_clear():
    try:
        sess = service().session
        if sess is None:
            return jsonify({"error": "没有打开的会话"}), 400
        n = sess.clear()
        return jsonify({"cleared": n, "stats": sess.stats()})
    except Exception as exc:
        return _fail(exc)


@bp.get("/api/calib/sample/<sample_id>/image")
def calib_sample_image(sample_id: str):
    """取样本原图或角点叠加图：?cam=0&overlay=1&scale=1.0"""
    import cv2
    try:
        sess = service().session
        if sess is None:
            return jsonify({"error": "没有打开的会话"}), 400
        sample = next((s for s in sess.samples if s.id == sample_id), None)
        if sample is None:
            return jsonify({"error": "没有样本 %s" % sample_id}), 404
        cam = str(request.args.get("cam", next(iter(sample.images or {"0": {}}))))
        meta = (sample.images or {}).get(cam)
        if not meta or not meta.get("file"):
            return jsonify({"error": "该样本没有相机 %s 的图像" % cam}), 404

        want_overlay = request.args.get("overlay") in ("1", "true")
        raw_path = os.path.join(sess.img_dir, meta["file"])
        if not want_overlay:
            if not os.path.isfile(raw_path):
                return jsonify({"error": "文件不存在：%s" % raw_path}), 404
            return send_file(raw_path, mimetype="image/png", max_age=0)

        over_path = os.path.join(sess.overlay_dir, meta["file"])
        if os.path.isfile(over_path):
            return send_file(over_path, mimetype="image/png", max_age=0)

        # 叠加图没落盘（旧样本或手工导入）时，用保存的角点现场生成，不臆造结果
        if not os.path.isfile(raw_path):
            return jsonify({"error": "原图不存在：%s" % raw_path}), 404
        img = cv2.imread(raw_path, cv2.IMREAD_UNCHANGED)
        d = sample.detections.get(cam) or {}
        det = board_mod.Detection(ok=bool(d.get("ok")),
                                  corners=_corners(d.get("corners")),
                                  method=d.get("method", ""),
                                  quality=d.get("quality") or {})
        scale = float(request.args.get("scale", 1.0))
        canvas = board_mod.draw_overlay(img, det, sess.spec, scale=scale, label_step=3)
        ok, buf = cv2.imencode(".png", canvas)
        if not ok:
            return jsonify({"error": "PNG 编码失败"}), 500
        return Response(buf.tobytes(), mimetype="image/png",
                        headers={"Cache-Control": "no-store"})
    except Exception as exc:
        return _fail(exc, 500)


# ---------------------------------------------------------------------------
# 采集
# ---------------------------------------------------------------------------
@bp.post("/api/calib/capture/single")
def calib_capture_single():
    b = _body()
    try:
        return jsonify(service().capture_single(
            int(b.get("camera", 0)), force=bool(b.get("force", False))))
    except Exception as exc:
        return _fail(exc, 409)


@bp.post("/api/calib/capture/stereo")
def calib_capture_stereo():
    b = _body()
    try:
        return jsonify(service().capture_stereo(force=bool(b.get("force", False))))
    except Exception as exc:
        return _fail(exc, 409)


# ---------------------------------------------------------------------------
# 世界系外参（固定棋盘格 + solvePnP）
# ---------------------------------------------------------------------------
@bp.post("/api/calib/capture/extrinsic")
def calib_capture_extrinsic():
    """外参阶段：棋盘格固定在中央地面，按一次存一张（不做姿态判重）。"""
    b = _body()
    try:
        return jsonify(service().capture_extrinsic(int(b.get("camera", 0))))
    except Exception as exc:
        return _fail(exc, 409)


@bp.post("/api/calib/solve/extrinsic")
def calib_solve_extrinsic():
    b = _body()
    try:
        return jsonify(service().solve_extrinsic(
            flip180=bool(b.get("flip180", False)),
            exclude=b.get("exclude") or ()))
    except Exception as exc:
        return _fail(exc)


@bp.get("/api/calib/extrinsic/overlay")
def calib_extrinsic_overlay():
    """世界原点 + X/Y/Z 轴 + 角点编号叠加到当前画面（JPEG），用于人工确认。"""
    try:
        cam = int(request.args.get("camera", 0))
        flip = request.args.get("flip180") in ("1", "true", "yes")
        scale = float(request.args.get("scale", 0.6))
        data, info = service().extrinsic_overlay(cam, flip180=flip, scale=scale)
        resp = Response(data, mimetype="image/jpeg")
        resp.headers["Cache-Control"] = "no-store"
        resp.headers["X-Calib-Correspondence"] = info["correspondence"]
        resp.headers["X-Calib-Flip"] = "1" if info["flip180"] else "0"
        return resp
    except Exception as exc:
        return _fail(exc, 409)


@bp.post("/api/calib/validate/rig")
def calib_validate_rig():
    """独立验证：板挪到多个位置/高度后的三角化精度。"""
    b = _body()
    try:
        return jsonify(service().validate_rig(holdout=bool(b.get("holdout", False))))
    except Exception as exc:
        return _fail(exc)


@bp.get("/api/calib/live")
def calib_live():
    """实况检测预览：返回叠加了角点的 JPEG（只画图，不入库）。"""
    try:
        cam = int(request.args.get("camera", 0))
        scale = float(request.args.get("scale", 0.6))
        data, info = service().detect_live(cam, scale=scale)
        resp = Response(data, mimetype="image/jpeg")
        resp.headers["Cache-Control"] = "no-store"
        resp.headers["X-Calib-Ok"] = "1" if info["ok"] else "0"
        resp.headers["X-Calib-Detect-Ms"] = str(info["detect_ms"])
        return resp
    except Exception as exc:
        return _fail(exc, 409)


@bp.get("/api/calib/live/info")
def calib_live_info():
    """与 /live 配合：只返回检测结论（JSON），供前端显示文字。"""
    try:
        cam = int(request.args.get("camera", 0))
        data, info = service().detect_live(cam, scale=0.25, quality=40)
        info.pop("jpeg", None)
        return jsonify(info)
    except Exception as exc:
        return _fail(exc, 409)


@bp.get("/api/calib/coverage")
def calib_coverage():
    try:
        sess = service().session
        if sess is None:
            return jsonify({"error": "没有打开的会话"}), 400
        cam = int(request.args.get("camera", 0))
        dets = []
        for s in sess.single_samples(cam):
            d = s.detections.get(str(cam)) or {}
            det = board_mod.Detection(ok=True,
                                      corners=_corners(d.get("corners")),
                                      method=d.get("method", ""), quality=d.get("quality") or {})
            dets.append(det)
        return jsonify(board_mod.coverage_map(dets, tuple(sess.image_size), sess.spec))
    except Exception as exc:
        return _fail(exc, 500)


def _corners(raw):
    import numpy as np
    if not raw:
        return None
    return np.asarray(raw, dtype=np.float32).reshape(-1, 1, 2)


# ---------------------------------------------------------------------------
# 求解 / 验证 / 保存
# ---------------------------------------------------------------------------
@bp.post("/api/calib/intrinsic")
def calib_intrinsic():
    b = _body()
    try:
        cams = b.get("cameras", b.get("camera"))
        if isinstance(cams, str):
            cams = cams if cams != "both" else None
        return jsonify(service().run_intrinsic(cams, exclude=b.get("exclude") or ()))
    except Exception as exc:
        return _fail(exc)


@bp.post("/api/calib/stereo")
def calib_stereo():
    b = _body()
    try:
        return jsonify(service().run_stereo(
            exclude=b.get("exclude") or (),
            auto_flip=bool(b.get("auto_flip", True)),
            flip_b=b.get("flip_b")))
    except Exception as exc:
        return _fail(exc)


@bp.post("/api/calib/validate")
def calib_validate():
    b = _body()
    try:
        cp = b.get("corner_pair")
        return jsonify(service().validate(
            exclude=b.get("exclude") or (),
            holdout=float(b.get("holdout", 0.3)),
            auto_flip=b.get("auto_flip"),
            expected_mm=b.get("expected_mm"),
            corner_pair=(int(cp[0]), int(cp[1])) if cp else None))
    except Exception as exc:
        return _fail(exc)


@bp.post("/api/calib/save")
def calib_save():
    b = _body()
    try:
        return jsonify(service().save_result(b.get("path")))
    except Exception as exc:
        return _fail(exc)


@bp.get("/api/calib/result")
def calib_result():
    """查看当前标定结果（优先内存里刚算的，没有就读已保存的文件）。"""
    try:
        st = service().status()
        if st.get("intrinsics") or st.get("stereo"):
            return jsonify(service().build_result())
        data = load_calibration(DEFAULT_RESULT_PATH)
        return jsonify(data)
    except FileNotFoundError as exc:
        return _fail(exc, 404)
    except Exception as exc:
        return _fail(exc)


@bp.get("/api/calib/matrices")
def calib_matrices():
    """给下游三维定位用：直接给出 (K0,D0,K1,D1,R,T)，可选按新分辨率缩放。"""
    try:
        svc = service()
        path = (request.args.get("path")
                or getattr(svc, "result_path", None) or DEFAULT_RESULT_PATH)
        data = load_calibration(path)
        want = request.args.get("size")
        size = tuple(int(v) for v in want.lower().split("x")) if want else None
        K0, D0, K1, D1, R, T = stereo_matrices(data, scaled_size=size)
        import numpy as np
        return jsonify({
            "K0": K0.tolist(), "D0": D0.tolist(),
            "K1": K1.tolist(), "D1": D1.tolist(),
            "R": R.tolist(), "T": T.tolist(),
            "baseline_m": float(np.linalg.norm(T)),
            "convention": "X_cam1 = R @ X_cam0 + T",
            "scaled_to": list(size) if size else None,
        })
    except Exception as exc:
        return _fail(exc)


@bp.get("/calibration")
def calibration_page():
    """标定界面集成在 /preview 里（#calib 区块），这里只做跳转，避免维护两套 UI。"""
    from flask import redirect
    return redirect("/preview#calib")


def register(app) -> None:
    app.register_blueprint(bp)
