"""红外相机与标定/LED 定位服务（当前唯一后端，端口 3002）。

    python api/preview_server.py                     # 默认启动 camera_config 里的全部相机
    python api/preview_server.py --indices 0 1       # 只开指定逻辑编号
    python api/preview_server.py --format MJPG       # 带宽不够时换压缩格式
    python api/preview_server.py --no-autostart      # 起来后等网页点 Start

浏览器打开 http://127.0.0.1:3002/preview 可以直接看多路画面，无需前端构建。
相机身份由 ``captures/calibration/camera_config.json`` 固定绑定（逻辑编号 ↔
USB 设备实例 key），枚举顺序变化不会改变编号。
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

from flask import Flask
from flask_cors import CORS

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import camera_config  # noqa: E402
import camera_preview  # noqa: E402
import calibration.routes as calibration_routes  # noqa: E402
import led.routes as led_routes  # noqa: E402
from camera_manager import manager  # noqa: E402
from led.recorder import recorder as led_recorder  # noqa: E402
from sync import sync_service  # noqa: E402


def build_app() -> Flask:
    app = Flask(__name__, static_folder=None)
    CORS(app, resources={r"/api/*": {"origins": "*"}})
    camera_preview.register(app)
    calibration_routes.register(app)   # /api/calib/*
    led_routes.register(app)           # /api/led/*

    @app.get("/")
    def index():
        return ('<meta charset="utf-8"><h3>IR camera preview server</h3>'
                '<p>预览页面：<a href="/preview">/preview</a>　'
                '状态接口：<a href="/api/cameras">/api/cameras</a></p>')

    return app


def _short_device(key: str | None) -> str:
    """设备实例 key → 便于肉眼区分的短标识（保留末尾 USB 端口段）。"""
    if not key:
        return "（无绑定）"
    tail = str(key).rsplit("#", 1)[-1] if "#" in str(key) else str(key)
    return "..." + tail[-20:]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="多路 USB 红外相机 MJPEG 预览服务")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=3002)
    p.add_argument("--indices", type=int, nargs="+", default=None,
                   help="要启动的逻辑相机编号，默认取 camera_config 全部")
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--height", type=int, default=None)
    p.add_argument("--fps", type=float, default=None)
    p.add_argument("--format", default=None,
                   help="NV12 / MJPG / YUY2 / AUTO；默认用每台相机自己的配置")
    p.add_argument("--no-autostart", action="store_true", help="不自动打开相机，等网页点 Start")
    p.add_argument("--debug", action="store_true")
    p.add_argument("--verbose", action="store_true",
                   help="输出每次 HTTP 请求的访问日志（默认关闭，避免轮询刷屏）")
    args = p.parse_args(argv)

    app = build_app()
    m = manager()

    try:
        cfg = camera_config.load()
    except Exception as exc:
        print("[警告] camera_config.json 读取失败：%s" % exc, file=sys.stderr)
        cfg = camera_config.default_config()

    # 帧同步分组线程常驻（空转开销可忽略），相机启动后立即开始分组
    sync_service().set_indices(camera_config.logical_indices(cfg))

    if not args.no_autostart:
        indices = args.indices or camera_config.logical_indices(cfg)
        body = {"width": args.width, "height": args.height,
                "fps": args.fps, "format": args.format}
        body = {k: v for k, v in body.items() if v is not None}
        try:
            entries = camera_preview._camera_start_configs(indices, body)
            m.start(entries)
            sync_service().set_indices(indices)
            sync_service().reset_pointers()
            print("已启动 %d 台相机：" % len(entries))
            for e in entries:
                print("  cam%d  %s %dx%d@%g  %s"
                      % (e["index"], e["format"], e["width"], e["height"],
                         e["fps"], _short_device(e.get("device_key"))))
        except Exception as exc:
            # 相机不在线不应该拦住服务，网页上会显示 error/offline 状态
            print("[警告] 相机启动失败：%s: %s" % (type(exc).__name__, exc),
                  file=sys.stderr)

    url = "http://%s:%d/preview" % (args.host, args.port)
    print("预览页面：%s" % url)
    print("按 Ctrl+C 结束（会自动释放相机）")
    if not args.verbose:
        # 页面每秒轮询多个接口 + 四路 MJPEG，默认关闭 werkzeug 访问日志
        logging.getLogger("werkzeug").setLevel(logging.ERROR)
        app.logger.setLevel(logging.ERROR)

    try:
        app.run(host=args.host, port=args.port, debug=args.debug, threaded=True,
                use_reloader=False)
    finally:
        try:
            led_recorder().stop()
        except Exception as exc:
            print("[警告] 停止 LED 录制失败：%s" % exc, file=sys.stderr)
        m.shutdown()
        print("\n相机已释放")
    return 0


if __name__ == "__main__":
    sys.exit(main())
