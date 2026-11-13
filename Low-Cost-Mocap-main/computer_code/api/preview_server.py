"""最小可用的红外相机预览服务（不依赖 serial / ruckig / 无人机相关代码）。

    python api/preview_server.py                     # 默认打开 0、1 两台，端口 3002
    python api/preview_server.py --indices 0 2       # 只开 0 和 2
    python api/preview_server.py --format MJPG       # 带宽不够时换压缩格式
    python api/preview_server.py --no-autostart      # 起来后等网页点 Start

浏览器打开 http://127.0.0.1:3002/preview 可以直接看双路画面（免 npm 构建）。

如果还需要原项目的 socket.io / Three.js 数据链路，改用 ``python api/index.py``，
它注册了同一套蓝图，预览接口完全一致。
"""

from __future__ import annotations

import argparse
import os
import sys

from flask import Flask
from flask_cors import CORS

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import camera_preview  # noqa: E402
from camera_manager import manager  # noqa: E402
from sync import sync_service  # noqa: E402


def build_app() -> Flask:
    app = Flask(__name__, static_folder=None)
    CORS(app, resources={r"/api/*": {"origins": "*"}})
    camera_preview.register(app)

    @app.get("/")
    def index():
        return ('<meta charset="utf-8"><h3>IR camera preview server</h3>'
                '<p>预览页面：<a href="/preview">/preview</a>　'
                '状态接口：<a href="/api/cameras">/api/cameras</a></p>')

    return app


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="多路 USB 红外相机 MJPEG 预览服务")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=3002)
    p.add_argument("--indices", type=int, nargs="+", default=None,
                   help="要打开的相机索引，默认取环境变量 MOCAP_CAM_INDICES 或 0 1")
    p.add_argument("--width", type=int, default=camera_preview.DEFAULT_WIDTH)
    p.add_argument("--height", type=int, default=camera_preview.DEFAULT_HEIGHT)
    p.add_argument("--fps", type=float, default=camera_preview.DEFAULT_FPS)
    p.add_argument("--format", default=camera_preview.DEFAULT_FORMAT,
                   help="NV12 / MJPG / YUY2 / AUTO")
    p.add_argument("--no-autostart", action="store_true", help="不自动打开相机，等网页点 Start")
    p.add_argument("--debug", action="store_true")
    args = p.parse_args(argv)

    app = build_app()
    m = manager()

    # 帧同步配对线程常驻（空转开销可忽略），相机启动后立即开始配对
    sync_service()

    if not args.no_autostart:
        indices = args.indices or camera_preview._parse_indices(None)
        try:
            m.start(indices, width=args.width, height=args.height,
                    fps=args.fps, pixel_format=args.format)
            sync_service().reset_pointers()
            print("已启动相机：%s  %dx%d@%g  %s"
                  % (indices, args.width, args.height, args.fps, args.format))
        except Exception as exc:
            # 相机不在线不应该拦住服务，网页上会显示 error 状态
            print("[警告] 相机启动失败：%s: %s" % (type(exc).__name__, exc), file=sys.stderr)

    url = "http://%s:%d/preview" % (args.host, args.port)
    print("预览页面：%s" % url)
    print("按 Ctrl+C 结束（会自动释放相机）")

    try:
        app.run(host=args.host, port=args.port, debug=args.debug, threaded=True,
                use_reloader=False)
    finally:
        m.shutdown()
        print("\n相机已释放")
    return 0


if __name__ == "__main__":
    sys.exit(main())
