"""红外 LED 检测、双相机匹配、三角化与三维跟踪。

模块分工
--------------------------------------------------------------------------
``detector``  单相机灰度阈值、连通域、峰值分离、亚像素中心
``geometry``  极线匹配、世界系三角化、34/79/79 mm 等腰三角形识别
``tracker``   基于真实时间戳的三维卡尔曼跟踪
``service``   复用 SyncService 的后台处理线程、参数持久化、二维叠加图
``recorder``  复用同步帧对的连续调试录制（PNG + JSONL）
``routes``    Flask 蓝图（/api/led/*）
"""

from __future__ import annotations

__all__ = ["detector", "geometry", "tracker", "service", "recorder",
           "routes"]
