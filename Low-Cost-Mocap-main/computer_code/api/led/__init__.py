"""红外 LED 检测、双相机匹配、三角化与三维跟踪。

模块分工
--------------------------------------------------------------------------
``detector``  单相机灰度阈值、连通域、峰值分离、亚像素中心
``geometry``  极线匹配、世界系三角化、33 mm 等边三角形识别
``tracker``   基于真实时间戳的三维卡尔曼跟踪
``service``   复用 SyncService 的后台处理线程、参数持久化、二维叠加图
``routes``    Flask 蓝图（/api/led/*）
"""

from __future__ import annotations

__all__ = ["detector", "geometry", "tracker", "service", "routes"]
