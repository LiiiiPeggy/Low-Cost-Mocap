"""双 USB 红外相机内外参标定（OpenCV calibrateCamera / stereoCalibrate 封装）。

模块分工
--------------------------------------------------------------------------
``board``      棋盘格规格、角点检测、质量评估、可视化
``dataset``    会话目录、样本清单、原始图落盘
``intrinsic``  单相机内参（K、D）+ 逐张重投影误差
``stereo``     双相机外参（R、T、E、F）+ 极线/三角化/已知尺寸校验
``service``    与 camera_manager / sync 对接的流程编排
``routes``     Flask 蓝图（/api/calib/*）
``selftest``   合成数据自检（无实机也能验证求解链路）

坐标约定（一句话）
--------------------------------------------------------------------------
物点单位是**米**，图像坐标是**原始灰度图像素坐标**，
外参满足 ``X_cam1 = R_10 @ X_cam0 + T_10``。

避免循环导入：这里只做极轻的再导出，真正实现按需 import。
"""

from __future__ import annotations

__all__ = [
    "BoardSpec", "DEFAULT_COLS", "DEFAULT_ROWS", "DEFAULT_SQUARE_MM",
    "detect", "detect_pair", "draw_overlay", "coverage_map",
    "Session", "Sample",
    "intrinsic", "stereo", "service", "routes",
]

__version__ = "1.0.0"


def __getattr__(name):  # 惰性导出，避免 import 链过深
    if name in ("BoardSpec", "DEFAULT_COLS", "DEFAULT_ROWS", "DEFAULT_SQUARE_MM",
                "detect", "detect_pair", "draw_overlay", "coverage_map"):
        from calibration import board
        return getattr(board, name)
    if name in ("Session", "Sample"):
        from calibration import dataset
        return getattr(dataset, name)
    raise AttributeError(name)
