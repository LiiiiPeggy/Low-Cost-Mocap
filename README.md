# Low-Cost-Mocap

低成本红外光学动捕方案。本仓库在原项目 [jyjblrd/Low-Cost-Mocap](https://github.com/jyjblrd/Low-Cost-Mocap)
（ESP32 红外标记 + 基站 + 上位机解算）基础上，补充了 **USB 红外相机的取帧与可视化工具**，
以及相机选型、Astra Pro 适配等调研笔记。

## 目录结构

| 路径 | 说明 |
|---|---|
| `ir_camera.py` | **USB 红外相机取帧 + 实时可视化**（Windows / Media Foundation，纯 ctypes 实现） |
| `Low-Cost-Mocap-main/` | 原始项目代码（3D 件、ESP32 收发端、上位机解算） |
| `docs/` | 调研笔记：方案总结、Astra Pro 适配、配图 |
| `image_driver/` | 相机厂商资料（说明书 PDF、AmCap 工具、DirectShow SDK 文档） |
| `Notes.md` | 物料清单 |
| `environment*.yml` | conda 环境定义 |

## ir_camera.py

不依赖厂商 SDK，也不需要 OpenCV 的 VideoCapture——直接通过 ctypes 调用 Windows 自带的
**Media Foundation**（`mfplat.dll` / `mf.dll` / `mfreadwrite.dll`）取流，仅依赖 `numpy` + `opencv-python`。

实测设备为 UVC 相机 `LRCP G720P`（OV9281 单色 CMOS），支持 NV12 / MJPG / YUY2，
最高 1280×960@30，实测 1280×720@30 NV12 可达 28 fps。

```bash
conda activate lowcost_mocap
python ir_camera.py                     # 实时显示（默认 NV12 1280x720@30）
python ir_camera.py --formats           # 列出相机支持的所有格式
python ir_camera.py --cmap inferno      # 伪彩色显示
python ir_camera.py --snapshot a.png    # 抓一帧存盘
python ir_camera.py --no-display --frames 120   # 无窗口压测
```

窗口快捷键：`g/j/t/h/i/v` 切换 colormap，`a` 开关自动对比度拉伸，`s` 存 PNG，`q` / `Esc` 退出。

## 注意

- 相机是**独占**的：预览窗口开着时再运行会报 `0xC00D3704`，需先关闭其他占用程序。
- 该相机输出 **8 bit 亮度**（非辐射亮度），**不能用来测温**。
- 本仓库未包含 267MB 的厂商 DirectShow SDK（`image_driver/camera windows sdk/`），
  因为代码不编译、不链接它，运行时也不需要。

## 许可

原始项目代码遵循其自带 `LICENSE`；本仓库新增的脚本与笔记见 `LICENSE`（如无则默认随原项目条款）。
