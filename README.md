# Low-Cost-Mocap

低成本红外光学动捕方案。本仓库在原项目 [jyjblrd/Low-Cost-Mocap](https://github.com/jyjblrd/Low-Cost-Mocap)
（ESP32 红外标记 + 基站 + 上位机解算）基础上，补充了 **USB 红外相机的取帧与可视化工具**，
以及相机选型、Astra Pro 适配等调研笔记。

## 目录结构

| 路径 | 说明 |
|---|---|
| `ir_camera.py` | **单相机**取帧 + 桌面实时可视化（Windows / Media Foundation，纯 ctypes） |
| `Low-Cost-Mocap-main/` | 原始项目代码（3D 件、ESP32 收发端、上位机解算） |
| `Low-Cost-Mocap-main/computer_code/api/camera_manager.py` | **双/多相机并发采集**（每相机一线程，latest-frame 缓存 + 时间戳） |
| `Low-Cost-Mocap-main/computer_code/api/camera_preview.py` | Flask 蓝图：状态 / MJPEG 推流 / 原始帧保存 |
| `Low-Cost-Mocap-main/computer_code/api/preview_server.py` | 只含相机功能的最小服务（无需 serial / ruckig） |
| `Low-Cost-Mocap-main/computer_code/src/components/CameraPreview.tsx` | React 双路预览面板 |
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

## Windows 双 USB 红外相机实时预览

两台 LRCP G720P 同时采集（1280×720@30 NV12），网页左右并排显示，可启停、可保存原始灰度。

### 架构（最小改动）

```
ir_camera.py                 ← 单相机取帧（未改动）
  └── api/camera_manager.py  ← 每相机一个线程；latest-frame 缓存；MF/主机双时间戳
        ├── api/camera_preview.py   ← Flask 蓝图：/api/cameras、MJPEG、save
        │     ├── api/preview_server.py   独立入口（3002，无需 serial/ruckig）
        │     └── api/index.py            原后端（3001）注册同一份蓝图
        └── src/components/CameraPreview.tsx ← 挂在 App.tsx 的一个新卡片里
```

关键设计：

- **一相机一线程**，`open()/grab()/close()` 全部在该线程内完成（COM STA 要求同线程创建/释放），
  主线程只读缓存，互不干扰。
- **只留最新一帧**：读端永远拿最新数据，浏览器卡顿或刷新都不会累积延迟。
- **同一帧只编码一次 JPEG**：多个 `<img>` / 多标签页共享缓存，不会重复开摄像头。
- **索引唯一**：重复 `start` 同一设备返回 `409`。
- **时间戳并存**：`mf_timestamp`（Media Foundation 流时间，100 ns 单位）+ `host_timestamp`
  （`time.time()`）+ `monotonic_timestamp`，供后续软件同步使用。

### 启动后端（二选一）

```bash
conda activate lowcost_mocap
cd Low-Cost-Mocap-main/computer_code

# A. 只做相机预览（推荐先跑这个，端口 3002）
python api/preview_server.py --indices 0 1
#   → 免构建页面： http://127.0.0.1:3002/preview
#   → 状态接口  ： http://127.0.0.1:3002/api/cameras

# B. 复用原项目 Flask 后端（端口 3001，含 socket.io / 三维可视化链路）
python api/index.py
```

常用参数：`--width 1280 --height 720 --fps 30 --format NV12|MJPG --no-autostart`。
带宽或 CPU 吃紧时可用环境变量 `MOCAP_JPEG_QUALITY=60` 降低推流单帧体积（默认 80）。

### 启动前端

```bash
cd Low-Cost-Mocap-main/computer_code
npm install
npm run dev          # http://localhost:5173
```

Vite 代理已在 `vite.config.ts` 配好：`/api` → `http://127.0.0.1:3002`（可用
`MOCAP_PREVIEW_URL` 改成 3001），`/socket.io` → `http://127.0.0.1:3001`。
原来的三维视图、工具栏、相机参数面板全部保留，双相机卡片插在 "Camera Stream" 下方。

### 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/cameras` | 各路状态：state / fps / 格式 / 帧号 / 双时间戳 |
| GET | `/api/cameras/devices` | 枚举系统视频设备（不打开它们） |
| POST | `/api/cameras/start` | `{"indices":[0,1],"width":1280,"height":720,"fps":30,"format":"NV12"}` |
| POST | `/api/cameras/stop` | 停止（可带 `{"indices":[0]}` 只停一路） |
| GET | `/api/camera/<i>/stream` | MJPEG 实时流（`multipart/x-mixed-replace`） |
| GET | `/api/camera/<i>/snapshot?raw=1` | 单帧；`raw=1` 返回 8bit 原始灰度 PNG |
| POST | `/api/cameras/save` | 保存各路最新帧为原始灰度 PNG，返回路径 + 时间戳 |

### 实测结果（Windows 11 + lowcost_mocap，两台 LRCP G720P）

| 项目 | 结果 |
|---|---|
| 双路并发 NV12 1280×720@30 | cam0 / cam1 各 **30.0 fps**（3 s 各 +90 帧） |
| 最新帧滞后 | **20.7 ms** ≈ 1 帧，不随时间增长 |
| MJPEG 单路 5 s | 151 帧 = **30.2 fps**，单帧 JPEG ≈ 100 KB（q=80，约 24 Mbit/s） |
| 三路并发（0 / 1 / 0 两个标签页看同一相机） | 仍只有 **2 个采集线程**，两路各 30 fps |
| 重复 open 同一设备 | `409 设备已在采集中` |
| 停止 / 重启 | `count 0` → 重新 start 正常出图 |
| 保存原始帧 | 1280×720 **uint8 单通道** PNG（无拉伸、无伪彩色） |
| 原后端 `api/index.py` | 缺 `pyserial` / `ruckig` 时可启动，`/api/cameras` 正常，`socket.io` 握手正常 |

> 单相机原有功能未受影响：`python ir_camera.py` 的桌面窗口、`--formats`、`--snapshot` 均保持原样。

### 尚未验证

- **React 前端未能在本机构建/运行**：该机器没有 `node_modules`，且 npm registry 不可达
  （`registry.npmjs.org`、`registry.npmmirror.com` 均连接失败）。`CameraPreview.tsx` 与 Vite 代理
  只是按现有代码风格写好，**没有在浏览器里跑过**，请在有网环境 `npm install && npm run dev` 后自查。
- 双相机同时拔插的热插拔恢复只做了断线重连逻辑（1 s 起指数退避），未做真实拔插实验。
- 未实现红外 LED 检测、相机标定、三角化、Go2 控制（按要求暂不做）。

## 注意

- 相机是**独占**的：预览窗口开着时再运行会报 `0xC00D3704`，需先关闭其他占用程序。
- 该相机输出 **8 bit 亮度**（非辐射亮度），**不能用来测温**。
- 本仓库未包含 267MB 的厂商 DirectShow SDK（`image_driver/camera windows sdk/`），
  因为代码不编译、不链接它，运行时也不需要。

## 许可

原始项目代码遵循其自带 `LICENSE`；本仓库新增的脚本与笔记见 `LICENSE`（如无则默认随原项目条款）。
