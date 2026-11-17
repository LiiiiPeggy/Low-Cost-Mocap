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
| `Low-Cost-Mocap-main/computer_code/api/calibration/` | **双相机内参 / 固定棋盘格世界系外参**（`calibrateCamera` + `solvePnP`；`stereoCalibrate` 仅作交叉校验） |
| `Low-Cost-Mocap-main/computer_code/api/led/` | **红外 LED 检测、极线匹配与三维定位**（三 LED 标记，L0–L2 临时编号） |
| `Low-Cost-Mocap-main/computer_code/api/calib_cli.py` | 标定命令行：采集 / 重检测 / 求解 / 验证 / 保存 |
| `Low-Cost-Mocap-main/computer_code/src/components/CameraPreview.tsx` | React 双路预览面板 |
| `AGENTS.md` / `PROGRESS.md` / `MEMORY.md` / `PROCESS.md` | AI Agent 公共工作规则与项目记忆 |
| `docs/calibration.md` | **标定使用说明、坐标约定、精度指标、局限** |
| `docs/` | 调研笔记：方案总结、Astra Pro 适配、配图 |
| `image_driver/` | 相机厂商资料（说明书 PDF、AmCap 工具、DirectShow SDK 文档） |
| `Notes.md` | 物料清单 |
| `environment*.yml` | conda 环境定义 |

## ir_camera.py

不依赖厂商 SDK，也不需要 OpenCV 的 VideoCapture——直接通过 ctypes 调用 Windows 自带的
**Media Foundation**（`mfplat.dll` / `mf.dll` / `mfreadwrite.dll`）取流，仅依赖 `numpy` + `opencv-python`。

实测设备为 UVC 相机 `LRCP G720P`（OV9281 单色 CMOS），支持 NV12 / MJPG / YUY2，
最高 1280×960@30；双路 1280×720@30 NV12 实测每路约 30 fps。

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
  └── api/camera_manager.py  ← 每相机一个线程；latest 缓存 + 定长历史缓存；三时间戳
        ├── api/camera_preview.py   ← Flask 蓝图：/api/cameras、MJPEG、save、sync
        │     ├── api/preview_server.py   独立入口（3002，无需 serial/ruckig）
        │     └── api/index.py            原后端（3001）注册同一份蓝图
        └── api/sync.py             ← 帧同步：主机单调时间戳最近邻配对 + 统计 + 保存成对原图
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
- **历史缓存与最新帧共享同一个 ndarray**：入缓存时把 NV12 的跨步 Y 平面（`[:, ::2]`）
  转成连续数组，45 帧只占 45×0.92 MB，且不会连带持有整块 1.38 MB 的 NV12 buffer。

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
| GET | `/api/sync/status` | 同步统计：Δt 均值 / P95 / 最大、配对率、丢弃数、缓存深度 |
| POST | `/api/sync/config` | `{"max_dt_ms":20}` 运行时改配对阈值（1–500 ms） |
| POST | `/api/sync/save` | `{"count":8}` 保存最近 N 对：两张原始灰度 PNG + 每对 JSON + `manifest.json` |

### 帧同步（软件级，非硬件同步）

`api/sync.py` 每 10 ms 扫一次两路历史缓存，按**主机单调时间戳**做流式最近邻配对。

**必须知道的边界**（否则会用错数据集）：

- 配对**只**用 `time.perf_counter()`（同一进程、同一时钟）。Wall clock 不参与，避免系统对时跳变。
- `mf_timestamp` 是**每台相机自己**的 MF 流时间轴，起点由驱动/会话决定，**跨设备不能直接相减**，
  因此完全不参与配对，只作溯源信息随图保存。实测反例：某一对帧两路 `mf_timestamp` 只差
  **0.135 ms**，而主机时间差是 **15.95 ms**。
- 时间戳记录的是**帧交付给用户态**的时刻（`ReadSample` 返回），**不是曝光中心**：
  `交付时刻 = 曝光中心 + 曝光时长 + 读出 + USB 传输 + 驱动缓冲`，这个差值随曝光/帧率/负载变化，
  **不是常数偏移**，无法用固定量补偿。
- 结论：**软件级近似同步**。30 fps 下帧周期 33.3 ms，仅时间量化就有 ±16.7 ms（半帧）量级。
  可用于静止标定板、低速运动；**不能**用于快速运动目标的亚毫秒三角化。要更高精度需要硬件
  触发（共享时钟）或 LED 检测标定偏差 —— 均未实现。

配对规则：两路各有一个按 `seq` 单调推进的"已消费"指针 → **帧不重复使用、配对时间单调不减**；
比较各自最老的未消费帧，`|Δt| ≤ max_dt` 则配对，否则丢弃更老的那一帧（流式最近邻）。
一路停流超过 0.5 s 时，另一路积压帧按丢弃计，避免配对率虚高。

配置（环境变量，也可在网页上实时改，改后只影响之后的新帧）：

| 变量 | 默认 | 说明 |
|---|---|---|
| `MOCAP_SYNC_MAX_DT_MS` | `20` | 最大允许时间差（ms） |
| `MOCAP_SYNC_BUFFER` | `45` | 每路历史缓存帧数（≈1.5 s@30fps，双路约 83 MB） |
| `MOCAP_SYNC_WINDOW` | `300` | 统计窗口（对数） |
| `MOCAP_SYNC_RECENT` | `16` | 内存里保留可保存的最近对数 |

网页顶部同步条实时显示：**累计配对率**与**最近 300 个配对/丢弃事件**的配对率、两路丢弃数、`|Δt|` 均值 / P95 / 最大、缓存深度，并可一键「保存同步对」。`/api/sync/status` 另外返回每路 FPS、最近帧间隔 mean/P95/max；这样能区分历史累计问题与当前实时状态。
保存目录默认 `captures/sync/sync_<时间戳>/`。

### 双相机内参 / 外参标定（固定棋盘格 + 世界系）

完整说明见 **[docs/calibration.md](docs/calibration.md)**。求解全部使用 OpenCV 成熟接口
（`findChessboardCornersSB`、`calibrateCamera`、`solvePnP` + `solvePnPRefineLM`、
`undistortPoints`、`triangulatePoints`），未手写标定算法，未引入 `cv2.sfm`，
也不需要 `opencv-contrib-python`。外参按 EasyMocap 的固定棋盘格思路：**板当世界基准，
每台相机各自 solvePnP**，`stereoCalibrate` 仅留作交叉校验。

**现场配置**：12×9 方格 → **11×8 内角点**，单格 **30 mm**，1280×720。

**世界系**：原点 = 图案中心；X 沿 12 格长边；Y 沿 9 格短边；Z 向上（右手，`Z = X × Y`）。
88 个角点，`index = row*11 + col`，行优先，`x∈[-0.150,0.150]`、`y∈[-0.105,0.105]` m。
板面高于地面时填「板面高」，地面即 `Z = -surface_height_m`。

**上手**

```bat
conda activate lowcost_mocap
cd Low-Cost-Mocap-main\computer_code
python api\preview_server.py --indices 0 1
:: 浏览器打开 http://127.0.0.1:3002/preview（标定面板默认展开，会话自动复用/创建）
```

面板内：页面自动打开 11×8 / 30mm 会话（已有则复用）→ 板移动采 30–40 张内参图 →「内参标定」查看 K/D/RMS/覆盖率
→ 板固定到中心地面 →「采集双相机图片」取同步帧对（可重复若干次）→「解外参 solvePnP」
→ 可视化模式切到「世界坐标轴」逐台确认 X 轴沿 12 格长边（不对就勾对应相机的「反向 180°」，
确认后勾「已确认世界坐标轴方向」）→ 板挪到 3 个不同位置/高度 →「采集并验证三维精度」。
内参与已确认外参会自动保存到 `captures/calibration/`，无需手动保存。

> 外参每台 1 张有效图即可求解，多采几张主要用于观察帧间抖动；修改 180° 会作废旧外参与方向确认。

**命令行**（采集默认走 HTTP，复用网页服务里的相机；求解不需要相机在线）

```bat
:: 先起服务：python api\preview_server.py --indices 0 1
python api\calib_cli.py new --cols 11 --rows 8 --square-mm 30 --surface-height-mm 3
python api\calib_cli.py capture-single --session <会话> --camera 0 --count 30
python api\calib_cli.py capture-single --session <会话> --camera 0 --count 6 --phase extrinsic
python api\calib_cli.py capture-single --session <会话> --camera 1 --count 6 --phase extrinsic
python api\calib_cli.py capture-stereo --session <会话> --count 3
python api\calib_cli.py extrinsic --session <会话> [--flip-cam0|--flip-cam1] [--orientation-confirmed]
python api\calib_cli.py validate-rig --session <会话>
python api\calib_cli.py save --session <会话> --validate --orientation-confirmed
```

**关键约定**

- 图像坐标 = **原始 1280×720 灰度图**像素坐标，未 `rot90`、未 `make_square`、未裁剪/缩放/去畸变。
- 物点单位 **米**。`X_cam = R_wc @ X_world + t_wc`；`X_cam1 = R_10 @ X_cam0 + T_10`，
  `||T||` = 基线（米）。
- **矩形板有 D2 四重对称**：4 种角点对应在几何上完全等价（重投影误差逐位相同），
  本模块只用「相机在图案正面一侧」筛掉镜像解，剩下的 180° 必须人工看世界轴确认。
  注意**看不出**来也要确认：合成案例中单侧反向会把基线从 2032 mm 变成 1635 mm，而三角化
  自洽残差只有约 0.6 mm，指标不会报警。两台相机可以各自独立翻转 180°。
- 结果自动保存：内参 → `captures/calibration/intrinsics.json`，外参 →
  `captures/calibration/world_extrinsics.json`，均可从网页重新载入；这两份新格式结果已纳入 Git。
  只有「两台内参 + 已确认外参」才会额外生成兼容文件 `api/calibration/calibration.json`
  （schema v2，供旧下游读取）。原始采集图仍在 `captures/` 下并被忽略；旧 `camera-params.json` **未改动**。
- 默认板面高 **3 mm**；数据目录为 `computer_code/captures/calibration/<session_id>/`。
- 下游取参数：`calibration.service.load_calibration()` + `stereo_matrices(data, scaled_size=...)`，
  或 `GET /api/calib/matrices?size=640x360`。

**接口**

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/calib/status` | 会话、样本计数（按 phase）、内外参状态、相机位姿摘要 |
| POST | `/api/calib/session` · `/session/load` · `/session/close` | 新建 / 打开 / 关闭会话（含板面高） |
| POST | `/api/calib/capture/single` | 采单相机内参图（判重 + 质量门限） |
| POST | `/api/calib/capture/extrinsic-pair` | 采一对同步帧作外参（双路检测，按 `pair_seq` 去重） |
| POST | `/api/calib/capture/extrinsic` | 采单相机外参图（CLI / 兼容路径） |
| POST | `/api/calib/capture/stereo` | 采同步帧对（旧验证路径） |
| GET | `/api/calib/live?camera=0` | 实时检测叠加（JPEG） |
| GET | `/api/calib/extrinsic/overlay?camera=0&flip180=0` | 世界原点 + XYZ 轴叠加 |
| POST | `/api/calib/intrinsic` · `/intrinsic/load` | 内参求解 / 载入已保存内参 |
| POST | `/api/calib/solve/extrinsic` · `/extrinsic/load` | 外参求解 / 载入已保存外参 |
| POST | `/api/calib/validate/capture` | 采同步帧并做三维精度验证 |
| POST | `/api/calib/validate/rig` | 用已有验证帧重新计算 |
| POST | `/api/calib/stereo` | stereoCalibrate 交叉校验（非主路线） |
| POST | `/api/calib/save` | 写兼容文件 `calibration.json` |
| GET | `/api/calib/result` · `/api/calib/matrices` | 查看结果 / 取下游矩阵 |
| GET | `/api/version` | Git SHA / 分支 / dirty / 启动时间 |
| GET/POST | `/api/led/state` · `/params` · `/detect/start` · `/locate/start` 等 | LED 检测、参数、ROI、调试保存与三维定位 |
| GET/POST | `/api/cameras/controls` | UVC 曝光/自动曝光/增益范围、当前值、设置与读回验证；POST `/controls/restore` 恢复保存值 |
| GET | `/api/led/overlay_pair` · `/api/led/diagnose` | 同一同步帧对的双相机叠加图；分阶段极线/三角化诊断 |
| POST | `/api/calib/session/ensure` | 页面初始化幂等会话：已有复用，没有才创建 |

**自检（无相机可跑）**

```bat
conda activate lowcost_mocap
cd Low-Cost-Mocap-main\computer_code\api
python -m calibration.selftest   :: 136/136
python -m led.selftest           :: 50/50
```

`calibration.selftest` 用已知 `K/D/R/T` 按真实畸变模型反向渲染虚拟棋盘格，再走正式代码求解；
同时用合成数据/假相机覆盖 IPPE 物理筛选、HTTP/CLI 采集、自动保存/载入、结果失效、兼容文件、
LED UI 与版本接口等流程。`led.selftest` 覆盖光斑检测、粘连拆分、过曝拒绝、细长反光过滤、
ROI、极线/三角化、33 mm 三边、自动相机映射、卡尔曼跟踪、同帧对叠加图、matched=0 极线视图、调试帧保存与接口。
最近一次（2026-11-17）：**136/136 + 50/50 通过**。
两者都只证明求解链路与流程，**不是实机精度**。

### 红外 LED 检测与三维定位（第二阶段）

标记灯统一采用 **850 nm 红外 LED**，不再评估 940 nm 方案。模块复用两台相机的同步帧对和已载入的标定参数：

- **二维检测**：阈值 + 连通域/面积/solidity/填充率/长宽比过滤；距离变换局部峰值拆分粘连光斑，拆分块至少占父亮区 15%；完全饱和/不可分的亮区标记为 unresolved/overexposed，不伪造三点；原始灰度强度加权亚像素中心；分别输出原始连通域和筛选候选。
- **三维定位**：`F` 由标定 R/t 计算，做极线/重投影约束后三角化到世界系；检查 LED 位于相机前方、三边约 33 mm、两相机三点绕向一致。
- **相机映射**：两台相机同名，USB/重启后枚举顺序可能互换。LED 服务同时评估 `live0→calib0/live1→calib1` 和反向映射，按正深度 + 33 mm 三角选择并锁定；状态和调试 JSON 会显示当前映射。
- **ROI**：每台相机可设归一化手动 ROI；匹配后动态跟踪 ROI（+80 px），连续失败 12 帧清除并全图重搜；ROI 不足 3 点自动全图回退。
- **极线诊断视图**：不依赖最终 matches；两台相机有候选点即绘制另一路每个候选的极线，同色标记来源，距离表显示最小极线距离与 OK/FAIL，支持水平/倾斜/近竖直极线裁剪。匹配门槛不变。
- **信息布局**：相机编号、候选/匹配数量、状态、ROI、帧对、Δt 统一放在每台图像**右下角**，按显示尺寸和文字宽度动态右对齐，带半透明深色背景；左下角保留颜色图例，局部放大后仍自适应。
- **限制**：`L0/L1/L2` 是临时编号；三个 LED 完全相同，120° 对称未解，**不输出唯一航向或 6DoF**。
- **参数**：默认阈值 200、面积 3–800 px²、峰距 6 px、三边容差 10 mm；可对 cam0/cam1 分别覆盖阈值/面积/峰距/分离；保存到 `captures/calibration/led_params.json`。
- **入口**：标定面板底部“红外 LED 检测与定位”；接口 `/api/led/state|params|detect/start|stop|locate/start|stop|overlay|overlay_pair|diagnose|debug/save`，以及 `/api/cameras/controls`。
- **曝光/增益**：LRCP G720P 的 UVC 曝光/自动曝光/增益由 DirectShow `IAMCameraControl`/`IAMVideoProcAmp` 实时读写，不另开取流、不打断 MF；控件已移动到“红外 LED 检测与定位”模块内，相机 0/1 独立。设置按 USB 设备实例路径保存到 `captures/calibration/camera_controls.json`，相机重新打开/服务重启后自动恢复并读回验证；不再每 5 秒轮询设备属性。装机时用实际 850 nm LED 确认亮度与检测阈值。

完整说明见 [docs/calibration.md](docs/calibration.md) §6.4。

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

**帧同步实测（阈值 20 ms；历史 60 s 连续运行，另见 2026-11-17 LED 运行统计）**

| 项目 | 结果 |
|---|---|
| 配对数速率 | **30.01 对/秒**（相机 30 fps，理论上限即 30） |
| 配对率 / 丢弃 | **100%**，丢弃 **0 / 0** |
| `\|Δt\|` 均值 | **0.2 – 1.6 ms**（随相位摆动） |
| `\|Δt\|` P95 / 最大 | 0.5 – 16.1 ms / **16.9 ms** |
| 相位漂移 | Δt 在 **0 – 16.7 ms** 之间往复（两相机独立时钟，半帧内的正常摆动） |
| 阈值 10 ms 对照实验 | 区间配对率降到 **71–82%**，每 5 s 丢 25–30 帧；恢复 20 ms 立刻回到 100% |
| 保存同步对 | 6 对 → 12 张 1280×720 uint8 PNG + 6 个 JSON + manifest（4.3 MB） |
| 停止 → 重启 | `pairs` 清零后 8 s 内重新累积 240 对（30/s），seq 归零自愈正常 |
| 2026-11-17 LED 运行 | 两路 30 fps；帧间隔 mean 33.3 ms、P95 约 47–48 ms；最近 300 事件配对率 94–100%，丢弃两路基本对称（如 16/16）；累计值含启动/控制切换暂态 |

> 注意：30.01 对/秒 与 100% 配对率说明**配对算法没有丢帧**，不代表**曝光同步**。
> 真实曝光时间差 = 报告 Δt + 两路交付延迟之差，这个差值未知且非常数。

> 单相机原有功能未受影响：`python ir_camera.py` 的桌面窗口、`--formats`、`--snapshot` 均保持原样。

### 尚未验证

- **React 前端未能在本机构建/运行**：该机器没有 `node_modules`，且 npm registry 不可达
  （`registry.npmjs.org`、`registry.npmmirror.com` 均连接失败）。`CameraPreview.tsx` 与 Vite 代理
  只是按现有代码风格写好，**没有在浏览器里跑过**，请在有网环境 `npm install && npm run dev` 后自查。
  同步状态显示同样只在 Flask 自带的 `/preview` 页上验证过。
- **帧同步只做了 60 s 连续验证**：小时级长跑时两路相位可能周期性穿越阈值（实测 60 s 内 Δt 已在
  0–16.7 ms 往复摆动），长跑时建议盯 `dropped` 是否持续增长。
- **四相机未实现**：`SyncService` 是"A/B 两路"的一对配对器，本身无全局耦合，扩展到四路需要为
  每对（0-1、0-2、0-3…）各建实例并汇总统计，调度层尚未实现。
- 双相机同时拔插的热插拔恢复只做了断线重连逻辑（1 s 起指数退避），未做真实拔插实验。
- **标定与 LED 仍只用合成数据 / 假相机验证**：曾采过实机标定图，但尚未形成“固定安装 + 人工方向确认 +
  多位置独立验证”的可信闭环，仓库不包含可用的实拍标定结果。旧版 `api/calibration/calibration.json`
  已删除；真实 RMS、基线、畸变系数必须现场采集后重新求解。
- 合成数据下画面**四角的畸变外推**误差约 1.6 px（补采四角后从 20 px 降到这里），
  实机需按 `radius_coverage.ratio` 自查是否覆盖足够。
- 标定 UI 只在 Flask 自带 `/preview` 页验证过；React 侧未加标定面板，也未构建。
- **棋盘格方向（180°）只能人工确认**：矩形板的 D2 四重对称在几何上不可分辨，
  单侧反向还会把基线改掉 20% 而不触发任何残差告警（详见 `docs/calibration.md` 第 6 节）。
  若需要完全自动，需要在板上加 ArUco / 缺角标记，本次未做。
- **LED 只完成链路与合成验证**：现场仍需调阈值/面积/峰距，并确认 850 nm LED 的实际亮度与曝光；
  L0–L2 为临时编号，尚不能输出唯一航向。
- Go2 / 机器狗控制与 ROS 2 输出未实现（标定与世界系已建立，见上文）。

## 注意

- 标记灯统一采用 **850 nm 红外 LED**；不再评估 940 nm 方案。
- 相机是**独占**的：预览窗口开着时再运行会报 `0xC00D3704`，需先关闭其他占用程序。
- 该相机输出 **8 bit 亮度**（非辐射亮度），**不能用来测温**。
- 本仓库未包含 267MB 的厂商 DirectShow SDK（`image_driver/camera windows sdk/`），
  因为代码不编译、不链接它，运行时也不需要。

## 项目记忆（面向 AI Agent）

开始实质任务前先读 [PROGRESS.md](PROGRESS.md)；按需读 [MEMORY.md](MEMORY.md)；涉及历史设计或方案比较时读 [PROCESS.md](PROCESS.md)。公共工作规则见 [AGENTS.md](AGENTS.md)。

## 许可

原始项目代码遵循其自带 `LICENSE`；本仓库新增的脚本与笔记见 `LICENSE`（如无则默认随原项目条款）。
