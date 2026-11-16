# Low-Cost-Mocap 项目记忆

> 更新至 **2026-11-16（晚）**。由 `2026-11-10.md`、`2026-11-12.md`、`2026-11-13.md`、`2026-11-16.md` 合并精简而成。以**最新已实现状态**为准；区分源码事实、合成测试、实机验证和待办事项。当前 HEAD 仍是 `cd7e991`，本轮界面/采集改版**尚未提交**。

## 1. 项目定位与当前状态

- **项目仓库**：[`LiiiiPeggy/Low-Cost-Mocap`](https://github.com/LiiiiPeggy/Low-Cost-Mocap)，`main`，截至 11 月 16 日最新已推送提交 **`cd7e991`**。
- **原项目**：[`jyjblrd/Low-Cost-Mocap`](https://github.com/jyjblrd/Low-Cost-Mocap)，参考 `main` 的 `8ac4d41`；为室内 *outside-in* 多相机低成本动捕与双无人机控制方案，MIT 许可。
- **当前方向**：把原项目的视觉定位思想迁移到四足机器人 Hypertron-T01，优先完成 **两台 LRCP G720P 相机的采集、软件同步、内参和固定棋盘格世界系外参标定**。后续是否接入定位主链路尚未完成。
- **本机路径**：`F:\Codes\Low-Cost-Mocap`；当前 conda 环境 `lowcost_mocap`（Python 3.10.22、NumPy 1.26.4、OpenCV 4.10.0、SciPy 1.13.1）。
- **2026-11-16 工作区状态**：HEAD 仍为 `cd7e991`；本轮“界面优化 + 采集流程合并 + 路径/单位修复”改动共 9 个文件，**未提交、未推送**。自检 **102/102**，使用合成图/假相机，**不是实机标定精度**。

### 已推送的重要提交

| 提交 | 内容 | 状态 |
| --- | --- | --- |
| `40e3b48` | 初始导入 | 已推送 |
| `63ff914` | 双相机并发采集、网页实时预览 | 已推送 |
| `96b6c7e` | 软件时间戳配对、同步帧保存 | 已推送 |
| `87fd19c` | 内参、固定棋盘格及 `solvePnP` 外参流程 | 已推送 |
| **`cd7e991`** | 标定流程修复与精简、自检 **77/77** | **最新已推送** |
| 工作区（未提交） | 标定界面/采集合并、三维视图、相对路径、间距单位修复；自检 **102/102** | **未提交** |

> 早期记录中的“未提交”状态已由后续提交覆盖；不能将任何合成测试指标称作真实相机的标定精度。

## 2. 原项目技术边界与机器人适配

原项目链路：4 台 PS3 Eye（320×240、90 FPS）采集红外亮点 → 极线匹配及 DLT 三角化 → 优化/滤波得到目标位置、速度和航向 → 串口 → ESP32 Sender（ESP-NOW）→ ESP32 Receiver（嵌套 PID）→ SBUS → Betaflight 飞控。

- 原项目核心是**外部视觉定位**，不是 SLAM，也不提供地图、障碍物检测、避障、路径搜索或动态重规划。
- `/api/trajectory-planning` 使用 **Ruckig** 将人工指定的航点时间参数化；这属于**轨迹生成**，不是自主路径规划。
- `pseyepy` 仅适用于 PS3 Eye；原项目没有硬件或软件级多相机同步：`Camera.read()` 批量取帧，但返回的时间戳被丢弃，各帧被假定同刻。
- 若部署到机器狗 RK3588，较合理的总体方向是固定相机端计算定位，以 Wi-Fi/ROS 2/UDP 传给 RK3588；**无需沿用两块 ESP32、ESP-NOW、SBUS 或无人机 PID**。这是适配建议，**尚非已实现接口**。ESP-NOW 不能由普通 RK3588 Wi-Fi 模组直接收发。
- 原项目用**主动发光 IR LED 标记**时，不必给相机加 IR 环形泛光灯；泛光灯主要供被动反光标记使用。LED 波长须与相机滤光片透过范围匹配，原项目未指明波长。
- 原项目 `cv2.sfm` 实际只用 `essentialFromFundamental`、`motionFromEssential`、`fundamentalFromProjections` 三处功能，可由常规 OpenCV/几何公式替换，避免编译完整 `opencv_contrib`；**替换建议不等于现有主链路已完成适配**。

**参考文档/配置**：`docs/low-cost-mocap-summary.md`（原仓库分析精简版）、`docs/astra-pro-sm-adaptation.md`、`environment.yml`（PS3 Eye 复现）、`environment-astra.yml`（Astra 方案）。原作者 demo 是外部 YouTube 视频，仓库没有示例数据集或可下载 demo 包。

### 相机方案变更

- **11 月 10 日探索过 Astra Pro SM**：推测需关结构光投射、调整 16-bit/分辨率/SDK 接入，OpenNI2/设备兼容性尚待确认；不是当前双相机管线的实测硬件。
- **11 月 12 日起实际使用两台 LRCP G720P**，后续采集、同步、标定实现均以此为准。不要混用 Astra 的规格和适配结论。

## 3. 当前相机：LRCP G720P

- USB 标识：`VID_BC15&PID_2E1B&MI_00`；标准 **UVC** 相机，不是专有 SDK 设备。
- 传感器：**OV9281 单色 CMOS**；可输出 NV12、MJPG、YUY2 等，最高配置中包括 1280×720@30 FPS；所枚举 47 种格式**不含 Y16**。当前得到的是 **8-bit 亮度图，不可测温**。
- 产品资料列出 400–650 nm 可见光版及多种定制窄带版，但**没有本机出厂滤光片配置单**。手机纯白屏实测中心亮度均值 251.1，证明可见光能通过，因此不是纯 850/940 nm 窄带相机；**是否透过 850/940 nm 近红外仍未确认**，需要 IR 遥控器/已知波长 LED 实测。不要直接认定它能检测近红外标记。
- 采集使用 Windows **Media Foundation + ctypes/COM**；DirectShow 设备 moniker 的 `BindToObject` 在本机异常。相机同一时刻只能被一个进程占用，重复打开可报 `0xC00D3704`。
- `ir_camera.py` 是独立取帧/显示工具，已精简为 **OpenCV 硬依赖**。命令示例：`python ir_camera.py --formats`、`python ir_camera.py`、`python ir_camera.py --no-display --frames 120`。

**Media Foundation 关键踩坑**：`QueryInterface` 不额外带 `pUnkOuter`；`IMFSourceReader::ReadSample` 的输出参数顺序是 `*index, *flags, *timestamp, **sample`；摄像头打开/读取/关闭应在同一 COM 工作线程中。关闭窗口时已修复 `_stop` 属性覆盖 `stop()` 和销毁已关闭 OpenCV 窗口的问题。

## 4. 双相机采集与实时预览（已实现）

| 文件 | 职责 |
| --- | --- |
| `computer_code/api/camera_manager.py` | 两个 `CameraWorker` 独立线程采集、latest/历史缓存、JPEG 缓存、重连、原始灰度保存 |
| `computer_code/api/camera_preview.py` | `/api/cameras`、启动/停止/保存、MJPEG、snapshot、同步及标定相关蓝图接入 |
| `computer_code/api/preview_server.py` | **3002 端口**的独立预览入口，不依赖无人机串口/Ruckig |
| `computer_code/api/static/preview.html` | 原生 HTML 网页，含双相机预览、同步状态、标定操作 |
| `computer_code/src/components/CameraPreview.tsx` | React 预览组件；`App.tsx` 和 Vite 代理已接入 |

- 本机两路 NV12 **1280×720@30 FPS** 同时稳定采集（约 30.0–30.2 FPS/路）；MJPEG 单路实测约 26.4 FPS。
- 采集独立于浏览器客户端：多客户端不会重复打开设备；最新帧延迟观察值约 **7–35 ms**，未累积。
- 预览页已修复“后端自动启动相机但 `<img>` 没有设置流地址”的问题：根据后端状态自动挂 MJPEG，首帧超时/解码失败时降级为 snapshot 轮询；静态页更新后需硬刷新。
- `index.py` 对本机缺失的串口、Ruckig、pseyepy 做了保护；关闭 Flask reloader 避免重复占用相机。
- **验证边界**：独立预览后端和 API 有实测；React/Vite 端曾因没有 `node_modules`、npm 不可达而**未完成浏览器实测**。

## 5. 双相机软件帧同步（已实现）

- `computer_code/api/sync.py` 的 `SyncService` 每约 **10 ms** 匹配两路历史帧，按 `time.perf_counter()` 主机时间最近邻配对，默认阈值 **20 ms**。
- **不能直接对比两台设备的 Media Foundation 时间戳**（不同设备时基不可比）。配对使用每相机严格单调的已消费序号，保证帧不重复、顺序单调；保存时包含两张原始 PNG、元数据 JSON 和 manifest。
- 历史帧入缓存前使用连续数组，防止 NV12 跨步视图无意持有大块底层 buffer。相机停止重启后序号归零可被检测并自恢复。
- 60 秒**实机**测试：约 **30.01 对/秒**、在 20 ms 阈值下配对率 **100%**、丢弃 0；配对时间差均值约 0.2–1.6 ms，P95 周期性上升至约 16 ms，最大 16.96 ms。阈值 10 ms 时配对率降至 71–82%。
- **性质**：这是**软件采集时间戳配对**，绝不是硬件同步或曝光时刻同步；两机独立时钟存在最高约半帧（16.7 ms）的相位漂移。适用于**静止棋盘格**标定；运动目标的几何误差需另行评估。

## 6. 标定系统（当前主路线）

### 6.1 物理配置与坐标约定

- 棋盘格为 **12×9 方格 = 11×8 内角点**；单格 **30 mm**；图案外尺寸 **360×270 mm**；采集原图 **1280×720**。
- **固定棋盘格定义世界系**：原点为图案中心，世界 X 沿 12 格长边，Y 沿 9 格短边，Z 朝上，右手系。表面高度由 `surface_height_m` 描述。
- 全流程仅使用相机**原始灰度图坐标**：不旋转、不裁成正方形、不缩放、不先去畸变；角点坐标和保存的 PNG 像素坐标一致。两台相机的内参、畸变参数**分别求解**。
- 固定板外参用 **`solvePnPGeneric(IPPE)` + `solvePnPRefineLM`** 求每台相机相对于同一世界系的位姿，再求两机相对变换；`X_cam1 = R_10 X_cam0 + T_10`。平移与基线单位为米。

### 6.2 代码和接口

| 位置 | 主要功能 |
| --- | --- |
| `api/calibration/board.py` | 棋盘格、SB 检测、角点质量、覆盖率、世界点及坐标轴叠加 |
| `api/calibration/dataset.py` | 会话、采集阶段、图片和元数据管理 |
| `api/calibration/intrinsic.py` | `calibrateCamera`、逐帧残差、覆盖率和异常提示 |
| `api/calibration/extrinsic.py` | IPPE/LM、单相机到世界系、相对位姿、自洽与验证 |
| `api/calibration/stereo.py` | 传统 `stereoCalibrate`；作为**独立离线能力**保留，非当前固定板外参主路线 |
| `api/calibration/service.py`、`routes.py` | 从已有相机工作线程和同步帧缓存读取、编排求解、结果加载/失效、HTTP 接口 |
| `api/calibration/selftest.py` | 合成数据、HTTP/CLI 回归自检 |
| `api/calib_cli.py` | `new/list/capture/detect/intrinsic/extrinsic/validate-rig/save/info` 等命令 |
| `docs/calibration.md`（仓库根） | 标定操作、坐标定义、遗留链路适配说明 |

**重要限制与设计**：

1. **角点极性**：`findChessboardCornersSB` 的正相/反相检测可能把整个角点序号旋转 180°。双相机同一帧对必须采用一致的检测极性；不能一台正相、一台反相后直接对应。
2. **棋盘对称歧义**：普通矩形棋盘格的 180° 或镜像对应不能仅凭各相机 `solvePnP` 重投影残差判定，错误世界方向可能仍有很低的三角化自洽误差。**最新实现支持每台相机独立 180° 开关，并要求人工核对世界坐标轴与 `orientation_confirmed`**。这是 11 月 16 日的新决策，**取代此前“只能全局 flip180”的旧设计**。
3. **IPPE 双解**：先筛相机处于棋盘正面（`Z>0`）、所有角点在相机前方的物理合理候选，再比较重投影误差；并列标为 `ambiguous`。不能只看 RMS，也不能只看高度。
4. **采集不重开相机**：从运行中的 `CameraManager`/`SyncService` 读数据；要求每次采集使用新序号，最多等待 **1.5 s**。validation 同步帧对按 `(seq_a, seq_b)` 去重；外参“采集双相机图片”写入两台 single 样本，按 `pair_seq` 去重。
5. **结果不会混用**：重载会话可恢复已保存的 intrinsic/extrinsic/validation/stereo 结果；新采集按阶段使依赖结果失效。CLI 默认经 HTTP 调现有服务获取相机数据，`--local` 才使用本进程；服务未运行时 `new` 可本地建会话。
6. **命名与单位**：原误称 `stereo_rms_px` 的指标已改名 `max_reprojection_rms_px`，另有 `consistency_max_mm` 表示三角化自洽最大误差（毫米）。这些指标不能互换。
7. **冗余删除**：已删除旧 `service.validate()`、`/api/calib/validate`、CLI `validate`、网页“独立验证（留出集）”按钮，以及重复 `cmd_save` / 旧 `cmd_validate`。固定板流程使用 `validate_rig`；`stereo.evaluate` 仍保留供离线方法和自检使用。

### 6.2.1 2026-11-16 工作区改版（未提交）

- **内参展示**：点“内参标定”后逐台显示 K、D、RMS、有效张数和**区域覆盖率**。覆盖率复用 `board.coverage_map` 的 6×6 网格，只统计该相机 `phase="intrinsic"` 的有效样本，不混入外参/验证图；不再输出冗长覆盖矩阵。
- **外参采集**：删除“采外参图 相机0/1”，改为**“采集双相机图片”**：一次取一对同步帧，分别检测、分别判断质量，写入两台相机的 `extrinsic` 阶段。接口 `POST /api/calib/capture/extrinsic-pair`；同一 `pair_seq` 拒绝重复采集。`extrinsic.MIN_FRAMES = 1`，每台 1 张有效图即可 solvePnP；单帧时 `stability.*.centre_std_m = null` 并说明原因。
- **覆盖率门限**：内参 `MIN_COVERAGE = 0.8%` 不变；固定板外参/验证据现场要求放宽为 **`MIN_COVERAGE_EXTRINSIC = 0.1%`**。0.1%–0.8% 的图会入库，但样本 `note`、采集响应、`warnings` 和按钮旁会显示“覆盖率偏低，位姿/深度精度会下降”。
- **三维验证**：删除“采集同步帧对”和“独立验证（多位置）”，改为**“采集并验证三维精度”**（`POST /api/calib/validate/capture`：取同步帧 → validation 阶段 → 复用 `validate_rig`）和**“重新计算”**（`POST /api/calib/validate/rig`，不重采）。界面显示最新一组、历史列表和汇总（重投影、网格间距、平面度、极线、有效位置数）。
- **可视化模式**：删除“实时检测叠加/世界轴叠加”，改为三种模式：无可视化 / **棋盘格角点可视化（默认）** / 世界坐标轴可视化（X 红、Y 绿、Z 蓝）。保留两台相机独立 180° 开关；改 180° 或重采外参会取消“已确认世界坐标轴方向”。
- **三维视图**：`preview.html` 的 2D 俯视图改为 **Canvas 2D 透视三维场景**（环境无本地 Three.js/node_modules，要求离线可用，故不引 CDN）。用 `R_cam_to_world + focal_px` 画真实相机朝向/视锥体；含地面网格、12×9 棋盘格、世界轴、基线、相机编号/位置/离地高度/重投影标签；左键旋转、滚轮缩放、右键平移、重置视角、俯视图。相机位姿来自已解外参，不重算。
- **日志与缓存**：标定日志改为逐行 DOM 节点（`white-space: pre-wrap`），保留时间戳、长行换行、无空行；`/preview` 增加 `Cache-Control: no-store`，页面版本标记 `v6 · 三维视图 + 合并采集`。
- **路径**：数据基准为 `computer_code/`；默认 `captures/calibration/<session_id>/` 与 `api/calibration/calibration.json`。界面/结果写项目相对路径，内部读写解析为绝对路径，不依赖启动时 CWD；旧绝对路径仍可读，历史目录不覆盖。默认板面高改为 **5 mm**（旧会话保持原值）。
- **单位修复**：`validate_rig` 汇总 `spacing_abs_error_mm` 的 mean/p95/max 原为米却标 mm；已乘 1000 并加 `unit: "mm"`。旧结果读回时若无 `unit` 标记，按米→毫米迁移一次（幂等）；逐位置字段本来就是 mm。
- **网页双目标定入口**：删除网页“外参标定（双目标定）”按钮和状态标签；`stereo.py`、CLI `stereo`、`/api/calib/stereo` 保留为离线交叉校验。
- **未提交文件（9 个）**：`api/calib_cli.py`、`api/calibration/{service,routes,extrinsic,dataset,selftest}.py`、`api/camera_preview.py`、`api/static/preview.html`、`docs/calibration.md`。`api/calibration/calibration.json` 与根目录 `memory.md` 未跟踪。

### 6.3 真实局限与结果

- **内参采集必须覆盖图像边角**：仅中央棋盘数据会让高阶畸变系数互相补偿，虽然 RMS 很低，但画面边缘外推可错约 **20 px**；补采四角后对应测试下降至约 **1.6 px**。已有 `radius_coverage.ratio < 0.75` 告警。
- 早期**合成数据**测试：内参 fx 误差约 0.13%、RMS 约 0.07 px；固定板 PnP 的合成案例基线约 2032 mm、三角化自洽 RMS 约 0.227 mm。上述均**不是实机标定结果**。
- `cd7e991` 当时：**77/77**。2026-11-16 工作区改版后：**102/102**（quick/full 均通过），新增内参覆盖率按相机/阶段隔离、同帧外参采集与单张求解、低覆盖图（0.15%）接受并告警、自动验证、间距单位迁移、默认角点可视化等检查；仍使用合成图/假相机，**未实机采集标定数据**。
- 实机会话 `captures/calibration/20261116_152637`：两台各 8 张内参图，已解 K/D（fx≈633/654，RMS 0.56/0.14 px）；外参尚未成功。`s0017`–`s0022` 是旧 0.8% 门限下的 invalid 外参样本（覆盖率 0.39–0.48%，原图已保存）；服务重启后 0.1% 门限生效，需重新点“采集双相机图片”。
- 未跟踪的 `api/calibration/calibration.json` 来自 10:58 的 `20261116_104419` 非正式测试：基线 **43.36 mm**、`orientation_confirmed=true`、内参 RMS 0.2259/0.077 px、10 个验证位置；旧汇总间距 0.0038/0.0112/0.2028 m 在加载时已迁移为 3.8/11.2/202.8 mm。该次相机/板未固定，**不能据此认定算法缺陷或标定精度**；文件未提交。
- 旧 `camera-params.json` 的 320 像素参数未被改写；当前仍**没有经过固定安装、人工方向确认和独立验证闭环的可信外参/基线结果**。
- 传统 `helpers.py` 的定位链路仍按 PS3 Eye 的 `rot90`、`make_square`、旧内参与 `cv2.sfm` 等假设编写，**未接入当前标定输出**；需以后单独适配。

## 7. 尚未完成、不能声称已完成

1. **实机标定闭环**：分别采集两台相机覆盖全画面的静止棋盘格，求真实 K/D；固定板求外参，人工逐台确认坐标轴方向，再核对实际基线、误差和重复性。
2. **UI 联调**：已用无头 Chrome 对 3002 页面验证加载、按钮绑定、真实鼠标点击建会话、无 JS 异常；“新建/打开会话无反应”多为**旧标签页缓存**（`/preview` 已加 `no-store`，但仍需重新导航）。尚未在真实浏览器人工完成整条标定流程；三维视图旋转/缩放/平移和视锥体尚未人工验收；React/Vite 页面未实测。
3. **红外兼容性**：LRCP G720P 的本机 IR-cut/850–940 nm 透过性尚未通过已知波长光源确认；定位标记的发光波长、曝光参数亦需实测。
4. **多相机/动态定位**：当前仅两机；四相机、运动目标时的非硬同步误差、三点标记识别、实时三角化以及向 RK3588/ROS 2 输出定位**尚未完成**。
5. **旧系统整合**：标定输出尚未替换原始 `helpers.py` 中的相机、内参及三角化流程，不要把预览/标定进展等同于“机器狗动捕定位已跑通”。

## 8. 维护与排障速查

- 设备被占用（`0xC00D3704`）：先确认是否有另一个预览服务或采集进程；不要开两个相机管理器。
- 浏览器黑屏但后端 30 FPS：检查 `/api/camera/<i>/stream` 是否真的被请求、预览页是否旧缓存；必要时硬刷新或使用 snapshot 降级。
- 点“新建/打开会话”无反应：先确认右上角版本是 `v6 · 三维视图 + 合并采集`、按钮旁有“就绪”徽标；没有就是旧标签页，关闭重开（`/preview` 已 `no-store`）。有徽标但无反应时看 F12 Console 和 `POST /api/calib/session`；改过 Python 后必须重启 preview_server。
- 外参图“没计入有效样本”：看“采集双相机图片”按钮旁的 `c-exmsg` 徽标。旧 0.8% 门限会把地面小板（0.4–0.5%）判为太小；2026-11-16 已把外参/验证下限调到 0.1%，但**运行中的服务需重启才生效**。低于 0.8% 仍带“覆盖率偏低”告警。
- 同步帧的 `mf_timestamp` 差不代表跨设备真实同步差；以主机 `perf_counter` 的软件配对指标为准。
- 双目外参方向看起来反转：不要通过降低 RMS 直接“自动修正”；逐台看坐标轴叠加与相机位置，人工确认各自 180° 方向。
- 原 Windows Git 曾因证书撤销检查报 `CRYPT_E_NO_REVOCATION_CHECK`；本仓库局部配置过 `http.schannelCheckRevoke=false`，不是建议全局关闭 TLS 校验。
- 未追踪大文件/临时件包括原项目 `upstream/`、DirectShow SDK、相机截图、日志、`__pycache__` 等；不要因为本地存在就误认为已推送。

---

**记忆使用原则**：后续讨论以 **`cd7e991` + 2026-11-16 工作区未提交改版 + 两台 LRCP G720P + 固定棋盘格 `solvePnP` 世界系** 为当前基线。优先参考最新代码与实测；旧方案及合成测试数据只作背景，发生冲突时以较新记录为准。工作区改版尚未提交，引用时请以实际文件为准。
