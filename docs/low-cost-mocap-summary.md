# Low-Cost-Mocap 仓库中文总结

> 分析对象：[github.com/jyjblrd/Low-Cost-Mocap](https://github.com/jyjblrd/Low-Cost-Mocap)，`main` 分支（commit `8ac4d41`），查阅时间 2026-11-10。
> 依据 README、源码、配置文件、架构图、Issue 模板与仓库元数据。仅整理仓库内可查证信息；外部来源单独标注，缺失或不确定项统一写明；不补充外部背景、不扩写为操作教程、不引用代码片段。

## 1. 项目目标

- README 标题为 **Low Cost Mocap (for drones)**，自述为「从零构建的通用动作捕捉系统，用于室内自主飞行多架无人机」；GitHub 描述为「房间尺度的低成本运动捕捉」。
- 系统范式是 **outside-in**：由外部相机跟踪固定空间内的目标。README 明确把 inside-out / 纯视觉 SLAM 划给作者的另一项目 `distributed_visual_SLAM`，**不属于本仓库**。
- topics（`motion-capture`、`bundle-adjustment`、`epipolar-geometry`、`quadcopter`、`esp32` 等）与「多相机视觉定位 + 无人机控制」定位一致。

## 2. 核心功能

- **多相机定位**：多相机取帧 → 亮点检测 → 极线匹配 → DLT 三角化 → 世界坐标变换 → 卡尔曼 / 低通滤波，输出每个目标的 `pos` / `vel` / `heading`。
- **相机自标定**：由标定点求基础矩阵 / 本质矩阵与相机间相对位姿，光束法平差优化参数并计算重投影误差；另含尺度、地平面、原点标定。
- **网页控制台**：相机流、FPS、三维场景与图表，以及标定、定位、轨迹、Arm / PID / Trim 等入口（`three.js` / `chart.js`，前后端走 Socket.IO）。
- **轨迹生成（不是路径规划）**：Ruckig 把前端**人工填写**的路点在速度 / 加速度 / 加加速度约束下做时间参数化插值，输出设定点序列；仓库不含环境地图、障碍物、避障或路径搜索逻辑。
- **无线闭环**：定位结果与设定点经串口 → ESP-NOW → 机载嵌套 PID → SBUS 输出给飞控。

## 3. 整体架构与工作流程

架构图 `images/architecture.png` 已取得并解析，图中标注：`4x Cameras` 经 `USB` 接入 `Computer`；`React frontend`（three.js / chart.js）与 `Python Backend`（OpenCV / SciPy Optimizer / Numpy）以 `Socket.IO` 通信；后端经 `Serial Port` 连 `ESP32 Sender`；机载 `ESP32 Receiver`（Nested PID Controllers）经 `SBUS Serial Protocol` 连 `Betaflight` 飞控；两侧走 `ESP-NOW Wireless Protocol`。

### 3.1 分层结构

| 层 | 位置 | 职责 |
| --- | --- | --- |
| 前端 | `computer_code/src` | Vite + React + TypeScript 控制台、3D/图表可视化、Socket.IO 交互 |
| 后端 | `computer_code/api` | Flask-SocketIO（端口 3001）：采集、视觉计算、滤波、串口输出、REST 轨迹接口 |
| 通信转发 | `sender_esp32` | 从串口读「目标编号 + JSON」，按编号用 ESP-NOW 转发 |
| 机载控制 | `receiver_esp32` | 解析 JSON，嵌套 PID 运算，SBUS 输出到飞控 |

### 3.2 主要工作流程（从源码可见）

1. `pseyepy` 以 90 FPS 批量读取多路帧（各相机自由运行，`read()` 返回的时间戳被丢弃，**无硬件快门同步，也无基于时间戳的软件对齐**），去畸变与亮点增强，横向拼接为 `/api/camera-stream` 的 MJPEG 流；定位计算挂在该 150 Hz 视频流循环内。
2. 灰度阈值二值化 → 轮廓质心，得到各相机的二维亮点。
3. 前端触发标定点采集，后端求基础 / 本质矩阵、在候选解中择优相机位姿、光束法平差优化，回传 `camera-pose`。
4. 用 0.15 m 已知距离定尺度、最小二乘拟合地平面、设置原点，得到「相机坐标 → 世界坐标」变换。
5. 实时定位：极线匹配跨相机点 → DLT / SVD 三角化 → 按重投影误差筛选 → 世界坐标变换。
6. 按三点固定间距（0.095 / 0.15 m）识别目标与朝向、区分编号 0/1，再经卡尔曼 + 低通滤波输出平滑状态。
7. 状态与设定点写入串口，经 ESP-NOW 到机载 ESP32，位置外环 + 速度内环 PID 以 50 Hz 输出 SBUS（控制循环标称 2000 Hz）。
8. 后端通过 `object-points`、`image-points`、`fps`、`camera-pose` 等 Socket.IO 事件刷新前端界面。

## 4. 主要目录与关键模块

| 路径 | 内容 |
| --- | --- |
| `README.md` / `LICENSE` | 仓库内唯一说明性文档；MIT 许可证 |
| `.github/ISSUE_TEMPLATE/bug-report.md` | Bug 反馈模板，要求复现步骤、预期行为、截图与 OS / 浏览器 / 版本信息 |
| `computer_code/api/index.py` | 后端主入口：Flask 应用、Socket.IO 事件、MJPEG 路由、Ruckig REST 接口 |
| `computer_code/api/helpers.py` | 视觉算法核心：`Cameras` 单例、亮点检测、极线匹配、三角化、光束法平差、目标识别 |
| `computer_code/api/KalmanFilter.py` / `LowPassFilter.py` / `Singleton.py` | 状态滤波、低通滤波、单例装饰器 |
| `computer_code/api/camera-params.json` / `camera-params copy.json` | 相机内参 / 畸变 / 旋转配置（4 组 / 3 组）；后者用途未说明 |
| `computer_code/api/test.py` | 最小 Flask-SocketIO 服务，**并非单元测试**；用途未说明 |
| `computer_code/src/App.tsx` 及 `src/components/` | 前端主控制台（集中绝大多数 UI 与交互逻辑）与展示组件 |
| `computer_code/src/shared/styles/scripts/socket.ts` | 建立到 `http://localhost:3001` 的 Socket.IO 连接 |
| `computer_code/package.json` / `vite.config.ts` / `tsconfig*.json` 等 | Node 依赖锁定与 Vite / TS / ESLint 配置；Vite 仅启用 React 插件，未配代理 |
| `sender_esp32/` / `receiver_esp32/` | 地面串口→ESP-NOW 转发；机载 ESP-NOW 接收、嵌套 PID、SBUS 输出 |
| `3d_files/` / `images/` | 相机外壳与支架 CAD / STL 模型；架构图与视频封面 |

> 仓库**没有 `docs/` 目录**，也不含数据集、模型权重或标定数据样本。

## 5. 依赖与运行环境

### 5.1 README 明确列出

- Python 库 `pseyepy`（**PS3 Eye 专用**驱动，基于 libusb / PS3EYEDriver）；OpenCV 的 **sfm 模块**需从源码编译 OpenCV（README 给出 SFM 依赖与 `opencv_contrib` 安装指南，作者称较麻烦）；`npm` 与 `yarn`。

### 5.2 从源码可见（版本大多未锁定）

- Python：`flask`、`flask_socketio`、`flask_cors`、`cv2`（含 `cv2.sfm`）、`numpy`、`scipy`、`pyserial`、`ruckig`、`pseyepy`。**未提供 `requirements.txt` / `pyproject.toml`**，版本与安装顺序均未说明。
- 前端：`vite` 4 / `react` 18 / `three` / `socket.io-client` 4 / `chart.js` / `plotly.js` / `mathjs` / `react-bootstrap` 等；脚本 `dev`、`build`、`lint`、`preview`，`version` 为 `0.0.0`。
- 固件：ESP32 Wi-Fi / ESP-NOW、`ArduinoJson`、`PID`、`EEPROM`、`sbus.h`；**未提供 Arduino / PlatformIO 库清单或板卡配置**。

### 5.3 环境线索与未说明项

- 后端串口硬编码 `/dev/cu.usbserial-02X2K2GE`（1,000,000 波特，macOS 风格路径）；前端 Socket.IO 与 REST 均指向 `localhost:3001`；后端以 `debug=True`、CORS 允许任意来源运行。
- 操作系统要求、Python / Node 版本、相机驱动的系统依赖均**未在仓库中说明**。

## 6. 仓库已有的构建与运行入口说明

- README 的「Runing the code」一节仅给出：在 `computer_code` 下 `yarn install` → `yarn run dev`（README 称会给出查看前端的 URL，**具体端口未写明**）→ 另一终端运行 `python3 api/index.py`。
- 后端 HTTP 路由：`/api/camera-stream`（MJPEG 流）、`POST /api/trajectory-planning`（轨迹设定点）。
- `package.json` 另有 `build`、`lint`、`preview`；`api/test.py` 可作为独立的最小 Socket.IO 服务启动（用途未说明）。
- **未提供** Dockerfile、CI 工作流、一键启动脚本、服务编排或跨平台安装说明；`.github` 下仅有 Issue 模板。

## 7. 数据、硬件与外部依赖

### 7.1 数据与配置

- `camera-params.json`：4 组相机参数（焦距 320、主点 160/160、5 个畸变系数、`rotation` 均为 0）；`camera-params copy.json`：3 组，内参与 `rotation` 各异，**用途未说明**。
- README 写 `camera_params.json`（下划线），与实际文件名不一致，是否同一文件未明确；计算方法指向 Discussion #11。
- 标定与定位依赖三点标记固定间距（0.095 / 0.15 m）；仓库**未提供**标记点物理规格、数量要求或布置说明。

### 7.2 硬件

- **4 台 USB 外部相机**（架构图标 `4x Cameras`，与 4 组参数一致）+ 一台电脑 + 每个目标 3 个主动发光标记点。采集层 `from pseyepy import Camera`（`fps=90`、`RES_SMALL` 320×240、`gain=10`、`exposure=100`）决定其**仅支持 PS3 Eye**；换相机须重写采集层并重新标定内参。亮点检测为固定阈值二值化 + 轮廓质心、不做形状判别，因此还需去除 IR cut、加装可见光截止滤光片（作者博客）；目标为主动发光 IR LED，**相机侧 IR 补光灯非必需**（Discussion #36：补光灯只用于被动逆反射球）。相机型号、是否必须 4 台及安装要求**未在 README 或配置中说明**。
- 地面 `sender_esp32` 仅做串口→ESP-NOW 转发，按两个预置 MAC 发送；机载 `receiver_esp32` **并非纯透传**——解析 `pos`/`vel`/`armed`/`setpoint`/`pid`/`trim` 六类消息，运行 7 个 PID 的嵌套环、地效补偿等飞行器常量及超时自动解锁保护。
- 飞控经 `SBUS` 接收控制量，架构图标注为 **Betaflight**（源码中未直接出现该名称）。ESP-NOW 是乐鑫定义的 Wi-Fi 数据链路层私有协议，仅乐鑫 SoC 支持（乐鑫官方文档）；**仓库未提供非 ESP32 机载平台的接入接口或替代无线链路**。
- `3d_files/` 提供相机外壳与球头 / 支架 / 螺丝模型，但**无装配说明或打印参数**。

### 7.3 外部资源与 demo（均为 README 引用，非仓库内文件）

- 作者博客 `joshuabird.com/blog/post/mocap-drones` 给出实际物料：PS3 Eye（需去除 IR cut 滤镜改装，配 USB 延长线；博客另列 IR 环形补光灯，按 Discussion #36 仅用于被动逆反射球）、F3 EVO Micro Brushed 飞控（旧版 Betaflight）+ 95 mm 有刷机架、3 mm 红外 LED + WEMOS D1 Mini ESP32 USB-C；称每架约 $50 USD、续航约 10 分钟。价格为发布时数值，博客所称「毫米级精度」**未在仓库 README 或源码中给出**。
- YouTube 演示视频 `youtu.be/0ql20JKrscQ`；Discussion #11（相机参数计算）、`pseyepy`、OpenCV SFM 与 `opencv_contrib` 安装指南、作者的 SLAM 论文项目。
- 仓库**不含**演示视频、示例数据集、可独立运行的 demo 工程或 Release 附件，仅存封面 `images/thumbnail.png`（画面文字「Indoor Drone Swarms」）与架构图。
- README 中的架构图、缩略图与 Discussion 链接指向旧仓库名 `Mocap-Drones`，与当前仓库名不一致，引用时需核对。

## 8. 适用场景

- 室内、房间尺度、固定空间内的 **outside-in** 多目标三维定位与跟踪。
- 多架无人机室内自主飞行、轨迹跟踪与控制实验（前端含圆形 / 方形 / 规划轨迹预设与 Arm / PID / Trim 控件）。
- 低成本自建动捕系统的研究 / 教学原型：源码涉及对极几何、三角化、光束法平差、卡尔曼滤波与 SfM 标定。
- **不适用**：无外部相机基础设施的场景、inside-out / 纯视觉 SLAM 需求。仓库未提供精度、延迟、跟踪范围、最大目标数等指标，适用边界**无法从仓库信息判断**。

## 9. 已知限制与信息缺失

### 9.1 README 明确说明

- 作者自述「文档相当缺乏」，希望补充 Python 类型定义；OpenCV SFM 需源码编译。
- 实验分支 `no-cv-sfm` 移除了该依赖，但被标注为 **completely untested（完全未经测试）**；本次分析仅覆盖 `main` 分支。

### 9.2 源码与仓库结构约束

- **强耦合双无人机场景**：`num_objects=2`、`NUM_DRONES=2`、`DRONE_INDEX`、第三点判 0/1 编号、两个预置 MAC，均硬编码。
- **强耦合特定硬件与标定物**：串口路径、4 组相机参数、三点间距 0.095 / 0.15 m、`localhost:3001` 均为硬编码；**无参数化配置或适配层**。
- **缺少环境与工程化声明**：无依赖清单、无版本锁定、无容器 / CI / 启动脚本；`App.tsx` 承担绝大部分前端逻辑，部分 UI 被注释，`index.py` 存在多个同名处理函数，`test.py` 不是测试。
- **部署姿态**：`debug=True` + CORS 全开、前端地址写死 localhost，无生产部署或访问控制说明。
- **维护状态**：仓库创建于 2023-07-29、最后代码推送 2024-05-25（本次查阅时）；版本号 `0.0.0`，无 Release / Tag。

### 9.3 未说明 / 未提供 / 不确定

- **未说明**：相机型号与最少数量、标记点规格与布置、操作系统与语言版本、Python 依赖版本、标定流程的完整前置条件。
- **未提供**：性能指标（精度、延迟、帧率稳定性、空间范围、目标数上限）、环境地图与障碍物表示、避障与全局路径搜索 / 重规划、自主导航栈、数据集、测试用例、装配说明、生产部署说明、机器狗 / 四足平台相关信息。
- **未查阅**：README 链接的 YouTube 演示视频在当前环境不可访问，仓库内亦无视频文件，故视频内容不在本文结论范围内。
- **不确定**：`camera-params copy.json` 的用途、README 与实际配置文件的命名差异、`test.py` 的意图、`no-cv-sfm` 分支的可用性。
- **关于机器狗 Hypertron-T01**：仓库中**不存在任何相关代码、配置或文档**。上述硬编码约束仅为源码事实，不构成适配判断；相关性能、接口与改造范围在仓库信息下**均不确定**。
