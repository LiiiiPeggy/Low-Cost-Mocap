# PROCESS — 工程决策与证据

> 记录影响架构或指标的重要问题、方案比较、实验依据与最终结论；不是日常操作日志。
> 原始逐日记录保存在本地 `.workbuddy/memory/2026-11-*.md`（已被 `.gitignore` 忽略）；此处只保留可复核的证据与结论。
> 当前状态见 `PROGRESS.md`；可复用结论见 `MEMORY.md`。

## 1. 项目边界与相机方案

- 原项目是 outside-in 外部视觉定位，不是 SLAM；Ruckig 只做轨迹时间参数化。迁移目标是把视觉定位思想用于 Hypertron-T01，而非复用无人机/ESP-NOW/SBUS 链路。
- 2026-11-10 曾评估 Astra Pro SM：推测需关闭结构光投射，并处理 16-bit、分辨率与 OpenNI2/SDK 兼容性；均未确认。
- 2026-11-12 起改用两台 **LRCP G720P**（UVC / OV9281）。全部采集、同步、标定结论以此硬件为准，Astra 的规格与适配推断不得混用。
- 原项目 `cv2.sfm` 实际只用 `essentialFromFundamental`、`motionFromEssential`、`fundamentalFromProjections` 三处，可用常规 OpenCV/几何公式替换，避免编译完整 `opencv_contrib`；这是替换建议，主链路尚未适配。
- 相关提交：`40e3b48`（初始导入与调研笔记）。

## 2. 双相机采集与网页预览（`63ff914`）

**问题**：浏览器多标签、刷新、断流会导致重复打开相机、帧延迟累积。
**方案**：一相机一线程（COM STA 要求 open/grab/close 同线程）+ 只保留 latest 帧 + 每帧只编码一次 JPEG + 重复 `start` 返回 409 + 断线重连。
**Media Foundation 踩坑（已修复）**：

- `QueryInterface` 不额外传 `pUnkOuter`；
- `IMFSourceReader::ReadSample` 输出参数顺序是 `*index, *flags, *timestamp, **sample`；
- 关闭窗口时 `_stop` 属性曾覆盖 `stop()`，且销毁了已关闭的 OpenCV 窗口；
- DirectShow 设备 moniker 的 `BindToObject` 在本机异常，故直接走 MF。

**实测**：两路 NV12 1280×720@30 各约 30 fps；最新帧滞后 20.7 ms 不增长；MJPEG 单路约 26–30 fps；设备独占，重复 open 报 `0xC00D3704`。
**预览页修复**：后端自动启动相机后，`<img>` 曾没有挂上 MJPEG 流地址；现按后端状态自动挂流，首帧超时/解码失败时降级为 snapshot 轮询。
**当前网页**：3002 `/preview`（`api/static/preview.html`，原生 JS + Canvas）。原 React/Vite 页面已删除，源码保留在 `upstream/`。

## 3. 软件帧同步（`96b6c7e`）

**备选方案**：

1. 直接比较两台设备的 `mf_timestamp`；
2. 用主机 `time.perf_counter()` 做流式最近邻配对；
3. 硬件触发 / 共享时钟。

**取舍**：方案 1 不可行——两台设备的 MF 时间轴独立，实测一对帧的 mf 差 0.135 ms，而主机时间差 15.95 ms；方案 3 硬件不具备。选择方案 2。

**实现要点**：

- 每约 10 ms 扫描两路历史缓存；两路各维护按 `seq` 单调推进的“已消费”指针；
- `|Δt| ≤ 20 ms` 则配对，否则丢弃更老一帧；不重复使用、配对时间单调；
- 一路停流 >0.5 s 时，另一路积压帧计为丢弃，避免配对率虚高；
- 保存同步对时写两张原始 PNG + 每对 JSON + `manifest.json`。

**60 s 实测**：30.01 对/s、100% 配对、0 丢弃；`|Δt|` 均值 0.2–1.6 ms，P95 周期升至约 16 ms，最大 16.9 ms；相位在 0–16.7 ms 往复。10 ms 阈值对照：配对率 71–82%，每 5 s 丢 25–30 帧，恢复到 20 ms 即回 100%。

**结论**：配对算法不丢帧，但**不解决曝光同步**；时间戳是交付用户态时刻，偏差随曝光/负载变化，不能当常数补偿。适用于静止标定板，不适用于快速运动目标。

## 4. 外参路线：固定板 solvePnP vs stereoCalibrate（`87fd19c`）

**需求**：相机固定、棋盘板可移动，需要统一世界系供后续 LED/机器人坐标使用。

**选择**：

- 板定义世界系，每台相机独立 `solvePnPGeneric(IPPE)` + `solvePnPRefineLM`；
- `stereoCalibrate` 只保留为离线交叉校验（`stereo.py`、CLI `stereo`、`/api/calib/stereo`），不作为网页主流程；
- 指标拆分为 `max_reprojection_rms_px` 与 `consistency_max_mm`，后者是三角化角点与已知世界坐标的最大偏差。

**D2 对称的发现**（见 §8）：仅凭重投影/自洽残差无法判断 180° 方向，因此流程要求人工核对世界轴并填 `orientation_confirmed`。

## 5. 标定流程修复与精简（`cd7e991`）

- 删除重复的 `cmd_save`、旧 `cmd_validate`、`service.validate()`、`/api/calib/validate` 和网页“独立验证（留出集）”；固定板流程统一用 `validate_rig`。
- CLI 采集默认改走 **HTTP**，复用运行中的相机；`--local` 才在当前进程采集，解决 CLI 与网页服务争抢相机的问题。
- 原误称 `stereo_rms_px` 的指标改名为 `max_reprojection_rms_px`，另设 `consistency_max_mm`；两者不允许互换。
- 会话重载可恢复 intrinsic/extrinsic/validation/stereo 结果；新采集按阶段使依赖结果失效。
- 自检达到 77/77（当时口径）。

## 6. 界面与采集整合（`c353edf`）

- **外参采集**：从“分别采两台相机”改为“一次取一对同步帧、分别检测/判质”，按 `pair_seq` 去重；`MIN_FRAMES = 1`，单帧时 `stability.*.centre_std_m = null`。
- **覆盖率门限**：内参 0.8%；外参/验证放宽到 **0.1%**——现场地面小板覆盖率只有 0.4–0.5%，会被旧门限判为无效。低覆盖仍告警“位姿/深度精度会下降”。
- **三维视图**：环境无 Three.js / `node_modules`，要求离线可用，故用 **Canvas 2D 透视投影**，按 `R_cam_to_world + focal_px` 画相机朝向/视锥体；不引 CDN。支持旋转、缩放、平移、重置和俯视图。
- **路径**：数据基准为 `computer_code/`，对外写项目相对路径，内部解析为绝对路径，不依赖启动 CWD；旧绝对路径仍可读。
- **单位 bug**：`validate_rig` 的 `spacing_abs_error_mm` 原为米却标 mm；已乘 1000 并加 `unit: "mm"`。旧结果读回时按米→毫米幂等迁移一次。
- **版本与缓存**：`/preview` 加 `no-store`；网页版本号后来改为从 `/api/version` 动态读取 Git SHA（旧的写死 `v6` 标签已废弃）。
- 自检从 77/77 提升到 102/102（当时口径）。

## 7. 自动保存、独立 180°、LED 改版（2026-11-16 起，2026-11-17 提交）

- **自动保存/载入**：内参/外参成功后分别写 `captures/calibration/intrinsics.json` 与 `world_extrinsics.json`；载入时校验相机号、K/D、图像尺寸、棋盘/世界系和方向确认状态。完整合并结果只按需写当前会话 `calibration.json`；旧全局兼容文件及其读写逻辑已移除。
- **每台独立 180°**：取代旧“全局 flip180”。理由见 §8：对称歧义可能只出现在单台相机；修改 flip 会作废旧外参与方向确认，避免状态轮询把“已确认”自动勾回。
- **默认板面高 3 mm**：新会话默认值；旧会话保持原值。
- **LED 模块**：阈值 + 连通域 → 距离变换峰值 + 最近峰值分配拆粘连 → 强度加权亚像素；过曝不可分不伪造。`F` 由标定 R/t 计算，`P=K[R|t]` 三角化到世界系，检查相机前方、重投影、34/79/79 mm 等腰三边；L0 为两长边公共顶点；卡尔曼按 `created_mono` 传播。
- **LED 波长定为 850 nm**：不再评估 940 nm 方案；装机时用实际 LED 确认亮度、曝光与检测阈值。
- **验证**：`calibration.selftest` 141/141、`led.selftest` 60/60（合成图/假相机，2026-11-17 复跑）。

## 8. 关键证据

### D2 四重对称

- 矩形角点阵列的 4 种对应在几何上完全等价：重投影误差逐位相同，三角化自洽误差都约 0.6 mm。
- 单侧 180° 反转的合成案例：基线由 **2032 mm** 变为 **1635 mm**（约 20%），而残差仍约 0.6 mm，任何自动指标都不会报警。
- 因此：必须人工查看世界轴叠加并显式确认；若要全自动，需要在板上加 ArUco / 缺角等非对称标记（未做）。

### 内参角点覆盖

- 仅用画面中央的棋盘数据：RMS 可以很低，但画面边缘外推误差可达约 **20 px**（高阶畸变系数互相补偿）。
- 补采四角后，同类测试边缘误差降到约 **1.6 px**。
- 已加入 `radius_coverage.ratio < 0.75` 告警。

### 实机会话（均不构成可信闭环）

- `20261116_152637`：两台各 8 张内参图，fx≈633/654，RMS 0.56/0.14 px；外参未成功。
- 内参迁移：`intrinsics.json` 的 RMS 0.2259/0.077 px 来自会话 `20261116_104419`，标记 `source=migrated`。该文件与世界系外参 `world_extrinsics.json` 已于 2026-11-17 纳入 Git；原始采集图仍忽略。
- `20261116_174926`：完成“已确认”外参求解，基线 **560.567 mm**，每台仅 1 帧且覆盖率偏低。
- 旧全局 `api/calibration/calibration.json` 于 2026-11-17 删除；其自动写入逻辑和 `/api/calib/result`、`/api/calib/matrices` 的默认回退读取也已移除，不再重新生成。完整结果只存在于会话目录 `calibration.json`。
- 旧 `camera-params.json`、`helpers.py`、3001 后端、React 网页和 ESP32 固件已从运行代码删除，原作者实现保留在 `upstream/`；当前标定输出不进入旧 PS3 Eye 图像流水线。

## 9. 相关提交与文档

| 提交 | 内容 |
| --- | --- |
| `40e3b48` | 初始导入：红外取帧工具、原项目代码、调研笔记 |
| `63ff914` | 双相机并发采集 + 网页实时预览 |
| `96b6c7e` | 时间戳软件帧同步 + 同步帧对保存 |
| `87fd19c` | 内参 + 固定棋盘格 + `solvePnP` 世界系外参 |
| `cd7e991` | 标定流程修复：去重、CLI 走 HTTP、状态恢复 |
| `c353edf` | 界面/采集整合：同帧采集、三维视图、相对路径、单位修复 |

详细操作、接口与局限见 `docs/calibration.md`；原项目分析见 `docs/low-cost-mocap-summary.md` 与 `docs/astra-pro-sm-adaptation.md`。
