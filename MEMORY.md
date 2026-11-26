# MEMORY — 长期知识

> 只保留**已验证、跨会话仍然有效**的事实、约束和设计原则。当前状态与待办见 `PROGRESS.md`；决策过程与实验证据见 `PROCESS.md`；操作步骤与接口细节见 `README.md`、`docs/calibration.md`。
> 引用指标时注明来源：实机实测 / 合成自检 / 单次会话。

## 项目边界

- 原项目 [`jyjblrd/Low-Cost-Mocap`](https://github.com/jyjblrd/Low-Cost-Mocap)（参考 `main` 的 `8ac4d41`，MIT）是 **outside-in 外部视觉定位**，不是 SLAM；不提供地图、障碍物检测、避障、路径搜索或动态重规划。
- 原项目 `/api/trajectory-planning` 用 **Ruckig** 做航点时间参数化，属于**轨迹生成**，不是自主路径规划。
- 原项目 4×PS3 Eye + `pseyepy`，没有硬件或软件多相机同步；`Camera.read()` 丢弃时间戳，各帧被假定同刻。
- 迁移到 Hypertron-T01 / RK3588 时，合理方向是固定相机端解算定位，再经 Wi-Fi / ROS 2 / UDP 传给机器人；无需沿用两块 ESP32、ESP-NOW、SBUS 或无人机 PID。ESP-NOW 不能由普通 RK3588 Wi-Fi 模组直接收发。这是适配建议，**尚未实现**。
- 主动发光 IR LED 标记不需要给相机加 IR 环形泛光灯（泛光灯用于被动反光标记）。标记灯统一采用 **850 nm 红外 LED**，不再评估 940 nm 方案；装机时按实际亮度确认曝光/阈值。
- 原项目 `cv2.sfm` 只用到 `essentialFromFundamental`、`motionFromEssential`、`fundamentalFromProjections` 三处，可用常规 OpenCV 替换以避免编译 `opencv_contrib`；**替换建议不等于主链路已适配**。详见 `PROCESS.md` §1。
- 旧 `camera-params.json`、`helpers.py`、3001 后端、React 网页和 ESP32 固件已从运行代码删除；原作者实现保留在 `upstream/`，当前标定输出不进入旧 PS3 Eye 图像流水线。

## 相机与采集

- 当前相机：**四台 LRCP G720P**，标准 UVC，`VID_BC15&PID_2E1B&MI_00`；OV9281 单色 CMOS，最高配置含 1280×720@30，支持 NV12 / MJPG / YUY2。四台格式列表完全相同（47 种），默认 NV12 1280×720@30，实测各 29.4–30.3 fps。
- **相机身份用 USB 设备实例 key 固定**（`camera_controls.device_key` 归一化，例如 `\\?\usb#vid_bc15&pid_2e1b&mi_00#6&5332b2&0&0000`），保存在 `captures/calibration/camera_config.json`；不用枚举索引、设备名或 VID/PID。枚举顺序变化不影响逻辑编号；离线相机保留编号不补位。`confirmed=false` 的相机不套用历史标定；旧标定文件没有设备信息时也要求人工确认。换 USB 口/加延长线会改变设备实例路径：只要用户在网页确认“这台设备就是逻辑相机 N”，即使标定文件记录的 device_key 与当前不同，也允许沿用该相机标定（2026-11-26 起）。原 cam1 的旧路径 `7&187a324` 已不在当前枚举中，其物理身份需人工确认。
- 所枚举 47 种格式**不含 Y16**；输出 **8-bit 亮度图，不可测温**。
- 产品有 400–650 nm 可见光版与定制窄带版，但**本机滤光片配置未确认**：白屏实测中心亮度均值 251.1，说明可透过可见光。标记灯按项目决定使用 850 nm LED，不再做 850/940 波长对比。
- **曝光与帧率**（2026-11-26 实测）：DirectShow 曝光值是 log2 秒；-3 = 125 ms → 8.00 fps，-5 = 31.25 ms → 30 fps，-6 = 15.6 ms → 30 fps，-8 = 3.9 ms → 30 fps。-1 = 0.5 s → 2 fps。**30 fps 要求曝光 ≤ 1/30 s**，-5 是仍能跑满 30 fps 的最亮档；增益 0–63 只放大信号、不改变曝光时间。
- **长曝光撕裂**：125 ms 曝光下该相机经 UVC/MF 交付的 NV12 帧约 3/4 出现横向断裂（单相机直连、绕过 Flask 也能复现，buffer 长度和 stride 正确），属相机/驱动在低帧率下的行为，不是解码或软件问题；-5/-6/-8 下无此现象。排查时先看曝光，不要先怀疑 stride 或线程。
- 相机**独占**：同一设备同时只能被一个进程打开，重复打开常见错误 `0xC00D3704`。跑两个 `preview_server` 会互相抢设备：表现为几路 `0xC00D3704`、一路离线；先确认只剩一个进程再重试。
- **重插/重启后的曝光复位**：相机重新枚举/MF 重新取流后，部分 UVC 曝光会回到默认 **-1（0.5 s，约 2 fps）**，此时“能连上但只有 2 fps”，不是连接失败；点“恢复上次保存”或重新应用 -5 即可。设备实例 key 没变时不需要重新绑定。
- 补光灯（IR LED 板）由**硬件光敏电路自动控制**：遮住镜头/感光区即点亮。UVC 只暴露曝光、增益、亮度、对比度、色调、饱和度、锐度、伽马、白平衡、背光补偿、电源频率（50/60 Hz 防闪烁）等标准项；`IKsTopologyInfo` 不可用（E_NOINTERFACE），无厂商扩展单元（XU），**没有任何软件参数可以开关补光灯**。需要可控补光只能外接独立 LED 与开关电路（2026-11-26 实机枚举确认）。
- 采集走 Windows **Media Foundation + ctypes/COM**。关键约束：`QueryInterface` 不额外传 `pUnkOuter`；`ReadSample` 输出参数顺序为 `*index, *flags, *timestamp, **sample`；打开/读取/关闭必须在同一 COM 线程。详见 `PROCESS.md` §2。
- 一相机一线程 + 只保留 latest + 每帧只编码一次 JPEG；重复 `start` 同一设备返回 409；历史缓存入队前转连续数组，避免持有整块 NV12 buffer。
- 旧 `index.py`（3001/Socket.IO/Ruckig）已删除；当前唯一后端是 `preview_server.py`（3002）。
- 单相机独立工具 `ir_camera.py` 仅依赖 NumPy + OpenCV；`--formats` / `--snapshot` / `--no-display` 可用。

## 帧同步

- 只用**主机 `time.perf_counter()`** 做流式最近邻配对，默认阈值 **20 ms**。`mf_timestamp` 是每台设备各自的时基，**跨设备不可减**，只随图保存作溯源。实测反例：某对帧两路 `mf_timestamp` 差 0.135 ms，而主机时间差 15.95 ms。
- 时间戳记录的是**帧交付用户态**时刻，不是曝光中心；差值随曝光、帧率、负载变化，不是常数偏移，无法固定补偿。
- 30 fps 帧周期 33.3 ms，仅量化误差就有 ±16.7 ms（半帧）。结论：**软件级近似同步**，适用于静止标定板 / 低速目标，**不可用于快速运动目标的亚毫秒三角化**。
- 实测（60 s，阈值 20 ms）：30.01 对/s、配对率 100%、丢弃 0；`|Δt|` 均值 0.2–1.6 ms，P95 最高约 16 ms，最大 16.9 ms。阈值 10 ms 时配对率降到 71–82%，因此保留 20 ms。详见 `PROCESS.md` §3。
- 配对按每路严格单调的 `seq` 推进，保证不重复、时间单调；一路停流 >0.5 s 时，另一路积压帧按丢弃计。
- **多相机帧组**：每台相机一个已消费指针，以每台最老未消费帧为候选窗口起点，选“参与相机最多、跨度最小”的一组；组内跨度 = max−min 交付时刻 ≤ `max_dt`（默认 20 ms）。优先 4 相机，自动降级 3/2 相机。四台 30 Hz 相机相位不同，只有相位差 ≤20 ms 才组成 4 相机组——实测帧组约 45 组/s、跨度均值约 3.5 ms、最大约 15 ms，4 相机组占 5–15%，其余为 3/2 相机组。这是软件同步的物理限制。

## 标定

- 现场配置：**12×9 方格 = 11×8 内角点**，单格 **30 mm**，图案 360×270 mm，采集原图 1280×720。2026-11-26 四相机标定（会话 `20261126_112734`）：四台各 12 张，内参 RMS 0.559/0.282/0.137/0.123 px；fx≈647/650/809/809 —— **相机 2/3 的镜头配置与 0/1 不同**，必须每台单独标定，不能互相套用。外参 `orientation_confirmed=true`，四相机联合三角化 max 2.28 mm；但相机 0/1 基线只有 **80 mm**、该对残差 max 53 mm，四台解出的中心彼此相距 ≤0.17 m，安装几何需要复核；尚无独立多位置验证。
- 图像坐标必须使用**原始灰度图像素坐标**：不 `rot90`、不裁成方形、不缩放、不先去畸变；内参/畸变两台分别求解。
- 世界系：原点 = 图案中心；X 沿 12 格长边，Y 沿 9 格短边，Z = X × Y 朝上，右手系；地面 `Z = -surface_height_m`（默认板面高 3 mm）。88 个角点按 row-major，`index = row*11 + col`，范围 X ±0.150 m、Y ±0.105 m。
- 外参主路线：板固定作世界基准，每台相机 `solvePnPGeneric(IPPE)` + `solvePnPRefineLM`；`X_cam1 = R_10 X_cam0 + T_10`，`||T||` 为基线（米）。`stereoCalibrate` 仅作离线交叉校验，不在网页主流程。详见 `PROCESS.md` §4。
- **矩形板 D2 四重对称**：4 种角点对应的重投影误差逐位相同，三角化自洽误差都约 0.6 mm。合成案例中，单侧 180° 反转把基线从 2032 mm 改成 1635 mm，而指标不报警。**必须逐台人工核对世界轴并显式确认方向**；当前实现支持每台独立 180° 开关。详见 `PROCESS.md` §8。
- IPPE 双解：先按“相机在图案正面（Z>0）、角点在相机前方”筛物理合理候选，再比重投影；并列标 `ambiguous`。不能只看 RMS 或高度。
- `findChessboardCornersSB` 的正相/反相检测可能把整幅角点旋转 180°；**同一帧对的两台相机必须采用一致极性**。
- 内参采集必须覆盖画面四角：只用中央棋盘会让高阶畸变互相补偿，RMS 很低但画面边缘外推可错约 **20 px**；补采四角后约 **1.6 px**。已有 `radius_coverage.ratio < 0.75` 告警。
- 覆盖率门限：内参 `MIN_COVERAGE = 0.8%`；固定板外参/验证 `MIN_COVERAGE_EXTRINSIC = 0.1%`。0.1%–0.8% 的样本会入库，但带“覆盖率偏低，位姿/深度精度会下降”告警。
- 指标名不可混用：`max_reprojection_rms_px`（重投影）与 `consistency_max_mm`（三角化角点 vs 已知世界点，毫米）含义不同；`spacing_abs_error_mm` 已修正为毫米并带 `unit`，旧结果读取时幂等迁移一次。
- 结果生命周期：新采集会使依赖结果失效；改 180° 会作废旧外参与方向确认；重载会话可恢复已保存结果。完整结果只按需写会话目录 `calibration.json`；旧全局兼容文件 `api/calibration/calibration.json` 及其读写逻辑已移除，不再重新生成。
- 采集不重开相机，直接从运行中的 `CameraManager` / `SyncService` 取帧；每次要求新序号，最多等待约 1.5 s。验证同步帧对按 `(seq_a, seq_b)` 去重，外参“采集双相机图片”按 `pair_seq` 去重。
- CLI 默认经 HTTP 复用运行中的 `preview_server.py` 相机；`--local` 才在当前进程采集；服务未运行时 `new` 可本地建会话。
- 数据默认在 `computer_code/captures/calibration/<session_id>/`；对外写项目相对路径，内部解析为绝对路径（不依赖启动 CWD）；旧绝对路径仍可读。
- **合成自检通过 ≠ 实机精度**。历史自检：`cd7e991` 77/77、`c353edf` 102/102、当前工作区 calibration 163/163 + LED 72/72（2026-11-26 复跑）。

## 红外 LED 检测与三角化

- `api/led/` 复用 `camera_manager`、`sync.SyncService`、`CalibrationService` 和现有三维视图。
- 检测：可选高斯模糊 → 阈值 → 连通域/面积/solidity/填充率/长宽比过滤；距离变换局部峰值 + 最近峰值分配（watershed 简化）拆粘连，拆分块至少占父亮区 15%；完全饱和/不可分亮区标 unresolved/overexposed，**不伪造三点**；强度加权亚像素中心；同时返回原始连通域与筛选后候选。
- 灯板几何为 **34/79/79 mm 等腰三角形**（短边 34、两腰 79）；L0 是两腰公共顶点，L1/L2 是短边端点。三维视图的橙色方向箭头从短边中点指向 L0，不代表机器人航向。
- **USB 截断帧**（2026-11-26）：偶发“半黑/上下两截”不是曝光或解码 bug，而是 USB 交付的 NV12 行被驱动填 0（MF 仍报完整长度），MJPG 则表现为 `Corrupt JPEG data: premature end of data segment`（OpenCV 仍返回部分图）。判据要与上一张有效帧比较（突然整段变全黑且 ≥10% 高度），不能只看大面积黑（暗场 LED 图本来就大部分为 0）。`CameraWorker` 会丢弃这类帧并累计 `corrupt_frames`。物理上优先直连口/供电 Hub/换短线。
- **多视角匹配**：任意相机对极线筛选（向量化）+ 双视角 DLT + 正深度 + 重投影；其它相机中重投影一致的观测加入同一 3D 假设并联合 DLT 精化；按 3D 距离合并。每个 LED 可用不同相机组合，不要求每台相机看到三个 LED。组合上限：3D 候选 24、三角形搜索 12 点，同一相机候选点不得复用。
- 默认阈值 200、面积 3–800 px²、峰距 6 px；极线/重投影/三角容差 **5/4/20**；可对 cam0/cam1 分别覆盖阈值/面积/峰距/分离；存 `captures/calibration/led_params.json`。**`max_area=10` 会把真实 LED 当“面积过大”拒绝，是实机 matched=0 的历史原因之一**；当前默认已重置。
- 2026-11-26 已移除手动 ROI、动态跟踪 ROI 和全图回退；`detect_leds` 始终处理整幅灰度图，状态/参数/调试 JSON 不再包含 `rois`、`roi_tracking`、`roi_mode`、`roi_fallback`。单相机全图检测实测 4–11 ms，四相机整体约 20–30 ms。
- **相机枚举顺序与标定编号互换的问题已由固定设备绑定解决**：逻辑编号直接对应标定编号，LED 服务不再评估 `live0/live1` 的排列；换 USB 口只会改变设备 key，需要在网页重新绑定并确认。
- `F` 由标定 R/t 计算；原始点只去畸变一次；`P = K[R|t]` 三角化直接得世界系米制坐标；检查相机前方、重投影、34/79/79 mm 三边 + 两相机三点绕向。L0 是两条 79 mm 边的公共顶点，L1/L2 是 34 mm 短边端点（物理身份不唯一）；方向箭头只是“短边中点 → L0”，**不输出唯一航向或 6DoF**。
- 9 状态卡尔曼按 `rec.created_mono` 真实时间传播；同一 `pair_seq` 只处理一次，参数变化允许对最新帧重算一次。
- 标记灯统一 **850 nm**。LRCP G720P 的 UVC 曝光/自动曝光/增益可由 DirectShow `IAMCameraControl`/`IAMVideoProcAmp` 实时读写，**不需要第二路取流、不打断 MF**；本机范围曝光 -13..-1（log2 s）、增益 0..63（仅手动）。设置按 USB 设备实例路径保存到 `captures/calibration/camera_controls.json`，相机重开/服务重启自动恢复并验证；控件已移到 LED 模块，每 5 秒轮询已移除。旧的 `helpers.py`/pseyepy 通路已删除。
- 同步 `/api/sync/status` 同时给出累计与最近 300 事件的配对率/丢弃数，以及每路 FPS、帧间隔 mean/P95/max；本机帧间隔均值 33.3 ms、P95 可到约 48 ms，启动/控制切换后的短暂丢帧不要用清零统计掩盖。
- 双相机叠加图改为 `/api/led/overlay_pair` 一次取同一 `pair_seq` 快照，避免前端分别刷新导致 cam0/cam1 显示不同 pair；`/api/led/diagnose` 输出两种映射的原始/理想像素、3×3 极线距离和各阶段计数。
- 极线视图不依赖 matches：候选点即可画另一路极线，同色标记来源，距离表显示最小距离与 OK/FAIL；`_clip_line` 处理水平/倾斜/近竖直极线。
- 叠加图相机信息整体移到底部右侧，按输出尺寸/文字宽度动态右对齐并使用半透明深色背景；左下角图例保留，所有视图/局部放大复用同一信息面板函数。
- “保存调试帧”输出 `captures/led_debug/<时间戳>/` 的原始 PNG、二值 PNG、叠加 JPG、`detections.json`、`diagnosis.json`、`state.json`；文件与诊断来自同一帧组。当前 LED 自检 **72/72**、标定自检 **163/163**（合成/假相机）；实机已有三点跟踪成功记录，但四相机三维精度尚未验收。
- 连续录制只轮询现有 `sync_service().recent_pairs()`，不新开相机/不重复同步；独立写盘线程保存 `captures/led_recordings/<时间戳>/frames.jsonl` 与 `frames/NNNNNN_cam0|1.png`。LED 线程 `note_result(pair_seq, state)` 只登记已有结果，按 pair_seq 对齐；未处理帧标 `not_processed`。本机用实机调试帧按 29.9 组/秒回放压测：90 对全部保存、队列为空、无丢帧；PNG 压缩等级 1。
- 本机实测：正确映射后 cam1 曝光 -10 + 阈值 210 覆盖，三边约 32.6–35.2 mm、重投影约 1.3 px；cam1 -11 时检测出 5 个候选但匹配能选出真实三点（三边 32.4/34.2/34.0）。这些是当前场景数据，不是通用默认值。

## 工程与运维

- 版本一致性：`GET /api/version` 返回 Git SHA / 分支 / dirty / 启动时间；页面右上角显示 `v<sha> [· dirty] [· 分支]`，检测到重启或 SHA 变化只提示刷新，不自动刷新。网页始终由 3002 同源提供，旧 3001 后端和端口回退已删除。
- 改过 Python 后必须重启 `preview_server.py`；静态页更新后需硬刷新。`/preview` 已加 `Cache-Control: no-store`。
- 浏览器黑屏但后端显示 30 fps：确认 `/api/camera/<i>/stream` 确实被请求；首帧超时/解码失败会降级为 snapshot 轮询。
- 页面默认展开标定面板并在初始化时自动复用/创建会话；原“标定面板/新建打开会话/收起”按钮已移除，多标签页调用 `/api/calib/session/ensure` 幂等。初始化失败看 `c-act` 与 F12 Console。
- 外参图“没计入有效样本”：看“采集双相机图片”旁的提示徽标；旧 0.8% 门限会拒绝地面小板（0.4–0.5%），改门限后需重启服务。
- 双目外参方向看似反转：不要靠降低 RMS“自动修正”；逐台看世界轴叠加和相机位置，人工确认 180°。
- 原 Windows Git 曾因证书撤销检查报 `CRYPT_E_NO_REVOCATION_CHECK`；本仓库局部配置 `http.schannelCheckRevoke=false`，不是全局建议。
- `upstream/`、厂商 DirectShow SDK、截图/日志/`__pycache__` 不纳入版本控制。`captures/` 默认忽略，例外是已跟踪的 `captures/calibration/intrinsics.json` 与 `world_extrinsics.json`（新格式标定结果）；其他本地采集结果仍不代表已推送。
