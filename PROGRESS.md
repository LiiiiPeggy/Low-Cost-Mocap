# PROGRESS — 当前状态

> 快照日期：**2026-11-24**。分支 `main`；本轮 2～4 相机统一扩展 + LED 实时性能优化**尚未提交**（等实机验收）。
> 本文只描述“现在做到哪里”；长期结论见 `MEMORY.md`，决策依据见 `PROCESS.md`，操作说明见 `README.md` 与 `docs/calibration.md`。

## 当前目标

用两台 LRCP G720P 完成 **采集 → 软件帧同步 → 内参 → 固定棋盘格世界系外参 → 三维验证**，再把定位结果接到四足机器人链路。

当前进度：**2～4 相机统一采集/同步/标定/多视角 LED 定位**已实现；四台实机可同时 1280×720@30 采集，同步帧组、网页预览、录制和二维检测已实测。**四相机内外参尚未重新标定**，因此四相机三维定位还没有实机精度结论。原作者旧后端、React 网页和 ESP32 链路已删除，源码只保留在 `upstream/`。

## 已完成并验证

- 双相机并发采集、MJPEG 预览、原始灰度保存：实机 1280×720@30 NV12 两路各约 30 fps；最新帧延迟约 7–35 ms，不累积。
- 软件帧同步：60 s 实机测试 30.01 对/s、20 ms 阈值下配对率 100%、丢弃 0；`|Δt|` 均值 0.2–1.6 ms、P95 最高约 16 ms。
- 标定链路：内参、固定板 `solvePnP` 外参、`validate_rig`、结果失效/恢复、CLI 与 HTTP 接口（已提交至 `c353edf`）。
- 已提交改版：内参/外参自动保存与载入、网页“采集双相机图片”、Canvas 2D 三维视图、红外 LED 检测与三角化、版本一致性、覆盖率门限放宽到 0.1%、单位修复、默认板面高 3 mm。
- 2026-11-17 实机 LED：极线阈值调整为 **5 px** 后成功完成三点三角化/跟踪；当前场景 cam1 使用阈值 210 覆盖，重投影约 1.3–1.4 px。该记录不代表全场景精度验收。
- 2026-11-17 灯板更新：识别几何改为 **34/79/79 mm 等腰三角形**，边长容差调整为 **20 mm**；L0 为两长边公共顶点；三维视图新增“短边中点 → L0”橙色方向箭头；相机模型与视锥体绘制缩小 50%（外参不变）。
- 2026-11-17 连续调试录制：复用现有同步帧对和 LED 状态，后台写盘线程保存 `captures/led_recordings/<时间戳>/` 的双相机原始 PNG 与 `frames.jsonl`；网页显示时长、帧组、丢帧和目录，刷新可恢复状态；未处理帧标 `not_processed`。
- 2026-11-24 多相机扩展：逻辑编号 ↔ USB 设备实例固定绑定（`camera_config.json`，未确认相机拒绝套用历史标定）；同步改为 2～4 相机帧组（优先 4 台，自动降级 3/2 台，组内跨度 ≤20 ms）；内参按相机独立保存、支持不同分辨率；外参每台独立 solvePnP + 多视角一致性；LED 任意相机对极线匹配 + 联合 DLT + 每个 LED 可用不同相机组合；网页 2×2 布局、去畸变视图、极线相机对选择；录制支持 2～4 路。
- 2026-11-24 四相机实机实测：四台 LRCP G720P 同时 NV12 1280×720@30，各约 29.4–30.3 fps；同步帧组约 45 组/s，组内跨度均值约 3.5 ms、最大约 15 ms；帧组构成随相位漂移，4 相机帧组约占 5–15%，其余为 3/2 相机组。录制 100 s：3041 组、0 丢帧、30.4 组/s、1.4 GB。LED 二维检测在“丢失后全图搜索”场景约 20–36 组/s，会统计跳过帧组；ROI 跟踪正常时单相机检测 1–2 ms，可跟上帧组。
- 2026-11-24 性能优化：取消 `MAX_PROCESS_FPS=25` 与固定休眠；检测器单灯跳过粘连分离/灰度峰值兜底、峰值质心改 `bincount`；极线矩阵向量化、三角形组合上限 12 点；录制 PNG 改 4 线程并行（约 3×）。
- 本轮代码精简：合并 LED 参数/跟踪重置/诊断/叠加图重复逻辑，统一标定检测分类与 JSON 载入，收紧前端重复事件与无效状态，合并测试中的假相机和几何辅助代码。
- 自检（2026-11-24 在本机复跑确认）：
  - `python -m calibration.selftest` → **156/156 通过**
  - `python -m led.selftest` → **76/76 通过**
  - 覆盖 2/3/4 相机、不同分辨率、设备身份校验、多视角三角化、录制一致性；
    均为合成图 / 假相机，只证明求解链路与流程，**不代表实机精度**。

运行方式（无相机也可）：

```bat
conda activate lowcost_mocap
cd Low-Cost-Mocap-main\computer_code\api
python -m calibration.selftest
python -m led.selftest
```

## 当前数据与结果

- **内参**：`computer_code/captures/calibration/intrinsics.json`（2026-11-16 17:34，`source=migrated`，源会话 `20261116_104419`）。两台各 8 张，RMS **0.2259 / 0.077 px**，1280×720。
- **外参**：`computer_code/captures/calibration/world_extrinsics.json`（2026-11-16 17:50，源会话 `20261116_174926`）。`orientation_confirmed=true`，基线 **560.567 mm**；每台仅 1 帧，带“外参图覆盖率偏低（s0001、s0002）”警告。
- 以上两份新格式结果已纳入 Git 跟踪；原始采集图与 `led_params.json` 仍被 `captures/` 忽略。
- 当前实机 LED 参数（已保存到 `captures/calibration/led_params.json`，不进入 Git）：threshold 150、cam1 覆盖 threshold 180、epi 5 px、reprojection 4.25 px、triangle 20 mm；实机换多个位置均可定位。**`camera_map` 自动映射已删除**，改为固定设备绑定。
- 相机身份：`captures/calibration/camera_config.json` 已生成。逻辑 0 = `6&5332b2`（历史记录可确认，`confirmed=true`）；逻辑 1–3 是当前另外三台，**尚未人工确认**（原 cam1 的设备路径 `7&187a324` 已不在当前枚举中，无法自动判断）。
- **旧版兼容文件已彻底移除**：`api/calibration/calibration.json` 及其自动写入逻辑、`/api/calib/result` 与 `/api/calib/matrices` 的默认回退读取均已删除，不再重新生成。完整结果只存在于当前会话目录的 `calibration.json`。
- **早期实机会话**：`computer_code/captures/calibration/20261116_152637`，两台各 8 张内参图；`s0017`–`s0022` 是旧 0.8% 门限下被拒的外参样本（覆盖率 0.39–0.48%，原图已保存）。
- 结论：**尚无四相机“固定安装 + 人工方向确认 + 独立多位置验证”闭环的可信外参/基线**。旧双相机外参仍可在四相机会话里载入，但四相机三维定位需要重新标定。旧 `camera-params.json`、`helpers.py`、3001 后端、React 网页和 ESP32 固件均已从运行代码删除；原作者实现见 `upstream/`。

## 本次提交内容

- 代码：`api/calibration/{dataset,routes,service,selftest}.py`、`api/calib_cli.py`、`api/{camera_preview,preview_server}.py`、`api/static/preview.html`、`docs/calibration.md`。
- 新增：`api/led/`（`detector/geometry/tracker/service/routes/selftest` 等 7 个文件）。
- 文档：`README.md` 同步到当前实现；`memory.md` 重命名为 `MEMORY.md`，新增 `AGENTS.md`、`PROGRESS.md`、`PROCESS.md`、`CLAUDE.md`。
- 清理：删除旧版全局兼容结果 `api/calibration/calibration.json` 及对应读写代码；删除原作者 `index.py`/`helpers.py`/`KalmanFilter`/`LowPassFilter`/`Singleton`/`test.py`、React/Vite 网页、ESP32 固件和旧相机参数文件。
- 运行产物 `captures/` 已被 `.gitignore` 忽略。

## 遗留问题与下一步

1. **相机身份确认**：在网页“相机身份绑定”里逐台确认逻辑 1–3 对应哪台物理相机（尤其原有 cam1）。
2. **四相机实机外参闭环**：每台重采覆盖率足够的固定板图（目标 ≥0.8%），人工逐台确认世界轴方向，记录基线与多视角一致性。
3. **独立验证**：把板放到 ≥3 个不同位置/高度采集 validation 数据，跑 `validate_rig`，检查重投影、网格间距、平面度、极线指标。
4. **850 nm LED 实机联调**：用实际 850 nm 标记灯确认亮度、曝光与检测阈值（波长已定为 850 nm，不再评估 940 nm）；当前 MF 接口没有可靠的运行时 exposure/gain 控制。
5. **UI 人工验收**：完整走一遍 3002 `/preview` 标定流程，验收三维视图旋转/缩放/平移和视锥体。
6. **动态目标**：软件同步对快速运动仍是半帧级误差；三点标记唯一编号、实时三角化精度未验收。
7. **下游整合**：当前系统只提供多相机标定与 LED 三点定位；四足机器人控制链路尚未实现。
