# PROGRESS — 当前状态

> 快照日期：**2026-11-17**。分支 `main`；本轮精简与旧系统清理已提交（`65b786e`）。
> 本文只描述“现在做到哪里”；长期结论见 `MEMORY.md`，决策依据见 `PROCESS.md`，操作说明见 `README.md` 与 `docs/calibration.md`。

## 当前目标

用两台 LRCP G720P 完成 **采集 → 软件帧同步 → 内参 → 固定棋盘格世界系外参 → 三维验证**，再把定位结果接到四足机器人链路。

当前进度：采集、同步、标定链路和 3002 网页/CLI 工具已可用；**尚未完成可信的实机标定闭环**。原作者旧后端、React 网页和 ESP32 链路已删除，源码只保留在 `upstream/`。

## 已完成并验证

- 双相机并发采集、MJPEG 预览、原始灰度保存：实机 1280×720@30 NV12 两路各约 30 fps；最新帧延迟约 7–35 ms，不累积。
- 软件帧同步：60 s 实机测试 30.01 对/s、20 ms 阈值下配对率 100%、丢弃 0；`|Δt|` 均值 0.2–1.6 ms、P95 最高约 16 ms。
- 标定链路：内参、固定板 `solvePnP` 外参、`validate_rig`、结果失效/恢复、CLI 与 HTTP 接口（已提交至 `c353edf`）。
- 已提交改版：内参/外参自动保存与载入、网页“采集双相机图片”、Canvas 2D 三维视图、红外 LED 检测与三角化、版本一致性、覆盖率门限放宽到 0.1%、单位修复、默认板面高 3 mm。
- 2026-11-17 实机 LED：极线阈值调整为 **5 px** 后成功完成三点三角化/跟踪；当前场景 cam1 使用阈值 210 覆盖，三边约 30–35 mm、重投影约 1.3–1.4 px。该记录不代表全场景精度验收。
- 本轮代码精简：合并 LED 参数/跟踪重置/诊断/叠加图重复逻辑，统一标定检测分类与 JSON 载入，收紧前端重复事件与无效状态，合并测试中的假相机和几何辅助代码。
- 自检（2026-11-17 在本机复跑确认）：
  - `python -m calibration.selftest` → **139/139 通过**
  - `python -m led.selftest` → **50/50 通过**
  - 均为合成图 / 假相机，只证明求解链路与流程，**不代表实机精度**。

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
- **旧版兼容文件已彻底移除**：`api/calibration/calibration.json` 及其自动写入逻辑、`/api/calib/result` 与 `/api/calib/matrices` 的默认回退读取均已删除，不再重新生成。完整结果只存在于当前会话目录的 `calibration.json`。
- **早期实机会话**：`computer_code/captures/calibration/20261116_152637`，两台各 8 张内参图；`s0017`–`s0022` 是旧 0.8% 门限下被拒的外参样本（覆盖率 0.39–0.48%，原图已保存）。
- 结论：**尚无“固定安装 + 人工方向确认 + 独立多位置验证”闭环的可信外参/基线**。旧 `camera-params.json`、`helpers.py`、3001 后端、React 网页和 ESP32 固件均已从运行代码删除；原作者实现见 `upstream/`。

## 本次提交内容

- 代码：`api/calibration/{dataset,routes,service,selftest}.py`、`api/calib_cli.py`、`api/{camera_preview,preview_server}.py`、`api/static/preview.html`、`docs/calibration.md`。
- 新增：`api/led/`（`detector/geometry/tracker/service/routes/selftest` 等 7 个文件）。
- 文档：`README.md` 同步到当前实现；`memory.md` 重命名为 `MEMORY.md`，新增 `AGENTS.md`、`PROGRESS.md`、`PROCESS.md`、`CLAUDE.md`。
- 清理：删除旧版全局兼容结果 `api/calibration/calibration.json` 及对应读写代码；删除原作者 `index.py`/`helpers.py`/`KalmanFilter`/`LowPassFilter`/`Singleton`/`test.py`、React/Vite 网页、ESP32 固件和旧相机参数文件。
- 运行产物 `captures/` 已被 `.gitignore` 忽略。

## 遗留问题与下一步

1. **实机外参闭环**：每台重采覆盖率足够的固定板图（目标 ≥0.8%），人工逐台确认世界轴方向，记录实际基线并复核重复性。
2. **独立验证**：把板放到 ≥3 个不同位置/高度采集 validation 数据，跑 `validate_rig`，检查重投影、网格间距、平面度、极线指标。
3. **850 nm LED 实机联调**：用实际 850 nm 标记灯确认亮度、曝光与检测阈值（波长已定为 850 nm，不再评估 940 nm）；当前 MF 接口没有可靠的运行时 exposure/gain 控制。
4. **UI 人工验收**：完整走一遍 3002 `/preview` 标定流程，验收三维视图旋转/缩放/平移和视锥体。
5. **多相机与动态目标**：四相机未实现；运动目标下软件同步误差、三点标记唯一编号、实时三角化均未完成。
6. **下游整合**：当前系统只提供双相机标定与 LED 三点定位；四足机器人控制链路尚未实现。
