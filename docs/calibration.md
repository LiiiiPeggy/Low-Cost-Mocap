# 双相机内外参标定（固定棋盘格 + 世界系）

本文件说明 `computer_code/api/calibration/` 怎么用、约定是什么、能声称什么、不能声称什么。

求解全部交给 OpenCV：`findChessboardCornersSB` / `findChessboardCorners` + `cornerSubPix`
（回退）、`calibrateCamera`、`solvePnP`（`SOLVEPNP_IPPE` 多候选取解 + `solvePnPRefineLM`）、
`undistortPoints`、`projectPoints`、`triangulatePoints`。**没有手写标定/优化算法，没有引入
`cv2.sfm`，没有依赖 `pseyepy`，也不需要 `opencv-contrib-python`。**

外参路线按 EasyMocap 的"固定棋盘格"思路：**把板当作世界基准，每台相机各自 solvePnP**，
而不是双目标定。这样每台相机的位姿可以单独检查，下游 LED 三维动捕直接拿到世界系。
`cv2.stereoCalibrate` 仅保留为交叉校验。

---

## 1. 现场配置（已确认）

| 项 | 值 |
|---|---|
| 相机 | 两台固定在不同位置，朝房间中心区域 |
| 标定板 | **12×9 个方格 → 11×8 内角点**，单格 **30 mm**，总尺寸 360×270 mm |
| 板面高 | 图案表面高于地面，**默认 3 mm**，按实际修改 |
| 图像 | 1280×720，NV12@30（MJPG 同样可用） |
| 环境 | conda `lowcost_mocap`：Python 3.10.22 / OpenCV 4.10.0 / NumPy 1.26.4；环境文件见 `docs/environment.yml` |

> 注意：**内角点数 = 方格数 − 1**。填 12×9 就错了，必须填 **11 × 8**。

### 世界坐标系

```
原点 O = 棋盘格图案中心（内角点阵列的几何中心，不是某个角点）
X 轴   = 沿 12 格长边（cols 方向）
Y 轴   = 沿 9 格短边（rows 方向）
Z 轴   = 板面法向，指向图案正面外侧；板正面朝上平放时 = 垂直地面向上
右手系  Z = X × Y
```

- 88 个角点，编号 `index = row * 11 + col`，行优先；`x = (col-5)*0.03` m，
  `y = (row-3.5)*0.03` m，即 **x ∈ [−0.150, 0.150] m，y ∈ [−0.105, 0.105] m**。
- 板面高于地面时填「板面高」`surface_height_m`。世界原点在板面上，所以
  **地面平面是 Z = −surface_height_m**；要地面基准坐标就把 Z 加上该值。
  结果文件里两者都写明（`world_frame.surface_height_m` / `floor_z_m`）。

---

## 2. 快速开始

```bat
:: 仓库根目录首次创建环境
conda env create -f docs\environment.yml
conda activate lowcost_mocap
cd Low-Cost-Mocap-main\computer_code
python api\preview_server.py --indices 0 1
:: 浏览器打开 http://127.0.0.1:3002/preview（标定面板默认展开）
```

面板默认展开，页面初始化会自动检查会话：已有有效会话直接复用，没有才按
内角点 `11 × 8`、方格 `30` mm、板面高 `3` mm 创建；多标签页/刷新不会重复创建，
也不会清空已载入的内外参。原「新建 / 打开会话」「收起」「标定面板」按钮已移除。

面板 5 步：

1. **会话**：页面加载时自动完成（见上）。
2. **内参**：让板在**两台相机前移动**，各采 30–40 张（角度/距离/位置都要变，尽量铺到画面四角）
   → 「内参标定」。成功后**自动保存**：
   - 会话：`captures/calibration/<session>/result_intrinsic.json`
   - 跨会话最新：`captures/calibration/intrinsics.json`
   页面会直接显示两台相机的 K、D、RMS、fx/fy/cx/cy、有效样本数、区域覆盖率和保存路径。
   重启程序或新建会话后，可点「载入内参」直接恢复 K/D，**不需要重新采集**。
   可视化模式默认是**棋盘格角点可视化**，确认方向时再切到世界坐标轴。
3. **外参**：把板**固定在中心地面** → 「采集双相机图片」按一次同时取一对同步帧，分别存入
   两台相机的 `extrinsic` 阶段。**每台 1 张有效图即可 solvePnP**；可重复按 3–5 次观察位姿
   抖动（单张时不会输出无意义的标准差）→「解外参 solvePnP」。成功后**自动保存**：
   - 会话：`captures/calibration/<session>/result_extrinsic.json`
   - 跨会话最新：`captures/calibration/world_extrinsics.json`
   重启后可点「载入世界坐标系外参」恢复三维位置、朝向、基线和三维视图。
4. **确认 180°**：把「可视化模式」切到**世界坐标轴可视化**，图上会画出世界原点、
   X（红）/Y（绿）/Z（蓝）轴和角点编号，0 号角点有红圈。**看 X 轴是否沿实物 12 格长边**；
   如果反了，勾对应相机的「反向 180°」再看一次。确认后勾「已确认世界坐标轴方向」，再点一次
   「解外参 solvePnP」把确认状态写入结果。**修改任一相机的 180° 会立即作废旧外参**，
   必须重新解外参；这一步只能人眼判断，原因见第 6 节。
5. **三维定位精度验证**：把板挪到**其它位置和高度**（建议至少 3 处），每处点
   「采集并验证三维精度」——自动取同步帧对、入库、三角化并更新历史统计；
   「重新计算」复用已保存的验证帧对，不再采集图像。

6. **红外 LED 检测与三维定位**（标定完成后的第二阶段）：确认双相机 streaming、内外参
   已载入且方向已确认。先在“红外 LED 检测与定位”模块内的 **相机曝光/增益** 把三个 LED 分开
   （避免完全饱和成一块；曝光值越小曝光越短）→ 在“红外 LED 检测与定位”区点
   **开始二维检测** → 把可视化模式切到 **红外 LED 检测可视化**，先用 **原始灰度**
   看真实光斑，再切 **二值化/最终检测叠加** 调整阈值/面积/峰值间距/光斑分离/形状过滤；
   必要时填写两台相机的**归一化手动 ROI**（0–1），或打开 **动态跟踪 ROI**；用 2×/4×
   **局部放大** 检查粘连 → 两台相机各稳定出现 3 个黄色候选后点 **开始三维定位** →
   在现有三维视图中查看三点、三角形、几何中心与轨迹 → 匹配异常时点 **保存调试帧**，
   把原始图/二值图/检测/诊断 JSON 存到 `captures/led_debug/<时间戳>/` → 结束时点
   **停止三维定位 / 停止二维检测**。参数保存在
   `captures/calibration/led_params.json`。

下方的三维世界视图用真实外参画出两台相机的位置、朝向与视锥体，并显示基线、世界轴、
棋盘格和相机编号/位置/离地高度/重投影误差。左键拖动旋转、滚轮缩放、右键拖动平移；
另有「重置视角」「俯视图」。标定结果变化后视图自动更新。

手动「查看结果」「保存 calibration.json」按钮已删除。两台内参与世界系外参分别自动保存到
`captures/calibration/intrinsics.json` 和 `world_extrinsics.json`；完整合并结果只写当前会话
目录的 `calibration.json`。旧版全局兼容文件 `api/calibration/calibration.json` 及其自动写入
逻辑已移除，未确认方向的外参只保存为待确认结果。

命令行等价流程见第 5 节。

---

## 3. 图像坐标约定（重点）

- 来源：`camera_manager` 的 `frame.gray`（NV12 的 Y 平面 / MJPG 解码），**8bit 单通道 1280×720**。
- **不**做 `np.rot90`、**不**做 `make_square`、**不**裁剪、**不**缩放、**不**预去畸变、**不**滤波增强。
  角点坐标就是这张原始图的像素坐标。
- 显示层（`draw_overlay` / `draw_world_overlay`）的缩放只作用于输出图，不回流到计算。
- 改分辨率/裁剪后，K 必须按 `fx*sx, fy*sy, cx*sx, cy*sy` 换算
  （`service.stereo_matrices(data, scaled_size=...)` 已实现）。
- **已去畸变的点不得再次去畸变**：内部只在三角化前对**原始像素点**调一次 `undistortPoints`。

---

## 4. 求解与指标

### 4.1 内参

`cv2.calibrateCamera`（针孔 + `[k1,k2,p1,p2,k3]`），输出 K、D、整体 RMS、逐张 RMS、
建议剔除样本、角点半径覆盖。另给出**区域覆盖率**（`board.coverage_map` 的 6×6 网格，
`coverage.covered_ratio`），只统计该相机 `phase="intrinsic"` 的有效样本，不会混入外参图
或验证图。

**为什么强调半径覆盖**：角点只集中在画面中心时，`k2/k3` 会互相补偿，重投影 RMS 仍可低到
0.07px，但把模型用到画面四角会出现明显偏差（合成实验里 20px → 补采四角后 1.6px）。
`radius_coverage.ratio < 0.75` 时会显式告警。

**自动保存与载入**：计算成功的相机会合并写入 `captures/calibration/intrinsics.json`
（包含 K、D、RMS、图像尺寸、有效样本数、覆盖比例、相机标识、计算时间）。失败的相机不会
覆盖文件里已有的有效结果。载入时校验 schema、相机编号、K/D、图像尺寸；不兼容的相机不会
清空当前内存结果，并在界面日志里明确报告“文件不存在 / 参数不兼容”。载入结果带
`source="loaded"` 和 `loaded_from`，不会被误认为本会话重新计算。

### 4.2 外参（solvePnP）

对每台相机：`solvePnPGeneric(SOLVEPNP_IPPE)` 得到两个候选，用
`cv2.solvePnPRefineLM` 精化，输出 `X_cam = R @ X_world + t`。

固定板 + 固定相机时，**每台相机最少 1 张有效图**（`extrinsic.MIN_FRAMES = 1`）即可求解。
重复采集多张只用于观察位姿稳定性；只有 1 张时 `stability.*.centre_std_m` 为 `null`，
并带说明文字，不会输出“标准差 0”这种误导性数字。

外参图同样有两台相机各自的质量门限（`MIN_SHARPNESS`、覆盖率）。覆盖率分两档：

- **内参**：`MIN_COVERAGE = 0.8%`，低于此直接拒绝。
- **固定板外参 / 验证**：`MIN_COVERAGE_EXTRINSIC = 0.1%`（现场要求，仅要求能检出 88 个
  角点并做 PnP）。板放在地面、离相机较远时通常只有 0.3%–0.6%，现在允许入库；但低于
  0.8% 时会在按钮旁和结果 `warnings` 里写入“覆盖率偏低”，提示位姿/深度精度会下降。
  条件允许时仍建议把板移近到画面约 **100 px 宽以上**，或让板面更正对两台相机。

强行把内参阶段也降到 0.1% 是不合适的：畸变参数需要画面边缘的角点覆盖。

**自动保存与载入**：`solvePnP` 成功后自动写会话 `result_extrinsic.json` 和跨会话
`captures/calibration/world_extrinsics.json`，包含两台相机的 R/t、三维位置、离地高度、
`R_10/T_10`、基线、重投影、棋盘格/世界系定义、内参快照和 `orientation_confirmed`。
未确认方向的结果会保存为**待确认**；无论是否确认，都不会再写全局兼容文件。
载入时校验棋盘格规格、世界系定义、相机编号、位姿字段完整性，以及文件内参快照与当前
已载入 K/D 的一致性；不兼容则明确报错且**不覆盖当前有效外参**。载入后三维视图和数值
自动刷新，并恢复 flip 与方向确认状态。修改 180° 会作废旧外参与确认，必须重新求解。

**候选解怎么选**：平面的两解互为"穿过板面的镜像"，重投影误差几乎相同，**不能按误差选**。
用物理先验——图案正面朝上，相机只能在其上方，所以取相机世界 Z 最大的解。

**坐标约定**（务必别转置/取反）：

```
世界 → 相机：X_cam   = R_wc @ X_world + t_wc
相机 → 世界：X_world = R_cw @ X_cam   + t_cw      (R_cw = R_wc^T, t_cw = -R_cw t_wc)
相机 1 相对相机 0：X_cam1 = R_10 @ X_cam0 + T_10
基线 = ||C1 - C0||，T 单位为米
```

### 4.3 独立验证

`validate_rig` 只用 **validation 阶段**的同步帧对（板在别的位置/高度），对每处：
三角化 88 个角点 → 重投影误差、极线误差、**已知网格间距误差**（30 mm）、重建平面度、
板心世界坐标；跨位置可给出实测距离与卷尺值对比。

网页上的「采集并验证三维精度」把“取一对同步帧 → 入库 → 验证”合成一次操作，只有内外参
已求解且本次采集通过质检时才自动计算；失败样本以 `invalid` 落盘、原图保留，但不计入有效
结果。「重新计算」只复用已保存的 validation 数据。

**单位**：`aggregate.spacing_abs_error_mm` 的 `mean/p95/max` 与逐位置字段一样都是**毫米**
（修复了早期汇总把米当毫米、数值小 1000 倍的问题），结果里另有 `unit: "mm"`。
读回旧版 `result_validation.json` / `calibration.json` 时，若缺少 `unit` 标记，会按
米→毫米**迁移一次**（幂等，不会重复放大）；逐位置字段本来就是毫米，不做转换。

**重投影误差小不等于三维精度高**。以间距误差与平面度为准，而且必须是独立数据。这些指标
是**三维几何一致性检查**：没有外部真实坐标时，不能称为绝对定位精度。

---

## 5. 命令行

**采集走 HTTP**：相机线程只存在于跑着 `preview_server` 的那个进程里，所以
`new / capture-single / capture-stereo` 默认请求 `http://127.0.0.1:3002`
（可用 `--api` 改地址）。CLI 会先 `POST /api/calib/session/load` 把会话挂到服务上，
再调用采集接口——不会重复打开相机，也不改 Media Foundation 驱动。
`--local` 只用于"相机确实由本进程打开"的特殊场景。

```bat
:: 先起服务（另一个窗口）
python api\preview_server.py --indices 0 1

:: 在线采集（HTTP）
python api\calib_cli.py new --cols 11 --rows 8 --square-mm 30 --surface-height-mm 3
python api\calib_cli.py capture-single --session captures\calibration\<会话> --camera 0 --count 30
python api\calib_cli.py capture-single --session captures\calibration\<会话> --camera 1 --count 30
:: 板固定到中心地面后（CLI 仍是逐相机采集；网页用「采集双相机图片」一次采一对）
python api\calib_cli.py capture-single --session <会话> --camera 0 --count 1 --phase extrinsic
python api\calib_cli.py capture-single --session <会话> --camera 1 --count 1 --phase extrinsic
:: 板挪到别处后采集验证帧对（等价于网页「采集并验证三维精度」的采集步骤）：
python api\calib_cli.py capture-stereo --session <会话> --count 3

:: 离线求解（不需要相机）
python api\calib_cli.py intrinsic    --session <会话>
python api\calib_cli.py extrinsic    --session <会话> [--flip180 | --flip-cam0 | --flip-cam1] [--orientation-confirmed]
python api\calib_cli.py validate-rig --session <会话>
python api\calib_cli.py save         --session <会话> --validate --orientation-confirmed
python api\calib_cli.py info         --result <会话>\calibration.json
```

`stereo` 子命令保留为 `stereoCalibrate` 交叉校验（不是主路线）。

`intrinsic / extrinsic / validate-rig / save` 会优先复用会话目录里已保存的
`result_intrinsic.json`，不会每次重算覆盖内参；`validate-rig` 只读已解出的外参，
**不会改写**它。

CLI 的 `intrinsic` / `extrinsic` 同样会自动保存 `captures/calibration/intrinsics.json`
和 `world_extrinsics.json`，并打印最新路径。`save` 子命令把完整结果写入当前会话目录的
`calibration.json`，可用 `--out` 额外导出；`info --result <文件>` 必须显式指定要查看的结果。
跨会话载入目前通过网页“载入内参 / 载入世界坐标系外参”按钮完成；对应的服务方法为
`service.load_intrinsics()` / `service.load_extrinsics()`。

---

## 6. 角点对应关系：D2 四重对称（必须知道）

矩形角点阵列的对称群是 **D2**，有 4 个元素：恒等、绕板法向 180°、沿行镜像、沿列镜像。
这 4 种对应在**纯几何上完全不可区分**——任何一种重排都能被某个合法位姿精确解释，
重投影误差逐位相同（实测 0.085100 px 对 0.085100 px）。

因此：

1. 两台相机从不同方向拍同一块板时，检测到的**0 号角点不一定是同一个物理角点**，
   不能默认对应。
2. 本模块只用「相机必须在图案正面一侧（世界 Z > 0）」筛掉两个镜像解，剩下一对相差 180°，
   并且**刻意不按重投影误差挑选**（两种解误差逐位相同，按误差选等于瞎猜）；结果里的
   `orientation_ambiguous` 会标出还有另一种解。
3. 剩下的 180° 只能**人工确认**：把「可视化模式」切到「世界坐标轴可视化」，逐台看
   X 轴是否沿 12 格长边。界面里 cam0 / cam1 各有独立的 180° 开关；确认后勾
   「已确认世界坐标轴方向」，再点一次「解外参 solvePnP」，状态才会写进结果文件
   （`orientation_confirmed`）。改任一相机的 180° 或重新采集外参图片会自动作废旧确认，
   必须重新求解。
4. **三角化一致性残差查不出 180° 错误**。实测：cam1 反向时基线从 **2032 mm 变成 1635 mm**，
   而自洽残差两种情况都只有 **0.6–0.8 mm**。也就是说，基线错了 20%，指标却看不出来。
   → 这就是为什么必须人眼看轴，也是为什么单翻一台相机的开关要单独给出。
5. 两台相机的 X 轴必须指向**同一物理方向**。如果一台对、一台反，说明两路检测顺序确实
   相差 180°，此时单翻那一台是对的；如果两台都反，两台一起翻即可。

更严谨的做法是在板上加非对称标记（ArUco / 缺角标记），本次未做。

### 6.4 红外 LED 检测与三维定位（第二阶段）

标定完成后，网页标定面板底部的“红外 LED 检测与定位”使用同一对同步帧和已载入标定参数。
代码在 `api/led/`：

| 文件 | 作用 |
|---|---|
| `detector.py` | 灰度阈值、连通域、距离变换峰值、光斑分离、亚像素中心、过曝/不可分状态 |
| `geometry.py` | 极线候选、世界系三角化、重投影筛选、33 mm 等边三角形、临时编号 |
| `tracker.py` | 9 状态（位置/速度/加速度）卡尔曼跟踪，按真实同步时间戳传播 |
| `service.py` | 后台线程、同步帧去重、参数持久化、二维叠加图、状态缓存 |
| `routes.py` | `/api/led/*` HTTP 接口 |
| `selftest.py` | 合成光斑/三角化/跟踪/服务去重测试（`python -m led.selftest`） |
| `../camera_controls.py` | DirectShow UVC 曝光/自动曝光/增益读取与设置（不另开流） |

**检测**：可选高斯模糊 → 灰度阈值 → 连通域 → 面积/形状（solidity、外接矩形填充率、
最小外接矩形长宽比）过滤。对每个亮区做距离变换，取局部峰值；多个峰值时用最近峰值
分配（watershed 的简化等价）拆分，并要求每个子块至少占父亮区 `split_min_area_ratio`
（默认 15%），防止单个 LED 的亮核/光晕被拆碎。距离峰值不够时用灰度局部极大值兜底。
完全饱和、无法分开的大亮区标记为 `unresolved / overexposed`，**不会伪造三个点**。
检测结果同时返回 `raw_components`（全部原始连通域）和 `leds`（通过筛选的候选），
网页据此区分原始点/筛选点/匹配点。

**参数**：默认灰度阈值 200、最小/最大面积 3/800 px²、最小峰值间距 6 px、光斑分离开启、
拆分最小面积比 0.15、最小 solidity 0.35、最小填充率 0.20、最大长宽比 5、最大候选数 24、
模糊核 3×3；另有极线 5 px、重投影 4 px、三角形边长容差 10 mm（实机质心噪声下先给 10 mm，诊断会输出实际三边与误差）。两台相机共用默认值，
可对阈值/最小面积/最大面积/最小峰值间距/光斑分离做相机 0/1 覆盖（留空=共用）。
参数保存到 `captures/calibration/led_params.json`，支持恢复默认；修改后对最新同步帧
重算一次。**注意**：`max_area=10` 这类过小值会把真实 LED 亮区当成“面积过大”拒绝，
只留下噪声点；默认 800 是 1–2.5 m 的初始值，应按实机调整。

**手动/动态 ROI**：两台相机可分别设置归一化 `[x0,y0,x1,y1]`（0–1，留空=全图）。
三维定位匹配成功后，用三个匹配点在各自图像上的包围框 + 80 px 边距更新动态 ROI；
连续失败超过 12 帧会清除动态 ROI，并强制下一帧全图重搜；手动 ROI 仍保留。ROI 内检测
不足 3 个候选时会自动回退全图搜索一次，避免目标移出 ROI 后永久丢失。

**匹配与三角化**：原始像素只去畸变一次，得到理想像素；用标定的 `R/t` 构造
`P = K [R|t]`，由 `R_10/T_10` 计算基础矩阵做极线筛选；`cv2.triangulatePoints` 直接得到
**世界系**米制坐标（不再重复施加世界变换）；检查两个相机前方、重投影误差；最后枚举
3 点组合，要求三边与 33 mm 的偏差在容差内，并优先选择边缘误差小/离散度小、两相机
三点绕向一致的组合。输出三边长度/误差、各 LED 重投影误差、极线误差和三角形评分。
候选对先按极线/重投影过滤，再做三组合枚举（最多 64 个候选对），不会对 24 个候选
做无约束的笛卡尔积三角化。

**相机映射（关键）**：两台 LRCP G720P 名称完全相同，USB 枚举顺序可能在重启/换口后
互换。若把实时 cam0 的图配到标定 cam1 的 K/D/R/t，极线距离会达到几十~几百像素，
三角化落到相机后方，表现就是“各有 3 个候选但 0 个匹配”。LED 服务现在同时构造
`live0→calib0/live1→calib1` 和反向映射，对两种映射分别做完整极线/三角化，按
“正深度 + 重投影通过 + 33 mm 三边”选择并锁定；状态、`/api/led/diagnose` 和调试 JSON
都会显示当前映射。映射是设备枚举顺序问题，不改标定文件本身。

**等边三角形**：三个 LED 完全相同，120° 对称无法仅凭几何消除；输出的 L0/L1/L2 只是
**临时显示编号**，利用时间连续性保持稳定，`labels_are_temporary=true`。本版本不输出
唯一航向或完整 6DoF。

**跟踪**：三维中心进入卡尔曼滤波；使用 `rec.created_mono`，不假定固定帧率。正常测量为
`tracking`；短时漏检/几何临时不一致为 `predicted`（明确标记“无当前测量”）；超过
0.6 s / 8 帧为 `lost` 并清除当前点，不把旧位置当作有效测量。

**二维可视化**：复用标定面板已有的两路图像。可视化模式新增“红外 LED 检测可视化”，
显示方式可选最终检测叠加/二值化/原始灰度/去畸变+候选极线诊断；原始连通域为小蓝色十字、
通过筛选的候选为黄色空心圆、成功匹配为绿色实心点并只给这 3 个点写 L0/L1/L2、
被拒绝亮区为细红框。这样不会再用 12 个黄色文字圆圈互相遮挡。支持 1×/2×/4× 局部放大
（以匹配点或 ROI 为中心）；相机编号/原始候选数/筛选候选数/匹配数/状态/ROI/帧对/Δt
固定显示在**右下角**，按实际输出尺寸和文字宽度动态右对齐，并带半透明深色背景；左下角
保留颜色图例，两者不会重叠。原始灰度、二值化、最终检测和极线视图布局一致，放大后也
按输出尺寸重新计算。只有该模式
请求 `/api/led/overlay_pair`（一次返回同一 `pair_seq` 的两台相机图）或单路 `/api/led/overlay`，
不会同时执行棋盘格检测；这样不会再出现 cam0 显示 pair 9464、cam1 显示 9466 的显示错位。

**极线视图不依赖 matches**：只要两台相机有候选点且内外参有效，就把另一台相机每个候选点
对应的极线画到当前去畸变图像上；每条极线/来源候选点使用同一种颜色，当前候选点按自身颜色
标记。左上角距离表列出每个候选点到所有极线的**最小距离**，绿色 `OK` 表示 ≤ 当前极线阈值，
橙色 `FAIL` 表示超阈值；`matched=0` 时也能直接看到候选关系为什么没配上。直线裁剪同时处理
水平、倾斜和接近竖直的极线，不会因为 `b≈0` 缺失。该视图只用可视化数据，不改变极线/三角化/
重投影/33 mm 的筛选条件。
定位运行时可点“匹配诊断”查看原始/去畸变像素、3×3 极线距离、各阶段通过数和最佳三边；
点“保存调试帧”把两台相机的原始 PNG、二值 PNG、检测叠加 JPG、`detections.json`、
`diagnosis.json`、`state.json` 写到 `captures/led_debug/<时间戳>/`，用于离线复现。

**三维可视化**：复用现有 Z 向上 Canvas 场景，增加三个 LED 点、三角形连线、几何中心、
中心 X/Y/Z 标签、历史轨迹和有效性状态；鼠标视角状态不会被定位刷新重置。

**启动条件**：相机 0/1 必须 `streaming`；二维检测需先启动；三维定位还要求两台内参、
世界系外参存在、`orientation_confirmed=true`，且当前分辨率与内参图像尺寸一致。标定在
运行中变化时，服务会重置跟踪并重新加载；不满足条件则停止三维定位并显示具体原因。

**接口**（返回 JSON，叠加图为 JPEG）：

```text
GET  /api/led/state          当前检测/定位状态、LED、三边、中心、速度、轨迹、ROI、相机映射
GET  /api/led/params         默认值/共用值/相机覆盖/有效参数/ROI/保存路径
POST /api/led/params         保存参数（body: {params, overrides, rois, roi_tracking, camera_map}）
POST /api/led/params/reset   恢复默认（含 ROI）
POST /api/led/detect/start|stop
POST /api/led/locate/start|stop
GET  /api/led/overlay?camera=0&view=detect|binary|raw|epipolar&zoom=1|2|4
GET  /api/led/overlay_pair   同一 pair_seq 的双相机叠加图（JSON + base64 JPEG）
GET  /api/led/diagnose       分阶段匹配诊断（两种相机映射）
POST /api/led/debug/save     保存原始图/二值图/检测/诊断 JSON
```

**曝光/增益**：本机 LRCP G720P 是标准 UVC 设备，DirectShow 的 `IAMCameraControl` /
`IAMVideoProcAmp` 实测可读可写，而且**不需要启动第二路取流**：创建 filter、
QueryInterface 后直接 GetRange/Get/Set，Media Foundation 的 SourceReader 继续推流
（已验证设置后 `streaming` 不中断）。控件已从相机预览卡片移到“红外 LED 检测与定位”模块内，两台相机独立：
曝光范围 **-13..-1**（log2 秒，约 0.12–500 ms，步长 1，支持自动曝光），
增益 **0..63**（设备只支持手动）。每次设置都读回设备值，失败会显示错误，不会假装
成功。`GET/POST /api/cameras/controls` 返回范围、当前值、自动/手动能力和读回结果；成功设置
按 USB 设备实例路径保存到 `captures/calibration/camera_controls.json`，相机重新打开或服务
重启时自动恢复并读回验证（暂停/自动模式也会恢复），不再每 5 秒轮询设备属性。
原作者的 `helpers.py` / pseyepy 通路已从运行代码删除，源码保留在 `upstream/`。

### 6.5 网页版本与刷新

- `GET /api/version` 返回当前进程的 Git 短 SHA、分支、dirty 状态、启动时间和 PID；
  页面顶部 `#build` 显示该版本，不再写死字符串。
- 页面每 2 s 检查一次版本；若 `git_sha` 或 `started_at` 变化（服务重启/更新），显示
  “服务已更新，请刷新页面”与刷新按钮，**不会自动刷新**。
- API 始终同源，网页只由 3002 提供；不再保留旧 3001 后端或端口回退。
- `/preview`、`/api/version`、`/api/led/state`、`/api/led/overlay_pair` 使用 `no-store`；
  标定面板默认展开、页面初始化自动复用/创建会话；LED 状态 4 Hz，后台标签页降频；
  同步帧/参数变化才重新处理，避免重复三角化。

LED 专项自检：`python -m led.selftest`，**50/50 通过**（合成光斑、粘连分离、过曝拒绝、
细长反光过滤、ROI 坐标偏移、极线匹配、33 mm 三角化、缺失/几何不一致、绕向一致性、
自动相机映射、卡尔曼预测/丢失、服务帧去重、同 pair 叠加图、matched=0 极线、水平/垂直
极线裁剪、局部放大、调试帧保存、参数/ROI/单相机覆盖）。
标定自检：**139/139 通过**（包含 LED UI/参数接口与版本接口检查）。

---

## 7. 输出

### 7.1 会话目录

```
captures/calibration/
  intrinsics.json             最新有效内参（跨会话，两台相机合并保存）
  world_extrinsics.json       最新有效世界系外参（跨会话，含方向确认状态）
  led_params.json             红外 LED 检测参数（共用 + 相机覆盖 + ROI + 相机映射）
  camera_controls.json        每台 USB 设备上次成功设置的曝光/自动曝光/增益
  <会话>/
    session.json              板规格（含 world_frame）、相机、分辨率、坐标约定
    samples.json              每条样本：phase、角点、质量、有效性、失败原因、Δt
    img/<id>_cam0.png         原始灰度（未旋转未缩放未去畸变）
    overlay/<id>_cam0.png     角点叠加图
    result_intrinsic.json     内参中间结果（含 loaded_from 时可区分历史载入）
    result_extrinsic.json     solvePnP 外参中间结果（含逐帧抖动）
    result_validation.json    独立验证结果
    calibration.json          完整结果副本（save / CLI save 按需写入）

captures/led_debug/<时间戳>/   “保存调试帧”输出
  cam0_raw.png / cam1_raw.png          原始灰度
  cam0_binary.png / cam1_binary.png    当前参数下的二值图
  cam0_detect.jpg / cam1_detect.jpg    检测叠加图
  detections.json                      检测候选/拒绝原因/参数/ROI/帧对
  diagnosis.json                       两种相机映射的原始/理想像素、极线矩阵、阶段计数
  state.json                           服务完整状态（跟踪、三边、轨迹）
```

`intrinsics.json` 按相机分别保存 K、D、RMS、图像尺寸、有效样本数、覆盖比例、相机标识、
计算时间；计算失败的相机不会覆盖其中已有的有效条目。`world_extrinsics.json` 保存两台相机
的 R/t、三维位置、离地高度、`R_10/T_10`、基线、重投影、棋盘格/世界系定义、内参快照和
`orientation_confirmed`；未确认方向的结果也会保存，但不会被当作已确认配置。

`samples.json` 里 `phase` 取值：`intrinsic`（板移动）/ `extrinsic`（板固定）/ `validation`（板换位置）。

**路径约定**：数据基准目录是 `computer_code/`。默认会话根目录 `captures/calibration/`；
完整合并结果写在该会话目录的 `calibration.json`。界面与结果文件里的 `session.root` 使用项目
相对路径；内部读写时再解析成绝对路径，因此不依赖启动进程时的当前工作目录。旧会话里保存
的绝对路径仍可直接读取，历史采集目录不会被覆盖。

### 7.2 会话目录 `calibration.json`（schema v2）

顶层：`schema` / `version` / `created` / `session` / `cameras` / `board` / `world_frame` /
`image_convention` / `intrinsics` / `extrinsics` / `validation` / `data` / `caveats`。

`extrinsics` 里：

| 字段 | 含义 |
|---|---|
| `method` | `solvePnP (IPPE candidates + LM refine) per camera against one fixed board` |
| `world_frame` | 世界系定义（原点、三轴、手性、板面高、地面 Z） |
| `flip_cam` | 每台相机的 180° 开关 `{"0":bool,"1":bool}` |
| `orientation_confirmed` | 是否已人工用世界坐标轴可视化确认方向（几何无法自动判定） |
| `cameras.{0,1}.R_world_to_cam` / `t_world_to_cam_m` | 世界 → 相机 |
| `cameras.{0,1}.R_cam_to_world` / `t_cam_to_world_m` | 相机 → 世界 |
| `cameras.{0,1}.camera_centre_world_m` | 相机光心世界坐标 |
| `cameras.{0,1}.height_above_floor_m` | 离地高度（已加板面高） |
| `cameras.{0,1}.optical_axis_world` | 光轴方向 |
| `cameras.{0,1}.correspondence` | 自动选定的格点对应类别 |
| `R_10` / `T_10` / `T_unit` / `baseline_m` | 相机 0 → 1 相对位姿（兼容字段） |
| `consistency` | 三角化角点与已知世界点的三维偏差 |
| `stability` | 同一静止板多帧之间的位姿抖动 |

`calibration.json` 是**当前会话的完整结果副本**，由 `/api/calib/save` 或 CLI `save`
按需写入；标定流程本身不再自动生成任何全局兼容文件。旧版
`api/calibration/calibration.json` 的写入逻辑和默认回退读取均已移除。

下游取参数：

```python
from calibration.service import load_calibration, stereo_matrices
data = load_calibration("captures/calibration/<会话>/calibration.json")
K0, D0, K1, D1, R, T = stereo_matrices(data)    # 或 scaled_size=(640, 360)
```

`load_calibration(path)` 必须显式给路径；或调用
`GET /api/calib/matrices?size=640x360`（默认读取当前会话已保存的结果，也可用
`?path=` 指定文件）。

旧的 `camera-params.json`（320×320 占位参数）**保持原样未改**，新模块不读它。

---

## 8. 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/calib/status` | 会话、样本计数（按 phase）、内外参状态、相机位姿摘要 |
| POST | `/api/calib/session` | 新建/打开会话（`cols/rows/square_mm/surface_height_mm`） |
| POST | `/api/calib/session/ensure` | 页面初始化幂等会话：已有复用，没有才创建（多标签页串行化） |
| POST | `/api/calib/capture/single` | 采一张单相机图（默认 intrinsic，判重 + 质量门限） |
| POST | `/api/calib/capture/extrinsic` | 采一张外参图（单相机兼容接口，固定板不判重） |
| POST | `/api/calib/capture/extrinsic-pair` | **采集双相机图片**：一对同步帧分别写入两台相机的 `extrinsic` 阶段 |
| POST | `/api/calib/capture/stereo` | 采一对同步帧（validation 用） |
| GET | `/api/calib/live?camera=0` | 角点检测可视化（JPEG） |
| GET | `/api/calib/extrinsic/overlay?camera=0&flip180=0` | **世界坐标轴可视化**（确认角点对应） |
| POST | `/api/calib/intrinsic` | 内参标定（含每相机区域覆盖率，成功后自动保存最新内参） |
| POST | `/api/calib/intrinsic/load` | **载入内参**：从 `intrinsics.json` 恢复 K/D，不重采/重算 |
| POST | `/api/calib/solve/extrinsic` | **外参标定（solvePnP）**，成功后自动保存最新外参 |
| POST | `/api/calib/extrinsic/load` | **载入世界坐标系外参**：校验后恢复三维位姿/基线 |
| POST | `/api/calib/extrinsic/invalidate` | 修改 180° 后作废旧外参与方向确认 |
| POST | `/api/calib/validate/rig` | 重新计算验证（复用已保存数据） |
| POST | `/api/calib/validate/capture` | **采集并验证三维精度**（采一对新帧 + 自动验证） |
| POST | `/api/calib/stereo` | stereoCalibrate 交叉校验（保留，网页不再入口） |
| POST | `/api/calib/save` | 写 `calibration.json` |
| GET | `/api/calib/result` · `/api/calib/matrices` | 查看结果 / 取矩阵 |
| GET | `/api/version` | 真实 Git 短 SHA、分支、dirty、启动时间 |
| GET/POST | `/api/cameras/controls` | 读/写 UVC 曝光、自动曝光、增益；按设备路径保存/恢复（见 6.4） |
| POST | `/api/cameras/controls/restore` | 手动恢复每台设备上次保存的曝光/增益 |
| GET | `/api/led/overlay_pair` · `/api/led/diagnose` | 同一帧对双相机叠加图 / 分阶段匹配诊断 |
| * | `/api/led/*` | 红外检测/定位状态、参数、启停、叠加图、调试保存（见 6.4） |

---

## 9. 自检（无相机也能跑）

```bat
cd Low-Cost-Mocap-main\computer_code\api
python -m calibration.selftest            :: 完整
python -m calibration.selftest --quick    :: 快速
python -m calibration.selftest --keep     :: 保留渲染图
```

用已知 `K/D/R/T` 按**真实畸变模型**逐像素反向渲染虚拟棋盘格（11×8 / 30 mm），再走正式代码
检测与求解，最后与真值比对。

最近一次标定自检：**139/139 通过**（`--quick` 与完整模式均通过）；LED 专项自检
`python -m led.selftest`：**50/50 通过**。合成数据（两台相机距板心
~1.8 m、基线 2032.78 mm）的代表性量级：

| 项目 | 结果 |
|---|---|
| 内参 fx 相对误差 | ~0.13% |
| 内参 RMS | ~0.07 px |
| solvePnP 重投影 | ~0.09 px |
| 三角化自洽（对已知世界点） | rms ~0.23 mm，max ~0.62 mm |
| 验证集网格间距误差（已修正为 mm） | mean ~0.17 mm，max ~1.1 mm |
| 验证集重投影 / 平面度 | ~0.06 px / ~0.18 mm |

除求解精度外，还覆盖了这些**流程**用例（全部用合成数据/假相机，不接硬件）：

- 代表位姿取重投影最好的那一帧，且一致性检查用**同一帧**的角点（防错配）
- IPPE 候选带物理合理性标记；选中解满足"相机在板面上方 + 角点全在前方"
- 自动格点对应不按误差挑选，仍报 `orientation_ambiguous`
- CLI `save` 端到端跑通（防重复定义/参数缺失）
- 重载会话能恢复 `result_intrinsic/extrinsic`；新增采集后旧外参被作废
- HTTP 采集（`/api/calib/capture/*`）用假相机验证；同一帧、同一同步帧对会被拒绝
- 内参区域覆盖率只统计本相机 `intrinsic` 样本，外参图不会混入；两台相机独立
- 一次同步采集把两张图分别写入相机 0/1 的 `extrinsic`，各 1 张即可 solvePnP；
  单帧时 `centre_std_m` 为 `None`
- 固定板外参/验证使用 0.1% 覆盖率下限，0.15% 的图被接受并带告警；内参仍要求 0.8%
- 「采集并验证三维精度」自动入库并计算，验证计数与外参/内参计数互不混算
- 汇总间距误差与逐位置结果单位一致（mm）；旧结果的米→毫米迁移幂等
- 网页包含三种可视化模式、三维视图/重置/俯视入口，旧的双目标定/覆盖检查/分散采集
  入口已移除
- 内参计算后自动写 `intrinsics.json`；新会话零样本执行 `load_intrinsics()` 可恢复两台 K/D，
  并标记 `source="loaded"` / `loaded_from`
- 无论是否确认方向都不再写全局兼容文件；确认后自动写 `world_extrinsics.json`，
  会话 `calibration.json` 可由 `save_result` / CLI `save` 生成，下游
  `load_calibration(path)`/`stereo_matrices`/`GET /api/calib/matrices` 正常
- 新会话载入内外参后基线/方向确认状态恢复；不兼容棋盘格、不兼容内参或缺失文件会被拒绝，
  且不清除当前有效结果
- 最新内参/外参文件写入失败时仍返回计算结果，并报告 `latest_error`/`save_errors`
- 修改 180° 后旧外参与方向确认被作废
- 网页已移除“查看结果 / 保存 calibration.json”，新增“载入内参 / 载入世界坐标系外参”
- LED 专项：分离三点、粘连两点/三点拆分、完全饱和不伪造、亚像素中心
- LED 专项：细长反光过滤、原始连通域保留、ROI 内检测坐标加回整图偏移
- LED 专项：极线匹配、世界系三角化、33 mm 三边、缺失检测、几何不一致、绕向一致
- LED 专项：卡尔曼匀速速度/位置、短时预测、超时丢失
- LED 服务：同一 `pair_seq` 不重复处理、新帧处理、二维叠加图、局部放大、调试帧保存
- LED 参数：共用参数/单相机分离覆盖/ROI/动态跟踪可保存并恢复默认
- 版本接口返回 Git 版本；页面含版本横幅/刷新按钮；前后端同源且无旧 3001 回退

**这不是实机精度**：合成图没有镜头杂散光、板面不平、运动模糊、滚动快门和 USB 传输抖动。
实机必须自己采数据重跑。

---

## 10. 局限（必须如实转述）

1. **软件级同步**：帧对来自主机单调时钟最近邻配对，时间戳是帧交付时刻而非曝光中心。
   对静止标定板无影响，但**不能**声称硬件同步，也不能用于高速运动目标的亚毫秒三角化。
2. **画面四角的畸变是外推**：11×8、30 mm 的板在 1280×720 里相对较大，很难铺满四角。
   内参 RMS 好看不代表边缘可信，请看 `radius_coverage` 告警。若需要边缘精度，
   建议内参单独用一块更小格距的板。
3. **D2 对称**：整体 180° 只能人工确认（第 6 节）。
4. **常规针孔模型**：鱼眼镜头需换 `cv2.fisheye`，本模块不自动切换。
5. **世界系锚在板的那一次摆放上**：板挪走后世界系不变（这正是本意），所以板所在平面
   必须与你声明的 `surface_height_m` 一致，否则所有高度会整体偏移。
6. **三角化精度依赖基线/距离**：基线越短、目标越远，深度误差越大。验证请覆盖实际工作距离。
7. **三维视图是检查工具，不是测量结果**：网页里的三维场景由 Canvas 2D 透视投影绘制
   （不依赖联网或额外前端工程），相机位置/朝向/视锥体来自已求解的外参和内参，用于检查
   方向与几何关系；数值精度仍以 `calibration.json` 和验证指标为准。
8. **板面高是声明值**：默认 3 mm 只描述图案表面高于地面的高度，世界原点仍在图案表面；
   如果实际板面高与填写值不符，基于地面的高度会整体偏移。
9. **最新文件是“最近一次成功求解”**：`world_extrinsics.json` 会被下一次 solvePnP 覆盖
   （包括待确认结果）；`intrinsics.json` 按相机合并，某台失败不会覆盖该相机旧值。
   完整 `calibration.json` 只存在于会话目录，按需保存，不再维护全局兼容副本。
10. **载入外参需要匹配当前内参与板规格**：文件里保存了内参快照和棋盘格/世界系定义；
    重新计算内参、改板面高或换会话规格后，旧外参文件会被判为不兼容，必须重新 solvePnP。
11. **LED 等边三角形 120° 对称未解**：L0/L1/L2 只是临时显示编号；本版本不输出唯一
    航向或完整 6DoF，也不使用原作者的 droneIndex 规则。
12. **LED 检测参数需要按现场调整**：阈值、面积、峰值间距、模糊核与 LED 亮度/距离相关；
    光斑完全饱和或粘连到无法分出峰值时只会提示，不会伪造三点。1–2.5 m 是设计工作段，
    更远距离的质心噪声和误匹配风险会显著增加。
13. **曝光/增益的边界**：当前通过 DirectShow UVC 控制（而非 Media Foundation 属性）
    可以实时设置曝光/自动曝光/增益，已实测不打断 MF 取流；但不同 UVC 固件的范围、
    自动曝光行为和读回延迟可能不同。页面显示的 range 来自设备 `GetRange`，不是硬编码；
    若某台相机没有 `IAMCameraControl/IAMVideoProcAmp`，会明确显示“不支持”，不会出现
    无效滑块。增益在本机只支持手动，自动增益不可用。
14. **运动目标仍受软件同步限制**：LED 定位使用同步帧对，但两机曝光时刻仍有半帧级
    不确定度；静止或低速目标可用，高速运动必须另行评估或做硬件同步。
15. **相机枚举顺序不是身份**：两台相机同名，重启/换 USB 口后实时 0/1 可能与标定
    0/1 互换。LED 服务会自动判别并锁定映射；如果两路都没有足够正深度/重投影候选，
    会保留上一映射并报 `lost/insufficient_detections`，不会在两个错误编号间跳变。
16. **同步统计分“累计”和“最近 300 事件”**：启动/相机重开、DirectShow 控制刚设置后
    可能有短暂丢帧，累计值会保留历史；实时状态请看 `recent` 和每路帧间隔。
    本机帧间隔均值约 33.3 ms、P95 可到 ~48 ms（USB 调度抖动），20 ms 配对阈值下
    偶发丢帧属于流式配对行为，不代表曝光设置失败；不要靠清零统计或放宽阈值掩盖。
17. **自动会话不会清空结果**：页面初始化调用 `/api/calib/session/ensure`，已有会话
    直接复用；没有才创建。服务重启后内存里的会话/结果会重新建立，最新内外参仍在
    `intrinsics.json` / `world_extrinsics.json`，需要点“载入内参 / 载入世界坐标系外参”
    恢复；载入外参只更新内存与最新外参文件，不再生成全局兼容文件。

---

## 11. 与原作者旧动捕代码的差异（历史记录）

原作者基于 pseyepy / PS3 Eye、Flask-SocketIO 3001 和 ESP32 的链路已从本仓库运行代码中删除，
完整源码见 `upstream/`。下表路径均指 upstream 原项目，只用于解释为什么新标定结果不能直接
套进旧图像流水线：

| upstream 位置 | 问题 |
|---|---|
| `helpers.py:79` | `np.rot90(frames[i], k=rotation)` 旋转图像，新 K 不适用 |
| `helpers.py:80` | `make_square()` 补边成方图，**主点被平移但 K 没改** |
| `helpers.py:81` | `cv.undistort()` 用旧 320×320 占位参数，对 1280×720 无效 |
| `helpers.py:82` | 去畸变后 `GaussianBlur`，会改变角点几何（标定链路没这一步） |
| `helpers.py:238-244` | `projectPoints(..., np.array([]))` 忽略畸变，仅在输入图已去畸变时自洽 |
| `helpers.py:314` / `361` | `P = intrinsic_matrix @ RT` 用旧 K，与当前分辨率不匹配 |
| `index.py:274` | `cv.sfm.essentialFromFundamental` 需 opencv-contrib，当前环境 `has sfm: False` |
| `index.py:266-297` | 旧外参对每对相机都取 `get_camera_params(0/1)`，四相机会用错内参 |

结论：下游请以当前会话 `calibration.json` 为唯一来源，并遵守“原始 1280×720 未旋转、未补边、
未去畸变”的约定；不要复用 upstream 旧流水线。
