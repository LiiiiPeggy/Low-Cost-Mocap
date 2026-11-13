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
| 标定板 | **12×9 个方格 → 11×8 内角点**，单格 **30 mm** |
| 图像 | 1280×720，NV12@30（MJPG 同样可用） |
| 环境 | conda `lowcost_mocap`：Python 3.10.22 / OpenCV 4.10.0 / NumPy 1.26.4 |

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
conda activate lowcost_mocap
cd Low-Cost-Mocap-main\computer_code
python api\preview_server.py --indices 0 1
:: 浏览器打开 http://127.0.0.1:3002/preview → 右上角「标定面板」
```

面板里 5 步：

1. **新建 / 打开会话**：内角点 `11 × 8`、方格 `30` mm、板面高按实际填（贴地就填 0）。
2. **内参**：让板在**两台相机前移动**，各采 30–40 张（角度/距离/位置都要变，尽量铺到画面四角）
   → 「内参标定」。看 RMS 与「角点半径覆盖」告警。
3. **外参**：把板**固定在中心地面** → 「采外参图 相机0 / 相机1」各拍 5–10 张（静止板，多拍
   几张用来看位姿抖动）→ 勾选「世界轴叠加」→「解外参 solvePnP」。
4. **确认 180°**：打开「世界轴叠加」，图上会画出世界原点、X（红）/Y（绿）/Z（蓝）轴和
   角点编号，0 号角点有红圈。**看 X 轴是否沿实物 12 格长边**；如果反了，勾「全局翻转 180°」
   再看一次。这一步只能人眼判断，原因见第 6 节。
5. **独立验证**：把板挪到**其它位置和高度**（至少 3 处），每处用「采集同步帧对」拍一对
   → 「独立验证（多位置）」。看网格间距误差、平面度、极差。

最后「保存 calibration.json」写 `api/calibration/calibration.json`。

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
建议剔除样本、角点半径覆盖。

**为什么强调半径覆盖**：角点只集中在画面中心时，`k2/k3` 会互相补偿，重投影 RMS 仍可低到
0.07px，但把模型用到画面四角会出现明显偏差（合成实验里 20px → 补采四角后 1.6px）。
`radius_coverage.ratio < 0.75` 时会显式告警。

### 4.2 外参（solvePnP）

对每台相机：`solvePnPGeneric(SOLVEPNP_IPPE)` 得到两个候选，用
`cv2.solvePnPRefineLM` 精化，输出 `X_cam = R @ X_world + t`。

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

**重投影误差小不等于三维精度高**。以间距误差与平面度为准，而且必须是独立数据。

---

## 5. 命令行

在 `Low-Cost-Mocap-main/computer_code` 下：

```bat
python api\calib_cli.py new --cols 11 --rows 8 --square-mm 30 --surface-height-mm 0
python api\calib_cli.py capture-single --camera 0 --count 30 --interval 1.5
python api\calib_cli.py capture-single --camera 1 --count 30 --interval 1.5
:: 板固定到中心地面后：
python api\calib_cli.py capture-single --camera 0 --count 6 --phase extrinsic
python api\calib_cli.py capture-single --camera 1 --count 6 --phase extrinsic
:: 板挪到别处后采集验证帧对：
python api\calib_cli.py capture-stereo --count 3

:: 以下不需要相机在线
python api\calib_cli.py intrinsic     --session captures\calibration\<会话>
python api\calib_cli.py extrinsic    --session captures\calibration\<会话> [--flip180]
python api\calib_cli.py validate-rig --session captures\calibration\<会话>
python api\calib_cli.py save         --session captures\calibration\<会话> --validate
python api\calib_cli.py info
```

`stereo` 子命令保留为 `stereoCalibrate` 交叉校验（不是主路线）。

---

## 6. 角点对应关系：D2 四重对称（必须知道）

矩形角点阵列的对称群是 **D2**，有 4 个元素：恒等、绕板法向 180°、沿行镜像、沿列镜像。
这 4 种对应在**纯几何上完全不可区分**——任何一种重排都能被某个合法位姿精确解释，
重投影误差逐位相同（实测 0.085100 px 对 0.085100 px）。

因此：

1. 两台相机从不同方向拍同一块板时，检测到的**0 号角点不一定是同一个物理角点**，
   不能默认对应。
2. 本模块只用「相机必须在图案正面一侧（世界 Z > 0）」筛掉两个镜像解，剩下一对相差 180°。
3. 剩下的 180° **必须人工确认**：把世界轴投影回图像，看 X 轴是否沿 12 格长边。
4. 180° 是**全局**开关，**必须两台一起翻**。只翻一台会让两台的世界系差 180°，
   相对位姿与三角化全错（实测基线会从 2032 mm 跳到 1635 mm）。

更严谨的做法是在板上加非对称标记（ArUco / 缺角标记），本次未做。

---

## 7. 输出

### 7.1 会话目录

```
captures/calibration/<会话>/
  session.json                板规格（含 world_frame）、相机、分辨率、坐标约定
  samples.json                每条样本：phase、角点、质量、有效性、失败原因、Δt
  img/<id>_cam0.png           原始灰度（未旋转未缩放未去畸变）
  overlay/<id>_cam0.png       角点叠加图
  result_intrinsic.json       内参中间结果
  result_extrinsic.json       solvePnP 外参中间结果（含逐帧抖动）
  result_validation.json      独立验证结果
  calibration.json            最终结果副本
```

`samples.json` 里 `phase` 取值：`intrinsic`（板移动）/ `extrinsic`（板固定）/ `validation`（板换位置）。

### 7.2 `api/calibration/calibration.json`（schema v2）

顶层：`schema` / `version` / `created` / `session` / `cameras` / `board` / `world_frame` /
`image_convention` / `intrinsics` / `extrinsics` / `validation` / `data` / `caveats`。

`extrinsics` 里：

| 字段 | 含义 |
|---|---|
| `method` | `solvePnP (IPPE candidates + LM refine) per camera against one fixed board` |
| `world_frame` | 世界系定义（原点、三轴、手性、板面高、地面 Z） |
| `flip180_global` | 人工确认的全局 180° |
| `cameras.{0,1}.R_world_to_cam` / `t_world_to_cam_m` | 世界 → 相机 |
| `cameras.{0,1}.R_cam_to_world` / `t_cam_to_world_m` | 相机 → 世界 |
| `cameras.{0,1}.camera_centre_world_m` | 相机光心世界坐标 |
| `cameras.{0,1}.height_above_floor_m` | 离地高度（已加板面高） |
| `cameras.{0,1}.optical_axis_world` | 光轴方向 |
| `cameras.{0,1}.correspondence` | 自动选定的格点对应类别 |
| `R_10` / `T_10` / `T_unit` / `baseline_m` | 相机 0 → 1 相对位姿（兼容字段） |
| `consistency` | 三角化角点与已知世界点的三维偏差 |
| `stability` | 同一静止板多帧之间的位姿抖动 |

下游取参数：

```python
from calibration.service import load_calibration, stereo_matrices
data = load_calibration()                       # 默认 api/calibration/calibration.json
K0, D0, K1, D1, R, T = stereo_matrices(data)    # 或 scaled_size=(640, 360)
```

或 `GET /api/calib/matrices?size=640x360`。

旧的 `camera-params.json`（320×320 占位参数）**保持原样未改**，新模块不读它。

---

## 8. 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/calib/status` | 会话、样本计数（按 phase）、内外参状态、相机位姿摘要 |
| POST | `/api/calib/session` | 新建/打开会话（`cols/rows/square_mm/surface_height_mm`） |
| POST | `/api/calib/capture/single` | 采一张内参图（判重 + 质量门限） |
| POST | `/api/calib/capture/extrinsic` | 采一张外参图（固定板，不判重） |
| POST | `/api/calib/capture/stereo` | 采一对同步帧（validation 用） |
| GET | `/api/calib/live?camera=0` | 实时检测叠加（JPEG） |
| GET | `/api/calib/extrinsic/overlay?camera=0&flip180=0` | **世界轴叠加**（确认角点对应） |
| POST | `/api/calib/intrinsic` | 内参标定 |
| POST | `/api/calib/solve/extrinsic` | **外参标定（solvePnP）** |
| POST | `/api/calib/validate/rig` | 独立验证（多位置/高度） |
| POST | `/api/calib/stereo` | stereoCalibrate 交叉校验 |
| POST | `/api/calib/save` | 写 `calibration.json` |
| GET | `/api/calib/result` · `/api/calib/matrices` | 查看结果 / 取矩阵 |

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

最近一次：**56/56 通过**。合成数据（两台相机距板心 ~1.8 m、基线 2032.78 mm）：

| 项目 | 结果 |
|---|---|
| 内参 fx 相对误差 | 0.13% |
| 内参 RMS | 0.07 px |
| solvePnP 重投影 | 0.096 / 0.085 px |
| 相机到板心距离误差 | 1.8 / 0.5 mm |
| 基线 | 2031.89 mm（误差 0.89 mm） |
| 三角化自洽（对已知世界点） | rms 0.227 mm，max 0.558 mm |
| 验证集网格间距误差 | mean 0.000 mm，max 0.001 mm |
| 验证集重投影 / 平面度 | 0.070 px / 0.227 mm |

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

---

## 11. 与旧动捕代码的差异（本次不改，但下游必须知道）

`api/helpers.py` 的 `Cameras` 是原项目（pseyepy + PS3 Eye）通路，与本模块互不依赖。
若要把标定结果接进旧动捕算法，以下位置必须适配，否则 K 与图像对不上：

| 位置 | 问题 |
|---|---|
| `helpers.py:79` | `np.rot90(frames[i], k=rotation)` 旋转图像，新 K 不适用 |
| `helpers.py:80` | `make_square()` 补边成方图，**主点被平移但 K 没改** |
| `helpers.py:81` | `cv.undistort()` 用 `camera-params.json` 的 320×320 占位参数，对 1280×720 无效 |
| `helpers.py:82` | 去畸变后 `GaussianBlur`，会改变角点几何（标定链路没这一步） |
| `helpers.py:238-244` | `projectPoints(..., np.array([]))` 忽略畸变，仅在输入图已去畸变时自洽 |
| `helpers.py:314` / `361` | `P = intrinsic_matrix @ RT` 用旧 K，与当前分辨率不匹配 |
| `index.py:274` | `cv.sfm.essentialFromFundamental` 需 opencv-contrib，当前环境 `has sfm: False` |
| `index.py:266-297` | 旧外参对每对相机都取 `get_camera_params(0/1)`，四相机会用错内参 |

结论：下游请以 `calibration.json` 为唯一来源，并遵守"原始 1280×720 未旋转、未补边、
未去畸变"的约定；不要在旧链路的图像上直接套新 K。
