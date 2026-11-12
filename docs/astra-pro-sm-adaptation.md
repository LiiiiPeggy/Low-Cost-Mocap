# Astra Pro SM 接入适配要点

> 本文是**部署侧工作笔记**，不是仓库内容总结。仓库总结见 `low-cost-mocap-summary.md`。
> 事实来源分三类：仓库源码（已核对行号）、厂商/SDK 官方文档、第三方资料（已标注可信度）。
> 仓库本身**没有任何 Astra、奥比中光、深度相机或 RK3588 相关信息**。

## 1. 型号确认与可信度

- 用户所述「Astra Pro S M」应为连写的 **Astra Pro SM**：Astra Pro 的衍生精简版，深圳蚂里奥技术有限公司有同名 Datasheet。
- 第三方技术博客（hqwc.cn、wmrh.cn）称 SM **砍掉 RGB 传感器，OpenNI2 下只剩 `SENSOR_DEPTH` 与 `SENSOR_IR`**，与 Astra Pro 接口一致、代码无需区分型号。**此项来自第三方博客，非官方规格书，需用 SDK 枚举设备名和可用流确认。**
- 该博客另称 SM 对黑色/镜面物体做了优化，此说法未获官方资料佐证，不作为依据。
- 奥比官网当前产品线为 Astra / Astra S / Astra Pro Plus / Astra Mini Pro 系列；`Astra Pro SM` 不在现行官网列表，属较老型号。

## 2. 结论

**可以用，但不能直接替换。** 相机是被动接收近红外、自带窄带滤光片，这一项比改装 PS3 Eye 更省事；但它是**主动结构光**设备，自带激光散斑投射器，与项目「画面里只能剩下几个亮点」的硬约束正面冲突。需改动采集层、图像格式、内参、循环节拍，以及 3 处 `cv2.sfm` 调用。

**好消息**：源码只用了 `cv2.sfm` 的 3 个函数，全部可用常规 OpenCV + numpy 等价实现，**不再需要源码编译 OpenCV + opencv_contrib**，conda 环境可一次装齐（见 `environment-astra.yml`）。

## 3. 规格与仓库假设的差距

| 项 | 仓库现有假设（PS3 Eye） | Astra Pro SM | 影响 |
| --- | --- | --- | --- |
| 采集接口 | `pseyepy`（PS3 Eye 专用） | USB 2.0 + **OpenNI 协议**（非 UVC） | 不能用 `cv2.VideoCapture` 直接读，采集层需重写 |
| 图像 | 320×240、RGB 三通道、8-bit | IR 流 640×480@30fps（也有 320×240），**单通道灰度，Y16/8UC1** | `make_square`、`cvtColor`、阈值都按 3 通道 8-bit 写死 |
| 帧率 | 90 FPS | 30 FPS（IR 与 depth 一致） | 后端 150 Hz 视频流循环需改；帧间隔从约 11 ms 增至约 33 ms |
| 内参 | `fx=fy=320`、`cx=cy=160`（320×240 中心） | 需按实际 IR 分辨率重标定 | `camera-params.json` 4 组参数全部失效 |
| 快门同步 | 无（PS3 Eye 有 3.5 mm 口但作者未启用） | 官方 Astra 系列规格表 `Trigger: N/A` | 上一轮确认的「无软/硬同步」问题依旧，且 30 FPS 下更明显 |
| 光源 | 需拆 IR cut + 外贴可见光截止片；目标用**主动 IR LED**，相机侧补光灯非必需 | **自带窄带滤光片**（省一步），但**自带激光散斑投射器**（必须关掉的干扰源） | 见第 4 节 |

## 4. 光源：三个角色别混在一起

`helpers.py:143-159` 的 `_find_dot` 只做「灰度 → 固定阈值 `255*0.2=51` 二值化 → 轮廓质心」，**没有形状、亮度、面积或数量上限判别**。Astra 的 IR 流默认满是激光散斑点，会被全部当成标记点，跨相机极线匹配（5 像素阈值）直接崩溃。

| 光源 | 装在哪 | 本方案怎么处理 |
| --- | --- | --- |
| ① 三点标记 IR LED | **目标（机器狗）身上**，硬导线兼作供电与支撑 | **必需**。这是定位信号的唯一来源，光来自目标自己，不是相机照明 |
| ② 激光散斑投射器 | 相机自带，850 nm 结构光 | **必须关掉**。否则产生成百上千个伪亮点，4 台之间还会串扰 |
| ③ 相机侧 IR 环形补光灯 | 每台相机镜头外围 | **不需要另买**。它只服务于被动逆反射球方案 |

**关于 ③ 的权威依据**：作者 jyjblrd 在仓库 Discussion #36 明确回复——"If you will only be using IR leds as the markers (as seen on the drones), then you do not need the floodlights on the cameras. The floodlight is only for passive markers (eg. retroreflective balls)."。博客 BOM 里那盏 $0.9 的 IR Ring Floodlight 是给被动标记准备的，本系统用不上；Discussion #20 也提醒被动标记球远距离不够亮（原装 PS3 Eye 镜头 1.5 m 外效果差），建议直接用主动 LED。Discussion #95（2026-06-11）正是有人问「只用滤光片 + 5 mm IR LED、不加补光灯能否检测」，尚未获回复。

对固定阈值二值化而言，**暗背景 + 3 个饱和亮点反而是最理想的**；补光灯把背景和机身照亮只会制造伪亮点。作者还提到环灯需从相机 PCB 5 V 取电、并要有源 USB Hub 才能带动 4 台，加了反而多一层供电负担。

**真正要处理的是曝光，不是照明**：Orbbec 官方论坛警告关掉散斑后「图像亮度大幅下降，可能接近全黑，取决于你的 setup」—— 这指的是环境缺少近红外照明，主动 IR LED 仍会自己发光；但需实测关断后 IR 流是否仍正常出帧、自动曝光是否把噪声拉爆。**应固定手动曝光**，而不是加补光灯。

落地顺序：

1. 关散斑：`OB_PROP_LASER_BOOL = False`（官方论坛；属性表另有 `OB_PROP_LDP_BOOL` 点阵投射器、`OB_PROP_FLOOD_BOOL` IR 泛光）；Orbbec Viewer 路径 Device Control → Laser Control → Laser Enable → Off；ROS 驱动有 `/camera/set_laser` 服务。**哪个属性对 SM 生效需实测。**
2. 固定 IR 手动曝光/增益，观察关断后画面是否为「暗背景 + 3 个亮点」。
3. 4 台相机全部关闭投射器。
4. 若 SM 上 `OB_PROP_FLOOD_BOOL` 确有独立泛光硬件（Astra Mini Datasheet 列有选配 830 nm IR Flood），**也无需开启**；仅调试需看清机身时可临时点亮。

## 5. 逐处代码改动清单

| 位置 | 现状 | 需改为 |
| --- | --- | --- |
| `helpers.py:13, 24-25` | `from pseyepy import Camera`；`Camera(fps=90, RES_SMALL, gain=10, exposure=100)`；`num_cameras = len(exposure)` | 换成 SDK/OpenNI2/ROS2 采集，按设备序列号枚举 4 台，固定 IR 流 640×480@30fps |
| `helpers.py:65-66` | 给 `cameras.exposure` / `gain` 直接赋值 | 映射到 SDK 的 IR 曝光/增益属性（需先关自动曝光） |
| `helpers.py:69` | `frames, _ = self.cameras.read()` | 返回单通道 16-bit；**保留被丢弃的时间戳**用于帧对齐（见第 3 节同步问题） |
| `helpers.py:72-74` | `np.rot90` → `make_square` → `cv.undistort` | 灰度图先转 3 通道或改 `make_square`（`helpers.py:508` 用 `x,y,_ = img.shape` 写死 3 通道）；补边后内参主点要相应平移；4 组新内参/畸变 |
| `helpers.py:75-82` | 高斯模糊 → `filter2D` 锐化 → `RGB2BGR` | 保留前处理，把 `RGB2BGR` 改为 `GRAY2BGR`；16-bit 需先缩放到 8-bit |
| `helpers.py:145-147` | `RGB2GRAY` + 阈值 51 | 单通道直接二值化；阈值按实测亮点亮度重调，建议补面积/圆度筛选与候选点数量上限 |
| `index.py:21` | 模块级 `serial.Serial("/dev/cu.usbserial-02X2K2GE", 1000000)` | 改为网络发布（WiFi / ROS2 / UDP）给 RK3588；否则 import 阶段在 Windows 必然抛异常 |
| `index.py:51-58` | `/api/camera-stream` 生成器按 150 Hz 目标频率串行执行采集→检测→三角化→滤波→串口 | 拆出独立采集线程，按 30 FPS 节拍驱动；`cv.imencode('.jpg', ...)` 需 3 通道图 |
| `index.py:34-37` | `num_objects = 2` + 串口锁 + `drone_armed` 数组 | 机器狗单目标；去掉 ESP32/SBUS 相关分支 |
| `camera-params.json` | 4 组 fx=320、cx=cy=160、5 个畸变系数 | 4 台各自棋盘格标定结果（IR 流内参，不是 RGB 内参） |

## 6. `cv2.sfm` 的 3 处等价替换（让 OpenCV 不再需要源码编译）

| 位置 | 原调用 | 等价实现 |
| --- | --- | --- |
| `index.py:247` | `cv.sfm.essentialFromFundamental(F, K1, K2)` | `E = K2.T @ F @ K1`（OpenCV sfm 文档所引公式 9.12），必要时归一化使奇异值为 `(1,1,0)` |
| `index.py:248` | `cv.sfm.motionFromEssential(E)` → `possible_Rs[4]`、`possible_ts[4]` | `R1, R2, t = cv.decomposeEssentialMat(E)`（常规 `calib3d`），组合成 `[(R1,t),(R1,-t),(R2,t),(R2,-t)]`，正好对上原代码 `for i in range(0,4)` 的手性检验 |
| `helpers.py:362` | `cv.sfm.fundamentalFromProjections(P1, P2)` | 由两个 3×4 投影矩阵按 4×4 行列式公式构造 F（opencv_contrib `fundamental.cpp` 同款算法，已有公开的 numpy 实现），或改用 `cv.computeCorrespondEpilines` 所需的等价 F |

> 亮点检测、极线匹配、DLT 三角化、`scipy.optimize` 光束法平差、卡尔曼/低通滤波**都不依赖 sfm，可原样复用**。

## 7. 采集层 SDK 三选一（按平台）

| 方案 | 适用 | 说明 |
| --- | --- | --- |
| `pyorbbecsdk==1.3.2`（Orbbec SDK **v1.x**） | Windows x64 + Python 3.8/3.9/3.11 | PyPI 有预编译 wheel（约 68 MB），`environment-astra.yml` 已含；**是否识别 SM 未验证** |
| OpenNI2 | Windows / Linux x64 / ARM64 | Astra 系列原生协议；第三方博客称在 RK3588 上实测可用；Python 侧需 ctypes 包装或系统级安装 |
| ROS2 `astra_camera` / `orbbec_camera` | RK3588 + ROS2 | 直接发布 `/camera/ir/image_raw`（640×480@30fps），并带 `/camera/set_laser` 服务关散斑；与用户现有 ROS2 栈最契合，但后端要改成订阅 topic |

**不要**装 `pyorbbecsdk2`（SDK v2）：官方文档明确 v2 不再支持传统 OpenNI 协议设备，且 2025-10 起公布的 OpenNI→UVC 固件升级名单只有 Astra Mini S Pro / Astra Mini Pro 及 Gemini E、DaBai 系列，**没有 Astra Pro / SM**。官方支持表中 Astra Pro Plus、Astra+ 在 v2 均为 `not supported`。

## 8. 待确认 / 不确定项

- SM 的官方 Datasheet 未取得（预览加载失败）；「无 RGB 流」「针对黑色物体优化」等细节来自第三方博客，需以 SDK 枚举结果与机身标签核对。
- IR 流的确切位深（Y16 / 8UC1）、FOV、内参、畸变系数、可用分辨率组合、曝光/增益范围均需实测或标定。
- 关闭散斑的可用属性（`LASER` / `LDP` / `FLOOD`）、关断后 IR 流是否仍出帧、以及固定手动曝光下的实际画面，需上机验证。
- **波长匹配**：Astra 系列结构光为 850 nm，Astra Mini Datasheet 的选配 IR Flood 为 830 nm，社区 Discussion #95 有人用 940 nm 滤光片 + 5 mm IR LED（未获回复）。仓库与博客**均未给出 IR LED 波长数值**，需确认所选 LED 落在相机窄带滤光片通带内。
- 4 台相机同机 USB 带宽与多设备枚举顺序，仓库与外部资料均未提供。
- 官方规格 `Trigger: N/A`，但经销商页面宣称可同步 30 台传感器，两者矛盾；按官方规格书采信「无硬件触发」，多机同步能力需实测。
- 仓库与厂商均未提供该组合下的精度、延迟、抖动或定位范围指标，本文不作性能承诺。
