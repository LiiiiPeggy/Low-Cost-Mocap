# Low-Cost-Mocap（原作者代码参考）

本目录保留原项目的基础资料和当前仍在使用的 USB 红外相机实现：

- `computer_code/api/`：当前 Media Foundation 双相机采集、标定和 LED 定位代码
- `3d_files/`、`images/`：原始 3D 文件与参考图片
- 当前运行说明、接口和标定流程见仓库根目录的 [README](../README.md)

原作者的 PS3 Eye、Flask-SocketIO（3001）、React/Vite 网页、Ruckig 无人机控制和
ESP32 `sender`/`receiver` 链路已从运行代码中删除，完整源码保留在 [upstream](../upstream/)。
