# GEM-X 视频动作服务

基于 NVIDIA [GEM-X](https://github.com/NVlabs/GEM-X) 的本地视频动作推理服务。与 `gem_blender_plugin` 分开运行，默认地址 **http://127.0.0.1:6700/**。Blender 只上传视频和导入动作，不安装 Torch。

## 本机使用

1. 双击 `start.bat`；Linux / macOS 使用 `bash start.sh`。实际推理需要 NVIDIA CUDA，目前没有 CPU / Apple GPU 推理路径。
2. 浏览器打开 http://127.0.0.1:6700/，可以管理端口、推理参数，下载 / 补全官方模型、上传视频和查看任务日志。
3. 在 Blender 中安装配套插件，侧栏 `GEM-X` 中选择视频，点击“视频推理并导入动作”。

本机已经配置了 `.venv`，复用全局 CUDA Torch，新增依赖装在服务端 `.venv` 中。全局环境原有包未被替换。模型下载后的状态可以在 WebUI 的模型管理中查看。

## 目录

```text
gem_inference_server/
  gem_server/       API、推理适配、动作导出、管理页面
  vendor/           随服务端提供的 GEM、SOMA、SAM-3D-Body、DINOv3 源码及许可
  requirements.txt
  start.bat / start.sh
  config.example.json
  config.json      首次启动生成，本机配置，不提交 Git
  models/          所有权重、检测器、SOMA 资源及 Torch/HF 缓存
  data/            上传文件、任务、日志、生成动作
  .venv/           可选的虚拟环境
```

**服务端是一个独立文件夹**：所需上游源码和配置已放在 `vendor/` 内，不依赖同级目录，不需要额外克隆 GEM-X、不需要初始化 Git 子模块。将整个服务端文件夹移动或单独发布即可；首次使用仍需按下方安装 Python 依赖、下载模型。虚拟环境在移动后建议重新创建，或通过 `GEM_PYTHON` 使用现有 Python。

## 新电脑安装

推荐 Python 3.12、较新的 NVIDIA 驱动。以下命令在服务端目录执行。`requirements.txt` 只声明版本下限；安装新版本后可用 `doctor` 检查兼容性。

### 使用虚拟环境

Windows PowerShell：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Linux：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

先到 [PyTorch 官方安装页](https://pytorch.org/get-started/locally/) 选择操作系统与 CUDA，执行其提供的 `torch torchvision` 安装命令。不要只安装 CPU 版 Torch。例：使用 CUDA 12.8 轮子时为：

```text
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

CUDA 版本应按官方可用轮子和驱动选择；不要求与本机安装的 CUDA Toolkit 一致。然后执行：

```text
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
python -m gem_server download
python -m gem_server doctor
python -m gem_server serve
```

`torch.cuda.is_available()` 应输出 `True`。模型也可以启动 WebUI 后点击下载，约需 7.3 GB 磁盘空间，另预留缓存和输出空间。首次运行 SOMA 的 Warp 内核可能需要编译，耗时会更长。

### 使用全局环境 / 已有环境

允许直接在已装好 CUDA Torch 的全局 Python 中执行 `pip install -r requirements.txt`，不强制虚拟环境。启动脚本的解释器优先级：

1. `GEM_PYTHON` 指定的解释器。
2. 已激活的 `VIRTUAL_ENV`。
3. 服务端 `.venv`。
4. PATH 中的 `python`（Windows）或 `python3`（Linux）。

例如强制用全局 Python：

```bat
set "GEM_PYTHON=D:\Python312\python.exe"
start.bat
```

```bash
GEM_PYTHON=/usr/bin/python3 bash start.sh
```

启动脚本不会自动安装环境或下载模型。可运行 `start.bat doctor` / `bash start.sh doctor` 排查依赖和缺失模型。

## 推理与显存

- 默认：完整 SAM-3D-Body 图像特征 + GEM 回归 + 官方动作后处理。
- DDIM：使用官方 50 步采样路径，耗时较长。
- 关闭图像特征：跳过 SAM-3D-Body，减少运行时间；质量会下降，推荐配合 DDIM。服务不会在失败时偷偷切换成这个模式。
- 人物检测使用 CPU ONNX Runtime。ViTPose、SAM、GEM 在 CUDA 上依次加载并释放，每个任务结束子进程退出，避免所有模型同时常驻显存。8 GB 显卡建议批大小 **1**。
- 视频会等比例缩小到指定最长边，按目标帧率采样，不会提高原视频帧率。从开头最多处理“最大处理帧数”帧；例如 30 FPS、300 帧约 10 秒，10 FPS 则约 30 秒。
- 默认服务端限制 600 帧、最长边 960 像素、上传 512 MB，可在管理页面调整；插件单次上传上限 512 MB。
- 同时只执行一个 GPU 任务，其他任务排队。取消会终止当前推理子进程，释放显存。

当前按**固定相机**估计，没有部署移动相机 SLAM / VO。尽量使用固定镜头、全身可见、少遮挡的单人视频。多人视频仅处理上游追踪器选中的主要人物。单目深度、接触点和手指仍可能不准确。

此项目没有把 Kimodo 的 ConvRot / NF4 量化方式直接用于 GEM。这些是不同网络，不会把未经验证的量化选项作为可用功能。

## 模型版本

`gem_server/model_manifest.json` 固定官方模型的修订版本、文件大小和 LFS SHA256。下载新文件时校验大小及摘要，避免下载不完整文件；支持断点续传。SOMA 资源固定为 GEM-X 的子模块所用版本，不能直接换成 SOMA-X 最新格式。

模型来源：`nvidia/GEM-X`、`nvidia/soma-x`、OpenMMLab 的 YOLOX HumanArt 检测器，均保存在 `models/` 下。ViTPose / SAM 使用随服务端一起提供的 `vendor/dinov3` 源码，通过本地 Torch Hub 加载；不会再从 GitHub 下载网络源码，也不额外下载 DINO 预训练权重。

模型 / 资源沿用各自上游许可，参见 [ATTRIBUTIONS.md](ATTRIBUTIONS.md)。

## 输出与 Blender

每个任务在 `data/jobs/<任务ID>/` 下保存：

- `motion.json`：包含 78 个关节（77 个 SOMA 关节加虚拟 Root）的父级、静止姿态和逐帧世界矩阵。插件直接使用它，正确转换 Y-up 到 Blender Z-up。
- `motion.bvh`：通用 BVH 动作，长度单位为米；其他工具导入时注意单位与轴向。
- `soma_params.npz`：原始 SOMA 参数（姿态、位移、体型、比例、帧率），供再次处理。
- `input.mp4`：实际处理的采样片段。
- `worker.log`：详细推理日志；WebUI 可以查看末尾日志。

插件创建新骨架和新动作。要驱动现有的 Steve / Minecraft / 其他骨架，还需通过 BoneAnimCopy 等工具做重定向。不会自动修改当前角色的权重或骨骼映射。

## API

交互文档：`http://127.0.0.1:6700/docs`。

```text
GET  /health
GET  /v1/config
PUT  /v1/config
GET  /v1/models
POST /v1/models/download
POST /v1/jobs                       multipart: video 文件、options JSON 字符串
GET  /v1/jobs
GET  /v1/jobs/{id}
POST /v1/jobs/{id}/cancel
GET  /v1/jobs/{id}/log
GET  /v1/jobs/{id}/files/motion.json
GET  /v1/jobs/{id}/files/motion.bvh
GET  /v1/jobs/{id}/files/soma_params.npz
```

监听地址和端口的更改需重启生效，插件服务地址也要对应修改。默认仅监听本机；此版本是本地管理服务，没有登录认证。

## 检查

```text
python -m gem_server doctor
python -m unittest discover -s tests -p "test_*.py" -v
```

Blender 导入测试随独立插件提供，见插件 README；服务端自身的测试无需插件文件夹。

验证单独移动服务端文件夹后的运行（使用本机已下载权重；完整模式还需要已有的验收片段）：

```text
python tests/standalone_check.py
python tests/standalone_check.py --infer
```

此检查将服务端源码复制到独立临时目录，使用独立的权重文件目录，禁止联网和读取原来的 GEM-X / 源码 / 模型目录，再验证内置模块加载；`--infer` 还会实际跑完整视频推理。验证完成后清理临时目录。

本机验收：RTX 4070 Laptop 8 GB、CUDA Torch 2.13.0、Blender 5.2。使用仓库 `assets/demo_2d_keypoints.gif` 的前 24 帧（10 FPS，已含关键点叠加），实际跑通完整 SAM 图像特征 + GEM 回归 + 后处理，以及无图像特征 + DDIM；完整流程还经过配套插件 HTTP 客户端上传、轮询、下载和后台 Blender 导入。API / 下载续传和独立目录检查通过。导出关节与 SOMA 计算核的最大位置差约 `2.4e-7` 米；Blender 导入误差见 `data/deployment` 验收日志。这些数值衡量动作传递的保真，视频动作恢复质量仍需用实际视频评估。
