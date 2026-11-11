# 安装、还原与本机配置

本包是当前部署代码的整理副本，原运行环境不会被复制。Linux + NVIDIA CUDA 驱动 + USB/串口权限 + systemd 用户服务是现有脚本的运行前提。
锁文件记录原 Linux CUDA 12.9 Pixi 依赖；新机器是否适配需独立验证。

## 1. 模型与图传环境

在仓库根目录：

```bash
cd so101-policy
pixi install -e dev
pixi run -e dev lerobot
cd ../so101-transport
pixi install
pixi run bash -c 'cd signal-server && go build -o webrtcssvr .'
cd ..
```

其中 so101-policy 的 lerobot task 安装的是旧数据接口依赖，不能直接作为串口控制环境。
如果带中文目录的路径影响某些二进制依赖，使用 Pixi detached environments 放到 ASCII 路径；不复制原机器 .pixi 缓存。

## 2. 机械臂驱动环境

已有可用的 lerobot 环境可继续使用。原机驱动来源提交为 `4aaff99be4a1d81568c08c8f0296b41b40c99ec4`，Python 3.12。
新机器参考创建（这一步下载和安装依赖，不控制设备）：

```bash
conda create -n lerobot python=3.12 pip
conda run -n lerobot pip install 'lerobot[feetech] @ git+https://github.com/huggingface/lerobot.git@4aaff99be4a1d81568c08c8f0296b41b40c99ec4'
cp deployment/so101/runtime.env.example deployment/so101/runtime.env
```

编辑 runtime.env，填写 Pixi 和该环境 Python 的实际绝对路径。原部署使用独立 Pixi 与 Conda 环境，互不替代。

## 3. 还原唯一模型

下载全部 local100000 分卷到仓库旁边 model_weights/，或指定其他目录。
原 shard 约 15.44 GiB；还原额外需要至少 17 GiB 可用空间。仅校验不落地不需要额外一份权重空间。

```bash
python3 tools/weights.py verify --weights-dir ../model_weights
python3 tools/weights.py restore --weights-dir ../model_weights
```

需要系统 zstd（Debian/Ubuntu 包名 zstd）。工具用流式解压，不创建中间完整 tar。
校验所有分卷、恢复后原文件哈希，拒绝覆盖已有 checkpoint。

## 4. 匹配数据与统计

**当前加载器仍验证训练数据身份。原训练数据没有随包开源，因此仅下载代码和权重不足以运行现有严格加载器。**
你自己机器的匹配数据可继续使用；其他使用者需要获得同一数据集，或另行实现并验证不依赖原数据的加载方式。本包不跳过这些校验。
该要求在实时推理启动时也存在，不只是离线样本测试。

在仓库根目录运行：

```bash
python3 tools/configure_local.py --dataset /absolute/path/to/matching_so101_dataset
```

工具调用原 relocation 脚本核对全部 source hashes，复制统计并更新路径；创建 config.json / machine.json。不会打开相机或串口。
公开原始模板在 so101-policy/assets；本机重定位资产在 local_assets（不提交）。
Paligemma tokenizer 首次需要联网从原配置地址缓存；在离线设备上应先准备缓存。

## 5. 设备和归位姿态

编辑 deployment/so101/config.json 内 YOUR_*：左右串口、三相机、robot_id。
用自己设备的 LeRobot 标定文件，不要套作者的标定。config.example.json 是接口模板，不是可直接运行的设备配置。

需要归位时，把 home_pose.example.json 复制成 home_pose.json，再将 start 填成**自己设备确认过**的 12 个值：左侧 shoulder_pan/lift/elbow/wrist_flex/wrist_roll/gripper，随后右侧同序。
关节用度，夹爪 0–100。空模板不会触发归位。原机器备份只在整理包 local_only/，没有进仓库。

## 6. 检查与使用

```bash
cd deployment/so101
bash robotctl.sh models
bash robotctl.sh run --model local100000 --rtc off --dry-run
bash robotctl.sh check-model --model local100000 --rtc off
```

--dry-run 只检查选择；check-model 校验张量和数据/统计，不操作机器人。
--offline 会进行记录样本 GPU 推理，也不操作机器人，但要求原数据和模型环境已准备好。
真实 run/home 会运动，命令见 ROBOT_COMMANDS.md。不要同时从两个副本运行同名 so101-deploy-* 服务，它们共享用户服务命名。

本次发布包只做 CPU、文件完整性、静态与导入检查，没有在新机器执行端到端安装或真机任务。

## Naming migration (2026-11-11)

Use `so101-policy/` for the model and `so101-transport/` for transport.
Environment variables now use the `SO101_` prefix; user services use `so101-deploy-*`.
Before upgrading a running installation, stop all services with the previous
checkout's `robotctl.sh stop`, then migrate the local paths and restart from
this checkout. The renamed stop command cannot manage services from an older
checkout. Do not run both versions against the same robot.

Historical evidence paths were normalized to the new directory names; recorded
metrics and model bytes are unchanged. Third-party attribution is in NOTICE.