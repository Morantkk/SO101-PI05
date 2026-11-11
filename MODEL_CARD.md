# local100000 — 冻结主干 SO101 衣物折叠模型

仅发布 fold81_197_b8_e10_20261108/step100000 raw 检查点。冻结 PaliGemma；动作专家与相关投影训练，见 so101-policy/scripts/train/train_pytorch_fsdp.py。
不是 cloud27010，不是 EMA，不是 LoRA/adapter。完整权重包含冻结主干，未量化、裁剪、转换张量或重导出。

## 训练与接口

- 基础：lerobot/pi05_base，记录修订 22c30e5b34f9e7ab738b1b34fc2fc5ecfe2d28f0。
- Batch 8，seed 42，bfloat16 训练配置，compile 开启，EMA 0.999（发布 raw）。
- 学习率 warmup 5000 / peak 5e-5 / decay 40000 / decay_lr 2e-5。
- episode 81–197 共 117 条，86,458 帧，horizon 30，无验证集。
- 原计划 108070 步/10 epochs，主动停在 100000，约 9.25 epochs；每 20000 步保存。
- 三相机 RGB 224×224；12 维状态/动作；左 6 维后右 6 维；关节度、夹爪 0–100。
- 内部 action_dim 32；离散 state；delta mask (5,-1,5,-1)；使用配套 norm_stats。
- 普通推理 40 去噪步；RTC 5 步、15 tick 窗口、引导 cap 5、延迟历史 10。

记录：[training_run.json](docs/evidence/training_run.json)、[release_scope.json](docs/evidence/release_scope.json)。
原训练数据、其他 checkpoints、EMA、优化器、基础 safetensors、云端全参权重不随包发布。

## 验证边界

作者有成功实拍视频，待补充到 media/videos/success_demo.mp4；未提供重复试验成功率。
现有普通/RTC 离线报告使用训练集样本，不能视为独立泛化测试。本包不虚构成功率。
本次打包不启动训练、推理或电机。

## 原文件身份

- shard：16574901365 bytes，SHA-256 `042e6ea3ddbcad174dcacdab4fef2c1d6b848e72fb3cf2cc5ddfdb4fcbd2c454`。
- .metadata：316367 bytes，SHA-256 `cf129832922185b1a1489dc21c1d71a3889c398b4379227f9ff633b7378dc1ce`。
- norm_stats SHA-256：`02bcec98c2125383b9b2fd266b00384f4c5137768d8682b38eecd1dc6389cce0`。

压缩分卷仅用于运输，解压后必须匹配原始 SHA-256。清单在 models/local100000/manifest.json。

## 许可

基础模型标记 license: gemma。完整 checkpoint 包含其衍生权重，不能用代码 Apache-2.0 替代权重许可。
保留 [基础模型卡](docs/licenses/BASE_MODEL_CARD.md)、[条款](docs/licenses/GEMMA_TERMS.txt)、[使用政策](docs/licenses/GEMMA_PROHIBITED_USE.txt)。
