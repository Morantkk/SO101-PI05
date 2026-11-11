# local100000 权重

唯一发布：冻结 PaliGemma 主干、训练动作专家及相关投影的本地 step100000 raw 模型。
训练时仅动作侧更新，但原始保存格式包含完整模型；不是仅动作头张量，也不是云端全参数权重。

实际文件在本整理目录上一级 model_weights/（压缩分卷）；GitHub 发布时作为 Release 附件。
下载全部分卷后，在仓库根目录运行：

```bash
python3 tools/weights.py verify --weights-dir ../model_weights
python3 tools/weights.py restore --weights-dir ../model_weights
```

restore 需要额外至少 17 GiB，拒绝覆盖，恢复原始 .metadata 与 __0_0.distcp。
不需要原文件所在机器或符号链接；分卷是真实独立文件。原始 checkpoint SHA-256 在 manifest.json。
未包含 EMA、优化器、RNG、训练恢复状态；这是推理权重发布，不是完整训练续跑备份。
