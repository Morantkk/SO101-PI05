# SO101 episode 81–197 训练数据准备

本目录是从 LeRobot v3 源数据生成的只读训练视图。源数据仍位于
`/path/to/so101_dataset`，没有改写源视频或 Parquet。

- 数据集当前有 198 条 episode（0–197）；本版本训练范围为 81–197，共 117 条。
- episode 0–80 被排除，不参加训练，也不作为验证集。`split.json` 用 `excluded` 明确记录它们。
- 训练帧只裁掉满足持续运动判据之外的开头和结尾静止段；内部停顿不删除、不拼接。
- 裁剪边界沿用相隔 6 帧、关节变化至少 1 度或夹爪变化至少 2 点、至少 3 个连续窗口；起点前保留 3 帧，末动作后保留 15 帧。
- SO101-PI05 使用 30 帧动作块和 30 FPS。SO101 状态/动作是 12 维；关节维做相对当前状态的 delta，两个夹爪维保持绝对值；模型输入补齐到 32 维。
- `norm_stats.json` 仅由上述 117 条训练 episode 的保留状态和完整 30 帧动作目标计算，不含 episode 0–80。
- 86,458 个训练帧样本；动作统计覆盖 2,593,740 个动作向量。所有原始源文件在准备前后 SHA-256 一致。
- 三路视频的文件编号和时间戳元数据逐 episode 对齐；每条训练 episode 的保留区间首/中/尾均成功解码三路精确帧。五条新增 episode 的画面样例在 `camera_sample_montage.jpg`。
- `verification.json` 汇总核验结果，`provenance.json` 记录数据与归一化来源，`trim_records.json` 记录 117 条裁剪边界。

对应训练配置已登记为 `pi05_so101_fold_81plus`：Batch 8、10 epochs、冻结 PaliGemma 主干、不启用验证。数据准备和核验状态不代表训练进程的运行状态；实际运行状态以训练面板和运行日志为准。

后续准备启动时，可在 `so101-policy` 目录运行：

```bash
WANDB_MODE=offline pixi run -e dev python scripts/train/local_training_panel.py \
  --config-name pi05_so101_fold_81plus \
  --exp-name fold81_197_b8_e10_20261108 \
  --port 8765 --batch-size 8 --num-epochs 10 --save-step-interval 20000
```
