# GitHub 发布步骤

## 上传源码

只把本整理目录里的 repo/ 作为 Git 仓库根，不上传外层 local_only/ 或 model_weights/。
README、模型卡、图片、测试、模型分卷清单和工具都在 repo/。

```bash
cd /path/to/LiOS_SO101_GitHub_20261110/repo
git init -b main
git add .
git status --short
git commit -m "Initial SO101 frozen-backbone deployment release"
git remote add origin https://github.com/YOUR_ACCOUNT/YOUR_REPOSITORY.git
git push -u origin main
```

YOUR_ACCOUNT/YOUR_REPOSITORY 由你替换。本次没有创建远端、提交或推送。

## 上传权重附件

在 GitHub 新建 Release，例如 local100000-v1，将 model_weights/ 内所有 part-000 至 part-008、SHA256SUMS、manifest.json、README.md 及随附的 Notice/条款/模型卡一起上传。
分卷每个至多 1000 MiB；最后一个较小；缺任意分卷都不能恢复。只要下载完整即可通过 tools/weights.py verify/restore。
权重不是普通 Git 文件，不要 git add、不要将 16.57GB 原 shard 直接上传 Git。
2026-11-10 官方文档说明 Release 单附件小于 2 GiB；Git LFS 单文件上限也随计划而定，原 shard 超过这些单文件限额，因此采用分卷 Release 附件。

来源（已于整理时核对）：
https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases
https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage

模型沿用基础 Gemma 条款，不把代码许可证误用于完整模型；相关条款副本在 docs/licenses/，放 Release 说明时链接 MODEL_CARD.md 和 WEIGHTS_NOTICE.md。

## 放入自己的视频

复制你录好的成功视频到 repo/media/videos/success_demo.mp4。
建议在 Release 或 README 附件区上传视频，再替换根 README 的 SUCCESS_DEMO_URL 占位；视频已被 .gitignore 排除。
如需封面，放真实成功视频截图到 media/images/ 并替换 README 图片。现有图片只有设备/训练/历史界面，不冒充成功帧。
填写录制模型、RTC 模式、是否加速/剪辑/人工干预；没有重复统计不要报告成功率。

## 发布前最后检查

确认 repo/ 不含 config.json、home_pose.json、串口标识、私钥、token、原始数据视频和缓存；查看 docs/VALIDATION.md。
这份公开包仍依赖匹配原数据集做身份校验，SETUP.md 已明确；未公开数据时，不要声称陌生用户只下载权重就能无配置运行。
