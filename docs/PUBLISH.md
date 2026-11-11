# GitHub 发布步骤

## 上传源码

只把本整理目录里的 repo/ 作为 Git 仓库根，不上传外层 local_only/ 或 model_weights/。
README、模型卡、图片、测试、模型分卷清单和工具都在 repo/。

```bash
cd /path/to/SO101-PI05/repo
git init -b main
git add .
git status --short
git commit -m "Initial SO101 frozen-backbone deployment release"
git remote add origin https://github.com/Morantkk/SO101-PI05.git
git push -u origin main
```

公开仓库：https://github.com/Morantkk/SO101-PI05 。源码已于 2026-11-11 首次发布。

## 上传权重附件

在 GitHub 新建 Release，例如 local100000-v1，将 model_weights/ 内所有 part-000 至 part-008、SHA256SUMS、manifest.json、README.md、Notice.txt 及随附条款/模型卡一起上传。
分卷每个至多 1000 MiB；最后一个较小；缺任意分卷都不能恢复。只要下载完整即可通过 tools/weights.py verify/restore。
权重不是普通 Git 文件，不要 git add、不要将 16.57GB 原 shard 直接上传 Git。
2026-11-10 官方文档说明 Release 单附件小于 2 GiB；Git LFS 单文件上限也随计划而定，原 shard 超过这些单文件限额，因此采用分卷 Release 附件。

来源（已于整理时核对）：
https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases
https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage

模型沿用基础 Gemma 条款，不把代码许可证误用于完整模型；相关条款副本在 docs/licenses/，放 Release 说明时链接 MODEL_CARD.md 和 WEIGHTS_NOTICE.md。

## 演示视频

浏览器兼容演示已放入 `media/videos/success_demo_h264.mp4`，封面为 `media/images/success_demo_poster.jpg`，根 README 已提供链接。
Git 仅收录这个约 6.75 MB 的演示；HEVC 备份不纳入 Git。具体检查点、RTC 模式、播放速度与人工介入未记录，不报告成功率。

## 发布前最后检查

确认 repo/ 不含 config.json、home_pose.json、串口标识、私钥、token、原始数据视频和缓存；查看 docs/VALIDATION.md。
这份公开包仍依赖匹配原数据集做身份校验，SETUP.md 已明确；未公开数据时，不要声称陌生用户只下载权重就能无配置运行。
