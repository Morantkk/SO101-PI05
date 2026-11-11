# 发布包验证（2026-11-10）

- 74 项相关 CPU 测试通过，使用现有 SO101-PI05 Pixi dev 环境、显式加载发布副本的代码与测试；未另装一套大环境。
- 9 个真实权重分卷均通过 SHA-256；已完整流式解压并核对原始 .metadata 和 __0_0.distcp 的 SHA-256。
- 因磁盘空间不足以额外存放原始 16.57 GB，没有在本机再落地一份解压权重；流式验证覆盖全部原始字节，不是仅列目录。
- 还原工具用合成小归档验证了恢复、拒绝覆盖、损坏检测；真实大归档的校验路径已通过。
- 全部 Python 文件语法检查、部署 Shell 语法检查通过；发布改动的入口和工具通过 ruff。
- 归一化 norm_stats 原始哈希不变；路径匿名化后的 split/provenance 校验关系一致。
- 发布版 models 只列 local100000；全参数云端权重没有包含。
- 标定、串口、原归位值在外层 local_only；源码只放模板。常见私钥/token 模式扫描未发现匹配（不是穷尽安全保证）。
- 用临时 Git 索引检查必要资产会上传，大权重/机器配置/视频不会误入普通 Git；未在发布目录创建远端或推送。
- 主页与模型卡的本地链接已核对；实际照片/曲线已查看；成功视频尚待作者加入。

home_sequence.py / execution_limits.py 与当前源工程逐字节一致；仍是半开等待 0.5 秒、恢复原限速和跟随保护。

新机器安装、GPU 推理、真机控制未执行，不据 CPU 测试宣称新机器已可直接运行。
当前加载器需要匹配原数据验证身份，原数据不在此包；这是公开复现的已知外部依赖，详见 SETUP.md。
证据见 evidence/release_validation.json。

## 2026-11-11 发布补充检查

- 发布源码共 284 个文件，134 个 Python 文件通过 AST 语法检查。
- 常见访问令牌与私钥模式扫描未发现匹配；未收录机器配置、原始训练数据或模型分卷。
- 9 个本地模型分卷的大小与 SHA-256 均与 manifest.json 匹配。
- 演示 H.264 视频和封面已加入，主页、模型卡、视频说明中的本地链接已检查。
- 此次发布未重新运行 Linux/GPU 或机械臂实机测试；上方 CPU 测试记录属于 2026-11-10 的原始验证。

## 2026-11-11 naming migration verification

- Renamed the policy and transport modules, runtime variables, services, UI labels, and documentation for SO101-PI05.
- Audited all 285 tracked paths and text files; upstream names remain only in license and attribution notices.
- All Python sources, JSON and TOML files parse; deployment Shell syntax checks pass.
- 16 service-initialization, manual-release and asset-relocation tests pass in an isolated Windows Pixi environment.
- Trimming-manifest, split and provenance hashes agree; normalization statistics retain their recorded SHA-256.
- PyTorch-dependent deployment tests could not be collected because this verification environment lacks torch. Linux service orchestration, GPU and robot execution were not tested.
- SOURCE_MANIFEST.json was regenerated for the current distribution and excludes itself.