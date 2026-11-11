# SO101 控制命令（本地动作专家 step100000 发布版）

先完成 ../../docs/SETUP.md。所有命令从本目录执行；脚本自动选择 Pixi 与 LeRobot 环境。
只包含 local100000，不包含 cloud27010 权重。

```bash
# 普通推理 / RTC 推理：二选一，均会实际控制机械臂
bash robotctl.sh run --model local100000 --rtc off
bash robotctl.sh run --model local100000 --rtc on

# 停止动作并保持，关闭推理及相机两端
bash robotctl.sh stop

# 停止；夹爪到 50%，等待 0.5 秒；恢复记录的归位姿态
bash robotctl.sh home

# 停止并归档运行状态，保留权重/标定/编译缓存
bash robotctl.sh clear-cache
bash robotctl.sh status
bash robotctl.sh models

# 不控制设备的选择校验和模型检查
bash robotctl.sh run --model local100000 --rtc off --dry-run
bash robotctl.sh check-model --model local100000 --rtc off
bash robotctl.sh check-model --model local100000 --rtc on --offline
```

run 成功后在 systemd 用户服务中持续运行，关终端不等于停止。相机和权重加载可能较慢，脚本打印阶段。
stop 保持当前测量位置，不释放全部电机力矩。启动失败/中断会执行清理。状态文件是历史记录，需结合服务状态。
home 张爪速度仍为 5 units/s，所以 0.5 秒是**到位后的等待**，不是要求张爪在半秒内完成。
张爪失败、超时或停止请求会阻止归位；最终夹爪回到 home_pose.json 中的值。

保护参数恢复原值：推理关节 45.63°/s、夹爪 36 units/s；跟随 6°/2；跟踪误差 8°/4。
归位关节 8°/s、夹爪 5 units/s，保留标定和停止检查。归位是有界过程，不自动开始推理。

代码放置变化：return_home.py 和 return_to_start.py 已从历史实验目录抽出；逻辑未改。
新增模型可走 register-model，但本发布包只附唯一 local100000。模型注册不赋予新设备的标定正确性。
