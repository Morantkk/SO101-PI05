# RTC 功能和参数

一边执行旧动作块，一边推理下一块；用旧计划的重叠段引导新动作，并跳过推理期间已执行的 tick。
逻辑在 lios-pi/src/pi/models_pytorch/rtc.py 和 deployment/so101/rtc_protocol.py。
这是论文推理时 RTC 在 LiOS/SO101 上的适配，不是原作者硬件实验的逐项复现。

| 参数 | 值 | 意思 |
|---|---:|---|
| horizon | 30 | 一次预测 30 个动作时间步 |
| fps | 30 | 控制目标频率 30Hz |
| rtc.num_steps | 5 | 去噪 5 次，不是动作只有 5 个 |
| execution_horizon | 15 | 当前块累计消耗 15 步后允许新计划 |
| initial_delay_steps | 5 | 初始延迟估计 |
| delay_buffer | 10 | 最近 10 次延迟的最大值 |
| max_guidance_weight | 5 | 重叠引导权重上限，不是速度倍率 |
| prefix_schedule | exp | 指数前缀权重 |

30Hz 下 30 步约 1 秒，15 步约 0.5 秒，5 步约 167ms。普通推理使用 40 次去噪。
RTC 用动作侧 autograd/VJP，但模型冻结，不做优化器更新；不能整个包进 inference_mode。
参考、原作者代码与 MIT 许可保留在 lios-pi/docs/rtc_reference；其中旧调参记录为历史证据。
