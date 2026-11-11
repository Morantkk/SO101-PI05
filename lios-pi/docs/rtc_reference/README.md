# Real-Time Chunking in LiOS SO101

Updated November 10, 2026: the unified `robotctl.sh run --rtc on|off` entry point
selects RTC or ordinary inference explicitly for either registered model.
Current RTC defaults are 5 denoising steps, minimum execution horizon 15,
guidance cap 5, initial delay 5 and delay history 10, with a trained horizon of
30 at 30 Hz. Both built-in models passed recorded-sample GPU inference in both
modes, with compilation enabled and no motor commands. Current commands and
limits are in `deployment/so101/ROBOT_COMMANDS.md`.

The parameter table, tuning history and measurements below are a historical
record from November 9. RTC was temporarily disabled that evening after behavior
regressions; see `deployment/so101/BASELINE_RESTORATION.md`. Those historical
measurements do not establish live performance for the new selectable modes.

This is a PyTorch/LiOS port of the complete **inference-time** RTC algorithm in
[Real-Time Execution of Action Chunking Flow Policies](https://arxiv.org/html/2506.07339v1),
Kevin Black, Manuel Y. Galliker and Sergey Levine, 2025, equations 2–5 and Algorithm 1.
It does not require retraining or the later training-time RTC method.

The author reference is
[Physical-Intelligence/real-time-chunking-kinetix](https://github.com/Physical-Intelligence/real-time-chunking-kinetix/tree/9296f31d62d5bfeb5779dcb2f9bcf71ca37f448b),
commit `9296f31d62d5bfeb5779dcb2f9bcf71ca37f448b`. Its public release targets
Kinetix simulation, not SO101 hardware. `author_model.py.txt` and `LICENSE` retain
the upstream reference and MIT notice. This integration is not an official author
robot deployment package.

## Implemented algorithm

- Full PiGDM vector-Jacobian product through the action expert, including its
  dependence on the noisy action. No detached-velocity approximation is used.
- Exponential soft prefix weights from equation 5, with full weight on the
  predicted-delay prefix, decaying weight on the overlap, and zero after it.
- Time-dependent guidance capped at beta, including the finite noise endpoint.
- An asynchronous inference worker and a control-clock-owned action queue.
  The old chunk keeps being consumed while the next chunk is generated.
- The consumer acknowledges session, chunk ID, origin tick, and consumed ticks.
  A result captured at tick s is installed at index `current_tick - s`, the
  distributed equivalent of Algorithm 1's `t = t - s`. Observed handoff delays
  include transport/consumer scheduling; their rolling maximum predicts delay.
- Previous absolute SO101 actions are rebased against the **new** measured state
  and normalized before guidance. Grippers remain absolute; padded dimensions
  have zero guidance residual.

LiOS uses noise time `t=1` down to data time `t=0`; the paper uses `tau=1-t`.
Thus the denoised estimate is `D(x)=x-t*v_LiOS(x,t)`, the cotangent is
`(previous-D(x))*W`, and `g=J_D(x)^T*cotangent`. The LiOS update is
`x <- x - (v_LiOS - lambda*g)/n`. The cotangent is detached, but the Jacobian
includes the full velocity network. Graphs are released between Euler steps.
The visual/language KV cache is computed once per observation without gradients.
RTC must run outside `torch.inference_mode()`. Model weights stay frozen.

## Active parameters and model

| Parameter | Paper real-robot setting | This checkpoint |
| --- | --- | --- |
| Prediction horizon H | 50 | 30 (trained model shape) |
| Minimum execution horizon | 25 | 6 (updated after grasp feedback) |
| Euler steps n | 5 | 10 (updated after grasp feedback) |
| Guidance cap beta | 5 | 5 |
| Delay-history length | 10 | 10 |
| Prefix schedule | exponential | exponential |

The deployment control rate is 30 Hz. Initial predicted delay is 5 ticks; it
adapts to the maximum of the last 10 observed handoff delays. Therefore this is
the full algorithm with checkpoint-specific horizon adaptation, not an exact
reproduction of the paper's robot experiments or horizon.

The active model is local raw `step100000` from
`fold81_197_b8_e10_20261108`, saved November 9, 2026, from the unfinished local
training run. Its SHA256 is
`042e6ea3ddbcad174dcacdab4fef2c1d6b848e72fb3cf2cc5ddfdb4fcbd2c454`.
No weights were retrained or modified for RTC.

The `rtc` section of `deployment/so101/config.json` enables the algorithm.
`rtc.num_steps=10` overrides the existing baseline `num_steps=40` only for RTC.
The compiled function is the action-expert denoising step, including autograd,
with compile mode `default`. Both bootstrap and guided paths are warmed up
before publishing usable chunks.

## Runtime and failure handling

The existing LiOS CUDA IPC image receiver and state/action WebSocket endpoints
are retained. `state_client.py` runs the actual RTC queue at 30 Hz without motor
writes. `joint_trial.py` uses the same queue and acknowledges a tick only after
its control command succeeds. Initial torque/hold preparation consumes no tick.

Sessions prevent previous runs from supplying actions. Reordered chunks, wrong
model identities, non-finite outputs, expired arrivals and queue underruns are
rejected. A real controller exits through its existing measured-position hold
on failure; the observation-only consumer reconnects with a new session.
Insufficient overlap for the predicted delay prevents another inference launch.

State freshness (0.2 s), image freshness (0.5 s), result publication age (0.35 s),
and incoming result age (0.4 s) checks are retained. After accepting an RTC chunk,
its validity is determined by its remaining control ticks: applying the old
0.4-second latest-packet TTL to every queued action would invalidate the paper's
0.5-second execution horizon. No action is extrapolated beyond its 30 ticks.
This is a change to action scheduling, not to motor position/speed protection.

Both original motor limiting functions are unchanged: full-task joint speed
45.63 degrees/s, gripper speed 36 units/s, calibration bounds, tracking limits,
current/torque protection, stop latch and loop-stall checks remain in force.
The current deployment has `execution_enabled=false` and `STOP_MOTION` set.

## Commands

Start the configured RTC observation/inference services (no motor commands):

```bash
bash /path/to/LiOS/deployment/so101/start.sh
```

The foreground inference entry point is below. Stop the existing inference
service first to release port 18081 and its GPU allocation; keep the camera
receiver running. A foreground process occupies this terminal until stopped.

```bash
systemctl --user stop lios-deploy-inference
cd /path/to/LiOS/lios-pi
pixi run --as-is -e dev python scripts/deployment/serve_so101.py \
  --config ../deployment/so101/config.json
```

CPU algorithm/protocol checks:

```bash
cd /path/to/LiOS/lios-pi
pixi run --as-is -e dev pytest -q tests/test_rtc.py tests/test_so101_deployment.py
```

To inspect an already running observation session without opening motor buses:

```bash
pixi run --as-is -e dev python ../deployment/so101/validate_rtc_observation.py --seconds 60
```

Offline checkpoint inference (requires GPU; stop the live inference process
before loading another copy of the model):

```bash
pixi run --as-is -e dev python scripts/deployment/serve_so101.py --offline \
  --config ../deployment/so101/local_step100000/rtc_config.json
```

## Validation recorded on November 9, 2026

- 16 targeted CPU tests passed, including a nonlinear analytical full-Jacobian
  comparison, guidance endpoints, overlap masks, padded dimensions, delta-frame
  conversion, asynchronous handoff, stale/reordered results, underrun and reset.
- The actual checkpoint produced finite 30x12 outputs through full VJP. Offline
  warmed samples took 70–72 ms. Initial forward/backward compilation took
  approximately 11.5/20.6 s and was not treated as usable inference.
- A 60-second live-camera/read-only-state window received 120 finite chunks in
  one session with zero queue resets. Median inference was 73.77 ms, P95 79.47 ms,
  maximum 85.04 ms; maximum observation-to-receipt age was 102.56 ms. The delay
  estimate converged to 3 control ticks. Motor commands sent: zero.
- Reports: `deployment/so101/local_step100000/offline_prediction.json` and
  `rtc_live_validation.json`; the latter contains per-chunk records.

The measurements above describe the initial 5-step / 15-tick RTC configuration.
After feedback about inaccurate grasping, the active configuration was changed
to 10 denoising steps and a 6-tick execution horizon (approximately 0.2 seconds).
Guidance cap 5, exponential prefix weights and the 10-entry delay buffer remain
unchanged. A 20-step / 8-tick candidate was rejected after one queue underrun.
The selected configuration passed a 60-second observation-only window with
299 chunks, one session and zero resets: median 101.76 ms, P95 108.87 ms and
maximum 117.72 ms inference time. Maximum receipt age was 136.43 ms.
Reports and the previous config are archived under
`deployment/so101/local_step100000/rtc_tuning_20261109_193502/`.
This establishes timing feasibility, not improved grasp success; moving-hardware
grasp accuracy with these tuned parameters still needs observation.

Full-task execution now defaults to continuous operation via
`execution_duration_s=0`. The CLI option `--duration 0` selects the same mode;
an explicit positive duration still selects a timed run. Small joint trials
remain limited to three seconds. Continuous operation still stops on the stop
latch, SIGINT/SIGTERM or an existing protection failure, followed by the measured
position hold. Reports retain the last 9,000 samples and total command count
to bound memory use during long runs. Changing this setting does not start motors.

These checks validate inference and queue operation. The arms were stationary;
task success, contact behavior and tracking quality under RTC have not been
validated on moving hardware. The offline sample belongs to the training set
because this training run contains no validation split.

The pre-RTC config, controller, protocol and server backups are under
`deployment/so101/local_step100000/rtc_before_20261109_190107/`. Setting
`rtc.enabled=false` and restarting both inference and state-client selects the
legacy sampler/protocol; stop any controller before changing deployment modes.
