"""RTC Algorithm 1 across LiOS WebSockets, using acknowledged control ticks.

The consumer owns the clock. A chunk's start_tick is the observation tick,
so installing a result skips exactly the ticks consumed during inference.
"""

import math
import uuid
from collections import deque

from action_protocol import validate_action


def validate_rtc_config(config):
    rtc = config["rtc"]
    horizon = config["horizon"]
    for name in ("execution_horizon", "initial_delay_steps", "delay_buffer"):
        if type(rtc.get(name)) is not int:
            raise ValueError(f"RTC {name} must be an integer")
    if not 0 < rtc["execution_horizon"] < horizon:
        raise ValueError("RTC execution horizon must be inside prediction horizon")
    if not 0 <= rtc["initial_delay_steps"] < horizon - rtc["execution_horizon"] or rtc["delay_buffer"] < 1:
        raise ValueError("RTC delay leaves insufficient overlap or invalid delay buffer")
    if rtc.get("prefix_schedule", "exp") != "exp":
        raise ValueError("Only the paper exponential RTC prefix schedule is supported")
    if type(rtc.get("num_steps", 5)) is not int or rtc.get("num_steps", 5) < 1:
        raise ValueError("RTC denoising steps must be positive")
    if not math.isfinite(rtc.get("max_guidance_weight", 5)) or rtc.get("max_guidance_weight", 5) <= 0:
        raise ValueError("RTC guidance cap must be finite and positive")


class RTCConsumer:
    def __init__(self, config):
        validate_rtc_config(config)
        self.config = config
        rtc = config["rtc"]
        self.session = uuid.uuid4().hex
        self.tick = 0
        self.chunk = None
        self.chunk_id = None
        self.start = 0
        self.delays = deque([int(rtc["initial_delay_steps"])], maxlen=int(rtc["delay_buffer"]))

    def state(self):
        return dict(session=self.session, tick=self.tick, chunk_id=self.chunk_id,
                    start_tick=self.start, delay_steps=max(self.delays))

    def accept(self, message, now, earliest):
        actions, _ = validate_action(message, self.config)
        meta = message.get("rtc", {})
        if meta.get("session") != self.session:
            return False
        if meta.get("chunk_id") == self.chunk_id:
            return False
        if meta.get("base_chunk_id") != self.chunk_id:
            raise ValueError("RTC chunk does not extend the acknowledged queue")
        stamp = float(message["timestamp/ms"]) / 1000
        if not math.isfinite(stamp) or stamp < earliest or not 0 <= now - stamp <= 0.4:
            raise ValueError("Expired RTC result")
        start = meta["start_tick"]
        if type(start) is not int or not self.start <= start <= self.tick:
            raise ValueError("Invalid RTC chunk tick")
        delay = self.tick - start
        if delay >= len(actions):
            raise ValueError("RTC result arrived after its horizon")
        if self.chunk is not None and self.tick >= self.start + len(self.chunk):
            raise ValueError("RTC old queue exhausted before handoff")
        if not isinstance(meta.get("chunk_id"), str) or not meta["chunk_id"]:
            raise ValueError("Missing RTC chunk identity")
        if self.chunk is not None:
            self.delays.append(delay)
        self.chunk, self.chunk_id, self.start = actions.copy(), meta["chunk_id"], start
        return True

    def peek(self):
        if self.chunk is None:
            return None
        index = self.tick - self.start
        if not 0 <= index < len(self.chunk):
            raise ValueError("RTC action queue underrun; hold and restart the session")
        return self.chunk[index].copy()

    def pop(self):
        result = self.peek()
        if result is not None:
            self.tick += 1
        return result


class RTCPlanner:
    """Single inference worker; a pending result must be acknowledged before reuse."""

    def __init__(self, config):
        validate_rtc_config(config)
        self.horizon = int(config["horizon"])
        self.minimum = int(config["rtc"]["execution_horizon"])
        if not 0 < self.minimum < self.horizon:
            raise ValueError("RTC execution horizon must be inside prediction horizon")
        self.session = None
        self.current = None
        self.last_tick = -1

    def plan(self, message):
        state = message.get("rtc")
        if not isinstance(state, dict):
            raise ValueError("RTC requires a tick-aware consumer")
        session, tick = state.get("session"), state.get("tick")
        if not isinstance(session, str) or not session or type(tick) is not int or tick < 0:
            raise ValueError("Invalid RTC session or tick")
        if session != self.session:
            self.session, self.current, self.last_tick = session, None, -1
        if tick < self.last_tick:
            raise ValueError("RTC tick moved backwards")
        self.last_tick = tick
        base = None
        previous = None
        delay = state.get("delay_steps")
        if type(delay) is not int or not 0 <= delay < self.horizon:
            raise ValueError("Invalid RTC delay estimate")
        if self.current is not None:
            meta, actions = self.current
            if state.get("chunk_id") != meta["chunk_id"]:
                return None
            if state.get("start_tick") != meta["start_tick"]:
                raise ValueError("RTC acknowledged chunk origin differs")
            consumed = tick - meta["start_tick"]
            if consumed < self.minimum:
                return None
            if consumed + delay >= self.horizon:
                raise ValueError("Insufficient RTC overlap for predicted delay")
            previous = actions[consumed:].copy()
            base = meta["chunk_id"]
        elif state.get("chunk_id") is not None or tick != 0:
            raise ValueError("RTC server restarted; reconnect consumer to reset queue")
        return dict(session=session, start_tick=tick, base_chunk_id=base,
                    chunk_id=uuid.uuid4().hex, delay_steps=delay, previous=previous)

    def commit(self, ticket, actions):
        meta = {key: value for key, value in ticket.items() if key != "previous"}
        self.current = meta, actions.copy()
        return meta
