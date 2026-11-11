"""Regression checks for disconnects between RTC observation and execution."""

import asyncio

from pi.services.ws_json_api import JSONWebSocketAPIServer


class Socket:
    def __init__(self, closed=False):
        self.close_code = 1000 if closed else None
        self.sent = []
        self.closed = False

    async def send(self, message):
        if self.close_code is not None:
            raise RuntimeError("closed")
        self.sent.append(message)

    async def close(self, **_kwargs):
        self.close_code = 1000
        self.closed = True


def test_closed_idle_consumer_cannot_steal_next_session_chunk():
    async def check():
        api = JSONWebSocketAPIServer(heartbeat_interval=0)
        api.send_action({"chunk": "new-session"})
        old = Socket(closed=True)
        await asyncio.wait_for(api._handle_to_client(old), timeout=1)
        assert api.action_out_queue.get_nowait() == {"chunk": "new-session"}
        assert old.sent == []
    asyncio.run(check())


def test_replacement_closes_previous_action_socket():
    async def check():
        api = JSONWebSocketAPIServer(heartbeat_interval=0)
        old, new = Socket(), Socket()
        api._ws_to_client = old
        api._stop_event = asyncio.Event()
        api._stop_event.set()
        api.send_action({"chunk": "new-session"})
        await asyncio.wait_for(api._handle_to_client(new), timeout=1)
        assert old.closed
        assert len(new.sent) == 1
        assert old.sent == []
    asyncio.run(check())
