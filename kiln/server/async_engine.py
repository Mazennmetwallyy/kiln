"""Runs the (synchronous, single-threaded) engine on its own thread and talks to asyncio.

Only the engine thread ever touches the engine. The asyncio side sends it commands (add, abort,
stats) through a thread-safe queue, and the engine thread posts each request's outputs back to that
request's asyncio.Queue via `call_soon_threadsafe`. While the model runs a step, the event loop keeps
accepting connections and streaming tokens.
"""

from __future__ import annotations

import asyncio
import queue
import threading
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any

from kiln.engine.engine import EngineStats, LLMEngine, RequestOutput
from kiln.sampling import SamplingParams


@dataclass
class _Stream:
    loop: asyncio.AbstractEventLoop
    queue: asyncio.Queue[RequestOutput | BaseException]


class AsyncEngine:
    def __init__(self, engine: LLMEngine):
        self.engine = engine
        self._cmds: queue.Queue[tuple[Any, ...]] = queue.Queue()
        self._streams: dict[str, _Stream] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.on_step: list[Any] = []  # callbacks(outputs, stats) run on the engine thread, e.g. metrics

    # ---- lifecycle -----------------------------------------------------------------------------
    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="kiln-engine", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._cmds.put(("wake",))
        if self._thread is not None:
            self._thread.join(timeout=10)

    # ---- asyncio-side API ----------------------------------------------------------------------
    async def submit(self, request_id: str, prompt_ids: list[int],
                     params: SamplingParams) -> AsyncGenerator[RequestOutput, None]:
        """Queue a request. Raises ValueError here, before any output, if the request is invalid."""
        loop = asyncio.get_running_loop()
        accepted: asyncio.Future[None] = loop.create_future()
        out_queue: asyncio.Queue[RequestOutput | BaseException] = asyncio.Queue()
        self._cmds.put(("add", request_id, prompt_ids, params, accepted, _Stream(loop, out_queue)))
        await accepted
        return self._iterate(request_id, out_queue)

    async def stats(self) -> EngineStats:
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[EngineStats] = loop.create_future()
        self._cmds.put(("stats", fut, loop))
        return await fut

    async def _iterate(self, request_id: str,
                       out_queue: asyncio.Queue[RequestOutput | BaseException]) -> AsyncGenerator[RequestOutput, None]:
        finished = False
        try:
            while True:
                item = await out_queue.get()
                if isinstance(item, BaseException):
                    finished = True
                    raise item
                if item.finished:
                    finished = True
                yield item
                if finished:
                    return
        finally:
            if not finished:  # the consumer left early (client disconnected, cancelled, ...)
                self._cmds.put(("abort", request_id))

    # ---- engine thread -------------------------------------------------------------------------
    def _run(self) -> None:
        while not self._stop.is_set():
            idle = not self.engine.has_unfinished()
            try:
                self._handle(self._cmds.get(timeout=0.1) if idle else self._cmds.get_nowait())
                while True:
                    self._handle(self._cmds.get_nowait())
            except queue.Empty:
                pass
            if self.engine.has_unfinished():
                self._step()

    def _step(self) -> None:
        try:
            outputs = self.engine.step()
        except Exception as e:  # don't kill the thread: fail the in-flight requests instead
            for rid, stream in list(self._streams.items()):
                stream.loop.call_soon_threadsafe(stream.queue.put_nowait, e)
                self.engine.abort(rid)
            self._streams.clear()
            return
        for out in outputs:
            target = self._streams.get(out.request_id)
            if target is not None:
                target.loop.call_soon_threadsafe(target.queue.put_nowait, out)
                if out.finished:
                    del self._streams[out.request_id]
        for cb in self.on_step:
            cb(outputs, self.engine.stats())

    def _handle(self, cmd: tuple[Any, ...]) -> None:
        kind = cmd[0]
        if kind == "add":
            _, rid, prompt_ids, params, accepted, stream = cmd
            try:
                self.engine.add_request(rid, prompt_ids, params)
            except Exception as e:
                stream.loop.call_soon_threadsafe(_set_exception, accepted, e)
                return
            self._streams[rid] = stream
            stream.loop.call_soon_threadsafe(_set_result, accepted, None)
        elif kind == "abort":
            self.engine.abort(cmd[1])
            self._streams.pop(cmd[1], None)
        elif kind == "stats":
            _, fut, loop = cmd
            loop.call_soon_threadsafe(_set_result, fut, self.engine.stats())


def _set_result(fut: asyncio.Future[Any], value: Any) -> None:
    if not fut.done():
        fut.set_result(value)


def _set_exception(fut: asyncio.Future[Any], exc: BaseException) -> None:
    if not fut.done():
        fut.set_exception(exc)
