"""Bounded asynchronous execution shared by benchmark runners."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable


async def map_concurrently[InputT, ResultT](
    items: Iterable[InputT],
    worker: Callable[[InputT], Awaitable[ResultT]],
    *,
    limit: int,
) -> AsyncIterator[ResultT]:
    """Yield worker results as they complete with at most ``limit`` active tasks."""
    if limit < 1:
        raise ValueError("concurrency limit must be at least 1")

    iterator = iter(items)
    pending: set[asyncio.Task[ResultT]] = set()

    def fill() -> None:
        while len(pending) < limit:
            try:
                item = next(iterator)
            except StopIteration:
                return
            pending.add(asyncio.create_task(worker(item)))

    fill()
    try:
        while pending:
            done, pending = await asyncio.wait(
                pending,
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in done:
                yield task.result()
            fill()
    finally:
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
