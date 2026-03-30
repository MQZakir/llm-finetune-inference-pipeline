"""
Streaming token generator utilities.

Provides higher-level streaming abstractions on top of the base engine,
including detokenisation buffering (to avoid printing partial unicode/BPE
tokens), stop-sequence detection, and async streaming for web server use.
"""

from __future__ import annotations

import asyncio
import re
from collections import deque
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from typing import Callable

from src.inference.engine import InferenceEngine, SamplingConfig


@dataclass
class StreamConfig:
    """Configuration for the streaming generator."""

    # Buffering — accumulate this many chars before yielding
    # Prevents sending incomplete multi-byte unicode characters
    min_yield_chars: int = 1

    # Stop sequences to detect mid-stream
    stop_sequences: list[str] = field(default_factory=list)

    # Whether to strip leading whitespace from the first token
    strip_leading_space: bool = True

    # Max chars to buffer for stop-sequence detection
    # (must be >= max stop sequence length)
    stop_detection_buffer: int = 64


class StreamingGenerator:
    """
    Wraps an InferenceEngine to provide buffered, stop-aware streaming.

    Handles edge cases that raw engine streaming misses:
      - Stop sequences that span multiple tokens
      - Partial BPE tokens at UTF-8 boundaries
      - Leading whitespace stripping
      - Per-token callbacks (e.g. for logging)

    Example
    -------
    >>> gen = StreamingGenerator(engine)
    >>> for chunk in gen.stream("Tell me a story", SamplingConfig()):
    ...     print(chunk, end='', flush=True)
    """

    def __init__(
        self,
        engine: InferenceEngine,
        on_token: Callable[[str], None] | None = None,
    ) -> None:
        self._engine = engine
        self._on_token = on_token

    def stream(
        self,
        prompt: str,
        sampling: SamplingConfig,
        config: StreamConfig | None = None,
    ) -> Iterator[str]:
        """
        Yield text chunks as they are generated.

        Unlike the raw engine stream, this handles stop-sequence detection
        across token boundaries and ensures clean unicode output.
        """
        cfg = config or StreamConfig(stop_sequences=sampling.stop_sequences)

        # Rolling buffer for stop-sequence detection
        buffer = ""
        is_first = True
        stop_patterns = [re.escape(s) for s in cfg.stop_sequences]
        stop_re = re.compile("|".join(stop_patterns)) if stop_patterns else None

        for raw_token in self._engine.stream(prompt, sampling):
            if self._on_token:
                self._on_token(raw_token)

            token = raw_token
            if is_first and cfg.strip_leading_space:
                token = token.lstrip()
                is_first = False
            elif is_first:
                is_first = False

            if not token:
                continue

            buffer += token

            # Check for stop sequences in the buffer
            if stop_re and stop_re.search(buffer):
                # Yield everything up to the stop sequence
                match = stop_re.search(buffer)
                if match:
                    yield_text = buffer[: match.start()]
                    if yield_text:
                        yield yield_text
                return

            # Yield from buffer when it's long enough to be safe
            if len(buffer) >= cfg.stop_detection_buffer + cfg.min_yield_chars:
                safe_len = len(buffer) - cfg.stop_detection_buffer
                yield buffer[:safe_len]
                buffer = buffer[safe_len:]

        # Flush remaining buffer
        if buffer:
            if stop_re:
                match = stop_re.search(buffer)
                if match:
                    buffer = buffer[: match.start()]
            if buffer:
                yield buffer

    async def astream(
        self,
        prompt: str,
        sampling: SamplingConfig,
        config: StreamConfig | None = None,
    ) -> AsyncIterator[str]:
        """
        Async streaming generator for use with FastAPI / websockets.

        Runs the synchronous engine in a thread pool to avoid blocking
        the event loop.
        """
        loop = asyncio.get_event_loop()
        queue: asyncio.Queue[str | None] = asyncio.Queue(maxsize=64)

        def _producer() -> None:
            try:
                for chunk in self.stream(prompt, sampling, config):
                    loop.call_soon_threadsafe(queue.put_nowait, chunk)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)  # sentinel

        thread = asyncio.to_thread(_producer)
        asyncio.create_task(thread)

        while True:
            chunk = await queue.get()
            if chunk is None:
                break
            yield chunk

    def collect(
        self,
        prompt: str,
        sampling: SamplingConfig,
        config: StreamConfig | None = None,
    ) -> str:
        """Generate and collect all output as a single string."""
        return "".join(self.stream(prompt, sampling, config))


class TokenCounter:
    """
    Counts tokens generated in a stream and emits periodic stats.

    Useful for monitoring token budgets in production.
    """

    def __init__(self, budget: int | None = None) -> None:
        self.budget = budget
        self.count = 0
        self._history: deque[tuple[float, int]] = deque(maxlen=100)

    def on_token(self, token: str) -> None:
        import time
        self.count += 1
        self._history.append((time.monotonic(), self.count))

    def tokens_per_second(self) -> float:
        if len(self._history) < 2:
            return 0.0
        t0, c0 = self._history[0]
        t1, c1 = self._history[-1]
        elapsed = t1 - t0
        return (c1 - c0) / elapsed if elapsed > 0 else 0.0

    @property
    def budget_remaining(self) -> int | None:
        if self.budget is None:
            return None
        return max(0, self.budget - self.count)

    @property
    def budget_exceeded(self) -> bool:
        return self.budget is not None and self.count >= self.budget
