"""Overlapped host<->device data feeding for the pretraining loops.

The training loop is bound by two things that have nothing to do with the GPU
maths: reading token windows out of the memory-mapped shards, and waiting for a
micro-batch to arrive in HBM.  Both are hidden here:

* batches are produced on a background thread so the mmap read of step *n+1*
  happens while step *n* is on the GPU,
* batches are staged in pinned host memory so ``tensor.to(device,
  non_blocking=True)`` is a real asynchronous copy instead of a synchronous one.

Neither changes *which* tokens a step sees.  Each produced batch carries the
position it was drawn from, so a checkpoint records the position of the last
batch the optimizer actually consumed -- not the one the prefetcher has raced
ahead to.
"""

from __future__ import annotations

import queue
import threading

import torch

_SENTINEL = object()


def pin(batch: torch.Tensor) -> torch.Tensor:
    """Move a batch into pinned host memory when that is possible."""

    if not torch.cuda.is_available() or not isinstance(batch, torch.Tensor):
        return batch

    try:

        return batch.pin_memory()

    except (RuntimeError, NotImplementedError):

        return batch


class PrefetchStream:
    """Wrap an iterable of batch tensors with a bounded, threaded prefetch queue.

    ``source`` only needs to implement ``__iter__``/``__next__``; the position
    attribute (if the source exposes one) or ``(source.epoch,
    source.batch_in_epoch)`` is captured for every batch so training state stays
    exactly resumable.
    """

    def __init__(
        self,
        source,
        depth: int = 2,
        pin_memory: bool = True,
        name: str = "prefetch",
        restart: bool | None = None,
        max_restarts: int = 100_000,
    ):
        """``source`` may be an iterator or a re-iterable (``__iter__`` only).

        For a re-iterable (e.g. the generator based ``StreamingDataLoader``) the
        producer starts a fresh pass whenever one ends, which is exactly the
        epoch rollover the training loop used to do by hand.  ``restart`` can
        force either behaviour; it defaults to "yes if the source is not its own
        iterator".
        """

        self.source = source

        self.depth = max(int(depth), 1)

        self.pin_memory = pin_memory

        self.position = getattr(source, "position", None)

        self.restart = (not hasattr(source, "__next__")) if restart is None else restart

        self.max_restarts = max(int(max_restarts), 1)

        self.restarts = 0

        self._queue: queue.Queue = queue.Queue(maxsize=self.depth)

        self._closed = threading.Event()

        self._thread = threading.Thread(target=self._produce, name=name, daemon=True)

        self._thread.start()

    # -- producer ---------------------------------------------------------
    def _produce(self):

        iterator = iter(self.source)

        while not self._closed.is_set():

            try:

                batch = next(iterator)

            except StopIteration:

                if not self.restart or self.restarts >= self.max_restarts:

                    self._put(_SENTINEL)

                    return

                self.restarts += 1

                iterator = iter(self.source)

                continue

            except Exception as error:  # surface real failures in the consumer

                self._put(error)

                return

            position = getattr(self.source, "position", None)

            if position is None:

                position = (getattr(self.source, "epoch", 0), getattr(self.source, "batch_in_epoch", 0))

            if self.pin_memory:

                batch = pin(batch)

            if not self._put((batch, position)):

                return

    def _put(self, item) -> bool:
        """Insert into the queue, staying interruptible while it is full."""

        while not self._closed.is_set():

            try:

                self._queue.put(item, timeout=0.25)

                return True

            except queue.Full:

                continue

        return False

    # -- consumer ---------------------------------------------------------
    def __iter__(self):

        return self

    def __next__(self) -> torch.Tensor:

        if self._closed.is_set():

            raise StopIteration

        item = self._queue.get()

        if item is _SENTINEL:

            self.close()

            raise StopIteration

        if isinstance(item, Exception):

            self.close()

            raise item

        batch, self.position = item

        return batch

    def close(self):

        self._closed.set()

        if self._thread.is_alive():

            self._thread.join(timeout=5.0)

    def __del__(self):

        try:

            self.close()

        except Exception:

            pass


def to_device(batch: torch.Tensor, device) -> torch.Tensor:
    """Non-blocking hop to the accelerator (falls back for CPU-only builds)."""

    if batch.device == device:

        return batch

    if torch.cuda.is_available() and batch.is_pinned():

        return batch.to(device, non_blocking=True)

    return batch.to(device)
