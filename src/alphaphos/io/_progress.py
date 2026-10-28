"""Byte-level read progress for the report readers.

The readers keep their single ``pd.read_csv`` / ``pd.read_parquet`` call; with a
progress callback they hand it a file object that counts the bytes it consumes.
Parsing is therefore unchanged -- only the source of the bytes differs.
"""

from __future__ import annotations

import io
from collections.abc import Callable, Sequence
from pathlib import Path

import pyarrow.parquet as pq

# progress(bytes_done, bytes_total); always ends with bytes_done == bytes_total.
ProgressCallback = Callable[[int, int], None]

_BUFFER_SIZE = 1 << 20


class _CountingReader(io.RawIOBase):
    def __init__(self, raw: io.RawIOBase, total: int, callback: ProgressCallback):
        self._raw = raw
        self._total = max(total, 1)
        self._done = 0
        self._callback = callback

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        return self._raw.seek(offset, whence)

    def tell(self) -> int:
        return self._raw.tell()

    def readinto(self, buffer) -> int:
        n = self._raw.readinto(buffer)
        if n:
            self._done += n
            # Parquet reads the footer more than once, so bytes read can exceed
            # the estimated total; never report past 100 %.
            self._callback(min(self._done, self._total), self._total)
        return n

    def close(self) -> None:
        self._raw.close()
        super().close()


def parquet_bytes_to_read(path: Path, columns: Sequence[str]) -> int:
    """Compressed size of the column chunks a pruned parquet read touches."""
    meta = pq.ParquetFile(path).metadata
    wanted = set(columns)
    return sum(
        meta.row_group(i).column(j).total_compressed_size
        for i in range(meta.num_row_groups)
        for j in range(meta.num_columns)
        if meta.row_group(i).column(j).path_in_schema in wanted
    )


def open_counting(path: Path, total: int, callback: ProgressCallback) -> io.BufferedReader:
    """Binary file object over ``path`` that reports bytes read to ``callback``."""
    return io.BufferedReader(
        _CountingReader(open(path, "rb", buffering=0), total, callback),
        buffer_size=_BUFFER_SIZE,
    )
