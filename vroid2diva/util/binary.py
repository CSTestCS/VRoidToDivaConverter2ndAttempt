"""Little helpers for reading and writing DIVA binary files.

The writer mirrors the behaviour of MikuMikuLibrary's EndianBinaryWriter:

* ``offset(fn)`` writes a 4-byte placeholder and *schedules* ``fn``.  Scheduled
  writes run depth-first once the parent block is finished, so a structure is
  followed by the data it points to.
* Offsets are relative to the base offset that was active when the offset was
  scheduled (``push_base`` / ``pop_base``), which is how object data inside an
  object set ends up relative to the start of each object.
* ``string_offset(s)`` writes a pointer into a de-duplicated string pool that
  is appended to the end of the file.
"""

from __future__ import annotations

import struct
from typing import Callable, Dict, List, Optional


def align_up(value: int, alignment: int) -> int:
    if alignment <= 1:
        return value
    return (value + alignment - 1) & ~(alignment - 1)


class _Scheduled:
    __slots__ = ("field", "fn", "align", "mode", "base")

    def __init__(self, field, fn, align, mode, base):
        self.field = field
        self.fn = fn
        self.align = align
        self.mode = mode  # "left", "center" or "none"
        self.base = base


class BinaryWriter:
    def __init__(self, endian: str = "<"):
        self.endian = endian
        self.buf = bytearray()
        self.pos = 0
        self._scheduled: List[_Scheduled] = []
        self._bases: List[int] = [0]
        self._strings: Dict[str, List[int]] = {}
        self._string_order: List[str] = []

    # -- primitive output -------------------------------------------------
    def tell(self) -> int:
        return self.pos

    def seek(self, pos: int) -> None:
        self.pos = pos

    def write(self, data: bytes) -> None:
        end = self.pos + len(data)
        if end > len(self.buf):
            self.buf.extend(b"\0" * (end - len(self.buf)))
        self.buf[self.pos:end] = data
        self.pos = end

    def _pack(self, fmt: str, *values) -> None:
        self.write(struct.pack(self.endian + fmt, *values))

    def u8(self, v: int) -> None:
        self._pack("B", v & 0xFF)

    def u16(self, v: int) -> None:
        self._pack("H", v & 0xFFFF)

    def i32(self, v: int) -> None:
        self._pack("i", v)

    def u32(self, v: int) -> None:
        self._pack("I", v & 0xFFFFFFFF)

    def f32(self, v: float) -> None:
        self._pack("f", float(v))

    def floats(self, values) -> None:
        values = [float(x) for x in values]
        self._pack("%df" % len(values), *values)

    def nulls(self, count: int) -> None:
        self.write(b"\0" * count)

    def align(self, alignment: int) -> None:
        self.nulls(align_up(self.pos, alignment) - self.pos)

    def fixed_string(self, s: str, length: int) -> None:
        data = s.encode("utf-8")[: length - 1]
        self.write(data + b"\0" * (length - len(data)))

    def cstring(self, s: str) -> None:
        self.write(s.encode("utf-8") + b"\0")

    def matrix(self, m) -> None:
        """Write a 4x4 matrix stored row-major in DIVA (row-vector) layout.

        ``m`` is expected to already be in the DIVA layout (rows are the axes,
        translation in the last row)."""
        self.floats([float(x) for row in m for x in row])

    # -- base offsets -----------------------------------------------------
    def push_base(self, pos: Optional[int] = None) -> None:
        self._bases.append(self.pos if pos is None else pos)

    def pop_base(self) -> None:
        self._bases.pop()

    @property
    def base(self) -> int:
        return self._bases[-1]

    # -- offsets ----------------------------------------------------------
    def offset(self, fn: Optional[Callable[[], None]], align: int = 0, mode: str = "left",
               condition: bool = True) -> None:
        """Write a placeholder offset and schedule ``fn`` to write its target."""
        field = self.pos
        self.u32(0)
        if fn is None or not condition:
            return
        self._scheduled.append(_Scheduled(field, fn, align, mode, self.base))

    def string_offset(self, s: Optional[str]) -> None:
        field = self.pos
        self.u32(0)
        if s is None:
            return
        if s not in self._strings:
            self._strings[s] = []
            self._string_order.append(s)
        self._strings[s].append(field)

    def _run_one(self, item: _Scheduled, base: int) -> None:
        # Remove nested items from the main list and run them right after.
        if item.mode in ("left", "center"):
            self.align(item.align)
        start = self.pos
        saved = self._scheduled
        self._scheduled = []
        item.fn()
        children = self._scheduled
        self._scheduled = saved
        if item.mode in ("right", "center"):
            self.align(item.align)
        if item.base > 0:
            base = item.base
        for child in children:
            self._run_one(child, base)
        end = self.pos
        self.pos = item.field
        self.u32(start - base)
        self.pos = end

    def flush(self) -> None:
        items = self._scheduled
        self._scheduled = []
        for item in items:
            self._run_one(item, 0)

    def finish(self, string_align: int = 16) -> bytes:
        """Run pending writes, append the string pool and return the bytes."""
        self.flush()
        self.pos = len(self.buf)
        if self._string_order:
            self.align(string_align)
            for s in self._string_order:
                at = self.pos
                self.cstring(s)
                end = self.pos
                for field in self._strings[s]:
                    self.pos = field
                    self.u32(at)
                self.pos = end
            self.align(string_align)
        return bytes(self.buf)


class BinaryReader:
    def __init__(self, data: bytes, endian: str = "<", pos: int = 0):
        self.data = data
        self.endian = endian
        self.pos = pos
        self._bases: List[int] = [0]

    def push_base(self, pos: Optional[int] = None) -> None:
        self._bases.append(self.pos if pos is None else pos)

    def pop_base(self) -> None:
        self._bases.pop()

    @property
    def base(self) -> int:
        return self._bases[-1]

    def read(self, n: int) -> bytes:
        out = self.data[self.pos:self.pos + n]
        if len(out) != n:
            raise EOFError("Unexpected end of data at 0x%X" % self.pos)
        self.pos += n
        return out

    def _unpack(self, fmt: str):
        size = struct.calcsize(fmt)
        return struct.unpack(self.endian + fmt, self.read(size))

    def u8(self) -> int:
        return self._unpack("B")[0]

    def u16(self) -> int:
        return self._unpack("H")[0]

    def i16(self) -> int:
        return self._unpack("h")[0]

    def u32(self) -> int:
        return self._unpack("I")[0]

    def i32(self) -> int:
        return self._unpack("i")[0]

    def u64(self) -> int:
        return self._unpack("Q")[0]

    def i64(self) -> int:
        return self._unpack("q")[0]

    def f32(self) -> float:
        return self._unpack("f")[0]

    def f64(self) -> float:
        return self._unpack("d")[0]

    def floats(self, n: int):
        return list(self._unpack("%df" % n))

    def matrix(self):
        v = self.floats(16)
        return [v[0:4], v[4:8], v[8:12], v[12:16]]

    def offset(self) -> int:
        """Read an offset; returns an absolute position (0 stays 0)."""
        value = self.u32()
        return value + self.base if value else 0

    def cstring_at(self, pos: int) -> str:
        end = self.data.index(b"\0", pos)
        return self.data[pos:end].decode("utf-8", errors="replace")

    def string_offset(self) -> Optional[str]:
        value = self.u32()
        if not value:
            return None
        return self.cstring_at(value + self.base)

    def fixed_string(self, length: int) -> str:
        raw = self.read(length)
        return raw.split(b"\0", 1)[0].decode("utf-8", errors="replace")

    def at(self, pos: int):
        return _At(self, pos)


class _At:
    def __init__(self, reader: BinaryReader, pos: int):
        self.reader = reader
        self.target = pos

    def __enter__(self):
        self.saved = self.reader.pos
        self.reader.pos = self.target
        return self.reader

    def __exit__(self, *exc):
        self.reader.pos = self.saved
        return False
