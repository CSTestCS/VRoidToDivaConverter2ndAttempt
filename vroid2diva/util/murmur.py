"""The MurmurHash variant DIVA tools use for IDs (same as MikuMikuLibrary)."""


def murmur_hash(value: str) -> int:
    data = value.upper().encode("utf-8")
    m = 0x7FD652AD
    mask = 0xFFFFFFFF
    h = 0xDEADBEEF
    n = len(data)
    i = 0
    while n - i >= 4:
        h = (h + int.from_bytes(data[i:i + 4], "little")) & mask
        h = (h * m) & mask
        h ^= h >> 16
        i += 4
    rest = n - i
    if rest:
        if rest == 3:
            h = (h + (data[i + 2] << 16)) & mask
        if rest >= 2:
            h = (h + (data[i + 1] << 8)) & mask
        h = (h + data[i]) & mask
        h = (h * m) & mask
        h ^= h >> 16
    h = (h * m) & mask
    h ^= h >> 10
    h = (h * m) & mask
    h ^= h >> 17
    return h
