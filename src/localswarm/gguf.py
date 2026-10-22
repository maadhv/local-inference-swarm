"""Small GGUF metadata reader used to calculate a layer-based split."""
import struct


_SCALAR_SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def _read_exact(f, size):
    data = f.read(size)
    if len(data) != size:
        raise ValueError("truncated GGUF metadata")
    return data


def _read_string(f):
    (size,) = struct.unpack("<Q", _read_exact(f, 8))
    if size > 1_000_000:
        raise ValueError("unreasonable GGUF metadata string length")
    return _read_exact(f, size).decode("utf-8")


def _skip_value(f, kind):
    if kind in _SCALAR_SIZES:
        f.seek(_SCALAR_SIZES[kind], 1)
    elif kind == 8:
        (size,) = struct.unpack("<Q", _read_exact(f, 8))
        if size > 1_000_000:
            raise ValueError("unreasonable GGUF metadata string length")
        f.seek(size, 1)
    elif kind == 9:
        element_type, count = struct.unpack("<IQ", _read_exact(f, 12))
        if count > 100_000_000:
            raise ValueError("unreasonable GGUF metadata array length")
        if element_type == 8:
            for _ in range(count):
                _skip_value(f, element_type)
        elif element_type in _SCALAR_SIZES:
            f.seek(_SCALAR_SIZES[element_type] * count, 1)
        else:
            raise ValueError(f"unsupported GGUF metadata array type {element_type}")
    else:
        raise ValueError(f"unsupported GGUF metadata type {kind}")


def get_layer_count(path):
    """Return the model's transformer block count from GGUF metadata."""
    with open(path, "rb") as f:
        if _read_exact(f, 4) != b"GGUF":
            raise ValueError("file does not start with GGUF magic")
        (version,) = struct.unpack("<I", _read_exact(f, 4))
        if version not in (2, 3):
            raise ValueError(f"unsupported GGUF version {version}; expected v2 or v3")
        _, metadata_count = struct.unpack("<QQ", _read_exact(f, 16))
        if metadata_count > 1_000_000:
            raise ValueError("unreasonable GGUF metadata entry count")
        for _ in range(metadata_count):
            key = _read_string(f)
            (kind,) = struct.unpack("<I", _read_exact(f, 4))
            if key.endswith(".block_count") and kind in (4, 5, 10, 11):
                sizes = {4: ("<I", 4), 5: ("<i", 4), 10: ("<Q", 8), 11: ("<q", 8)}
                fmt, size = sizes[kind]
                (count,) = struct.unpack(fmt, _read_exact(f, size))
                if count <= 0 or count > 100_000:
                    raise ValueError(f"invalid block_count value {count}")
                return count
            _skip_value(f, kind)
    raise ValueError("no *.block_count entry found in GGUF metadata")
