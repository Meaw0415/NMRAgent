from __future__ import annotations

from pathlib import Path

import lmdb

_ENV_CACHE: dict[tuple[str, bool, bool, bool], lmdb.Environment] = {}


def open_shared_lmdb(
    path: str | Path,
    *,
    readonly: bool = True,
    lock: bool = False,
    readahead: bool = False,
    subdir: bool = False,
) -> lmdb.Environment:
    key = (str(Path(path).resolve()), readonly, lock, readahead, subdir)
    env = _ENV_CACHE.get(key)
    if env is not None:
        return env
    env = lmdb.open(
        key[0],
        readonly=readonly,
        lock=lock,
        readahead=readahead,
        subdir=subdir,
    )
    _ENV_CACHE[key] = env
    return env
