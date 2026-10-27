"""Host-wide native mount reservations, outside every generated-code mount.

The authority is shared by all registries in this OS account. Reservations are
permanent audit records: deleting an output tree cannot grant mount reuse.
"""

from __future__ import annotations

import os
import sqlite3
import stat
from contextlib import closing
from pathlib import Path

from src.research.registry import canonical, content_hash


def _authority_path() -> Path:
    return Path.home() / ".local/state/quantagentlab/native-mounts.sqlite3"


def canonical_path(path: str | Path) -> Path:
    value = Path(path).absolute()
    if any(parent.is_symlink() for parent in (value, *value.parents)):
        raise ValueError("native reserved paths cannot use symbolic links")
    if value.exists():
        info = value.stat()
        if not stat.S_ISDIR(info.st_mode) and (
            not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
        ):
            raise ValueError("native source/control files must be ordinary files without hardlinks")
    return value.resolve()


def control_path(path: str | Path) -> Path:
    value = canonical_path(path)
    if value.is_dir():
        for child in value.rglob("*"):
            canonical_path(child)
    return value


def overlaps(first: Path, second: Path) -> bool:
    return first.is_relative_to(second) or second.is_relative_to(first)


class NativeMountAuthority:
    def __init__(self, path: Path | None = None):
        self.path = canonical_path(path or _authority_path())
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)

    @staticmethod
    def _schema(db):
        db.execute("PRAGMA busy_timeout=30000")
        db.execute("CREATE TABLE IF NOT EXISTS mounts(hash TEXT PRIMARY KEY,payload BLOB NOT NULL)")
        db.execute("CREATE TABLE IF NOT EXISTS controls(hash TEXT PRIMARY KEY,path TEXT NOT NULL)")
        for table in ("mounts", "controls"):
            for action in ("UPDATE", "DELETE"):
                db.execute(
                    f"CREATE TRIGGER IF NOT EXISTS immutable_{table}_{action} BEFORE {action} ON {table} BEGIN SELECT RAISE(ABORT,'native mount history is immutable'); END"
                )
        db.commit()

    def protect(self, *paths: Path) -> None:
        import json

        values = {control_path(path) for path in (*paths, self.path.parent)}
        with closing(sqlite3.connect(self.path, timeout=30)) as db, db:
            self._schema(db)
            db.execute("BEGIN IMMEDIATE")
            for key, raw in db.execute("SELECT hash,payload FROM mounts"):
                if content_hash(raw) != key:
                    raise ValueError("native mount authority integrity differs")
                old = json.loads(raw)
                exposed = [
                    Path(old["writable"]),
                    *map(Path, old.get("inputs", old.get("protected", []))),
                ]
                if any(overlaps(exposure, path) for path in values for exposure in exposed):
                    raise ValueError(
                        "control state overlaps a reserved native writable mount or source input"
                    )
            for path in values:
                db.execute(
                    "INSERT OR IGNORE INTO controls VALUES (?,?)",
                    (content_hash(str(path).encode()), str(path)),
                )
        os.chmod(self.path, 0o600)

    def reserve(
        self,
        identity: dict,
        *,
        writable: Path,
        inputs: tuple[Path, ...],
        controls: tuple[Path, ...] = (),
    ) -> str:
        work = canonical_path(writable)
        readonly = {canonical_path(path) for path in inputs}
        private = {control_path(path) for path in (*controls, self.path.parent)}
        if any(overlaps(work, path) for path in readonly | private):
            raise ValueError("native writable mount overlaps input, source or control state")
        if any(overlaps(source, control) for source in readonly for control in private):
            raise ValueError("native source input overlaps never-expose control state")
        payload = {
            "identity": identity,
            "writable": str(work),
            "inputs": sorted(map(str, readonly)),
            "controls": sorted(map(str, private)),
        }
        raw = canonical(payload)
        key = content_hash(raw)
        with closing(sqlite3.connect(self.path, timeout=30)) as db, db:
            self._schema(db)
            db.execute("BEGIN IMMEDIATE")
            import json

            for control_key, control in db.execute("SELECT hash,path FROM controls"):
                if content_hash(control.encode()) != control_key or overlaps(work, Path(control)):
                    raise ValueError("native writable mount overlaps reserved control state")
                if any(overlaps(source, Path(control)) for source in readonly):
                    raise ValueError("native source input overlaps never-expose control state")

            for old_key, old_raw in db.execute("SELECT hash,payload FROM mounts"):
                if content_hash(old_raw) != old_key:
                    raise ValueError("native mount authority integrity differs")
                if old_key == key:
                    return key
                old = json.loads(old_raw)
                old_work = Path(old["writable"])
                old_inputs = set(map(Path, old.get("inputs", old.get("protected", []))))
                old_controls = set(map(Path, old.get("controls", old.get("protected", []))))
                if any(
                    overlaps(source, control) for source in readonly for control in old_controls
                ) or any(overlaps(source, control) for source in old_inputs for control in private):
                    raise ValueError("native source input overlaps never-expose control state")
                if old["identity"] == identity and old_work == work:
                    continue
                if (
                    overlaps(work, old_work)
                    or any(overlaps(work, path) for path in old_inputs | old_controls)
                    or any(overlaps(old_work, path) for path in readonly | private)
                ):
                    raise ValueError(
                        "native mount overlaps another reserved writable/control boundary"
                    )
            db.execute("INSERT INTO mounts VALUES (?,?)", (key, raw))
        os.chmod(self.path, 0o600)
        return key
