"""
LocalStorageManager — drop-in replacement for AzureFileStorageManager.
Mirrors the CommunityRAPP storage layout:
  shared_memories/memory.json   — shared memories
  memory/{guid}/user_memory.json — per-user memories
Data lives in .brainstem_data/ next to this file.
"""

import os
import re
import json
import tempfile
import threading
import hashlib
from datetime import datetime
from pathlib import Path

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".brainstem_data")
_path_locks = {}
_path_locks_guard = threading.Lock()

# CommunityRAPP's user-context contract: only a strict GUID gets a per-user
# store; anything else falls back to shared memory (see set_memory_context).
_GUID_RE = re.compile(
    r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
    re.IGNORECASE,
)


class _FileEntry(str):
    """A directory listing entry that satisfies BOTH client styles: cloud-ported
    agents read `.name` (Azure SDK objects), pre-parity local agents treat
    entries as plain strings (`"x.txt" in list_files(...)`)."""

    is_directory = False

    def __new__(cls, name, is_directory=False):
        obj = super().__new__(cls, name)
        obj.is_directory = is_directory
        return obj

    @property
    def name(self):
        return str(self)


def _safe_join(*parts):
    """Join path parts under _DATA_DIR and refuse anything that escapes it.

    user_guid and agent-supplied file paths are attacker-influenced (they come from
    LLM tool-call arguments), so a value like '../../.env' or an absolute path must
    not be able to read or write outside the data directory. Returns an absolute path
    guaranteed to live under _DATA_DIR, or raises ValueError."""
    base = os.path.abspath(_DATA_DIR)
    target = os.path.abspath(os.path.join(base, *[str(p) for p in parts]))
    try:
        contained = os.path.commonpath(
            [os.path.normcase(base), os.path.normcase(target)]) == os.path.normcase(base)
    except ValueError:
        contained = False
    if not contained:
        raise ValueError(f"path escapes data directory: {os.path.join(*[str(p) for p in parts])}")

    # Resolve only components that already exist. Resolving a destination while
    # another thread creates its parent can yield inconsistent Windows path
    # prefixes; the existing parent is enough to detect a symlink/junction escape.
    existing = target
    while not os.path.exists(existing):
        parent = os.path.dirname(existing)
        if parent == existing:
            break
        existing = parent
    real_base = os.path.realpath(base)
    real_existing = os.path.realpath(existing)
    try:
        contained = os.path.commonpath([
            os.path.normcase(real_base), os.path.normcase(real_existing)
        ]) == os.path.normcase(real_base)
    except ValueError:
        contained = False
    if not contained:
        raise ValueError(f"path escapes data directory: {os.path.join(*[str(p) for p in parts])}")
    return target


def _ensure_private_dir(path):
    os.makedirs(path, exist_ok=True)
    try:
        os.chmod(path, 0o700)
    except OSError:
        pass


def _lock_for(path):
    """Return a process-local lock shared by all managers writing this path."""
    key = os.path.normcase(os.path.abspath(path))
    with _path_locks_guard:
        return _path_locks.setdefault(key, threading.RLock())


def _atomic_write(path, write_fn, binary=False):
    """Write via a temp file in the same directory + os.replace, so a crash or a
    concurrent reader never sees a half-written (and on the next write, silently
    wiped) file. write_fn receives the open file handle."""
    directory = os.path.dirname(os.path.abspath(path))
    _ensure_private_dir(directory)
    fd, tmp = tempfile.mkstemp(
        prefix=f".{os.path.basename(path)}.", suffix=".tmp", dir=directory)
    try:
        mode = "wb" if binary else "w"
        encoding = None if binary else "utf-8"
        with os.fdopen(fd, mode, encoding=encoding) as f:
            write_fn(f)
            f.flush()
            os.fsync(f.fileno())
        with _lock_for(path):
            os.replace(tmp, path)
            try:
                os.chmod(path, 0o600)
            except OSError:
                pass
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


class AzureFileStorageManager:
    """
    Local-first shim that mirrors the AzureFileStorageManager API from
    CommunityRAPP.  Agents import this transparently via the shim in brainstem.py.
    """

    DEFAULT_MARKER_GUID = "c0p110t0-aaaa-bbbb-cccc-123456789abc"

    def __init__(self, share_name=None, **kwargs):
        self.current_guid = None
        normalized_share = str(share_name or "").strip().lower()
        self.share_name = normalized_share or None
        # Preserve the historical unnamed layout for bundled agents. Named Azure
        # shares receive deterministic, non-overlapping roots so cartridges cannot
        # accidentally read or overwrite another share's local data.
        self.storage_root = (
            os.path.join("shares", hashlib.sha256(normalized_share.encode("utf-8")).hexdigest())
            if normalized_share else ""
        )
        # Matches CommunityRAPP paths
        self.shared_memory_path = os.path.join(self.storage_root, "shared_memories")
        self.default_file_name = "memory.json"
        self.current_memory_path = self.shared_memory_path
        _ensure_private_dir(_DATA_DIR)

    def _scoped_path(self, file_path=""):
        return _safe_join(self.storage_root, file_path)

    # ── Context ───────────────────────────────────────────────────────────

    def set_memory_context(self, guid=None, user_guid=None):
        """Set the memory context — CommunityRAPP's exact contract: falsy or the
        default marker → shared (True); anything that is not a strict GUID →
        fall back to shared and return False, NEVER raise. The strict GUID
        format also guarantees a single safe path component, so a traversal
        attempt ('a/../b') lands in shared memory, not another user's store.

        The cloud names this parameter `guid`; the pre-parity local shim said
        `user_guid` — accept both keyword spellings."""
        if guid is None and user_guid is not None:
            guid = user_guid
        if not guid or guid == self.DEFAULT_MARKER_GUID:
            self.current_guid = None
            self.current_memory_path = self.shared_memory_path
            return True

        if not isinstance(guid, str) or not _GUID_RE.match(guid):
            self.current_guid = None
            self.current_memory_path = self.shared_memory_path
            return False

        self.current_guid = guid
        self.current_memory_path = os.path.join(self.storage_root, "memory", guid)
        return True

    # ── Core I/O ──────────────────────────────────────────────────────────

    def _file_path(self):
        """Return the absolute path for the current memory file.
        Shared:  .brainstem_data/shared_memories/memory.json
        User:    .brainstem_data/memory/{guid}/user_memory.json
        current_guid is regex-validated by set_memory_context; _safe_join
        contains it anyway (defense in depth).
        """
        if self.current_guid:
            rel = os.path.join(self.storage_root, "memory", self.current_guid, "user_memory.json")
        else:
            rel = os.path.join(self.shared_memory_path, self.default_file_name)
        path = _safe_join(rel)
        _ensure_private_dir(os.path.dirname(path))
        return path

    def read_json(self, file_path=None):
        """Read JSON data from local storage."""
        path = self._scoped_path(file_path) if file_path else self._file_path()
        if not os.path.exists(path):
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}

    def write_json(self, data, file_path=None):
        """Write JSON data to local storage (atomically)."""
        path = self._scoped_path(file_path) if file_path else self._file_path()
        with _lock_for(path):
            _atomic_write(path, lambda f: json.dump(data, f, indent=2, default=str))
        return True

    def update_json(self, update_fn, file_path=None):
        """Atomically read, transform, and replace a JSON document.

        The callback runs under a per-path lock and receives the current decoded
        value (or {} for a missing file). Decode/read failures are raised so a
        subsequent save cannot silently erase recoverable bytes.
        """
        path = self._scoped_path(file_path) if file_path else self._file_path()
        with _lock_for(path):
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    current = json.load(f)
            else:
                current = {}
            updated = update_fn(current)
            _atomic_write(path, lambda f: json.dump(updated, f, indent=2, default=str))
            return updated

    # ── File API — CommunityRAPP signatures ───────────────────────────────
    # Cloud shape: op(directory_name, file_name[, content]) — what every RAR
    # registry agent calls. The pre-parity local shape op(file_path[, content])
    # is still accepted (detected by arity) so user-authored agents from older
    # installs keep working. Cloud error contract: log-and-return failure
    # values (False/None/[]), never raise.

    _BINARY_EXTENSIONS = (
        '.pptx', '.docx', '.xlsx', '.pdf', '.zip',
        '.jpg', '.png', '.gif', '.jpeg', '.webp',
    )

    def _entry_path(self, directory_name, file_name):
        if file_name is None:
            return self._scoped_path(directory_name)  # legacy single-path form
        return self._scoped_path(os.path.join(str(directory_name), str(file_name)))

    def ensure_directory_exists(self, directory_name):
        """Create a (possibly nested) directory under the data root."""
        if not directory_name:
            return False
        try:
            _ensure_private_dir(self._scoped_path(directory_name))
            return True
        except (ValueError, OSError) as e:
            print(f"[local_storage] ensure_directory_exists failed: {e}")
            return False

    def read_file(self, directory_name, file_name=None):
        """Text content, bytes for known-binary extensions, None when missing."""
        try:
            full = self._entry_path(directory_name, file_name)
            if not os.path.exists(full):
                return None
            name = file_name if file_name is not None else directory_name
            if str(name).lower().endswith(self._BINARY_EXTENSIONS):
                with open(full, "rb") as f:
                    return f.read()
            with open(full, "rb") as f:
                raw = f.read()
            try:
                return raw.decode("utf-8")
            except UnicodeDecodeError:
                return raw
        except (ValueError, OSError) as e:
            print(f"[local_storage] read_file failed: {e}")
            return None

    def read_file_binary(self, directory_name, file_name=None):
        try:
            full = self._entry_path(directory_name, file_name)
            if not os.path.exists(full):
                return None
            with open(full, "rb") as f:
                return f.read()
        except (ValueError, OSError) as e:
            print(f"[local_storage] read_file_binary failed: {e}")
            return None

    def write_file(self, directory_name, file_name, content=None):
        # Legacy two-arg form: write_file(file_path, content).
        if content is None:
            directory_name, file_name, content = "", directory_name, file_name
        try:
            full = self._entry_path(directory_name, file_name)
            # Mirror the cloud's content coercion: bytes pass through, file-like
            # objects are drained, everything else goes through str().
            if isinstance(content, (bytes, bytearray)):
                data = bytes(content)
            elif hasattr(content, "read") and callable(content.read):
                content.seek(0)
                data = content.read()
                if not isinstance(data, (bytes, bytearray)):
                    data = str(data).encode("utf-8")
            else:
                data = str(content).encode("utf-8")
            with _lock_for(full):
                _atomic_write(full, lambda f: f.write(data), binary=True)
            return True
        except (ValueError, OSError) as e:
            print(f"[local_storage] write_file failed: {e}")
            return False

    def list_files(self, directory_name="", auto_create=True):
        """Entries carry .name (cloud style) and compare as strings (legacy)."""
        try:
            full = self._scoped_path(directory_name)
            if not os.path.exists(full):
                if auto_create and directory_name:
                    self.ensure_directory_exists(directory_name)
                return []
            return [
                _FileEntry(name, is_directory=os.path.isdir(os.path.join(full, name)))
                for name in os.listdir(full)
            ]
        except (ValueError, OSError) as e:
            print(f"[local_storage] list_files failed: {e}")
            return []

    def delete_file(self, directory_name, file_name=None):
        try:
            full = self._entry_path(directory_name, file_name)
            if os.path.exists(full):
                os.remove(full)
                return True
            return False
        except (ValueError, OSError) as e:
            print(f"[local_storage] delete_file failed: {e}")
            return False

    def file_exists(self, directory_name, file_name=None):
        try:
            return os.path.exists(self._entry_path(directory_name, file_name))
        except ValueError:
            return False

    def get_file_properties(self, directory_name, file_name=None):
        try:
            full = self._entry_path(directory_name, file_name)
            if not os.path.exists(full):
                return None
            st = os.stat(full)
            return {
                'name': str(file_name if file_name is not None
                            else os.path.basename(str(directory_name))),
                'size': st.st_size,
                'content_type': None,
                'last_modified': datetime.fromtimestamp(st.st_mtime),
                'etag': None,
            }
        except (ValueError, OSError) as e:
            print(f"[local_storage] get_file_properties failed: {e}")
            return None

    def generate_download_url(self, directory, filename, expiry_minutes=30):
        """No URL service locally — return a file:// URI when the file exists,
        None otherwise (cloud returns None on failure too)."""
        try:
            full = self._entry_path(directory, filename)
            if not os.path.exists(full):
                return None
            return Path(full).as_uri()
        except (ValueError, OSError):
            return None

    def refresh_credentials(self):
        """Cloud re-authenticates here; local storage has no credentials."""
        return None
