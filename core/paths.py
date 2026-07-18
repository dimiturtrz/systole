"""Cross-platform data-root resolution.

`paths.yaml` (or CARDIAC_DATA) holds ONE machine's data root. This module adapts it to the OS the
process is actually running on — translating between a Windows drive path (`D:/data`) and its WSL
mount (`/mnt/d/data`) when it can, and failing LOUD when it can't. The point: a Windows path used on
POSIX is otherwise treated as *relative* and silently creates a repo-local `D:/…` dir — which once
leaked dataset metadata into git. Translate-when-possible, guard-when-not; never silently relative.

Pure functions, OS detection dependency-injected (`os_name` / `wsl`) so the matrix is unit-testable
on any host. pathlib (PureWindows/PurePosixPath) does the parsing — no slash-regex.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path, PurePosixPath, PureWindowsPath


class Paths:
    """Cross-platform data-root resolution (free helpers folded in as staticmethods; public names kept)."""

    @staticmethod
    @lru_cache(maxsize=None)
    def _wslpath(flag: str, raw: str) -> str | None:
        """Translate `raw` via the NATIVE WSL `wslpath` tool (-u Windows->mount, -w mount->Windows) —
        authoritative about the actual mount config. None if wslpath is absent or fails (then callers fall
        back to the pure-pathlib mapping). Memoized: data_root is hit many times per run."""
        if not shutil.which("wslpath"):
            return None
        try:
            # trusted: fixed argv (no shell); `raw` is a path passed as an argv element, not interpolated
            out = subprocess.run(["wslpath", flag, raw], capture_output=True, text=True,  # noqa: S603, S607
                                 timeout=5, check=True).stdout.strip()
            return out or None
        except (subprocess.SubprocessError, OSError):          # nonzero exit / timeout / wslpath vanished
            return None

    @staticmethod
    def detect_wsl() -> bool:
        """True iff running under WSL (POSIX kernel exposing Windows drives at /mnt). Heuristic: the
        WSL_DISTRO_NAME env, else 'microsoft' in /proc/version."""
        if os.name == "nt":
            return False
        if os.environ.get("WSL_DISTRO_NAME"):
            return True
        try:
            return "microsoft" in Path("/proc/version").read_text().lower()
        except OSError:
            return False

    @staticmethod
    def win_drive(raw: str) -> str | None:
        """Lowercase drive letter if `raw` is a Windows drive path ('D:/x' -> 'd'), else None.
        Uses PureWindowsPath so it detects a drive even when called on POSIX."""
        drive = PureWindowsPath(raw).drive            # 'D:' for 'D:/x'; '' for a posix path
        return drive[0].lower() if len(drive) == 2 and drive[1] == ":" else None  # noqa: PLR2004 ('D:' is 2 chars)

    @staticmethod
    def mnt_drive(raw: str) -> str | None:
        """Lowercase drive letter if `raw` is a WSL mount path ('/mnt/d/x' -> 'd'), else None."""
        parts = PurePosixPath(raw).parts              # ('/', 'mnt', 'd', 'x')
        if len(parts) >= 3 and parts[1] == "mnt" and len(parts[2]) == 1 and parts[2].isalpha():  # noqa: PLR2004 ('/mnt/d/...' >= 3 parts)
            return parts[2].lower()
        return None

    @staticmethod
    def to_wsl(raw: str) -> str:
        """Windows drive path -> WSL mount path ('D:/data/x' -> '/mnt/d/data/x'). Native `wslpath -u`
        when available (honors the real mount config), else a pure-pathlib /mnt/<drive> fallback."""
        native = Paths._wslpath("-u", raw)
        if native:
            return native
        d = Paths.win_drive(raw)
        if d is None:
            raise ValueError(f"to_wsl expects a Windows drive path, got {raw!r}")
        rest = PureWindowsPath(raw).parts[1:]         # drop the 'D:\\' anchor
        return str(PurePosixPath("/mnt", d, *rest))

    @staticmethod
    def to_windows(raw: str) -> str:
        """WSL mount path -> Windows drive path ('/mnt/d/data/x' -> 'D:/data/x'). Native `wslpath -w`
        when available, else a pure-pathlib fallback (as-posix, forward slashes)."""
        native = Paths._wslpath("-w", raw)
        if native:
            return native.replace("\\", "/")          # wslpath -w yields backslashes; normalize
        parts = PurePosixPath(raw).parts
        return PureWindowsPath(f"{parts[2].upper()}:/", *parts[3:]).as_posix()

    @staticmethod
    def resolve_data_root(raw: str, *, os_name: str | None = None, wsl: bool | None = None) -> str:
        """Adapt the configured data root `raw` to the current OS.

        - same family (Windows path on Windows / POSIX path on POSIX) -> unchanged
        - Windows path on WSL            -> translate to the /mnt mount
        - WSL/POSIX-mount path on Windows -> translate back to the drive path
        - Windows path on NON-WSL POSIX  -> RAISE (can't translate; set CARDIAC_DATA) — never silently
          become a relative 'D:/' dir
        """
        os_name = os.name if os_name is None else os_name
        wsl = Paths.detect_wsl() if wsl is None else wsl
        drive = Paths.win_drive(raw)
        if os_name == "nt":
            return Paths.to_windows(raw) if (drive is None and Paths.mnt_drive(raw)) else raw
        # POSIX
        if drive is None:
            return raw                                # already a posix path
        if wsl:
            return Paths.to_wsl(raw)                   # D:/x -> /mnt/d/x
        raise RuntimeError(
            f"data root {raw!r} is a Windows drive path but this is non-WSL POSIX — can't translate it to "
            f"a real path here. Set CARDIAC_DATA to a path on this machine. (Refusing to treat it as "
            f"relative, which would silently create a repo-local '{drive.upper()}:/' dir — a past leak.)")
