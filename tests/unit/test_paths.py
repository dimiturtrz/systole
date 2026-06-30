"""Cross-platform data-root resolution (core.paths) — the full OS × path-kind matrix, with OS
detection injected so it runs identically on any host. The leak this prevents: a Windows drive path
on POSIX silently becoming a relative repo-local 'D:/' dir."""
import pytest

from core.paths import resolve_data_root, win_drive, mnt_drive, to_wsl, to_windows


# --- detection ---
def test_win_drive_detects_drive_paths():
    assert win_drive("D:/data/x") == "d"
    assert win_drive(r"C:\data\x") == "c"
    assert win_drive("/mnt/d/x") is None
    assert win_drive("/data/x") is None
    assert win_drive("data") is None            # repo-relative fallback


def test_mnt_drive_detects_wsl_mounts():
    assert mnt_drive("/mnt/d/data/x") == "d"
    assert mnt_drive("/mnt/c/x") == "c"
    assert mnt_drive("/data/x") is None
    assert mnt_drive("/mnt/share/x") is None     # not a single-letter drive


# --- translation (round-trips) ---
def test_translation_roundtrip():
    assert to_wsl("D:/foo/bar") == "/mnt/d/foo/bar"
    assert to_windows("/mnt/d/foo/bar") == "D:/foo/bar"
    assert to_windows(to_wsl("E:/a/b/c")) == "E:/a/b/c"


# --- resolve matrix (os_name + wsl injected) ---
def test_windows_path_on_windows_unchanged():
    assert resolve_data_root("D:/data/x", os_name="nt") == "D:/data/x"


def test_posix_path_on_posix_unchanged():
    assert resolve_data_root("/data/x", os_name="posix", wsl=False) == "/data/x"


def test_windows_path_on_wsl_translates():
    assert resolve_data_root("D:/data/x", os_name="posix", wsl=True) == "/mnt/d/data/x"


def test_mount_path_on_windows_translates_back():
    assert resolve_data_root("/mnt/d/data/x", os_name="nt") == "D:/data/x"


def test_windows_path_on_native_linux_raises():
    """The leak guard: no WSL mount to translate to -> must RAISE, never go relative."""
    with pytest.raises(RuntimeError, match="non-WSL POSIX"):
        resolve_data_root("D:/data/x", os_name="posix", wsl=False)


def test_repo_relative_fallback_untouched():
    assert resolve_data_root("data", os_name="posix", wsl=False) == "data"
    assert resolve_data_root("data", os_name="nt") == "data"
