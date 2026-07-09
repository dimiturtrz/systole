"""Cross-platform data-root resolution (core.paths) — the full OS × path-kind matrix, with OS
detection injected so it runs identically on any host. The leak this prevents: a Windows drive path
on POSIX silently becoming a relative repo-local 'D:/' dir."""
import pytest

from core.paths import Paths


# --- detection ---
def test_win_drive_detects_drive_paths():
    assert Paths.win_drive("D:/data/x") == "d"
    assert Paths.win_drive(r"C:\data\x") == "c"
    assert Paths.win_drive("/mnt/d/x") is None
    assert Paths.win_drive("/data/x") is None
    assert Paths.win_drive("data") is None            # repo-relative fallback


def test_mnt_drive_detects_wsl_mounts():
    assert Paths.mnt_drive("/mnt/d/data/x") == "d"
    assert Paths.mnt_drive("/mnt/c/x") == "c"
    assert Paths.mnt_drive("/data/x") is None
    assert Paths.mnt_drive("/mnt/share/x") is None     # not a single-letter drive


# --- translation (round-trips) ---
def test_translation_roundtrip():
    assert Paths.to_wsl("D:/foo/bar") == "/mnt/d/foo/bar"
    assert Paths.to_windows("/mnt/d/foo/bar") == "D:/foo/bar"
    assert Paths.to_windows(Paths.to_wsl("E:/a/b/c")) == "E:/a/b/c"


# --- resolve matrix (os_name + wsl injected) ---
def test_windows_path_on_windows_unchanged():
    assert Paths.resolve_data_root("D:/data/x", os_name="nt") == "D:/data/x"


def test_posix_path_on_posix_unchanged():
    assert Paths.resolve_data_root("/data/x", os_name="posix", wsl=False) == "/data/x"


def test_windows_path_on_wsl_translates():
    assert Paths.resolve_data_root("D:/data/x", os_name="posix", wsl=True) == "/mnt/d/data/x"


def test_mount_path_on_windows_translates_back():
    assert Paths.resolve_data_root("/mnt/d/data/x", os_name="nt") == "D:/data/x"


def test_windows_path_on_native_linux_raises():
    """The leak guard: no WSL mount to translate to -> must RAISE, never go relative."""
    with pytest.raises(RuntimeError, match="non-WSL POSIX"):
        Paths.resolve_data_root("D:/data/x", os_name="posix", wsl=False)


def test_repo_relative_fallback_untouched():
    assert Paths.resolve_data_root("data", os_name="posix", wsl=False) == "data"
    assert Paths.resolve_data_root("data", os_name="nt") == "data"
