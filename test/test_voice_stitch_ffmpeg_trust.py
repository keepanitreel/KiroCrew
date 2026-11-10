"""Which ffmpeg ``stitch_mp3s`` runs.

The stitch runs in the gateway, outside the agent sandbox, so the program it
starts must come from the same fixed directories the transcriber trusts
(``kiro_crew.transcribe._find_system_ffmpeg``), never from the gateway's
ambient ``PATH``. A directory such as ``~/.local/bin`` is on that ``PATH`` in
the service unit and is writable from inside the sandbox.
"""

from __future__ import annotations

import os
import stat
from unittest.mock import patch

import pytest
from voice_reply_helpers import _no_argv_prefixers  # noqa: F401
from voice_reply_helpers import _capture_mkstemp, _mock_subprocess

from kiro_crew import transcribe, transcribe_ffmpeg
from kiro_crew.voice_reply import stitch_mp3s

_TRUSTED_FFMPEG = os.path.join(os.sep, "usr", "bin", "ffmpeg")


def _two_inputs(tmp_path) -> list[str]:
    paths = []
    for name in ("a.mp3", "b.mp3"):
        p = tmp_path / name
        p.write_bytes(b"fake-mp3")
        paths.append(str(p))
    return paths


def _plant_on_path(tmp_path) -> str:
    """Put an executable named ffmpeg in a directory and make it all of PATH."""
    local_bin = tmp_path / "home" / ".local" / "bin"
    local_bin.mkdir(parents=True)
    name = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
    planted = local_bin / name
    planted.write_bytes(b"not a real ffmpeg")
    planted.chmod(planted.stat().st_mode | stat.S_IXUSR)
    return str(local_bin)


@pytest.mark.asyncio
async def test_stitch_runs_the_trusted_ffmpeg_not_the_path_one(tmp_path) -> None:
    allocated: list[str] = []
    argv: list[str] = []

    async def fake_exec(*cmd, **kwargs):
        argv.extend(cmd)
        with open(cmd[-1], "wb") as f:
            f.write(b"stitched-mp3-data")
        return _mock_subprocess(returncode=0)

    with pytest.MonkeyPatch.context() as mp:
        allocated = _capture_mkstemp(mp)
        mp.setenv("PATH", _plant_on_path(tmp_path))
        mp.setattr(transcribe, "_find_system_ffmpeg", lambda: _TRUSTED_FFMPEG)
        with patch("asyncio.create_subprocess_exec", side_effect=fake_exec):
            result = await stitch_mp3s(_two_inputs(tmp_path))

    try:
        assert result == allocated[0]
        assert argv[0] == _TRUSTED_FFMPEG
        assert os.path.isabs(argv[0])
    finally:
        for path in allocated:
            if os.path.exists(path):
                os.unlink(path)


@pytest.mark.asyncio
async def test_stitch_starts_nothing_when_no_trusted_ffmpeg(tmp_path) -> None:
    async def fake_exec(*cmd, **kwargs):
        raise AssertionError(f"no ffmpeg may start, got {cmd[0]!r}")

    with pytest.MonkeyPatch.context() as mp:
        allocated = _capture_mkstemp(mp)
        mp.setenv("PATH", _plant_on_path(tmp_path))
        mp.setattr(transcribe, "_find_system_ffmpeg", lambda: None)
        with patch("asyncio.create_subprocess_exec", side_effect=fake_exec) as spawn:
            result = await stitch_mp3s(_two_inputs(tmp_path))

    assert result is None
    spawn.assert_not_called()
    assert len(allocated) == 1
    assert not os.path.exists(allocated[0])


@pytest.mark.asyncio
async def test_stitch_ignores_an_ffmpeg_only_on_path(tmp_path) -> None:
    """End to end through the real resolver: with no ffmpeg in a trusted
    directory, a copy that only PATH can reach is never started."""
    empty = tmp_path / "empty"
    empty.mkdir()

    async def fake_exec(*cmd, **kwargs):
        raise AssertionError(f"no ffmpeg may start, got {cmd[0]!r}")

    with pytest.MonkeyPatch.context() as mp:
        allocated = _capture_mkstemp(mp)
        mp.setenv("PATH", _plant_on_path(tmp_path))
        mp.setattr(transcribe_ffmpeg.platform_compat, "trusted_system_path", lambda: str(empty))
        mp.setattr(transcribe_ffmpeg, "_FFMPEG_CANDIDATE_DIRS", [str(empty)])
        with patch("asyncio.create_subprocess_exec", side_effect=fake_exec) as spawn:
            result = await stitch_mp3s(_two_inputs(tmp_path))

    assert result is None
    spawn.assert_not_called()
    assert not os.path.exists(allocated[0])
