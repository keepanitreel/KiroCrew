"""The cwd of a pooled backend is the daemon's own, not the register frame's.

``PoolKey.work_dir`` comes from a stub's register frame, which any same-user
process on the gateway socket can send. The launch approval covers command,
arguments and env, so the daemon must not let the frame choose the directory an
approved launch runs in. These tests pin that the default resolver always names
the gateway's own default work dir, ``workspace_root()/_default``, and that the
pre-fork re-check treats a moved directory as a changed target.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging

import pytest
from mcp_launch_approval_helpers import (
    _APPROVED_CMD,
    _approval_module,
    _approvals_for,
    _pool_key,
    _target_spec,
)

from kiro_crew.config.loader_locations import workspace_root
from kiro_crew.testing.wait import async_wait_until

#: Opts this module out of the conftest fixture that approves every gatewayd launch.
ENFORCE_LAUNCH_APPROVAL = True


@pytest.fixture
def approved_srv(tmp_path, monkeypatch):
    """An approved ``srv`` launch, with a workspace root this test owns."""
    mod = _approval_module()
    assert mod is not None
    monkeypatch.setenv("KIROCREW_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("KIROCREW_WORKSPACE", str(tmp_path / "workspace"))
    (tmp_path / "workspace" / "_default").mkdir(parents=True)
    monkeypatch.setenv("KIROCREW_MCP_TARGET_SRV", _target_spec(_APPROVED_CMD))
    approvals = _approvals_for(mod, "srv", _APPROVED_CMD, [])
    mod._write(mod.approvals_path(), mod._document(approvals, {}))
    return str(workspace_root(create=False) / "_default")


def _key_with_work_dir(work_dir: str):
    return dataclasses.replace(_pool_key("srv"), work_dir=work_dir)


def test_the_rewriters_default_work_dir_is_the_spawn_cwd(approved_srv):
    from kiro_crew.mcp_gateway import gatewayd

    target = gatewayd.env_target_resolver(_key_with_work_dir(approved_srv))

    assert target is not None
    assert target[3] == approved_srv


def test_a_register_frame_cannot_choose_the_spawn_cwd(approved_srv, tmp_path):
    from kiro_crew.mcp_gateway import gatewayd

    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()

    target = gatewayd.env_target_resolver(_key_with_work_dir(str(elsewhere)))

    assert target is not None
    assert target[0] == _APPROVED_CMD
    assert target[3] == approved_srv


def test_a_frame_naming_another_work_dir_is_logged(approved_srv, tmp_path, caplog):
    from kiro_crew.mcp_gateway import gatewayd

    other = str(tmp_path / "elsewhere")
    with caplog.at_level(logging.WARNING, logger="kiro_crew.mcp_gateway"):
        gatewayd.env_target_resolver(_key_with_work_dir(other))

    assert any(
        "registered work dir" in record.getMessage() and other in record.getMessage()
        for record in caplog.records
    )


def test_the_default_work_dir_is_not_logged(approved_srv, caplog):
    from kiro_crew.mcp_gateway import gatewayd

    with caplog.at_level(logging.WARNING, logger="kiro_crew.mcp_gateway"):
        gatewayd.env_target_resolver(_key_with_work_dir(approved_srv))

    assert not any("registered work dir" in r.getMessage() for r in caplog.records)


def test_an_unresolvable_work_dir_refuses_the_spawn(approved_srv, monkeypatch):
    from kiro_crew.mcp_gateway import gatewayd
    from kiro_crew.mcp_gateway.daemon import launch

    def _unresolvable(*, create: bool = True):
        raise OSError("no workspace")

    monkeypatch.setattr(launch, "workspace_root", _unresolvable)

    assert gatewayd.env_target_resolver(_key_with_work_dir(approved_srv)) is None


@pytest.mark.asyncio
async def test_a_work_dir_that_moves_while_queued_is_a_changed_target(
    approved_srv, tmp_path, monkeypatch
):
    from unittest.mock import AsyncMock

    from kiro_crew.mcp_gateway import admission as admission_mod
    from kiro_crew.mcp_gateway import gatewayd, host_budget
    from kiro_crew.mcp_gateway.pool import BackendPool

    spawned = AsyncMock(side_effect=AssertionError("spawned in a directory never judged"))
    monkeypatch.setattr(gatewayd, "spawn_backend", spawned)
    pool = BackendPool(max_backends=2)
    admission = admission_mod.Admission(
        gate=admission_mod.SpawnGate(1),
        budget=host_budget.HostBudget(host_budget.HostBudgetLimits(max_procs=2)),
        initialize_timeout_secs=1.0,
        spawn_queue_wait_secs=30.0,
    )
    blocker = await admission.gate.acquire(label="blocker")
    task = asyncio.create_task(
        gatewayd._acquire_backend(
            pool,
            _key_with_work_dir(approved_srv),
            gatewayd.env_target_resolver,
            admission=admission,
        )
    )
    try:
        await async_wait_until(
            lambda: admission.gate.queued == 1, describe=lambda: admission.gate.queued
        )
        moved = tmp_path / "moved-workspace"
        (moved / "_default").mkdir(parents=True)
        monkeypatch.setenv("KIROCREW_WORKSPACE", str(moved))
        blocker.release()
        with pytest.raises(gatewayd._TargetUnknown, match="approval changed"):
            await task
    finally:
        blocker.release()
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    spawned.assert_not_awaited()
    assert pool.resident_pending == 0
    assert admission.gate.in_flight == 0
    assert admission.budget.procs_in_use == 0
