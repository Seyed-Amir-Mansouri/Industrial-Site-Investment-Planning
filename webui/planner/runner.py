"""Runs plan_capacity.py in a background thread so the web request returns immediately."""
from __future__ import annotations

import subprocess
import threading
from django.conf import settings
from django.db import close_old_connections
from django.utils import timezone

from . import services
from .models import PlanRun


def start(run: PlanRun) -> None:
    threading.Thread(target=_execute, args=(run.pk,), daemon=True).start()


def _execute(run_pk: int) -> None:
    close_old_connections()
    run = PlanRun.objects.get(pk=run_pk)
    prefix = services.output_prefix_for(run_pk)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    cmd = services.build_command(run.params, prefix)
    try:
        proc = subprocess.run(cmd, cwd=settings.PROJECT_ROOT, capture_output=True, text=True,
                              encoding="utf-8", errors="replace")
        log = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
        run.log = log
        run.command = " ".join(cmd)
        if proc.returncode == 0:
            run.summary = services.parse_summary(prefix, log)
            run.status = PlanRun.Status.DONE
        else:
            run.status = PlanRun.Status.FAILED
            run.error = "\n".join(log.strip().splitlines()[-25:])
    except Exception as exc:  # noqa: BLE001 - surface any failure to the user, don't kill the thread silently
        run.status = PlanRun.Status.FAILED
        run.error = f"{type(exc).__name__}: {exc}"
    run.finished_at = timezone.now()
    run.save()
    close_old_connections()
