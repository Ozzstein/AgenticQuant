"""Programmatic facade over the authoritative durable schedule supervisor."""
from __future__ import annotations

from datetime import datetime, timezone
from threading import Event

from src.operations.application import execute_cycle, schedule_supervisor
from src.operations.scheduler import ScheduleSpec
from src.utils.config import get_config


class PipelineScheduler:
    def __init__(self, config=None, *, lane="operating", authorities=None):
        self.config = config or get_config()
        self.lane = lane
        self.supervisor = schedule_supervisor(self.config, lane=lane, authorities=authorities)
        self._stop = Event()

    def register(self, spec: ScheduleSpec):
        if spec.execution_lane != self.lane:
            raise ValueError("schedule belongs to another worker lane")
        self.supervisor.register(spec)

    def run_once(self, plan=None):
        if plan is not None:
            if self.lane != "operating":
                raise ValueError("paper cycles require the operating worker")
            return execute_cycle(plan, self.config)
        return self.supervisor.tick(datetime.now(timezone.utc), lane=self.lane)

    def start(self):
        self._stop.clear()
        while not self._stop.is_set():
            self.run_once()
            self._stop.wait(self.config.operations.poll_seconds)

    def start_intraday(self, spec: ScheduleSpec):
        """Register an explicit interval input schedule using the same cycle."""
        if not isinstance(spec, ScheduleSpec) or spec.kind != "interval":
            raise ValueError("intraday operation requires an explicit interval ScheduleSpec")
        self.register(spec)

    def stop(self):
        self._stop.set()
