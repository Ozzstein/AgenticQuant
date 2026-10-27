"""The monitoring facade uses the shared durable schedule and actual inputs."""
from datetime import datetime, timezone

import pytest

from src.monitoring.scheduler import PipelineScheduler
from src.operations.scheduler import ScheduleSpec
from src.utils.config import AppConfig


def test_interval_registration_is_durable_and_requires_an_explicit_schedule(tmp_path):
    cfg=AppConfig(_env_file=None,output_dir=str(tmp_path/'out'),log_dir=str(tmp_path/'logs'))
    scheduler=PipelineScheduler(cfg)
    spec=ScheduleSpec(schedule_id='intraday',starts_at=datetime.now(timezone.utc),kind='interval',
                      interval_seconds=3600,input_directory=str(tmp_path/'inputs'))
    scheduler.start_intraday(spec)
    restarted=PipelineScheduler(cfg)
    restarted.register(spec)
    assert restarted.run_once()[0]['status']=='waiting'
    with pytest.raises(ValueError,match='explicit interval'):
        scheduler.start_intraday(object())
    with pytest.raises(ValueError,match='another worker'):
        scheduler.register(spec.model_copy(update={'execution_lane':'research'}))
    scheduler.stop()
    assert scheduler._stop.is_set()
