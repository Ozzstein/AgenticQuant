"""Compatibility CLI for the shared durable scheduler.

Register explicit schedules and immutable job inputs with ``python -m
src.scheduler register``; separate operating and research workers use ``serve``.
The retired date-only nine-step scheduler cannot authorize economic work.
"""
from scripts.run_operations import schedule_app as app

if __name__ == "__main__":
    app()
