"""
Модуль для запуска всего пайплайна раз в неделю через APScheduler.
"""
import asyncio
from zoneinfo import ZoneInfo
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from typing import Callable, Dict, Any
import logging
from abc import ABC, abstractmethod

# Интерфейс планировщика
class IScheduler(ABC):
    @abstractmethod
    def add_cron_job(self, func: Callable, cron: Dict[str, Any], job_id: str = None):
        pass
    @abstractmethod
    def remove_job(self, job_id: str):
        pass
    @abstractmethod
    def shutdown(self):
        pass

# Реализация через APScheduler
class AioScheduler(IScheduler):
    def __init__(self, timezone: str = "Europe/Moscow"):
        self.scheduler = AsyncIOScheduler(timezone=ZoneInfo(timezone))
        # self.scheduler.start()  # Удалено, запускать только из event loop

    async def start(self):
        self.scheduler.start()

    def add_cron_job(self, func: Callable, cron: Dict[str, Any], job_id: str = None):
        trigger = CronTrigger(**cron)
        self.scheduler.add_job(func, trigger, id=job_id)

    def remove_job(self, job_id: str):
        self.scheduler.remove_job(job_id)

    def shutdown(self):
        self.scheduler.shutdown()
