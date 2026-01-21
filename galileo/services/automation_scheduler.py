from __future__ import annotations
from contextlib import contextmanager

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.cron import CronTrigger
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from models import Training
from services.automation import run_automation_for_training


class TrainingAutomationScheduler:
    """
    Schedules automations based on Training.automation_enabled + Training.automation_schedule.
    If schedule is all digits ⇒ seconds (interval). Otherwise, crontab.
    """

    def __init__(self, db_url: str):
        self.engine = create_engine(db_url)
        self.SessionLocal = sessionmaker(bind=self.engine)
        self.sched = BackgroundScheduler()

    @contextmanager
    def session(self):
        s = self.SessionLocal()
        try:
            yield s
        finally:
            s.close()

    def _parse_trigger(self, schedule: str):
        if schedule and schedule.isdigit():
            return IntervalTrigger(seconds=int(schedule))
        return CronTrigger.from_crontab(schedule)

    def _job(self, training_id: str):
        with self.session() as db:
            run_automation_for_training(db, training_id)

    def add_or_update_training(self, training: Training):
        job_id = f"auto:{training.id}"
        if self.sched.get_job(job_id):
            self.sched.remove_job(job_id)

        if not training.automation_enabled or not training.automation_schedule:
            return

        trigger = self._parse_trigger(training.automation_schedule)
        self.sched.add_job(
            self._job,
            trigger,
            id=job_id,
            args=[training.id],
            replace_existing=True,
        )

    def remove_training(self, training_id: str):
        job_id = f"auto:{training_id}"
        if self.sched.get_job(job_id):
            self.sched.remove_job(job_id)

    def warm_boot(self):
        with self.session() as db:
            trainings = (
                db.query(Training)
                .filter(Training.automation_enabled == True)
                .all()
            )
            for tr in trainings:
                if tr.automation_schedule:
                    self.add_or_update_training(tr)

    def start(self):
        if not self.sched.running:
            self.sched.start()
