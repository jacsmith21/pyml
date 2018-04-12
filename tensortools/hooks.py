import tensorflow as tf
from tensorflow.python.training.basic_session_run_hooks import SecondOrStepTimer

from tensortools import abc
from tensortools import logging

logger = logging.get_logger(__name__)


class IntervalHook(tf.train.SessionRunHook):
    def __init__(self, interval):
        self.global_step = None
        self.interval = interval

        if interval is not None:
            self.timer = SecondOrStepTimer(every_steps=interval)
        else:
            self.timer = None

    def begin(self):
        self.global_step = tf.train.get_or_create_global_step()

    def before_run(self, run_context):
        return tf.train.SessionRunArgs([self.global_step, *self.session_run_args(run_context)])

    # noinspection PyMethodMayBeStatic, PyUnusedLocal
    def session_run_args(self, run_context):  # pylint: disable=unused-argument
        return list()

    def after_run(self, run_context, run_values):
        if self.interval is None:
            return

        global_step = run_values.results[0]
        if self.timer.should_trigger_for_step(global_step):
            self.timer.update_last_triggered_step(global_step)
            self.run_interval_operations(run_context, run_values.results[1:], global_step)

    @abc.abstract
    def run_interval_operations(self, run_context, results, global_step):
        pass


class GlobalStepIncrementor(IntervalHook):
    def __init__(self, log_interval=None):
        super().__init__(log_interval)
        self.step_incrementor = None

    def begin(self):
        super().begin()
        self.step_incrementor = tf.assign_add(self.global_step, 1)

    def session_run_args(self, run_context):
        return [self.step_incrementor]
