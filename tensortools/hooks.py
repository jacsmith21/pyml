import tensorflow as tf
from tensorflow.python.training.basic_session_run_hooks import SecondOrStepTimer

from tensortools import abc
from tensortools import logging

logger = logging.get_logger(__name__)


class IntervalHook(tf.train.SessionRunHook):
    """
    A hook which runs every # of iterations. Useful for subclassing.
    """
    def __init__(self, interval):
        """
        Construct the interval.

        :param interval: The interval.
        """
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
        """
        Create the session run arguments.

        :param run_context: The run context.
        :return: The list of arguments to run.
        """
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
        """
        The method to override.

        :param run_context: The run context.
        :param results: The results of running the given arguments.
        :param global_step: The evaluated global step tensor.
        """
        pass


class GlobalStepIncrementor(tf.train.SessionRunHook):
    """
    Increments the global step after each `Session` `run` call. Useful for models which do not use optimizers.
    """
    def __init__(self):
        self.step_incrementor = None

    def begin(self):
        self.step_incrementor = tf.assign_add(self.global_step, 1)

    def before_run(self, run_context):
        return tf.train.SessionRunArgs([self.step_incrementor])
