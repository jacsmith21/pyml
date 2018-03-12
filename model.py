from abc import abstractmethod

import os
import tensorflow as tf
from tensorflow.python.estimator.model_fn import ModeKeys
from tensorflow.python.tools import freeze_graph

from dataset import Dataset
from config import BaseConfig


class BaseModel:
    def __init__(self, config):
        """

        :param config:
        :type config: BaseConfig
        """
        self.config = config

    @abstractmethod
    def build(self, inputs, mode):
        pass

    def train(self):
        dataset = Dataset.build_dataset(ModeKeys.TRAIN)
        outputs = self.build(dataset.inputs, mode=ModeKeys.TRAIN)
        loss = self.loss(dataset.inputs, outputs)
        optimizer = self.optimize(loss)

        saver = tf.train.Saver()
        with tf.Session() as sess:
            if self.config.retrain:
                saver.restore(sess, self.config.checkpoint_path)

            for epoch in self.config.epochs:
                sess.run(optimizer)

                if (epoch+1) % self.config.validation_interval == 0:
                    self.evaluate('validation')

    @abstractmethod
    def loss(self, inputs, outputs):
        pass

    @abstractmethod
    def optimize(self, loss):
        pass

    @abstractmethod
    def evaluate(self, mode):
        pass

    def freeze_graph(self):
        """Function that 'freezes' a graph. A frozen graph can be used by C/C++/Java/tensorflow serving"""
        sess = tf.Session()
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        input_placeholders = Dataset.build_placeholders()

        self.build(input_placeholders, mode=ModeKeys.PREDICT)

        sess.run(init_op)
        saver = tf.train.Saver()
        saver.restore(sess, self.config.checkpoint_path)

        graph_path = '/home/eigen/Desktop'
        ckpt = tf.train.get_checkpoint_state(graph_path)
        tf.train.write_graph(sess.graph_def, os.path.join(graph_path), 'graph_structure.ph', as_text=False)

        output_node_names = []
        freeze_graph.freeze_graph(
            os.path.join(graph_path, 'graph_structure.pb'),
            input_saver='',
            input_binary=True,
            input_checkpoint=ckpt.model_checkpoint_path,
            output_node_names=",".join(output_node_names),
            restore_op_name='save/restore_all',
            filename_tensor_name='save/Const:0',
            output_graph=os.path.join(graph_path, 'output_graph.pb'),
            clear_devices=False,
            initializer_nodes=False
        )
