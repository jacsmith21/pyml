import importlib.machinery

import os

import utils
from data import Data


class Trainer:
    def __init__(self, file):
        application_root = os.path.dirname(file)

        config_path = os.path.join(application_root, 'config.py')
        self.config = importlib.machinery.SourceFileLoader('config', config_path).load_module().Config()

        model_path = os.path.join(application_root, 'model.py')
        self.model = importlib.machinery.SourceFileLoader('model', model_path) \
            .load_module().Model(self.config)

    def run(self):
        if self.config.preprocess:
            dataset = self.model.get_data()
            train, valid, test = utils.split_dataset(dataset, self.config.train_valid_test_ratio)
            for set, set_type in zip([train, valid, test], [Data.TRAIN, Data.VALID, Data.TEST]):
                utils.create_tfrecord(self.config.dataset, set, set_type, self.config.dataset_path)

        if self.config.train or self.config.retrain:
            self.model.train()

        if self.config.evaluate:
            self.model.evaluate()

        if self.config.freeze_graph:
            self.model.freeze_graph()

