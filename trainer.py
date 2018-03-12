import importlib.machinery

import os


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
            print('preprocess')

        if self.config.train or self.config.retrain:
            self.model.train()
            self.model.freeze_graph()

        if self.config.evaluate:
            self.model.evaluate(evaluate_on_validation=self.config.evaluate_on_validation)

        if self.config.predict:
            self.model.predict()

        if self.config.predict_chov:
            self.model.predict_chov(file_path=self.config.chov_file_path)

        if self.config.cloud_backup:
            self.model.cloud_backup()

        if self.config.freeze_graph:
            self.model.freeze_graph()

