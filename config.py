from decorators import param


class BaseConfig:
    @param(abstract=True)
    def epochs(self):
        pass

    @param(abstract=True)
    def learning_rate(self):
        pass

    @param(abstract=True)
    def validation_interval(self):
        pass

    @param
    def checkpoint_path(self):
        pass

    @param
    def train(self):
        return False

    @param
    def retrain(self):
        return False

    @param(abstract=True)
    def train_valid_test_ratio(self):
        pass
