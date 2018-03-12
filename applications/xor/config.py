from decorators import param
from models.rnn.config import RNNConfig


class Config(RNNConfig):
    @param
    def num_samples(self):
        return 100000
