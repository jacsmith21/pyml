import numpy as np

from models.rnn.model import RNN
from utils import one_hot


class Model(RNN):
    def get_data(self):
        data = np.random.randint(2, 50, size=50*self.config.num_samples)
        data = np.array_split(data, self.config.num_samples)
        labels = np.sum(data, axis=-1) % 2

        return {
            'streams': data,
            'labels': one_hot(labels)
        }
