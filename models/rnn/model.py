from model import BaseModel


class RNN(BaseModel):
    def loss(self, inputs, outputs):
        pass

    def optimize(self, loss):
        pass

    def build(self, inputs, mode):
        pass

    def train(self):
        pass

    def evaluate(self, mode):
        pass
