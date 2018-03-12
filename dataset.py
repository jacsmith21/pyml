class Dataset:
    def __init__(self, mode):
        self.mode = mode
        self.inputs = None

    @staticmethod
    def build_dataset(mode):
        return Dataset(mode)

    @staticmethod
    def build_placeholders():
        return [5]