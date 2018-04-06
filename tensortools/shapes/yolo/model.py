from common.templates.model import BaseModel


class Model(BaseModel):
    def inference(self, inputs, mode):
        images = inputs['images']

