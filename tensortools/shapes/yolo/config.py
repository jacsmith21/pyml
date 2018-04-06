from clients.eigen.shapes.config import ShapeConfig
from common.decorators import Param


class Config(ShapeConfig):
    @Param
    def n_images(self):
        return 100

    @Param
    def shape(self):
        return 50, 50

    @Param
    def max_shapes(self):
        return 3

    @Param
    def grid_shape(self):
        return 13, 13
