from common.decorators import Param
from common.templates.config import BaseConfig


class ShapeConfig(BaseConfig):
    @Param(abstract=True)
    def n_images(self):
        pass

    @Param(abstract=True)
    def shape(self):
        """The shape, ie. `[height, width]`"""
        pass

    @Param(abstract=True)
    def max_shapes(self):
        pass

    @Param(flag=True)
    def allow_overlap(self):
        pass

    @Param
    def min_shapes(self):
        return 1

    @Param(abstract=True)
    def max_shapes(self):
        pass

    @Param
    def min_size(self):
        return 2

    @Param
    def max_size(self):
        return None
