import unittest

import pytest

from tensortools.abc import abstract


class TestAbc(unittest.TestCase):
    @abstract
    def an_abstract_method(self):
        pass

    def test_abstract(self):
        with pytest.raises(NotImplementedError):
            self.an_abstract_method()
