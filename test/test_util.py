import unittest
from kaggle_donors_choose_2014 import util


class Test(unittest.TestCase):

    def test_sample(self):
        def gen():
            for i in range(10):
                yield i

        s = util.sample(gen(), 10)
        self.assertEqual(range(10), s)

    def test_sample(self):
        def gen():
            for i in range(10):
                yield i

        s = util.sample(gen(), 5)
        self.assertEqual(5, len(s))

    def test_larger(self):
        def gen():
            for i in range(1000):
                yield i

        s = util.sample(gen(), 5)
        self.assertEqual(5, len(s))
