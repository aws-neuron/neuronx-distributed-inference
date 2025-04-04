import unittest

from neuronx_distributed_inference.placeholder import get_hello_message


class PlaceholderTest(unittest.TestCase):
    def test_get_hello_message(self):
        hello_message = get_hello_message()
        assert hello_message == "Hello, Neuron!"


if __name__ == "__main__":
    unittest.main()
