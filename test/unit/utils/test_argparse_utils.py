import argparse
import unittest
from neuronx_distributed_inference.utils.argparse_utils import StringOrIntegers


class TestStringOrIntegers(unittest.TestCase):
    def setUp(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--values', action=StringOrIntegers, nargs='+')

    def test_single_integer(self):
        args = self.parser.parse_args(['--values', '10'])
        self.assertEqual(args.values, [10])

    def test_multiple_integers(self):
        args = self.parser.parse_args(['--values', '1', '2', '3'])
        self.assertEqual(args.values, [1, 2, 3])

    def test_auto_string(self):
        args = self.parser.parse_args(['--values', 'auto'])
        self.assertEqual(args.values, 'auto')

    def test_auto_with_other_values(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            self.parser.parse_args(['--values', 'auto', '5'])
    
    def test_duplicate_auto(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            self.parser.parse_args(['--values', 'auto', 'auto'])

    def test_invalid_string(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            self.parser.parse_args(['--values', 'invalid'])

    def test_mixed_valid_and_invalid(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            self.parser.parse_args(['--values', '10', 'invalid'])

if __name__ == '__main__':
    unittest.main()
