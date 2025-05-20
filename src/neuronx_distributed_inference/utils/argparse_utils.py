import argparse

AUTO = 'auto'


class StringOrIntegers(argparse.Action):
    def __init__(self, option_strings, dest=None, nargs=None, **kwargs):
        super(StringOrIntegers, self).__init__(option_strings, dest=dest, nargs=nargs, **kwargs)
        self.string_value = AUTO
        self.string_value_found = False

    def __call__(self, parser, namespace, values, option_string=None):
        result = []
        for value in values:
            if value.lower() == self.string_value:
                if self.string_value_found or len(values) > 1:
                    raise argparse.ArgumentTypeError(f"'{self.string_value}' can only be used once and not with other arguments.")
                self.string_value_found = True
                result = value.lower()
            else:
                try:
                    result.append(int(value))
                except ValueError:
                    raise argparse.ArgumentTypeError(f"'{value}' is not a valid input. It must be '{self.string_value}' or an integer.")

        setattr(namespace, self.dest, result)
