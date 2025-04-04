class LogitMatchingValidationError(AssertionError):
    """
    Custom LogitMatchingValidationError class to throw an error when logit validation fails and returns the results map.
    """

    def __init__(self, message: str, results: dict):
        super().__init__(message)

        self.message = message
        self.results = results
