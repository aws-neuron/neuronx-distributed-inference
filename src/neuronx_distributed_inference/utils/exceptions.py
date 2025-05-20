class LogitMatchingValidationError(AssertionError):
    """
    Custom LogitMatchingValidationError class to throw an error when logit validation fails and returns the results map.
    """

    def __init__(self, message: str, results: dict):
        super().__init__(message)

        self.message = message
        self.results = results

    def get_divergence_index(self):
        """
        If a token at zero failed the logits validation for any sample in the batch, returns 0.
        Otherwise, returns the index of the largest divergence of failed logits validation across the batch.
        If no token failed, returns -1.
        """
        batch_size = len(self.results)
        last_divergence_index = -1
        for batch_index in range(batch_size):
            for token_index, token_result in enumerate(self.results[batch_index]):
                if not token_result["passed"] and token_index > last_divergence_index:
                    last_divergence_index = token_index
                    if token_index == 0:
                        return 0
        return last_divergence_index
