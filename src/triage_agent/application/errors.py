"""Technical failures. These are exceptions; business errors travel as Result values."""


class TechnicalError(Exception):
    """Infrastructure failure the caller cannot fix by changing the request."""


class LlmUnavailableError(TechnicalError):
    """The language model could not be reached or returned an unusable answer."""
