
class UnrecognizedInputError(Exception):
    """Use this when a use input reaches an 'else' statement, when it should not be possible to."""

class UnrecognizedGateType(Exception):
    "Use this when an unrecognized gate type has been detected."

class UnknownError(Exception):
    "Use this when the cause of the error is unclear."
