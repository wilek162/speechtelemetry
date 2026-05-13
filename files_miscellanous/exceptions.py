"""Custom exceptions for speechtelemetry.

All exceptions raised by the library are subclasses of SpeechTelemetryError.
Every exception includes an actionable message telling the user exactly what to do.
"""


class SpeechTelemetryError(Exception):
    """Base class for all speechtelemetry exceptions."""


class BackendNotAvailableError(SpeechTelemetryError):
    """Raised when a backend's optional dependency is not installed.

    Always includes the pip install command needed to fix the problem.
    """


class BackendNotFoundError(SpeechTelemetryError):
    """Raised when an unknown backend name is requested from the registry."""


class EnvironmentCheckError(SpeechTelemetryError):
    """Raised by pre-flight checks when a mandatory prerequisite is missing.

    Hard fail — the pipeline cannot proceed.
    """


class BackendError(SpeechTelemetryError):
    """Raised when a backend fails to initialize (e.g. model download error)."""
