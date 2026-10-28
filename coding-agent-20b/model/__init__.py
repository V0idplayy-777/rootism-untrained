"""Model package.

Imports are lazy: ``model.tokenizer`` (and the config dataclass) are pure
Python and must be importable on machines without ``torch`` — the tokenizer
training / data pipeline / agent harness all depend on this. The heavy
``CodingAgentModel`` is only imported on first attribute access.
"""

from .config import CodingAgentConfig
from .tokenizer import CodeTokenizer

__all__ = ["CodeTokenizer", "CodingAgentConfig", "CodingAgentModel"]


def __getattr__(name: str):
    if name == "CodingAgentModel":
        from .model import CodingAgentModel

        return CodingAgentModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
