"""Type definitions and taxonomy contracts for data provenance."""

from enum import Enum
from typing import Tuple, Union

class ProvenanceType(str, Enum):
    """Four-tier provenance classification for all data and metrics."""
    OBSERVED = "OBSERVED"
    USER_SUPPLIED = "USER_SUPPLIED"
    SYNTHETIC_TEST = "SYNTHETIC_TEST"
    SIMULATION_ASSUMPTION = "SIMULATION_ASSUMPTION"


# Strictly immutable primitive value types
PrimitiveValue = Union[str, int, float, bool, bytes, None]

# Supported immutable key types (single primitive or tuple of primitives)
ImmutableKey = Union[PrimitiveValue, Tuple[PrimitiveValue, ...]]
