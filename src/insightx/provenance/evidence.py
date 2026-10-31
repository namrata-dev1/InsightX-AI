"""Structured Evidence and Audit Trail Data Contracts."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, List, Literal, Optional, Tuple, Union

from insightx.provenance.types import ImmutableKey, PrimitiveValue, ProvenanceType


def _validate_primitive(value: Any, field_name: str) -> None:
    """Validates that a value is an immutable primitive.
    
    Rejects mutable types (dict, list, set) with TypeError.
    """
    if not (isinstance(value, (str, int, float, bool, bytes)) or value is None):
        raise TypeError(
            f"Unsupported type for {field_name}: {type(value).__name__}. "
            "Evidence requires an immutable primitive (str, int, float, bool, bytes, None)."
        )


def _validate_immutable_key(key: Any, field_name: str) -> None:
    """Validates that a key is a primitive or a tuple of primitives."""
    if isinstance(key, tuple):
        for idx, elem in enumerate(key):
            _validate_primitive(elem, f"{field_name}[{idx}]")
    else:
        _validate_primitive(key, field_name)


@dataclass(frozen=True)
class StructuredEvidence:
    """Immutable audit record linking an agent claim to exact data provenance.
    
    Every field is verified to be immutable upon initialization.
    """
    evidence_id: str                      # Unique UUID for this evidence record
    claim_id: str                         # Identifier of claim in narrative
    dataset_version_id: str               # Git commit SHA or data checksum
    query_hash: str                       # SHA-256 of normalized SQL
    execution_id: str                     # UUID of the query execution event
    source_type: Literal["SQL_VIEW", "RECSYS_CORE", "DEMAND_FORECAST"]
    table_or_view_name: str               # e.g., 'v_orders'
    column_name: str                      # e.g., 'payment_value'
    result_row_or_key: ImmutableKey       # Immutable row index or composite key
    raw_value: PrimitiveValue             # Immutable raw value (e.g., 142.50)
    unit: str                             # 'BRL', 'days', 'count', 'ratio', 'percentage'
    calculation_provenance: str           # Derivation rule (e.g., 'ROUND(SUM(payment_value), 2)')
    provenance_type: ProvenanceType       # OBSERVED | USER_SUPPLIED | SYNTHETIC_TEST | SIMULATION_ASSUMPTION
    evidence_to_claim_linkage: str        # Natural-language sentence citing this evidence
    created_at: datetime

    def __post_init__(self) -> None:
        # Validate raw_value strictly: reject mutable types with TypeError
        _validate_primitive(self.raw_value, "raw_value")
        
        # Validate result_row_or_key strictly: reject mutable types
        _validate_immutable_key(self.result_row_or_key, "result_row_or_key")

        # Validate provenance_type strictly
        if not isinstance(self.provenance_type, ProvenanceType):
            raise TypeError(
                f"Invalid provenance_type: {self.provenance_type}. "
                "Must be an instance of ProvenanceType enum."
            )


@dataclass(frozen=True)
class AgentReport:
    """Immutable report presenting synthesized findings with complete evidence audit trails."""
    narrative_markdown: str
    evidence_trail: Tuple[StructuredEvidence, ...]
    assumptions_declared: Tuple[str, ...]
    execution_time_ms: float
    groundedness_verified: bool
