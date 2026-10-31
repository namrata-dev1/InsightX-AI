"""Tests for Data and Evidence Provenance Subsystem."""

from dataclasses import FrozenInstanceError
from datetime import datetime
import pytest

from insightx.provenance.types import ProvenanceType
from insightx.provenance.evidence import StructuredEvidence, AgentReport


def test_provenance_types_coverage():
    """Verifies that all 4 required provenance classes exist and are unique."""
    expected = {"OBSERVED", "USER_SUPPLIED", "SYNTHETIC_TEST", "SIMULATION_ASSUMPTION"}
    actual = {p.value for p in ProvenanceType}
    assert actual == expected


def test_structured_evidence_valid_instantiation():
    """Verifies valid creation across all 4 provenance types."""
    for prov in ProvenanceType:
        evidence = StructuredEvidence(
            evidence_id="ev-12345",
            claim_id="claim-1",
            dataset_version_id="git-sha-abc1234",
            query_hash="sha256-hash-xyz",
            execution_id="exec-9876",
            source_type="SQL_VIEW",
            table_or_view_name="v_orders",
            column_name="payment_value",
            result_row_or_key=("order_id_100", "item_1"),
            raw_value=150.75,
            unit="BRL",
            calculation_provenance="ROUND(SUM(payment_value), 2)",
            provenance_type=prov,
            evidence_to_claim_linkage="Total revenue for order was R$ 150.75.",
            created_at=datetime(2026, 1, 1, 12, 0, 0),
        )
        assert evidence.provenance_type == prov
        assert evidence.raw_value == 150.75
        assert evidence.result_row_or_key == ("order_id_100", "item_1")


def test_structured_evidence_immutability():
    """Verifies that modifying attributes raises FrozenInstanceError."""
    evidence = StructuredEvidence(
        evidence_id="ev-1",
        claim_id="c-1",
        dataset_version_id="v1",
        query_hash="qhash",
        execution_id="ex-1",
        source_type="SQL_VIEW",
        table_or_view_name="v_orders",
        column_name="order_status",
        result_row_or_key="ord-1",
        raw_value="delivered",
        unit="status",
        calculation_provenance="raw",
        provenance_type=ProvenanceType.OBSERVED,
        evidence_to_claim_linkage="Order is delivered",
        created_at=datetime.utcnow(),
    )
    with pytest.raises(FrozenInstanceError):
        evidence.raw_value = "cancelled"  # type: ignore

    with pytest.raises(FrozenInstanceError):
        evidence.table_or_view_name = "v_customers"  # type: ignore


def test_structured_evidence_rejects_mutable_raw_value():
    """Verifies that passing mutable structures (dict, list, set) to raw_value raises TypeError."""
    base_kwargs = dict(
        evidence_id="ev-1",
        claim_id="c-1",
        dataset_version_id="v1",
        query_hash="qhash",
        execution_id="ex-1",
        source_type="SQL_VIEW",
        table_or_view_name="v_orders",
        column_name="metric",
        result_row_or_key="key-1",
        unit="count",
        calculation_provenance="raw",
        provenance_type=ProvenanceType.SYNTHETIC_TEST,
        evidence_to_claim_linkage="Claim text",
        created_at=datetime.utcnow(),
    )

    # Passing a list must be rejected
    with pytest.raises(TypeError, match="Unsupported type for raw_value: list"):
        StructuredEvidence(raw_value=[1, 2, 3], **base_kwargs)

    # Passing a dict must be rejected
    with pytest.raises(TypeError, match="Unsupported type for raw_value: dict"):
        StructuredEvidence(raw_value={"score": 95}, **base_kwargs)

    # Passing a set must be rejected
    with pytest.raises(TypeError, match="Unsupported type for raw_value: set"):
        StructuredEvidence(raw_value={1, 2, 3}, **base_kwargs)


def test_structured_evidence_rejects_mutable_key():
    """Verifies that passing mutable structures to result_row_or_key raises TypeError."""
    base_kwargs = dict(
        evidence_id="ev-1",
        claim_id="c-1",
        dataset_version_id="v1",
        query_hash="qhash",
        execution_id="ex-1",
        source_type="SQL_VIEW",
        table_or_view_name="v_orders",
        column_name="metric",
        raw_value=42,
        unit="count",
        calculation_provenance="raw",
        provenance_type=ProvenanceType.SYNTHETIC_TEST,
        evidence_to_claim_linkage="Claim text",
        created_at=datetime.utcnow(),
    )

    # Passing a list as key must be rejected
    with pytest.raises(TypeError, match="Unsupported type for result_row_or_key: list"):
        StructuredEvidence(result_row_or_key=["order_1", "item_2"], **base_kwargs)

    # Passing a tuple containing a list must also be rejected
    with pytest.raises(TypeError, match="Unsupported type for result_row_or_key\\[1\\]: list"):
        StructuredEvidence(result_row_or_key=("order_1", [1, 2]), **base_kwargs)


def test_agent_report_immutability():
    """Verifies that AgentReport is deeply frozen."""
    report = AgentReport(
        narrative_markdown="Executive Brief",
        evidence_trail=(),
        assumptions_declared=("Assumption 1",),
        execution_time_ms=12.5,
        groundedness_verified=True,
    )
    with pytest.raises(FrozenInstanceError):
        report.groundedness_verified = False  # type: ignore
