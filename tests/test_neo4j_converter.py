from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from backend.neo4j_converter import (
    ConversionStats,
    _build_merge_node_query,
    _build_merge_relationship_query,
    _relationship_identity,
    _semantic_relationship_binding,
    _sanitize_identifier,
)
from backend.schema_introspection import TableSchema


def test_sanitize_identifier_handles_special_chars_and_digits():
    assert _sanitize_identifier("table-name") == "table_name"
    assert _sanitize_identifier("123abc") == "_123abc"


def test_build_merge_node_query_uses_merge_and_props():
    query = _build_merge_node_query("housing_metrics", ["id", "zip"])
    assert "MERGE" in query
    assert "SET n += $props" in query
    assert "`id`: $key.id" in query
    assert "`zip`: $key.zip" in query


def test_build_merge_node_query_can_use_semantic_node_label():
    query = _build_merge_node_query("students", ["student_id"], node_label="Student")
    assert "MERGE (n:`Student`" in query


def test_build_merge_relationship_query_is_generic_and_dedup_friendly():
    query = _build_merge_relationship_query(
        source_table="housing_metrics",
        target_table="zip_codes",
        source_key_columns=["zip"],
        target_key_columns=["zip"],
    )
    assert "MATCH (s:`housing_metrics`" in query
    assert "MATCH (t:`zip_codes`" in query
    assert "MERGE (s)-[r:`RELATED_TO` {mapping: $mapping_name, identity: $rel_identity}]->(t)" in query
    assert "SET r += $rel_props" in query


def test_conversion_stats_readable_report_contains_required_sections():
    stats = ConversionStats(
        nodes_created=7,
        relationships_created=5,
        unmatched_records=2,
        warnings=["Example warning"],
        unmatched_examples=[
            {
                "mapping_name": "students->courses:ENROLLED_IN",
                "reason": "target node missing",
                "source_table": "students",
                "target_table": "courses",
                "source_key": {"student_id": 1},
                "target_key": {"course_id": 999},
            }
        ],
    )
    report = stats.to_readable_report()
    assert "Nodes created: 7" in report
    assert "Relationships created: 5" in report
    assert "Unmatched records: 2" in report
    assert "Join tables collapsed: 0" in report
    assert "Direct FK relationships used: 0" in report
    assert "Unmatched Diagnostics (1):" in report
    assert "Example 1" in report
    assert "mapping: students->courses:ENROLLED_IN" in report
    assert 'source: students {"student_id": 1}' in report
    assert 'target: courses {"course_id": 999}' in report
    assert "students->courses:ENROLLED_IN" in report
    assert "target node missing" in report
    assert "Warnings (1):" in report
    assert "Example warning" in report


def test_conversion_stats_compat_properties_map_to_new_fields():
    stats = ConversionStats(nodes_created=3, relationships_created=4)
    assert stats.nodes_processed == 3
    assert stats.relationships_processed == 4


def test_semantic_relationship_binding_uses_join_columns_and_node_keys_correctly_when_reversed():
    schema_by_table = {
        "instructors": TableSchema(name="instructors", columns=[], primary_keys=["instructor_id"], indexes=[]),
        "courses": TableSchema(name="courses", columns=[], primary_keys=["course_id"], indexes=[]),
    }
    rel = {
        "classification": "collapsed_join_table",
        "from_table": "instructors",
        "to_table": "courses",
        "raw_from_table": "courses",
        "raw_to_table": "instructors",
        "raw_via_table": "course_instructors",
        "raw_from_join_columns": ["course_id"],
        "raw_to_join_columns": ["instructor_id"],
        "raw_from_columns": ["course_id"],
        "raw_to_columns": ["instructor_id"],
    }
    binding = _semantic_relationship_binding(rel, schema_by_table)
    assert binding["fetch_table"] == "course_instructors"
    assert binding["source_row_columns"] == ["instructor_id"]
    assert binding["source_node_key_columns"] == ["instructor_id"]
    assert binding["target_row_columns"] == ["course_id"]
    assert binding["target_node_key_columns"] == ["course_id"]


def test_semantic_relationship_binding_uses_fk_holder_primary_key_for_direct_fk():
    schema_by_table = {
        "invoice": TableSchema(name="invoice", columns=[], primary_keys=["invoice_id"], indexes=[]),
        "customer": TableSchema(name="customer", columns=[], primary_keys=["customer_id"], indexes=[]),
    }
    rel = {
        "classification": "direct_fk_relationship",
        "from_table": "customer",
        "to_table": "invoice",
        "raw_from_table": "invoice",
        "raw_to_table": "customer",
        "raw_from_columns": ["customer_id"],
        "raw_to_columns": ["customer_id"],
    }
    binding = _semantic_relationship_binding(rel, schema_by_table)
    assert binding["fetch_table"] == "invoice"
    assert binding["source_row_columns"] == ["customer_id"]
    assert binding["source_node_key_columns"] == ["customer_id"]
    assert binding["target_row_columns"] == ["invoice_id"]
    assert binding["target_node_key_columns"] == ["invoice_id"]


def test_relationship_identity_uses_join_table_primary_key_or_payload_to_preserve_distinct_edges():
    schema_by_table = {
        "course_instructors": TableSchema(name="course_instructors", columns=[], primary_keys=["assignment_id"], indexes=[]),
    }
    rel = {
        "classification": "collapsed_join_table",
        "raw_via_table": "course_instructors",
        "properties": ["semester", "role"],
    }
    identity_a = _relationship_identity(
        rel,
        {"assignment_id": 1, "semester": "Fall", "role": "Lead"},
        schema_by_table,
        "instructors->courses:TEACHES",
    )
    identity_b = _relationship_identity(
        rel,
        {"assignment_id": 2, "semester": "Spring", "role": "Guest"},
        schema_by_table,
        "instructors->courses:TEACHES",
    )
    assert identity_a != identity_b
