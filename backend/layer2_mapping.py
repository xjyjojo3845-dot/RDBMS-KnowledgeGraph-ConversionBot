from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from backend.schema_introspection import ColumnInfo, IndexInfo, TableSchema

AUDIT_COLUMNS = {
    "created_at",
    "updated_at",
    "created_on",
    "updated_on",
    "deleted_at",
    "last_modified",
    "timestamp",
}

GENERIC_REL_NAMES = {"HAS", "RELATED_TO", "LINKS_TO"}


@dataclass
class ForeignKeyInfo:
    source_table: str
    source_columns: list[str]
    target_table: str
    target_columns: list[str]


def _singularize(name: str) -> str:
    cleaned = name.replace("_", " ").strip()
    if cleaned.endswith("s") and len(cleaned) > 1:
        cleaned = cleaned[:-1]
    return cleaned.title().replace(" ", "")


def _upper_snake(name: str) -> str:
    return name.replace("-", "_").replace(" ", "_").upper()


def _table_role(name: str) -> str:
    normalized = name.lower().strip()
    singular = normalized[:-1] if normalized.endswith("s") and len(normalized) > 1 else normalized

    if singular.startswith("student"):
        return "student"
    if singular.startswith("course") or singular.startswith("class") or singular.startswith("section"):
        return "course"
    if singular.startswith("instructor") or singular.startswith("teacher") or singular.startswith("faculty"):
        return "instructor"
    if singular.startswith("department"):
        return "department"
    return singular


def _load_schema_from_json(schema_path: Path) -> list[TableSchema]:
    raw = json.loads(schema_path.read_text())
    schema: list[TableSchema] = []
    for table in raw:
        columns = table.get("columns", [])
        schema.append(
            TableSchema(
                name=table["name"],
                columns=[
                    ColumnInfo(
                        name=col["name"],
                        data_type=col.get("data_type", ""),
                        not_null=bool(col.get("not_null", False)),
                        default_value=col.get("default_value"),
                        is_primary_key=bool(col.get("is_primary_key", False)),
                    )
                    for col in columns
                ],
                primary_keys=table.get("primary_keys", []),
                indexes=[
                    IndexInfo(
                        name=idx["name"],
                        unique=bool(idx.get("unique", False)),
                        columns=list(idx.get("columns", [])),
                    )
                    for idx in table.get("indexes", [])
                ],
            )
        )
    return schema


def load_schema(db_path: Path, schema_path: Path | None = None) -> list[TableSchema]:
    if schema_path and schema_path.exists():
        return _load_schema_from_json(schema_path)
    from backend.schema_introspection import introspect_sqlite_schema

    return introspect_sqlite_schema(db_path)


def load_foreign_keys(db_path: Path, table_names: list[str]) -> list[ForeignKeyInfo]:
    fks: list[ForeignKeyInfo] = []
    with sqlite3.connect(str(db_path)) as conn:
        for table in table_names:
            rows = conn.execute(f"PRAGMA foreign_key_list('{table}')").fetchall()
            grouped: dict[int, dict[str, Any]] = {}
            for row in rows:
                # id, seq, table, from, to, on_update, on_delete, match
                fk_id = int(row[0])
                grouped.setdefault(
                    fk_id,
                    {
                        "target_table": row[2],
                        "source_columns": [],
                        "target_columns": [],
                    },
                )
                grouped[fk_id]["source_columns"].append(row[3])
                grouped[fk_id]["target_columns"].append(row[4])

            for grouped_fk in grouped.values():
                fks.append(
                    ForeignKeyInfo(
                        source_table=table,
                        source_columns=grouped_fk["source_columns"],
                        target_table=grouped_fk["target_table"],
                        target_columns=grouped_fk["target_columns"],
                    )
                )
    return fks


def load_postgres_foreign_keys(
    *,
    host: str,
    port: int,
    database: str,
    user: str,
    password: str,
    schema_name: str,
    table_names: list[str],
) -> list[ForeignKeyInfo]:
    import psycopg

    fks: list[ForeignKeyInfo] = []
    with psycopg.connect(
        host=host,
        port=port,
        dbname=database,
        user=user,
        password=password,
    ) as conn:
        with conn.cursor() as cur:
            for table in table_names:
                cur.execute(
                    """
                    SELECT
                        tc.constraint_name,
                        kcu.column_name,
                        ccu.table_name AS foreign_table_name,
                        ccu.column_name AS foreign_column_name,
                        kcu.ordinal_position
                    FROM information_schema.table_constraints AS tc
                    JOIN information_schema.key_column_usage AS kcu
                      ON tc.constraint_name = kcu.constraint_name
                     AND tc.table_schema = kcu.table_schema
                    JOIN information_schema.constraint_column_usage AS ccu
                      ON ccu.constraint_name = tc.constraint_name
                     AND ccu.table_schema = tc.table_schema
                    WHERE tc.constraint_type = 'FOREIGN KEY'
                      AND tc.table_schema = %s
                      AND tc.table_name = %s
                    ORDER BY tc.constraint_name, kcu.ordinal_position
                    """,
                    (schema_name, table),
                )
                grouped: dict[str, dict[str, Any]] = {}
                for row in cur.fetchall():
                    constraint_name = row[0]
                    grouped.setdefault(
                        constraint_name,
                        {
                            "target_table": row[2],
                            "source_columns": [],
                            "target_columns": [],
                        },
                    )
                    grouped[constraint_name]["source_columns"].append(row[1])
                    grouped[constraint_name]["target_columns"].append(row[3])

                for grouped_fk in grouped.values():
                    fks.append(
                        ForeignKeyInfo(
                            source_table=table,
                            source_columns=grouped_fk["source_columns"],
                            target_table=grouped_fk["target_table"],
                            target_columns=grouped_fk["target_columns"],
                        )
                    )
    return fks


def _rule_based_relationship_name(from_table: str, to_table: str, via_table: str | None = None) -> str:
    via = (via_table or "").lower()
    pair = (from_table.lower(), to_table.lower())

    if "enroll" in via:
        return "ENROLLED_IN"
    if "instructor" in via or "teach" in via:
        return "TEACHES"
    if pair[1].startswith("department") and pair[0].startswith("course"):
        return "OFFERED_BY"
    if pair[1].startswith("department") and pair[0].startswith("instructor"):
        return "WORKS_FOR"
    if pair[1].startswith("department") and pair[0].startswith("student"):
        return "MAJORS_IN"

    if via_table:
        return _upper_snake(via_table)

    if pair[1].startswith("department"):
        return "BELONGS_TO"

    return f"{_upper_snake(_singularize(from_table))}_TO_{_upper_snake(_singularize(to_table))}"


def _resolve_join_relationship(
    left_fk: ForeignKeyInfo,
    right_fk: ForeignKeyInfo,
    via_table: str,
) -> tuple[ForeignKeyInfo, ForeignKeyInfo, str]:
    left_role = _table_role(left_fk.target_table)
    right_role = _table_role(right_fk.target_table)
    via = via_table.lower()

    if "enroll" in via and {left_role, right_role} == {"student", "course"}:
        if left_role == "student":
            return left_fk, right_fk, "ENROLLED_IN"
        return right_fk, left_fk, "ENROLLED_IN"

    if any(token in via for token in ("teach", "instruction", "assignment")) and {left_role, right_role} == {"instructor", "course"}:
        if left_role == "instructor":
            return left_fk, right_fk, "TEACHES"
        return right_fk, left_fk, "TEACHES"

    rule_name = _rule_based_relationship_name(left_fk.target_table, right_fk.target_table, via_table)
    return left_fk, right_fk, rule_name


def _classify_columns(
    table: TableSchema,
    fk_columns: set[str],
    for_relationship: bool,
) -> dict[str, list[str]]:
    pk_set = set(table.primary_keys)
    metadata = [c.name for c in table.columns if c.name.lower() in AUDIT_COLUMNS]
    ignorable = [c.name for c in table.columns if c.name.lower().startswith("sys_")]
    non_key_columns = [
        c.name
        for c in table.columns
        if c.name not in pk_set and c.name not in fk_columns and c.name not in metadata and c.name not in ignorable
    ]

    return {
        "primary_key_columns": list(table.primary_keys),
        "foreign_key_columns": sorted(fk_columns),
        "candidate_node_properties": non_key_columns if not for_relationship else [],
        "candidate_relationship_properties": non_key_columns if for_relationship else [],
        "metadata_columns": metadata,
        "ignorable_columns": ignorable,
    }


def _build_display_text(from_table: str, rel_name: str, to_table: str, props: list[str]) -> str:
    base = f"{_singularize(from_table)} {rel_name} {_singularize(to_table)}"
    if props:
        return f"{base} {{{', '.join(props)}}}"
    return base


def build_graph_mapping(
    schema: list[TableSchema],
    foreign_keys: list[ForeignKeyInfo],
    structural_mapping: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build deterministic Layer 2 mapping (entities, join collapse, semantic relationships)."""
    del structural_mapping  # reserved for future weighting from existing pipeline

    schema_by_name = {table.name: table for table in schema}
    fk_by_source: dict[str, list[ForeignKeyInfo]] = {}
    referenced_counts: dict[str, int] = {table.name: 0 for table in schema}

    for fk in foreign_keys:
        fk_by_source.setdefault(fk.source_table, []).append(fk)
        referenced_counts[fk.target_table] = referenced_counts.get(fk.target_table, 0) + 1

    nodes: list[dict[str, Any]] = []
    relationships: list[dict[str, Any]] = []
    join_tables: list[dict[str, Any]] = []
    ambiguous_cases: list[dict[str, Any]] = []

    entity_tables: set[str] = set()

    for table in schema:
        table_fks = fk_by_source.get(table.name, [])
        fk_columns = {col for fk in table_fks for col in fk.source_columns}
        classification = _classify_columns(table, fk_columns, for_relationship=False)

        non_fk_non_pk = [
            col.name
            for col in table.columns
            if col.name not in fk_columns and col.name not in set(table.primary_keys)
        ]
        confidence = 0.65
        signals: list[str] = []

        is_join_candidate = len(table_fks) >= 2
        if len(table.primary_keys) >= 1:
            confidence += 0.1
            signals.append("strong_primary_key")
        if referenced_counts.get(table.name, 0) > 0:
            confidence += 0.1
            signals.append("referenced_by_other_tables")
        if non_fk_non_pk:
            confidence += 0.1
            signals.append("non_fk_attributes_present")

        join_confidence = 0.0
        join_signals: list[str] = []
        if is_join_candidate:
            join_confidence += 0.45
            join_signals.append("two_or_more_foreign_keys")
            if len(non_fk_non_pk) <= 3:
                join_confidence += 0.25
                join_signals.append("few_non_fk_attributes")
            if len(table_fks) == 2:
                join_confidence += 0.2
                join_signals.append("primarily_connects_two_tables")
            if any(set(idx.columns) >= fk_columns for idx in table.indexes if getattr(idx, "unique", False)):
                join_confidence += 0.1
                join_signals.append("composite_uniqueness_on_link_columns")

        if join_confidence >= 0.65 and len(table_fks) >= 2:
            join_tables.append(
                {
                    "table": table.name,
                    "confidence": round(min(join_confidence, 0.99), 2),
                    "signals": join_signals,
                    "linked_tables": [fk.target_table for fk in table_fks[:2]],
                    "join_columns": sorted(fk_columns),
                    "non_key_columns": non_fk_non_pk,
                }
            )
            # collapse first pair deterministically, keep extras as ambiguous cases
            primary_fk, secondary_fk = table_fks[0], table_fks[1]
            resolved_from_fk, resolved_to_fk, rule_name = _resolve_join_relationship(primary_fk, secondary_fk, table.name)
            rel_props = _classify_columns(table, fk_columns, for_relationship=True)["candidate_relationship_properties"]
            rel = {
                "id": f"{resolved_from_fk.target_table}->{resolved_to_fk.target_table}::via::{table.name}",
                "from_table": resolved_from_fk.target_table,
                "to_table": resolved_to_fk.target_table,
                "via_table": table.name,
                "from_join_columns": resolved_from_fk.source_columns,
                "to_join_columns": resolved_to_fk.source_columns,
                "from_columns": resolved_from_fk.target_columns,
                "to_columns": resolved_to_fk.target_columns,
                "properties": rel_props,
                "classification": "collapsed_join_table",
                "cardinality": "many_to_many",
                "source_is_join_table": True,
                "has_payload_columns": bool(rel_props),
                "confidence": round(min(join_confidence, 0.99), 2),
                "signals": join_signals,
                "rule_based_name": rule_name,
                "ai_suggested_name": None,
                "final_name": rule_name,
                "display_text": _build_display_text(resolved_from_fk.target_table, rule_name, resolved_to_fk.target_table, rel_props),
                "review_status": "rule_based",
                "reviewable": len(table_fks) > 2,
            }
            relationships.append(rel)
            if len(table_fks) > 2:
                ambiguous_cases.append(
                    {
                        "object_type": "join_table",
                        "object_id": table.name,
                        "reason": "join table references more than two parent tables",
                        "fallback_name": rule_name,
                    }
                )
            continue

        entity_tables.add(table.name)
        nodes.append(
            {
                "table": table.name,
                "label": _singularize(table.name),
                "primary_key": list(table.primary_keys),
                "properties": classification["candidate_node_properties"],
                "column_classification": classification,
                "classification": "entity_table",
                "confidence": round(min(confidence, 0.99), 2),
                "signals": signals,
            }
        )

    # direct relationships from entity tables
    for fk in foreign_keys:
        if fk.source_table not in entity_tables or fk.target_table not in entity_tables:
            continue
        rule_name = _rule_based_relationship_name(fk.source_table, fk.target_table)
        if rule_name in GENERIC_REL_NAMES:
            ambiguous_cases.append(
                {
                    "object_type": "relationship",
                    "object_id": f"{fk.source_table}->{fk.target_table}",
                    "reason": "generic relationship naming fallback used",
                    "fallback_name": rule_name,
                }
            )

        relationships.append(
            {
                "id": f"{fk.source_table}->{fk.target_table}",
                "type": rule_name,
                "from_table": fk.source_table,
                "to_table": fk.target_table,
                "from_columns": fk.source_columns,
                "to_columns": fk.target_columns,
                "properties": [],
                "classification": "direct_fk_relationship",
                "cardinality": "many_to_one",
                "source_is_join_table": False,
                "has_payload_columns": False,
                "confidence": 0.95,
                "signals": ["explicit_foreign_key"],
                "rule_based_name": rule_name,
                "ai_suggested_name": None,
                "final_name": rule_name,
                "display_text": _build_display_text(fk.source_table, rule_name, fk.target_table, []),
                "review_status": "rule_based",
                "reviewable": False,
            }
        )

    return {
        "nodes": sorted(nodes, key=lambda item: item["table"]),
        "relationships": sorted(relationships, key=lambda item: (item["from_table"], item["to_table"], item.get("via_table", ""))),
        "join_tables": sorted(join_tables, key=lambda item: item["table"]),
        "ambiguous_cases": ambiguous_cases,
    }
