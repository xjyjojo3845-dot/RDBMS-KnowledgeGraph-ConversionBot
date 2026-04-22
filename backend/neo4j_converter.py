from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

from backend.schema_introspection import TableSchema


@dataclass
class ConversionStats:
    nodes_created: int = 0
    relationships_created: int = 0
    unmatched_records: int = 0
    existing_relationships: int = 0
    warnings: list[str] | None = None
    join_tables_collapsed: int = 0
    direct_fk_relationships_used: int = 0
    semantic_relationship_summaries: list[str] | None = None
    node_labels_created: list[str] | None = None
    unmatched_examples: list[dict[str, Any]] | None = None

    def __post_init__(self) -> None:
        if self.warnings is None:
            self.warnings = []
        if self.semantic_relationship_summaries is None:
            self.semantic_relationship_summaries = []
        if self.node_labels_created is None:
            self.node_labels_created = []
        if self.unmatched_examples is None:
            self.unmatched_examples = []

    @property
    def nodes_processed(self) -> int:
        return self.nodes_created

    @property
    def relationships_processed(self) -> int:
        return self.relationships_created

    def to_readable_report(self) -> str:
        warning_lines = "\n".join(f"  - {warning}" for warning in self.warnings) if self.warnings else "  - None"
        node_labels = ", ".join(sorted(set(self.node_labels_created))) if self.node_labels_created else "None"
        rel_lines = (
            "\n".join(f"  - {item}" for item in self.semantic_relationship_summaries)
            if self.semantic_relationship_summaries
            else "  - None"
        )
        unmatched_lines = (
            "\n".join(
                "\n".join(
                    [
                        f"  - Example {index + 1}",
                        f"    mapping: {item.get('mapping_name')}",
                        f"    reason: {item.get('reason')}",
                        f"    source: {item.get('source_table')} {json.dumps(item.get('source_key', {}), default=str, sort_keys=True)}",
                        f"    target: {item.get('target_table')} {json.dumps(item.get('target_key', {}), default=str, sort_keys=True)}",
                    ]
                )
                for index, item in enumerate(self.unmatched_examples)
            )
            if self.unmatched_examples
            else "  - None"
        )
        return (
            "Neo4j Conversion Report\n"
            "=======================\n"
            f"Nodes created: {self.nodes_created}\n"
            f"Relationships created: {self.relationships_created}\n"
            f"Existing relationships reused: {self.existing_relationships}\n"
            f"Join tables collapsed: {self.join_tables_collapsed}\n"
            f"Direct FK relationships used: {self.direct_fk_relationships_used}\n"
            f"Unmatched records: {self.unmatched_records}\n"
            f"Node labels created: {node_labels}\n"
            "Graph Mapping Summary:\n"
            f"{rel_lines}\n"
            f"Unmatched Diagnostics ({len(self.unmatched_examples)}):\n"
            f"{unmatched_lines}\n"
            f"Warnings ({len(self.warnings)}):\n"
            f"{warning_lines}"
        )


def _sanitize_identifier(name: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z_]", "_", name)
    if not cleaned:
        return "X"
    if cleaned[0].isdigit():
        return f"_{cleaned}"
    return cleaned


def _table_key_columns(table_schema: TableSchema, row: dict[str, Any]) -> list[str]:
    if table_schema.primary_keys:
        return list(table_schema.primary_keys)
    return list(row.keys())


def _normalize_neo4j_value(value: Any) -> Any:
    if isinstance(value, Decimal):
        return int(value) if value == value.to_integral_value() else float(value)
    if isinstance(value, dict):
        return {key: _normalize_neo4j_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_neo4j_value(item) for item in value]
    return value


def _build_merge_node_query(table_name: str, key_columns: list[str], node_label: str | None = None) -> str:
    label = _sanitize_identifier(node_label or table_name)
    key_props = ", ".join([f"`{col}`: $key.{col}" for col in key_columns])
    return f"MERGE (n:`{label}` {{{key_props}}}) SET n += $props"


def _build_merge_relationship_query(
    source_table: str,
    target_table: str,
    source_key_columns: list[str],
    target_key_columns: list[str],
    relationship_type: str = "RELATED_TO",
    source_label: str | None = None,
    target_label: str | None = None,
) -> str:
    source_label = _sanitize_identifier(source_label or source_table)
    target_label = _sanitize_identifier(target_label or target_table)
    relationship_label = _sanitize_identifier(relationship_type)
    source_key_props = ", ".join([f"`{col}`: $source_key.{col}" for col in source_key_columns])
    target_key_props = ", ".join([f"`{col}`: $target_key.{col}" for col in target_key_columns])

    return (
        f"MATCH (s:`{source_label}` {{{source_key_props}}}) "
        f"MATCH (t:`{target_label}` {{{target_key_props}}}) "
        f"MERGE (s)-[r:`{relationship_label}` {{mapping: $mapping_name, identity: $rel_identity}}]->(t) "
        "SET r += $rel_props"
    )


def _build_relationship_diagnostic_query(
    source_table: str,
    target_table: str,
    source_key_columns: list[str],
    target_key_columns: list[str],
    relationship_type: str,
    source_label: str | None = None,
    target_label: str | None = None,
) -> str:
    source_label = _sanitize_identifier(source_label or source_table)
    target_label = _sanitize_identifier(target_label or target_table)
    relationship_label = _sanitize_identifier(relationship_type)
    source_key_props = ", ".join([f"`{col}`: $source_key.{col}" for col in source_key_columns])
    target_key_props = ", ".join([f"`{col}`: $target_key.{col}" for col in target_key_columns])
    return (
        f"OPTIONAL MATCH (s:`{source_label}` {{{source_key_props}}}) "
        f"OPTIONAL MATCH (t:`{target_label}` {{{target_key_props}}}) "
        f"OPTIONAL MATCH (s)-[r:`{relationship_label}` {{mapping: $mapping_name, identity: $rel_identity}}]->(t) "
        "RETURN s IS NOT NULL AS source_found, "
        "t IS NOT NULL AS target_found, "
        "r IS NOT NULL AS relationship_found"
    )


def _semantic_relationship_binding(
    rel: dict[str, Any],
    schema_by_table: dict[str, TableSchema],
) -> dict[str, Any]:
    semantic_from = str(rel.get("from_table"))
    semantic_to = str(rel.get("to_table"))
    raw_from = str(rel.get("raw_from_table") or rel.get("from_table"))
    raw_to = str(rel.get("raw_to_table") or rel.get("to_table"))

    if rel.get("classification") == "collapsed_join_table":
        raw_from_join = list(rel.get("raw_from_join_columns", rel.get("from_join_columns", [])))
        raw_to_join = list(rel.get("raw_to_join_columns", rel.get("to_join_columns", [])))
        raw_from_keys = list(rel.get("raw_from_columns", rel.get("from_columns", [])))
        raw_to_keys = list(rel.get("raw_to_columns", rel.get("to_columns", [])))

        if semantic_from == raw_from and semantic_to == raw_to:
            return {
                "fetch_table": str(rel.get("raw_via_table") or rel.get("via_table")),
                "source_row_columns": raw_from_join,
                "source_node_key_columns": raw_from_keys,
                "target_row_columns": raw_to_join,
                "target_node_key_columns": raw_to_keys,
            }
        return {
            "fetch_table": str(rel.get("raw_via_table") or rel.get("via_table")),
            "source_row_columns": raw_to_join,
            "source_node_key_columns": raw_to_keys,
            "target_row_columns": raw_from_join,
            "target_node_key_columns": raw_from_keys,
        }

    fk_holder_schema = schema_by_table.get(raw_from)
    fk_holder_primary_keys = list(fk_holder_schema.primary_keys) if fk_holder_schema else []
    if not fk_holder_primary_keys:
        fk_holder_primary_keys = list(rel.get("raw_source_primary_keys", []))

    fk_holder_fk_columns = list(rel.get("raw_from_columns", rel.get("from_columns", [])))
    referenced_key_columns = list(rel.get("raw_to_columns", rel.get("to_columns", [])))

    if semantic_from == raw_from and semantic_to == raw_to:
        return {
            "fetch_table": raw_from,
            "source_row_columns": fk_holder_primary_keys,
            "source_node_key_columns": fk_holder_primary_keys,
            "target_row_columns": fk_holder_fk_columns,
            "target_node_key_columns": referenced_key_columns,
        }
    return {
        "fetch_table": raw_from,
        "source_row_columns": fk_holder_fk_columns,
        "source_node_key_columns": referenced_key_columns,
        "target_row_columns": fk_holder_primary_keys,
        "target_node_key_columns": fk_holder_primary_keys,
    }


def _relationship_identity(
    rel: dict[str, Any],
    row_dict: dict[str, Any],
    schema_by_table: dict[str, TableSchema],
    mapping_name: str,
) -> str:
    if rel.get("classification") == "collapsed_join_table":
        via_table = str(rel.get("raw_via_table") or rel.get("via_table") or "")
        via_schema = schema_by_table.get(via_table)
        identity_columns = list(via_schema.primary_keys) if via_schema and via_schema.primary_keys else []
        if not identity_columns:
            identity_columns = list(rel.get("properties", []))
        if not identity_columns:
            return mapping_name
        identity_payload = {
            "table": via_table,
            "columns": {column: row_dict.get(column) for column in identity_columns},
        }
        return json.dumps(_normalize_neo4j_value(identity_payload), sort_keys=True, default=str)

    fetch_table = str(rel.get("raw_from_table") or rel.get("from_table") or "")
    fetch_schema = schema_by_table.get(fetch_table)
    identity_columns = list(fetch_schema.primary_keys) if fetch_schema and fetch_schema.primary_keys else []
    if not identity_columns:
        return mapping_name
    identity_payload = {
        "table": fetch_table,
        "columns": {column: row_dict.get(column) for column in identity_columns},
    }
    return json.dumps(_normalize_neo4j_value(identity_payload), sort_keys=True, default=str)


def _diagnose_relationship_result(
    session: Any,
    *,
    source_table: str,
    target_table: str,
    source_key_columns: list[str],
    target_key_columns: list[str],
    relationship_type: str,
    mapping_name: str,
    rel_identity: str,
    source_key: dict[str, Any],
    target_key: dict[str, Any],
    source_label: str | None = None,
    target_label: str | None = None,
) -> dict[str, bool]:
    query = _build_relationship_diagnostic_query(
        source_table=source_table,
        target_table=target_table,
        source_key_columns=source_key_columns,
        target_key_columns=target_key_columns,
        relationship_type=relationship_type,
        source_label=source_label,
        target_label=target_label,
    )
    record = session.run(
        query,
        source_key=source_key,
        target_key=target_key,
        mapping_name=mapping_name,
        rel_identity=rel_identity,
    ).single()
    return {
        "source_found": bool(record["source_found"]) if record else False,
        "target_found": bool(record["target_found"]) if record else False,
        "relationship_found": bool(record["relationship_found"]) if record else False,
    }


def _append_unmatched_example(
    stats: ConversionStats,
    *,
    mapping_name: str,
    reason: str,
    source_table: str,
    target_table: str,
    source_key: dict[str, Any],
    target_key: dict[str, Any],
) -> None:
    if len(stats.unmatched_examples or []) >= 12:
        return
    stats.unmatched_examples.append(
        {
            "mapping_name": mapping_name,
            "reason": reason,
            "source_table": source_table,
            "target_table": target_table,
            "source_key": source_key,
            "target_key": target_key,
        }
    )


def _upsert_nodes(
    fetch_rows: Any,
    session: Any,
    schema: list[TableSchema],
    stats: ConversionStats,
    node_tables: set[str] | None = None,
    node_labels_by_table: dict[str, str] | None = None,
) -> None:
    for table in schema:
        if node_tables is not None and table.name not in node_tables:
            continue
        rows = fetch_rows(table.name)
        for row in rows:
            row_dict = _normalize_neo4j_value(dict(row))
            key_columns = _table_key_columns(table, row_dict)
            key = {col: row_dict.get(col) for col in key_columns}
            label = (node_labels_by_table or {}).get(table.name, table.name)
            query = _build_merge_node_query(table.name, key_columns, node_label=label)
            session.run(query, key=key, props=row_dict)
            stats.nodes_created += 1
        stats.node_labels_created.append((node_labels_by_table or {}).get(table.name, table.name.rstrip("s").title()))


def _sqlite_fetcher(sqlite_db_path: str | Path) -> tuple[Any, Any]:
    sqlite_db_path = Path(sqlite_db_path)
    conn = sqlite3.connect(str(sqlite_db_path))
    conn.row_factory = sqlite3.Row

    def fetch_rows(table_name: str) -> list[dict[str, Any]]:
        return conn.execute(f'SELECT * FROM "{table_name}"').fetchall()

    return conn, fetch_rows


def _postgres_fetcher(
    *,
    host: str,
    port: int,
    database: str,
    user: str,
    password: str,
    schema_name: str,
) -> tuple[Any, Any]:
    import psycopg
    from psycopg.rows import dict_row

    conn = psycopg.connect(
        host=host,
        port=port,
        dbname=database,
        user=user,
        password=password,
        row_factory=dict_row,
    )

    def fetch_rows(table_name: str) -> list[dict[str, Any]]:
        with conn.cursor() as cur:
            cur.execute(f'SELECT * FROM "{schema_name}"."{table_name}"')
            return cur.fetchall()

    return conn, fetch_rows


def convert_sql_to_neo4j(
    schema: list[TableSchema],
    mapping_config: dict[str, Any],
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
    mode: str = "structural",
    graph_mapping: dict[str, Any] | None = None,
    source_kind: str = "sqlite",
    sqlite_db_path: str | Path | None = None,
    postgres_config: dict[str, Any] | None = None,
) -> ConversionStats:
    schema_by_table = {table.name: table for table in schema}
    stats = ConversionStats()
    null_source_key_skips: dict[str, int] = {}
    null_target_key_skips: dict[str, int] = {}

    if source_kind == "postgres":
        if not postgres_config:
            raise ValueError("postgres_config is required for PostgreSQL conversion.")
        source_conn, fetch_rows = _postgres_fetcher(**postgres_config)
    else:
        if sqlite_db_path is None:
            raise ValueError("sqlite_db_path is required for SQLite conversion.")
        source_conn, fetch_rows = _sqlite_fetcher(sqlite_db_path)

    with source_conn:
        from neo4j import GraphDatabase

        with GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password)) as driver:
            with driver.session() as session:
                if mode == "semantic" and graph_mapping:
                    node_labels_by_table = {
                        str(node["table"]): str(node.get("label") or node["table"])
                        for node in graph_mapping.get("nodes", [])
                    }
                    node_tables = {node["table"] for node in graph_mapping.get("nodes", [])}
                    _upsert_nodes(
                        fetch_rows,
                        session,
                        schema,
                        stats,
                        node_tables=node_tables,
                        node_labels_by_table=node_labels_by_table,
                    )

                    for rel in graph_mapping.get("relationships", []):
                        from_table = rel["from_table"]
                        to_table = rel["to_table"]
                        from_label = node_labels_by_table.get(from_table, from_table)
                        to_label = node_labels_by_table.get(to_table, to_table)
                        rel_name = rel.get("final_name", rel.get("rule_based_name", "RELATED_TO"))
                        rel_props_cols = rel.get("properties", [])
                        binding = _semantic_relationship_binding(rel, schema_by_table)
                        join_rows = fetch_rows(binding["fetch_table"])
                        source_row_columns = binding["source_row_columns"]
                        source_columns = binding["source_node_key_columns"]
                        target_row_columns = binding["target_row_columns"]
                        target_columns = binding["target_node_key_columns"]
                        if rel.get("classification") == "collapsed_join_table":
                            stats.join_tables_collapsed += 1
                        else:
                            stats.direct_fk_relationships_used += 1

                        relationship_query = _build_merge_relationship_query(
                            source_table=from_table,
                            target_table=to_table,
                            source_key_columns=source_columns,
                            target_key_columns=target_columns,
                            relationship_type=rel_name,
                            source_label=from_label,
                            target_label=to_label,
                        )

                        for source_row in join_rows:
                            source_dict = _normalize_neo4j_value(dict(source_row))
                            source_key = {
                                source_columns[idx]: source_dict.get(source_row_columns[idx])
                                for idx in range(min(len(source_row_columns), len(source_columns)))
                            }
                            target_key = {
                                target_columns[idx]: source_dict.get(target_row_columns[idx])
                                for idx in range(min(len(target_row_columns), len(target_columns)))
                            }
                            rel_props = {col: source_dict.get(col) for col in rel_props_cols}

                            mapping_name = f"{from_table}->{to_table}:{rel_name}"
                            rel_identity = _relationship_identity(rel, source_dict, schema_by_table, mapping_name)
                            if any(value is None for value in source_key.values()):
                                null_source_key_skips[mapping_name] = null_source_key_skips.get(mapping_name, 0) + 1
                                continue
                            if any(value is None for value in target_key.values()):
                                null_target_key_skips[mapping_name] = null_target_key_skips.get(mapping_name, 0) + 1
                                continue

                            result = session.run(
                                relationship_query,
                                source_key=source_key,
                                target_key=target_key,
                                mapping_name=mapping_name,
                                rel_identity=rel_identity,
                                rel_props=rel_props,
                            )
                            created_count = result.consume().counters.relationships_created
                            stats.relationships_created += created_count
                            if created_count == 0:
                                diagnosis = _diagnose_relationship_result(
                                    session,
                                    source_table=from_table,
                                    target_table=to_table,
                                    source_key_columns=source_columns,
                                    target_key_columns=target_columns,
                                    relationship_type=rel_name,
                                    mapping_name=mapping_name,
                                    rel_identity=rel_identity,
                                    source_key=source_key,
                                    target_key=target_key,
                                    source_label=from_label,
                                    target_label=to_label,
                                )
                                if diagnosis["relationship_found"]:
                                    stats.existing_relationships += 1
                                else:
                                    stats.unmatched_records += 1
                                    if diagnosis["source_found"] and not diagnosis["target_found"]:
                                        reason = "target node missing"
                                    elif not diagnosis["source_found"] and diagnosis["target_found"]:
                                        reason = "source node missing"
                                    elif not diagnosis["source_found"] and not diagnosis["target_found"]:
                                        reason = "source and target nodes missing"
                                    else:
                                        reason = "relationship could not be matched"
                                    _append_unmatched_example(
                                        stats,
                                        mapping_name=mapping_name,
                                        reason=reason,
                                        source_table=from_table,
                                        target_table=to_table,
                                        source_key=source_key,
                                        target_key=target_key,
                                    )

                        stats.semantic_relationship_summaries.append(
                            rel.get("display_text", f"{from_table} -[:{rel_name}]-> {to_table}")
                        )
                else:
                    _upsert_nodes(fetch_rows, session, schema, stats)

                    for source_table, targets in mapping_config.get("mapping", {}).items():
                        source_schema = schema_by_table.get(source_table)
                        if not source_schema:
                            continue

                        source_rows = fetch_rows(source_table)
                        for rel in targets:
                            target_table = rel["to_table"]
                            source_columns = rel["from_columns"]
                            target_columns = rel["to_columns"]
                            mapping_name = (
                                f"{source_table}.{','.join(source_columns)}"
                                f"->{target_table}.{','.join(target_columns)}"
                            )
                            target_schema = schema_by_table.get(target_table)
                            if not target_schema:
                                stats.warnings.append(
                                    f"Skipped mapping {mapping_name}: target table `{target_table}` not found in schema."
                                )
                                continue
                            if len(source_columns) != len(target_columns):
                                stats.warnings.append(
                                    f"Skipped mapping {mapping_name}: source/target column counts differ."
                                )
                                continue

                            relationship_query = _build_merge_relationship_query(
                                source_table=source_table,
                                target_table=target_table,
                                source_key_columns=source_columns,
                                target_key_columns=target_columns,
                            )

                            for source_row in source_rows:
                                source_dict = _normalize_neo4j_value(dict(source_row))
                                source_key = {col: source_dict.get(col) for col in source_columns}
                                target_key = {
                                    target_columns[idx]: source_dict.get(source_columns[idx])
                                    for idx in range(min(len(source_columns), len(target_columns)))
                                }
                                rel_identity = _relationship_identity(
                                    {
                                        "classification": "direct_fk_relationship",
                                        "raw_from_table": source_table,
                                    },
                                    source_dict,
                                    schema_by_table,
                                    mapping_name,
                                )

                                if any(value is None for value in source_key.values()):
                                    null_source_key_skips[mapping_name] = null_source_key_skips.get(mapping_name, 0) + 1
                                    continue
                                if any(value is None for value in target_key.values()):
                                    null_target_key_skips[mapping_name] = null_target_key_skips.get(mapping_name, 0) + 1
                                    continue

                                result = session.run(
                                    relationship_query,
                                    source_key=source_key,
                                    target_key=target_key,
                                    mapping_name=mapping_name,
                                    rel_identity=rel_identity,
                                    rel_props={},
                                )
                                created_count = result.consume().counters.relationships_created
                                stats.relationships_created += created_count
                                if created_count == 0:
                                    diagnosis = _diagnose_relationship_result(
                                        session,
                                        source_table=source_table,
                                        target_table=target_table,
                                        source_key_columns=source_columns,
                                        target_key_columns=target_columns,
                                        relationship_type="RELATED_TO",
                                        mapping_name=mapping_name,
                                        rel_identity=rel_identity,
                                        source_key=source_key,
                                        target_key=target_key,
                                    )
                                    if diagnosis["relationship_found"]:
                                        stats.existing_relationships += 1
                                    else:
                                        stats.unmatched_records += 1
                                        if diagnosis["source_found"] and not diagnosis["target_found"]:
                                            reason = "target node missing"
                                        elif not diagnosis["source_found"] and diagnosis["target_found"]:
                                            reason = "source node missing"
                                        elif not diagnosis["source_found"] and not diagnosis["target_found"]:
                                            reason = "source and target nodes missing"
                                        else:
                                            reason = "relationship could not be matched"
                                        _append_unmatched_example(
                                            stats,
                                            mapping_name=mapping_name,
                                            reason=reason,
                                            source_table=source_table,
                                            target_table=target_table,
                                            source_key=source_key,
                                            target_key=target_key,
                                        )

    for mapping_name, count in null_source_key_skips.items():
        stats.warnings.append(f"Skipped {count} row(s) for mapping {mapping_name}: NULL in source key.")
    for mapping_name, count in null_target_key_skips.items():
        stats.warnings.append(f"Skipped {count} row(s) for mapping {mapping_name}: NULL in target key.")

    return stats


def convert_sqlite_to_neo4j(
    sqlite_db_path: str | Path,
    schema: list[TableSchema],
    mapping_config: dict[str, Any],
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
    mode: str = "structural",
    graph_mapping: dict[str, Any] | None = None,
) -> ConversionStats:
    return convert_sql_to_neo4j(
        sqlite_db_path=sqlite_db_path,
        schema=schema,
        mapping_config=mapping_config,
        neo4j_uri=neo4j_uri,
        neo4j_user=neo4j_user,
        neo4j_password=neo4j_password,
        mode=mode,
        graph_mapping=graph_mapping,
        source_kind="sqlite",
    )


def convert_postgres_to_neo4j(
    postgres_config: dict[str, Any],
    schema: list[TableSchema],
    mapping_config: dict[str, Any],
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
    mode: str = "structural",
    graph_mapping: dict[str, Any] | None = None,
) -> ConversionStats:
    return convert_sql_to_neo4j(
        postgres_config=postgres_config,
        schema=schema,
        mapping_config=mapping_config,
        neo4j_uri=neo4j_uri,
        neo4j_user=neo4j_user,
        neo4j_password=neo4j_password,
        mode=mode,
        graph_mapping=graph_mapping,
        source_kind="postgres",
    )
