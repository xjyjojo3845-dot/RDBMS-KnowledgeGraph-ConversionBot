from __future__ import annotations

import sqlite3
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ColumnInfo:
    name: str
    data_type: str
    not_null: bool
    default_value: Any
    is_primary_key: bool


@dataclass
class IndexInfo:
    name: str
    unique: bool
    columns: list[str]


@dataclass
class TableSchema:
    name: str
    columns: list[ColumnInfo]
    primary_keys: list[str]
    indexes: list[IndexInfo]


def _list_user_tables(conn: sqlite3.Connection) -> list[str]:
    cursor = conn.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table'
          AND name NOT LIKE 'sqlite_%'
        ORDER BY name
        """
    )
    return [row[0] for row in cursor.fetchall()]


def _get_columns(conn: sqlite3.Connection, table_name: str) -> list[ColumnInfo]:
    cursor = conn.execute(f"PRAGMA table_info('{table_name}')")
    columns: list[ColumnInfo] = []

    for row in cursor.fetchall():
        # Row format: cid, name, type, notnull, dflt_value, pk
        columns.append(
            ColumnInfo(
                name=row[1],
                data_type=row[2] or "",
                not_null=bool(row[3]),
                default_value=row[4],
                is_primary_key=bool(row[5]),
            )
        )

    return columns


def _get_indexes(conn: sqlite3.Connection, table_name: str) -> list[IndexInfo]:
    index_list_cursor = conn.execute(f"PRAGMA index_list('{table_name}')")
    indexes: list[IndexInfo] = []

    for idx_row in index_list_cursor.fetchall():
        # Row format: seq, name, unique, origin, partial
        index_name = idx_row[1]
        unique = bool(idx_row[2])

        index_info_cursor = conn.execute(f"PRAGMA index_info('{index_name}')")
        index_columns = [index_col_row[2] for index_col_row in index_info_cursor.fetchall()]

        indexes.append(IndexInfo(name=index_name, unique=unique, columns=index_columns))

    return indexes


def introspect_sqlite_schema(db_path: str | Path) -> list[TableSchema]:
    """Read table, column, primary key, and index metadata from a SQLite database."""
    db_path = Path(db_path)

    if not db_path.exists():
        raise FileNotFoundError(f"Database file not found: {db_path}")

    with sqlite3.connect(str(db_path)) as conn:
        table_names = _list_user_tables(conn)
        schema: list[TableSchema] = []

        for table_name in table_names:
            columns = _get_columns(conn, table_name)
            primary_keys = [column.name for column in columns if column.is_primary_key]
            indexes = _get_indexes(conn, table_name)

            schema.append(
                TableSchema(
                    name=table_name,
                    columns=columns,
                    primary_keys=primary_keys,
                    indexes=indexes,
                )
            )

    return schema


def _parse_postgres_index_columns(index_definition: str) -> list[str]:
    match = re.search(r"\((.*)\)", index_definition)
    if not match:
        return []
    return [part.strip().strip('"') for part in match.group(1).split(",")]


def introspect_postgres_schema(
    *,
    host: str,
    port: int,
    database: str,
    user: str,
    password: str,
    schema_name: str = "public",
) -> list[TableSchema]:
    """Read table, column, primary key, and index metadata from a PostgreSQL schema."""
    import psycopg

    with psycopg.connect(
        host=host,
        port=port,
        dbname=database,
        user=user,
        password=password,
    ) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = %s
                  AND table_type = 'BASE TABLE'
                ORDER BY table_name
                """,
                (schema_name,),
            )
            table_names = [row[0] for row in cur.fetchall()]

            schema: list[TableSchema] = []
            for table_name in table_names:
                cur.execute(
                    """
                    SELECT
                        c.column_name,
                        c.data_type,
                        c.is_nullable,
                        c.column_default
                    FROM information_schema.columns c
                    WHERE c.table_schema = %s
                      AND c.table_name = %s
                    ORDER BY c.ordinal_position
                    """,
                    (schema_name, table_name),
                )
                column_rows = cur.fetchall()

                cur.execute(
                    """
                    SELECT kcu.column_name
                    FROM information_schema.table_constraints tc
                    JOIN information_schema.key_column_usage kcu
                      ON tc.constraint_name = kcu.constraint_name
                     AND tc.table_schema = kcu.table_schema
                    WHERE tc.table_schema = %s
                      AND tc.table_name = %s
                      AND tc.constraint_type = 'PRIMARY KEY'
                    ORDER BY kcu.ordinal_position
                    """,
                    (schema_name, table_name),
                )
                primary_keys = [row[0] for row in cur.fetchall()]

                columns = [
                    ColumnInfo(
                        name=row[0],
                        data_type=row[1] or "",
                        not_null=(row[2] == "NO"),
                        default_value=row[3],
                        is_primary_key=row[0] in primary_keys,
                    )
                    for row in column_rows
                ]

                cur.execute(
                    """
                    SELECT indexname, indexdef
                    FROM pg_indexes
                    WHERE schemaname = %s
                      AND tablename = %s
                    ORDER BY indexname
                    """,
                    (schema_name, table_name),
                )
                indexes = [
                    IndexInfo(
                        name=row[0],
                        unique="UNIQUE INDEX" in row[1].upper(),
                        columns=_parse_postgres_index_columns(row[1]),
                    )
                    for row in cur.fetchall()
                ]

                schema.append(
                    TableSchema(
                        name=table_name,
                        columns=columns,
                        primary_keys=primary_keys,
                        indexes=indexes,
                    )
                )

    return schema
