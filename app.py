from __future__ import annotations

import json
import re
import sqlite3
import tempfile
import zipfile
from dataclasses import asdict
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from xml.etree import ElementTree as ET

import pandas as pd
import streamlit as st

from backend.layer2_mapping import build_graph_mapping, load_foreign_keys, load_postgres_foreign_keys
from backend.graph_query import build_query_schema_registry, plan_graph_query_full_ai, run_graph_query
from backend.neo4j_converter import convert_postgres_to_neo4j, convert_sqlite_to_neo4j
from backend.schema_introspection import TableSchema, introspect_postgres_schema, introspect_sqlite_schema
from ai.config import load_ai_settings
from ai.reviewer import apply_ai_review, run_ai_mapping_review
from semantic_relationship_reviewer import apply_semantic_relationship_review, run_semantic_relationship_review
from relationship_detector import (
    build_mapping_config,
    detect_relationships,
)


st.set_page_config(page_title="RDBMS to Graph Explorer", layout="wide")
st.markdown(
    """
    <style>
    .section-banner {
        border-radius: 18px;
        padding: 1.1rem 1.25rem;
        margin: 1.2rem 0 0.8rem 0;
        border: 1px solid rgba(15, 23, 42, 0.08);
    }
    .section-banner h2 {
        margin: 0;
        font-size: 1.35rem;
        font-weight: 700;
        color: #0f172a;
    }
    .section-banner p {
        margin: 0.35rem 0 0 0;
        color: #334155;
        font-size: 0.97rem;
    }
    .section-banner.schema {
        background: linear-gradient(135deg, #fff7ed 0%, #ffedd5 100%);
    }
    .section-banner.graph {
        background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
    }
    .section-banner.convert {
        background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
    }
    .status-strip {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        padding: 0.8rem 1rem;
        border-radius: 14px;
        border: 1px solid rgba(15, 23, 42, 0.08);
        margin: 0.5rem 0 0.9rem 0;
        background: #f8fafc;
    }
    .status-badge {
        display: inline-block;
        padding: 0.28rem 0.6rem;
        border-radius: 999px;
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.01em;
        white-space: nowrap;
    }
    .status-badge.ai {
        background: #dbeafe;
        color: #1d4ed8;
    }
    .status-badge.det {
        background: #e2e8f0;
        color: #334155;
    }
    .status-text {
        color: #0f172a;
        font-size: 0.95rem;
    }
    .metric-row {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 0.9rem;
        margin: 0.4rem 0 1rem 0;
    }
    .metric-card {
        background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
        border: 1px solid rgba(15, 23, 42, 0.08);
        border-radius: 16px;
        padding: 1rem 1rem 0.9rem 1rem;
    }
    .metric-label {
        color: #475569;
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 0.35rem;
    }
    .metric-value {
        color: #0f172a;
        font-size: 1.8rem;
        font-weight: 800;
        line-height: 1;
    }
    @media (max-width: 900px) {
        .metric-row {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }
    }
    .summary-pills {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin: 0.2rem 0 0.9rem 0;
    }
    .summary-pill {
        display: inline-block;
        padding: 0.26rem 0.62rem;
        border-radius: 999px;
        background: #fff7ed;
        color: #9a3412;
        border: 1px solid rgba(154, 52, 18, 0.10);
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.01em;
    }
    div[data-testid="stGraphVizChart"] > div {
        display: flex;
        justify-content: center;
    }
    div[data-testid="stGraphVizChart"] svg {
        width: 64% !important;
        height: auto !important;
        max-width: none;
        margin: 0 auto;
        display: block;
    }
    .pipeline-progress-card {
        max-width: 760px;
        margin: 1rem auto 1.25rem auto;
        padding: 1rem 1.1rem;
        border-radius: 18px;
        border: 1px solid rgba(15, 23, 42, 0.08);
        background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
        box-shadow: 0 14px 32px rgba(15, 23, 42, 0.06);
    }
    .pipeline-progress-label {
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #475569;
        margin-bottom: 0.25rem;
    }
    .pipeline-progress-title {
        font-size: 1.15rem;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 0.18rem;
    }
    .pipeline-progress-note {
        font-size: 0.95rem;
        color: #475569;
        margin-bottom: 0.75rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("RDBMS to Graph Explorer")
st.write("Choose a database source to inspect the relational schema, review the graph mapping, and export or convert the final result.")
st.caption("Run with `streamlit run app.py`. The default local URL is usually `http://localhost:8501`.")

PIPELINE_CACHE_VERSION = "semantic-review-v2"
EXAMPLE_QUESTIONS_PATH = Path("example questions.docx")

source_type = st.radio("Source database", options=["SQLite file", "PostgreSQL connection"], horizontal=True)

uploaded_file = None
postgres_config: dict[str, object] | None = st.session_state.get("postgres_config")
postgres_ready = False

if source_type == "SQLite file":
    uploaded_file = st.file_uploader("Upload SQLite database", type=["db", "sqlite", "sqlite3"])
else:
    pg_input_mode = st.radio(
        "PostgreSQL input mode",
        options=["Connection fields", "Connection string"],
        horizontal=True,
    )
    with st.form("postgres_connection_form"):
        if pg_input_mode == "Connection fields":
            st.caption("Connection details")
            pg_cols = st.columns(3)
            pg_host = pg_cols[0].text_input("Host", value="localhost")
            pg_port = pg_cols[1].number_input("Port", min_value=1, max_value=65535, value=5432, step=1)
            pg_schema = pg_cols[2].text_input("Schema", value="public")
            auth_cols = st.columns(3)
            pg_db = auth_cols[0].text_input("Database")
            pg_user = auth_cols[1].text_input("Username")
            pg_password = auth_cols[2].text_input("Password", type="password")
            pg_url = ""
        else:
            st.caption("Connection string")
            pg_url = st.text_input(
                "PostgreSQL URL",
                placeholder="postgresql://username:password@host:5432/database_name?schema=public",
            )
            override_cols = st.columns(2)
            pg_schema = override_cols[0].text_input("Schema override (optional)", value="")
            override_cols[1].caption("Use this to override the schema in the URL if needed.")
            pg_host = ""
            pg_port = 5432
            pg_db = ""
            pg_user = ""
            pg_password = ""
        postgres_ready = st.form_submit_button("Connect PostgreSQL")

    if pg_input_mode == "Connection fields":
        st.caption("Provide host, port, database, username, password, and schema. Connection or permission problems will be shown with a specific error message below.")
    else:
        st.caption("Provide a PostgreSQL connection string. You can optionally override the schema separately.")

    if postgres_ready:
        try:
            if pg_input_mode == "Connection string":
                postgres_config = _postgres_config_from_url(pg_url.strip())
                if pg_schema.strip():
                    postgres_config["schema_name"] = pg_schema.strip()
            else:
                postgres_config = {
                    "host": pg_host.strip(),
                    "port": int(pg_port),
                    "database": pg_db.strip(),
                    "user": pg_user.strip(),
                    "password": pg_password,
                    "schema_name": pg_schema.strip() or "public",
                }
            st.session_state["postgres_config"] = postgres_config
        except ValueError as exc:
            st.error(str(exc))
            postgres_config = None
            st.session_state["postgres_config"] = None


def _render_section_banner(title: str, description: str, tone: str) -> None:
    st.markdown(
        f"""
        <div class="section-banner {tone}">
            <h2>{title}</h2>
            <p>{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _update_pipeline_progress(
    placeholder: object,
    percent: int,
    title: str,
    note: str,
) -> None:
    bounded_percent = max(0, min(100, int(percent)))
    with placeholder.container():
        st.markdown(
            f"""
            <div class="pipeline-progress-card">
                <div class="pipeline-progress-label">Processing</div>
                <div class="pipeline-progress-title">{title}</div>
                <div class="pipeline-progress-note">{note}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.progress(bounded_percent / 100, text=f"{bounded_percent}% complete")


@st.cache_data(show_spinner=False)
def _load_example_questions() -> dict[str, list[str]]:
    if not EXAMPLE_QUESTIONS_PATH.exists():
        return {"Teaching database": [], "Chinook database": []}

    try:
        with zipfile.ZipFile(EXAMPLE_QUESTIONS_PATH) as archive:
            xml_bytes = archive.read("word/document.xml")
    except (OSError, KeyError, zipfile.BadZipFile):
        return {"Teaching database": [], "Chinook database": []}

    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        return {"Teaching database": [], "Chinook database": []}

    namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    paragraphs: list[str] = []
    for paragraph in root.findall(".//w:body/w:p", namespace):
        fragments = [node.text or "" for node in paragraph.findall(".//w:t", namespace)]
        text = "".join(fragments).strip()
        if text:
            paragraphs.append(text)

    sections = {"Teaching database": [], "Chinook database": []}
    current_section: str | None = None
    for line in paragraphs:
        normalized = line.rstrip(":").strip().lower()
        if normalized == "teaching database":
            current_section = "Teaching database"
            continue
        if normalized == "chinook database":
            current_section = "Chinook database"
            continue
        if current_section:
            sections[current_section].append(line)

    return sections


def _render_schema_graph(schema: list[TableSchema], foreign_keys: list[object]) -> None:
    if not schema:
        return

    def _escape(value: str) -> str:
        return (
            value.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )

    lines = [
        "digraph schema_map {",
        '  graph [rankdir=LR, splines=ortho, nodesep=0.65, ranksep=1.15, pad=0.2];',
        '  node [shape=plain, fontname="Helvetica"];',
        '  edge [color="#64748B", fontname="Helvetica", fontsize=8, arrowsize=0.55, minlen=2];',
    ]
    for table in schema:
        column_rows: list[str] = []
        max_columns = 10
        for idx, column in enumerate(table.columns[:max_columns]):
            pk_marker = "PK" if column.name in set(table.primary_keys) else ""
            fk_marker = "FK" if any(column.name in fk.source_columns for fk in foreign_keys if fk.source_table == table.name) else ""
            marker = " / ".join([item for item in (pk_marker, fk_marker) if item]) or "&nbsp;"
            column_rows.append(
                "<TR>"
                f'<TD ALIGN="LEFT" BORDER="0" CELLPADDING="2"><FONT POINT-SIZE="8" COLOR="#475569">{marker}</FONT></TD>'
                f'<TD ALIGN="LEFT" BORDER="0" CELLPADDING="2"><FONT POINT-SIZE="10">{_escape(column.name)}</FONT></TD>'
                "</TR>"
            )
        if len(table.columns) > max_columns:
            column_rows.append(
                '<TR><TD BORDER="0"></TD><TD ALIGN="LEFT" BORDER="0" CELLPADDING="2"><FONT POINT-SIZE="9" COLOR="#64748B">...</FONT></TD></TR>'
            )
        table_label = (
            "<<TABLE BORDER=\"1\" CELLBORDER=\"0\" CELLSPACING=\"0\" CELLPADDING=\"0\" COLOR=\"#2563EB\">"
            f"<TR><TD COLSPAN=\"2\" BGCOLOR=\"#DBEAFE\" CELLPADDING=\"4\"><FONT POINT-SIZE=\"11\"><B>{_escape(table.name)}</B></FONT></TD></TR>"
            + "".join(column_rows)
            + "</TABLE>>"
        )
        lines.append(f'  "{table.name}" [label={table_label}];')
    for fk in foreign_keys:
        label = ", ".join(fk.source_columns)
        lines.append(
            f'  "{fk.source_table}" -> "{fk.target_table}" [xlabel="{_escape(label)}", color="#94A3B8", fontcolor="#475569", fontsize=7];'
        )
    lines.append("}")
    st.graphviz_chart("\n".join(lines), use_container_width=True)



def _render_graph_model_diagram(nodes: list[dict[str, object]], relationships: list[dict[str, object]]) -> None:
    if not nodes or not relationships:
        st.info("No graph model diagram is available yet.")
        return

    def _escape(value: str) -> str:
        return (
            value.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )

    table_to_label = {
        str(node.get("table")): str(node.get("label") or _singularize(str(node.get("table"))))
        for node in nodes
    }

    unique_nodes: dict[str, dict[str, object]] = {}
    for node in nodes:
        label = str(node.get("label") or _singularize(str(node.get("table"))))
        unique_nodes.setdefault(label, node)

    unique_relationships: dict[tuple[str, str, str], dict[str, object]] = {}
    for rel in relationships:
        from_label = table_to_label.get(str(rel.get("from_table")), _singularize(str(rel.get("from_table"))))
        to_label = table_to_label.get(str(rel.get("to_table")), _singularize(str(rel.get("to_table"))))
        rel_type = str(rel.get("final_name", "RELATED_TO"))
        key = (from_label, rel_type, to_label)
        unique_relationships.setdefault(
            key,
            {
                "from_label": from_label,
                "to_label": to_label,
                "type": rel_type,
                "properties": list(rel.get("properties", [])),
            },
        )

    lines = [
        "digraph graph_model {",
        '  graph [rankdir=LR, splines=true, nodesep=1.0, ranksep=1.45, pad=0.35];',
        '  node [shape=plain, fontname="Helvetica"];',
        '  edge [color="#64748B", fontname="Helvetica", fontsize=13, arrowsize=0.8, minlen=2, penwidth=1.2];',
    ]

    for label, node in unique_nodes.items():
        props = list(node.get("properties", []))
        prop_rows: list[str] = []
        for prop in props[:6]:
            prop_rows.append(
                "<TR>"
                f'<TD ALIGN="LEFT" BORDER="0" CELLPADDING="5"><FONT POINT-SIZE="15">{_escape(str(prop))}</FONT></TD>'
                "</TR>"
            )
        if len(props) > 6:
            prop_rows.append(
                '<TR><TD ALIGN="LEFT" BORDER="0" CELLPADDING="5"><FONT POINT-SIZE="14" COLOR="#64748B">...</FONT></TD></TR>'
            )
        node_label = (
            "<<TABLE BORDER=\"1\" CELLBORDER=\"0\" CELLSPACING=\"0\" CELLPADDING=\"0\" COLOR=\"#2563EB\">"
            f"<TR><TD BGCOLOR=\"#EFF6FF\" CELLPADDING=\"7\"><FONT POINT-SIZE=\"18\"><B>{_escape(label)}</B></FONT></TD></TR>"
            + "".join(prop_rows)
            + "</TABLE>>"
        )
        lines.append(f'  "{label}" [label={node_label}];')

    for rel in unique_relationships.values():
        props = list(rel.get("properties", []))
        edge_label = rel["type"]
        if props:
            preview = ", ".join(str(prop) for prop in props[:4])
            if len(props) > 4:
                preview += ", ..."
            edge_label += f" ({preview})"
        lines.append(
            f'  "{rel["from_label"]}" -> "{rel["to_label"]}" [xlabel="{_escape(edge_label)}", color="#94A3B8", fontcolor="#475569", fontsize=13];'
        )

    lines.append("}")
    st.graphviz_chart("\n".join(lines), use_container_width=True)


def _render_table_schema(table_schema: TableSchema, foreign_keys_by_table: dict[str, list[dict[str, object]]]) -> None:
    table_fks = foreign_keys_by_table.get(table_schema.name, [])
    column_count = len(table_schema.columns)
    pk_count = len(table_schema.primary_keys)
    fk_count = len(table_fks)
    index_count = len(table_schema.indexes)

    expander_label = f"{table_schema.name}  |  COLS {column_count}  |  PK {pk_count}  |  FK {fk_count}"
    with st.expander(expander_label, expanded=False):
        st.subheader(f"Table: {table_schema.name}")
        _render_summary_pills(
            [
                ("Columns", column_count),
                ("Primary Keys", pk_count),
                ("Foreign Keys", fk_count),
                ("Indexes", index_count),
            ]
        )

        if table_schema.columns:
            columns_df = pd.DataFrame(
                [
                    {
                        "column": column.name,
                        "type": column.data_type,
                        "not_null": column.not_null,
                        "default": column.default_value,
                        "is_primary_key": column.is_primary_key,
                    }
                    for column in table_schema.columns
                ]
            )
            st.markdown("**Columns**")
            st.dataframe(columns_df, use_container_width=True, hide_index=True)
        else:
            st.info("No columns found.")

        st.markdown("**Primary Keys**")
        if table_schema.primary_keys:
            st.write(", ".join(table_schema.primary_keys))
        else:
            st.write("None")

        st.markdown("**Indexes**")
        if table_schema.indexes:
            indexes_df = pd.DataFrame(
                [
                    {
                        "index_name": index.name,
                        "unique": index.unique,
                        "columns": ", ".join(index.columns),
                    }
                    for index in table_schema.indexes
                ]
            )
            st.dataframe(indexes_df, use_container_width=True, hide_index=True)
        else:
            st.write("None")

        st.markdown("**Foreign Keys**")
        if table_fks:
            st.dataframe(pd.DataFrame(table_fks), use_container_width=True, hide_index=True)
        else:
            st.write("None")


def _read_schema_sql(db_path: Path) -> str:
    with sqlite3.connect(str(db_path)) as conn:
        cursor = conn.execute(
            """
            SELECT sql
            FROM sqlite_master
            WHERE sql IS NOT NULL
              AND type IN ('table', 'index')
              AND name NOT LIKE 'sqlite_%'
            ORDER BY type, name
            """
        )
        statements = [row[0].strip().rstrip(";") + ";" for row in cursor.fetchall() if row[0]]
    return "\n".join(statements)


def _db_signature(uploaded_name: str, payload: bytes) -> str:
    return f"{PIPELINE_CACHE_VERSION}:{uploaded_name}:{len(payload)}"


def _postgres_signature(config: dict[str, object]) -> str:
    return f"{PIPELINE_CACHE_VERSION}:postgres:{config['host']}:{config['port']}:{config['database']}:{config['schema_name']}:{config['user']}"


def _source_columns(rel: dict[str, object]) -> list[str]:
    return list(rel.get("from_join_columns", rel.get("from_columns", [])))


def _target_columns(rel: dict[str, object]) -> list[str]:
    return list(rel.get("to_join_columns", rel.get("to_columns", [])))


def _raw_source_columns(rel: dict[str, object]) -> list[str]:
    return list(rel.get("raw_from_join_columns", rel.get("raw_from_columns", [])))


def _raw_target_columns(rel: dict[str, object]) -> list[str]:
    return list(rel.get("raw_to_join_columns", rel.get("raw_to_columns", [])))


def _normalize_relationship(rel: dict[str, object]) -> dict[str, object]:
    normalized = dict(rel)
    normalized["final_name"] = normalized.get("final_name") or normalized.get("type") or "RELATED_TO"
    normalized["origin"] = normalized.get("origin") or normalized.get("classification") or "deterministic"
    normalized["raw_from_table"] = normalized.get("raw_from_table") or normalized.get("from_table")
    normalized["raw_to_table"] = normalized.get("raw_to_table") or normalized.get("to_table")
    normalized["raw_final_name"] = normalized.get("raw_final_name") or normalized["final_name"]
    normalized["raw_display_text"] = normalized.get("raw_display_text") or _build_display_text(
        str(normalized["raw_from_table"]),
        str(normalized["raw_final_name"]),
        str(normalized["raw_to_table"]),
        list(normalized.get("properties", [])),
    )
    normalized["semantic_from_table"] = normalized.get("semantic_from_table") or normalized.get("from_table")
    normalized["semantic_to_table"] = normalized.get("semantic_to_table") or normalized.get("to_table")
    normalized["semantic_relationship_type"] = normalized.get("semantic_relationship_type") or normalized["final_name"]
    normalized["semantic_display_text"] = normalized.get("semantic_display_text") or _build_display_text(
        str(normalized["semantic_from_table"]),
        str(normalized["semantic_relationship_type"]),
        str(normalized["semantic_to_table"]),
        list(normalized.get("properties", [])),
    )
    normalized["display_text"] = _build_display_text(
        str(normalized["from_table"]),
        str(normalized["final_name"]),
        str(normalized["to_table"]),
        list(normalized.get("properties", [])),
    )
    return normalized


def _editable_relationships() -> list[dict[str, object]]:
    return list(st.session_state.get("editable_relationships", []))


def _set_editable_relationships(relationships: list[dict[str, object]]) -> None:
    st.session_state["editable_relationships"] = [_normalize_relationship(rel) for rel in relationships]


def _ensure_relationship_state(signature: str, graph_mapping: dict[str, object]) -> None:
    if st.session_state.get("editable_relationships_signature") != signature:
        st.session_state["editable_relationships_signature"] = signature
        _set_editable_relationships(list(graph_mapping.get("relationships", [])))
        st.session_state["editing_relationship_id"] = None
        st.session_state["relationship_revision"] = 0


def _relationship_config_from_graph(graph_mapping: dict[str, object]) -> dict[str, object]:
    structural_relationships: list[dict[str, object]] = []
    for rel in graph_mapping.get("relationships", []):
        if rel.get("classification") == "collapsed_join_table":
            continue
        structural_relationships.append(
            {
                "source_table": rel["from_table"],
                "source_columns": _source_columns(rel),
                "target_table": rel["to_table"],
                "target_columns": _target_columns(rel),
                "relationship_type": rel.get("classification", "final"),
                "confidence": rel.get("confidence", 1.0),
                "signals": rel.get("signals", []),
                "id": rel.get("id"),
            }
        )
    return build_mapping_config(structural_relationships)


def _graph_mapping_with_edits(base_graph_mapping: dict[str, object]) -> dict[str, object]:
    updated = dict(base_graph_mapping)
    updated["relationships"] = sorted(
        [_normalize_relationship(rel) for rel in _editable_relationships()],
        key=lambda item: (item["from_table"], item["to_table"], item.get("via_table", ""), item.get("final_name", "")),
    )
    return updated


def _singularize(name: str) -> str:
    cleaned = name.replace("_", " ").strip()
    if cleaned.endswith("s") and len(cleaned) > 1:
        cleaned = cleaned[:-1]
    return cleaned.title().replace(" ", "")


def _humanize_relationship_name(name: str) -> str:
    return name.replace("_", " ").strip().lower()


def _normalize_relationship_name_input(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", name.strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned.upper() or "RELATED_TO"


def _build_display_text(from_table: str, rel_name: str, to_table: str, props: list[str]) -> str:
    base = f"{_singularize(from_table)} {_humanize_relationship_name(rel_name)} {_singularize(to_table)}"
    if props:
        return f"{base} {{{', '.join(props)}}}"
    return base


def _format_reason(signal: str) -> str:
    return signal.replace("_", " ")


def _classification_reasons(item: dict[str, object]) -> list[str]:
    return [_format_reason(signal) for signal in item.get("signals", [])]


def _friendly_label(value: str | None) -> str:
    mapping = {
        "direct_fk_relationship": "Direct FK",
        "collapsed_join_table": "Join Table",
        "join_table": "Join Table",
        "relationship": "Relationship",
        "ambiguous_case": "Ambiguous Case",
        "manual_relationship": "Manual",
        "entity_table": "Entity Table",
        "user_edited": "User Edited",
        "user_added": "User Added",
        "ai_refined": "AI Refined",
        "ai_accepted": "AI Accepted",
        "ai_reviewed": "AI Reviewed",
        "deterministic": "Deterministic",
        "deterministic_only": "Deterministic Only",
        "final": "Final",
        "rule_based": "Rule Based",
        "kept": "Kept",
        "reversed": "Reversed",
    }
    if not value:
        return "Unknown"
    return mapping.get(value, value.replace("_", " ").title())


def _postgres_config_from_url(url: str) -> dict[str, object]:
    parsed = urlparse(url)
    if parsed.scheme not in {"postgres", "postgresql"}:
        raise ValueError("Connection string must start with `postgres://` or `postgresql://`.")
    query = parse_qs(parsed.query)
    return {
        "host": parsed.hostname or "",
        "port": parsed.port or 5432,
        "database": (parsed.path or "").lstrip("/"),
        "user": parsed.username or "",
        "password": parsed.password or "",
        "schema_name": query.get("schema", ["public"])[0] or "public",
    }


def _friendly_source_error(source_info: dict[str, object] | None, exc: Exception) -> str:
    message = str(exc).strip()
    lowered = message.lower()

    if source_info and source_info.get("kind") == "postgres":
        config = source_info.get("config", {})
        target = f"{config.get('host')}:{config.get('port')} / db={config.get('database')} / schema={config.get('schema_name')}"
        if "password authentication failed" in lowered or "authentication failed" in lowered:
            return f"PostgreSQL authentication failed for {target}. Check username and password."
        if "connection refused" in lowered:
            return f"PostgreSQL connection was refused for {target}. Check host, port, and whether the server is running."
        if "could not translate host name" in lowered or "name or service not known" in lowered:
            return f"PostgreSQL host could not be resolved for {target}. Check the host name."
        if "database" in lowered and "does not exist" in lowered:
            return f"PostgreSQL database not found for {target}. Check the database name."
        if "schema" in lowered and "does not exist" in lowered:
            return f"PostgreSQL schema not found for {target}. Check the schema name."
        if "timeout" in lowered:
            return f"PostgreSQL connection timed out for {target}. Check network access and server availability."
        return f"Failed to read PostgreSQL schema from {target}: {message}"

    return f"Failed to read schema: {message}"


def _load_table_counts(db_path: Path, table_names: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    with sqlite3.connect(str(db_path)) as conn:
        for table_name in table_names:
            counts[table_name] = int(conn.execute(f'SELECT COUNT(*) FROM "{table_name}"').fetchone()[0])
    return counts


def _load_postgres_table_counts(config: dict[str, object], table_names: list[str]) -> dict[str, int]:
    import psycopg

    counts: dict[str, int] = {}
    with psycopg.connect(
        host=str(config["host"]),
        port=int(config["port"]),
        dbname=str(config["database"]),
        user=str(config["user"]),
        password=str(config["password"]),
    ) as conn:
        with conn.cursor() as cur:
            for table_name in table_names:
                cur.execute(
                    f'SELECT COUNT(*) FROM "{config["schema_name"]}"."{table_name}"'
                )
                counts[table_name] = int(cur.fetchone()[0])
    return counts


def _estimate_relationship_count(db_path: Path, relationship: dict[str, object]) -> int:
    with sqlite3.connect(str(db_path)) as conn:
        basis_classification = relationship.get("raw_classification", relationship.get("classification"))
        if basis_classification == "collapsed_join_table":
            via_table = relationship.get("raw_via_table", relationship.get("via_table"))
            if not via_table:
                return 0
            return int(conn.execute(f'SELECT COUNT(*) FROM "{via_table}"').fetchone()[0])

        source_table = relationship.get("raw_from_table", relationship["from_table"])
        source_columns = _raw_source_columns(relationship) or _source_columns(relationship)
        if not source_columns:
            return 0
        where_clause = " AND ".join([f'"{col}" IS NOT NULL' for col in source_columns])
        query = f'SELECT COUNT(*) FROM "{source_table}"'
        if where_clause:
            query += f" WHERE {where_clause}"
        return int(conn.execute(query).fetchone()[0])


def _estimate_postgres_relationship_count(config: dict[str, object], relationship: dict[str, object]) -> int:
    import psycopg

    with psycopg.connect(
        host=str(config["host"]),
        port=int(config["port"]),
        dbname=str(config["database"]),
        user=str(config["user"]),
        password=str(config["password"]),
    ) as conn:
        with conn.cursor() as cur:
            schema_name = str(config["schema_name"])
            basis_classification = relationship.get("raw_classification", relationship.get("classification"))
            if basis_classification == "collapsed_join_table":
                via_table = relationship.get("raw_via_table", relationship.get("via_table"))
                if not via_table:
                    return 0
                cur.execute(f'SELECT COUNT(*) FROM "{schema_name}"."{via_table}"')
                return int(cur.fetchone()[0])

            source_table = relationship.get("raw_from_table", relationship["from_table"])
            source_columns = _raw_source_columns(relationship) or _source_columns(relationship)
            if not source_columns:
                return 0
            where_clause = " AND ".join([f'"{col}" IS NOT NULL' for col in source_columns])
            query = f'SELECT COUNT(*) FROM "{schema_name}"."{source_table}"'
            if where_clause:
                query += f" WHERE {where_clause}"
            cur.execute(query)
            return int(cur.fetchone()[0])


def _build_schema_sql_from_metadata(schema: list[TableSchema], foreign_keys: list[object]) -> str:
    fk_by_source: dict[str, list[object]] = {}
    for fk in foreign_keys:
        fk_by_source.setdefault(fk.source_table, []).append(fk)

    statements: list[str] = []
    for table in schema:
        column_lines: list[str] = []
        pk_set = set(table.primary_keys)
        for column in table.columns:
            line = f'{column.name} {column.data_type or "TEXT"}'
            if column.not_null:
                line += " NOT NULL"
            if column.default_value is not None:
                line += f" DEFAULT {column.default_value}"
            if column.name in pk_set and len(pk_set) == 1:
                line += " PRIMARY KEY"
            column_lines.append(line)

        if len(pk_set) > 1:
            column_lines.append(f"PRIMARY KEY ({', '.join(table.primary_keys)})")

        for fk in fk_by_source.get(table.name, []):
            column_lines.append(
                f"FOREIGN KEY ({', '.join(fk.source_columns)}) REFERENCES {fk.target_table} ({', '.join(fk.target_columns)})"
            )

        statements.append(f"CREATE TABLE {table.name} (\n  " + ",\n  ".join(column_lines) + "\n);")

        for index in table.indexes:
            unique_sql = "UNIQUE " if index.unique else ""
            statements.append(
                f"CREATE {unique_sql}INDEX {index.name} ON {table.name} ({', '.join(index.columns)});"
            )

    return "\n".join(statements)


def _node_count_rows(graph_mapping: dict[str, object], source_info: dict[str, object]) -> list[dict[str, object]]:
    if source_info["kind"] == "postgres":
        table_counts = _load_postgres_table_counts(
            source_info["config"],
            [node["table"] for node in graph_mapping.get("nodes", [])],
        )
    else:
        table_counts = _load_table_counts(source_info["db_path"], [node["table"] for node in graph_mapping.get("nodes", [])])
    return [
        {"label": node["label"], "table": node["table"], "count": table_counts.get(node["table"], 0)}
        for node in graph_mapping.get("nodes", [])
    ]


def _relationship_count_rows(graph_mapping: dict[str, object], source_info: dict[str, object]) -> list[dict[str, object]]:
    aggregated: dict[str, int] = {}
    for rel in graph_mapping.get("relationships", []):
        rel_type = str(rel.get("final_name"))
        if source_info["kind"] == "postgres":
            count = _estimate_postgres_relationship_count(source_info["config"], rel)
        else:
            count = _estimate_relationship_count(source_info["db_path"], rel)
        aggregated[rel_type] = aggregated.get(rel_type, 0) + count
    return [{"type": rel_type, "count": count} for rel_type, count in sorted(aggregated.items())]


def _render_ai_review_summary(graph_mapping: dict[str, object], review_provider: str | None, review_model: str | None) -> None:
    summary = graph_mapping.get("ai_review_summary", {})
    semantic_summary = graph_mapping.get("semantic_review_summary", {})
    nodes_in_scope = summary.get("nodes_in_scope", len(graph_mapping.get("nodes", [])))
    join_tables_in_scope = summary.get("join_tables_in_scope", len(graph_mapping.get("join_tables", [])))
    relationships_in_scope = summary.get("relationships_in_scope", len(graph_mapping.get("relationships", [])))
    node_refinements = summary.get("node_refinements", 0)
    join_table_refinements = summary.get("join_table_refinements", 0)
    relationship_refinements = summary.get("relationship_refinements", 0)
    semantic_text = (
        f"Relationship semantics used {semantic_summary.get('provider') or 'AI'}"
        + (f" ({semantic_summary.get('model')})" if semantic_summary.get("model") else "")
        + f" for {semantic_summary.get('ai_reviewed_relationships', 0)} of "
        f"{semantic_summary.get('relationships_total', len(graph_mapping.get('relationships', [])))} relationships; "
        f"renamed {semantic_summary.get('renamed_relationships', 0)} and reversed "
        f"{semantic_summary.get('reversed_relationships', 0)}. "
        f"Fallbacks: {semantic_summary.get('fallback_relationships', 0)}."
        if semantic_summary.get("enabled")
        else f"Relationship semantics are using deterministic fallback for {len(graph_mapping.get('relationships', []))} relationships."
    )

    if summary.get("ai_reviewed"):
        text = (
            f"Using {review_provider or 'AI'}"
            + (f" ({review_model})" if review_model else "")
            + ". "
            f"Checked the full mapping for {nodes_in_scope} nodes, "
            f"{join_tables_in_scope} join tables, and "
            f"{relationships_in_scope} relationships. "
            f"Applied explicit refinements to {node_refinements} nodes, "
            f"{join_table_refinements} join tables, and "
            f"{relationship_refinements} relationships; "
            f"renamed {summary.get('renamed_relationships', 0)} relationships and "
            f"kept {summary.get('ambiguous_cases_flagged', 0)} ambiguous cases visible. "
            f"{semantic_text}"
        )
        badge_text = "AI Reviewed"
    else:
        text = (
            "No AI review was applied. "
            f"Showing deterministic output for {nodes_in_scope} nodes, "
            f"{join_tables_in_scope} join tables, and "
            f"{relationships_in_scope} relationships. {semantic_text}"
        )
        badge_text = "Deterministic"
    container = st.container(border=True)
    badge_col, text_col = container.columns([1, 8])
    badge_col.markdown(f"**{badge_text}**")
    text_col.write(text)
    if graph_mapping.get("manual_edits_applied"):
        text_col.caption("Manual relationship edits are being used as the final result. AI review is not re-run after those edits.")


def _render_metric_cards(items: list[tuple[str, int]]) -> None:
    columns = st.columns(len(items))
    for column, (label, value) in zip(columns, items):
        column.metric(label, value)


def _render_summary_pills(items: list[tuple[str, object]]) -> None:
    pills_html = "".join(
        f'<span class="summary-pill">{label}: {value}</span>'
        for label, value in items
    )
    st.markdown(f'<div class="summary-pills">{pills_html}</div>', unsafe_allow_html=True)


def _render_bullet_list(items: list[str], empty_text: str = "None") -> None:
    if not items:
        st.write(f"- {empty_text}")
        return
    for item in items:
        st.write(f"- {item}")


def _dataframe_height(row_count: int, max_visible_rows: int = 8) -> int:
    visible_rows = max(1, min(row_count, max_visible_rows))
    return 40 + (visible_rows * 35)


def _schema_to_jsonable(
    schema: list[TableSchema],
    foreign_keys_by_table: dict[str, list[dict[str, object]]] | None = None,
) -> list[dict[str, object]]:
    foreign_keys_by_table = foreign_keys_by_table or {}
    return [
        {
            "name": table.name,
            "primary_keys": table.primary_keys,
            "columns": [
                {
                    "name": column.name,
                    "data_type": column.data_type,
                    "not_null": column.not_null,
                    "default_value": column.default_value,
                    "is_primary_key": column.is_primary_key,
                }
                for column in table.columns
            ],
            "indexes": [
                {
                    "name": index.name,
                    "unique": index.unique,
                    "columns": index.columns,
                }
                for index in table.indexes
            ],
            "foreign_keys": foreign_keys_by_table.get(table.name, []),
        }
        for table in schema
    ]


def _delete_relationship(relationship_id: str) -> None:
    st.session_state["editable_relationships"] = [
        rel for rel in _editable_relationships() if rel.get("id") != relationship_id
    ]
    if st.session_state.get("editing_relationship_id") == relationship_id:
        st.session_state["editing_relationship_id"] = None
    st.session_state["relationship_revision"] = int(st.session_state.get("relationship_revision", 0)) + 1
    st.rerun()


def _save_relationship(
    original_relationship: dict[str, object] | None,
    source_table: str,
    source_columns: list[str],
    target_table: str,
    target_columns: list[str],
    final_name: str,
    *,
    swap_direction: bool = False,
) -> None:
    updated = dict(original_relationship or {})
    if swap_direction:
        source_table, target_table = target_table, source_table
        source_columns, target_columns = target_columns, source_columns

    updated["id"] = updated.get("id") or f"manual::{len(_editable_relationships())}"
    updated["from_table"] = source_table
    updated["to_table"] = target_table
    updated["final_name"] = _normalize_relationship_name_input(final_name)
    updated["type"] = updated["final_name"]
    updated["properties"] = list(updated.get("properties", []))
    updated["classification"] = updated.get("classification", "manual_relationship")
    updated["signals"] = list(updated.get("signals", [])) or ["manual_edit"]
    updated["confidence"] = float(updated.get("confidence", 1.0))
    updated["review_status"] = "final"
    updated["reviewable"] = False
    updated["origin"] = "user_edited" if original_relationship else "user_added"
    updated["semantic_review_source"] = updated["origin"]
    updated["semantic_explanation"] = "Manually edited in the UI."
    updated["semantic_fallback_reason"] = None
    updated["semantic_confidence"] = updated["confidence"]
    updated["final_from_table"] = source_table
    updated["final_to_table"] = target_table

    if updated.get("classification") == "collapsed_join_table":
        updated["from_join_columns"] = source_columns
        updated["to_join_columns"] = target_columns
    else:
        updated["from_columns"] = source_columns
        updated["to_columns"] = target_columns

    raw_from = str(updated.get("raw_from_table") or source_table)
    raw_to = str(updated.get("raw_to_table") or target_table)
    updated["semantic_direction"] = "reversed" if (raw_from != source_table or raw_to != target_table) else "kept"
    updated["semantic_from_table"] = source_table
    updated["semantic_to_table"] = target_table
    updated["semantic_relationship_type"] = updated["final_name"]
    updated["semantic_display_text"] = _build_display_text(source_table, updated["final_name"], target_table, list(updated.get("properties", [])))

    updated = _normalize_relationship(updated)

    relationships = _editable_relationships()
    replaced = False
    for idx, rel in enumerate(relationships):
        if rel.get("id") == updated["id"]:
            relationships[idx] = updated
            replaced = True
            break
    if not replaced:
        relationships.append(updated)
    _set_editable_relationships(relationships)
    st.session_state["editing_relationship_id"] = None
    st.session_state["relationship_revision"] = int(st.session_state.get("relationship_revision", 0)) + 1
    st.rerun()


def _render_relationship_editor(schema: list[TableSchema], relationship: dict[str, object] | None = None) -> None:
    table_columns = {table.name: [column.name for column in table.columns] for table in schema}
    table_names = list(table_columns.keys())

    source_table_default = relationship.get("from_table") if relationship else table_names[0]
    target_table_default = relationship.get("to_table") if relationship else (table_names[1] if len(table_names) > 1 else table_names[0])
    table_cols = st.columns(2)
    source_table = table_cols[0].selectbox(
        "Source table",
        options=table_names,
        index=table_names.index(source_table_default) if source_table_default in table_names else 0,
        key=f"edit_source_table_{relationship.get('id', 'new') if relationship else 'new'}",
    )
    target_table = table_cols[1].selectbox(
        "Target table",
        options=table_names,
        index=table_names.index(target_table_default) if target_table_default in table_names else 0,
        key=f"edit_target_table_{relationship.get('id', 'new') if relationship else 'new'}",
    )
    column_cols = st.columns(2)
    source_columns = column_cols[0].multiselect(
        "Source columns",
        options=table_columns[source_table],
        default=[col for col in _source_columns(relationship or {}) if col in table_columns[source_table]],
        key=f"edit_source_columns_{relationship.get('id', 'new') if relationship else 'new'}",
    )
    target_columns = column_cols[1].multiselect(
        "Target columns",
        options=table_columns[target_table],
        default=[col for col in _target_columns(relationship or {}) if col in table_columns[target_table]],
        key=f"edit_target_columns_{relationship.get('id', 'new') if relationship else 'new'}",
    )
    final_name = st.text_input(
        "Relationship name",
        value=str((relationship or {}).get("final_name", "RELATED_TO")),
        key=f"edit_relationship_name_{relationship.get('id', 'new') if relationship else 'new'}",
    )
    swap_direction = st.checkbox(
        "Swap source and target direction before saving",
        value=False,
        key=f"swap_relationship_direction_{relationship.get('id', 'new') if relationship else 'new'}",
        help="Use this when the semantic relationship should point the other way around.",
    )

    action_cols = st.columns(2)
    save_clicked = action_cols[0].form_submit_button("Save")
    cancel_clicked = action_cols[1].form_submit_button("Cancel")

    if cancel_clicked:
        st.session_state["editing_relationship_id"] = None
        st.rerun()

    if save_clicked:
        if not source_columns or not target_columns:
            st.error("Source columns and target columns are required.")
            return
        if len(source_columns) != len(target_columns):
            st.error("Source and target column counts must match.")
            return
        _save_relationship(
            relationship,
            source_table,
            source_columns,
            target_table,
            target_columns,
            final_name,
            swap_direction=swap_direction,
        )


def _relationship_option_label(rel: dict[str, object]) -> str:
    return str(rel.get("semantic_display_text") or rel.get("display_text") or rel.get("id"))


def _apply_relationship_summary_table_edits(edited_rows: list[dict[str, object]]) -> None:
    existing = {str(rel["id"]): rel for rel in _editable_relationships()}
    updated_relationships: list[dict[str, object]] = []

    for row in edited_rows:
        relationship_id = str(row["relationship_id"])
        relationship = existing.get(relationship_id)
        if not relationship:
            continue

        source_table = str(row["source_table"]).strip()
        target_table = str(row["target_table"]).strip()
        relationship_name = str(row["relationship_phrase"]).strip()
        swap_direction = bool(row.get("swap_direction", False))

        source_columns = _source_columns(relationship)
        target_columns = _target_columns(relationship)
        if swap_direction:
            source_table, target_table = target_table, source_table
            source_columns, target_columns = target_columns, source_columns

        updated = dict(relationship)
        updated["from_table"] = source_table
        updated["to_table"] = target_table
        updated["final_name"] = _normalize_relationship_name_input(relationship_name)
        updated["type"] = updated["final_name"]
        updated["origin"] = "user_edited"
        updated["semantic_review_source"] = "user_edited"
        updated["semantic_explanation"] = "Manually edited in the summary table."
        updated["semantic_fallback_reason"] = None
        updated["semantic_confidence"] = updated.get("confidence", 1.0)
        updated["final_from_table"] = source_table
        updated["final_to_table"] = target_table
        updated["semantic_from_table"] = source_table
        updated["semantic_to_table"] = target_table
        updated["semantic_relationship_type"] = updated["final_name"]
        updated["semantic_display_text"] = _build_display_text(
            source_table,
            updated["final_name"],
            target_table,
            list(updated.get("properties", [])),
        )
        raw_from = str(updated.get("raw_from_table") or source_table)
        raw_to = str(updated.get("raw_to_table") or target_table)
        updated["semantic_direction"] = "reversed" if (raw_from != source_table or raw_to != target_table) else "kept"

        if updated.get("classification") == "collapsed_join_table":
            updated["from_join_columns"] = source_columns
            updated["to_join_columns"] = target_columns
        else:
            updated["from_columns"] = source_columns
            updated["to_columns"] = target_columns

        updated_relationships.append(_normalize_relationship(updated))

    st.session_state["editable_relationships"] = updated_relationships
    st.session_state["relationship_revision"] = int(st.session_state.get("relationship_revision", 0)) + 1
    st.rerun()


def _render_relationship_summary_editor(schema: list[TableSchema], relationships: list[dict[str, object]]) -> None:
    st.markdown("**Batch Semantic Edit**")
    st.caption("This is the final relationship result table. Edit multiple rows here, then apply once. Relationship phrases will be normalized automatically.")
    table_names = [table.name for table in schema]
    editor_rows = []
    for rel in relationships:
        editor_rows.append(
            {
                "relationship_id": str(rel["id"]),
                "source_table": str(rel.get("from_table")),
                "relationship_phrase": _humanize_relationship_name(str(rel.get("final_name", "RELATED_TO"))),
                "target_table": str(rel.get("to_table")),
                "final_preview": str(
                    rel.get("semantic_display_text")
                    or rel.get("display_text")
                    or _build_display_text(
                        str(rel.get("from_table")),
                        str(rel.get("final_name", "RELATED_TO")),
                        str(rel.get("to_table")),
                        list(rel.get("properties", [])),
                    )
                ),
                "swap_direction": False,
            }
        )
    with st.form("relationship_summary_batch_edit_form"):
        edited_df = st.data_editor(
            pd.DataFrame(editor_rows),
            use_container_width=True,
            hide_index=True,
            num_rows="fixed",
            column_config={
                "relationship_id": st.column_config.TextColumn("Relationship ID", disabled=True),
                "source_table": st.column_config.SelectboxColumn("Source", options=table_names, required=True),
                "relationship_phrase": st.column_config.TextColumn("Relationship Phrase", required=True),
                "target_table": st.column_config.SelectboxColumn("Target", options=table_names, required=True),
                "final_preview": st.column_config.TextColumn("Final Preview", disabled=True),
                "swap_direction": st.column_config.CheckboxColumn("Swap Direction"),
            },
            disabled=["relationship_id", "final_preview"],
            key="relationship_summary_batch_editor",
        )
        submit_cols = st.columns([1, 4])
        apply_clicked = submit_cols[0].form_submit_button("Apply all edits")
        if apply_clicked:
            _apply_relationship_summary_table_edits(edited_df.to_dict("records"))


def _render_graph_mapping_summary(
    graph_mapping: dict[str, object],
    source_info: dict[str, object],
    query_registry: dict[str, object] | None = None,
) -> None:
    st.subheader("Graph Mapping Result")
    st.caption("Derived node labels, final relationships, explainability details, and graph export options.")
    nodes = graph_mapping.get("nodes", [])
    relationships = graph_mapping.get("relationships", []) or _editable_relationships()
    join_tables = graph_mapping.get("join_tables", [])
    ambiguous_cases = graph_mapping.get("ambiguous_cases", [])
    review_status = graph_mapping.get("review_status", "deterministic_only")
    review_provider = graph_mapping.get("review_provider")
    review_model = graph_mapping.get("review_model")

    _render_metric_cards(
        [
            ("Nodes", len(nodes)),
            ("Relationships", len(relationships)),
            ("Join Tables", len(join_tables)),
            ("Ambiguous Cases", len(ambiguous_cases)),
        ]
    )
    _render_ai_review_summary(graph_mapping, review_provider, review_model)

    download_cols = st.columns(3)
    download_cols[0].download_button(
        label="Download schema.json",
        data=json.dumps(
            _schema_to_jsonable(
                st.session_state["current_schema"],
                st.session_state.get("current_schema_foreign_keys", {}),
            ),
            indent=2,
        ),
        file_name="schema.json",
        mime="application/json",
    )
    download_cols[1].download_button(
        label="Download graph_mapping.json",
        data=json.dumps(graph_mapping, indent=2),
        file_name="graph_mapping.json",
        mime="application/json",
    )
    download_cols[2].download_button(
        label="Download query_schema_registry.json",
        data=json.dumps(query_registry or {}, indent=2),
        file_name="query_schema_registry.json",
        mime="application/json",
    )

    if nodes:
        node_counts = {row["table"]: row["count"] for row in _node_count_rows(graph_mapping, source_info)}
        st.subheader("Node Summary")
        node_rows = [{
                "table": n["table"],
                "label": n["label"],
                "count": node_counts.get(n["table"], 0),
                "primary_key": ", ".join(n.get("primary_key", [])),
                "properties": ", ".join(n.get("properties", [])),
                "classification": _friendly_label(n.get("classification")),
            } for n in nodes]
        st.dataframe(
            pd.DataFrame(node_rows),
            use_container_width=True,
            hide_index=True,
            height=_dataframe_height(len(node_rows), max_visible_rows=7),
        )
        st.caption("Expand a node below if you want to see why it was treated as a graph node.")
        for node in nodes:
            with st.expander(f"{node['table']} -> {node.get('label')}", expanded=False):
                _render_summary_pills(
                    [
                        ("Count", node_counts.get(node["table"], 0)),
                        ("Type", _friendly_label(node.get("classification"))),
                        ("Properties", len(node.get("properties", []))),
                    ]
                )
                st.write(f"Label: {node.get('label')}")
                st.write("Reasoning:")
                _render_bullet_list(_classification_reasons(node))

    st.subheader("Relationship Summary")
    action_col, _ = st.columns([1, 4])
    if action_col.button("Add relationship"):
        st.session_state["editing_relationship_id"] = "__new__"
        st.rerun()

    if st.session_state.get("editing_relationship_id") == "__new__":
        with st.form("add_relationship_form"):
            _render_relationship_editor(schema=st.session_state["current_schema"], relationship=None)

    if relationships:
        relationship_counts = {row["type"]: row["count"] for row in _relationship_count_rows(graph_mapping, source_info)}
        semantic_summary = graph_mapping.get("semantic_review_summary", {})
        if semantic_summary.get("fallback_relationships", 0):
            st.warning(
                f"Semantic review fell back to rule-based output for {semantic_summary.get('fallback_relationships', 0)} relationship(s). "
                "Some relationships are still using the rule-based result."
            )
        _render_relationship_summary_editor(st.session_state["current_schema"], relationships)
        for rel in relationships:
            rel_count = relationship_counts.get(str(rel.get("final_name")), 0)
            relationship_text = rel.get("semantic_display_text") or rel.get("display_text") or _build_display_text(
                str(rel.get("from_table")),
                str(rel.get("final_name")),
                str(rel.get("to_table")),
                list(rel.get("properties", [])),
            )
            expander_title = (
                f"{relationship_text} "
                f"| count={rel_count}"
            )
            with st.expander(expander_title, expanded=False):
                header_cols = st.columns([7, 1, 1])
                with header_cols[0]:
                    _render_summary_pills(
                        [
                            ("Count", rel_count),
                            ("Source Type", _friendly_label(rel.get("classification", "final"))),
                            ("Properties", len(rel.get("properties", []))),
                        ]
                    )
                if header_cols[1].button("Advanced edit", key=f"edit_{rel['id']}"):
                    st.session_state["editing_relationship_id"] = rel["id"]
                    st.rerun()
                if header_cols[2].button("Delete", key=f"delete_{rel['id']}"):
                    _delete_relationship(str(rel["id"]))

                if st.session_state.get("editing_relationship_id") == rel["id"]:
                    with st.form(f"edit_relationship_form_{rel['id']}"):
                        _render_relationship_editor(schema=st.session_state["current_schema"], relationship=rel)

                st.markdown("**Relationship Explanation**")
                st.write(f"Final relationship: {rel.get('semantic_display_text') or relationship_text}")
                st.write(f"Explanation: {rel.get('semantic_explanation', 'Using deterministic rule-based relationship.')}")

                st.markdown("**Why This Mapping**")
                st.write(f"Type: {_friendly_label(rel.get('classification'))}")
                if rel.get("via_table"):
                    st.write(f"Via: {rel.get('via_table')}")
                if rel.get("classification") == "collapsed_join_table":
                    st.write("Join columns")
                    for join_col, target_col in zip(rel.get("from_join_columns", []), rel.get("from_columns", [])):
                        st.write(f"{rel.get('via_table')}.{join_col} -> {rel.get('from_table')}.{target_col}")
                    for join_col, target_col in zip(rel.get("to_join_columns", []), rel.get("to_columns", [])):
                        st.write(f"{rel.get('via_table')}.{join_col} -> {rel.get('to_table')}.{target_col}")
                else:
                    st.write("Column mapping")
                    for source_col, target_col in zip(_source_columns(rel), _target_columns(rel)):
                        st.write(f"{rel.get('from_table')}.{source_col} -> {rel.get('to_table')}.{target_col}")
                st.write("Properties")
                if rel.get("properties"):
                    _render_bullet_list(list(rel.get("properties", [])))
                else:
                    st.write("- none")
                if rel.get("ai_reasoning_summary"):
                    st.write(f"AI note: {rel.get('ai_reasoning_summary')}")
                reasons = _classification_reasons(rel)
                if reasons:
                    st.write("Signals")
                    _render_bullet_list(reasons)
    else:
        st.info("No relationships are currently available. You can add one manually.")

    if join_tables:
        st.subheader(f"Join Table Transformations ({len(join_tables)})")
        for jt in join_tables:
            rel = next((r for r in relationships if r.get("via_table") == jt.get("table")), None)
            with st.expander(jt["table"], expanded=False):
                _render_summary_pills(
                    [
                        ("Confidence", jt.get("confidence", "-")),
                        ("Linked Tables", len(jt.get("linked_tables", []))),
                        ("Join Columns", len(jt.get("join_columns", []))),
                        ("Metadata Columns", len(jt.get("non_key_columns", []))),
                    ]
                )
                if rel:
                    st.write(f"Mapped to: {rel['display_text']}")
                st.write("Linked tables")
                _render_bullet_list(list(jt.get("linked_tables", [])))
                st.write("Join columns")
                _render_bullet_list(list(jt.get("join_columns", [])))
                st.write("Relationship metadata")
                _render_bullet_list(list(jt.get("non_key_columns", [])), empty_text="none")

    if ambiguous_cases:
        st.subheader(f"Ambiguous Cases ({len(ambiguous_cases)})")
        for case in ambiguous_cases:
            label = case.get("object_id") or case.get("object_type", "case")
            with st.expander(str(label), expanded=False):
                _render_summary_pills(
                    [
                        ("Type", _friendly_label(case.get("object_type"))),
                        ("Status", _friendly_label(case.get("review_status", graph_mapping.get("review_status")))),
                    ]
                )
                if case.get("reason"):
                    st.write(f"Reason: {case.get('reason')}")
                if case.get("fallback_name"):
                    st.write(f"Fallback: {case.get('fallback_name')}")

    st.subheader("Graph Preview")
    with st.expander(f"Graph Preview ({len(relationships)} relationships)", expanded=False):
        st.markdown("**Graph Model Diagram**")
        st.caption("A schema-level graph view showing node labels, relationship types, direction, and relationship properties.")
        _render_graph_model_diagram(nodes, relationships)

        st.markdown("**Relationship Preview**")
        for rel in relationships:
            props = rel.get("properties", [])
            props_display = f" {{{', '.join(props)}}}" if props else ""
            st.code(
                f"{_singularize(str(rel.get('from_table')))} "
                f"-[:{str(rel.get('final_name'))}]"
                f"{props_display}-> "
                f"{_singularize(str(rel.get('to_table')))}"
            )



def _render_convert_section(
    source_info: dict[str, object],
    schema: list[TableSchema],
    graph_mapping: dict[str, object],
    conversion_context_key: str,
) -> None:
    st.subheader("Neo4j Conversion")
    st.caption("Load source rows as nodes and apply the approved mapping to create graph relationships in Neo4j.")

    relationship_config = st.session_state.get("relationship_config")
    if not relationship_config:
        st.info("Build or review relationship mapping first.")
        return

    _render_summary_pills(
        [
            ("Source", "PostgreSQL" if source_info["kind"] == "postgres" else "SQLite"),
            ("Tables", len(schema)),
            ("Relationships", len(graph_mapping.get("relationships", []))),
        ]
    )
    mode_cols = st.columns([1, 2])
    mode = mode_cols[0].selectbox("Conversion mode", options=["structural", "semantic"], index=0, help="Use semantic mode to apply Layer 2 graph mapping semantics.")
    mode_cols[1].caption("Structural uses the structural mapping. Semantic uses the reviewed graph mapping and semantic relationship names.")

    existing_conn = st.session_state.get("neo4j_connection", {})
    conn_cols = st.columns(3)
    neo4j_uri = conn_cols[0].text_input("Neo4j URI", value=str(existing_conn.get("uri", "bolt://localhost:7687")), key="neo4j_uri")
    neo4j_user = conn_cols[1].text_input("Neo4j username", value=str(existing_conn.get("user", "neo4j")), key="neo4j_user")
    neo4j_password = conn_cols[2].text_input("Neo4j password", value=str(existing_conn.get("password", "")), type="password", key="neo4j_password")
    st.session_state["neo4j_connection"] = {"uri": neo4j_uri, "user": neo4j_user, "password": neo4j_password}

    def _render_conversion_result(result: dict[str, object]) -> None:
        metric_columns = st.columns(7)
        metric_columns[0].metric("Nodes created", int(result.get("nodes_created", 0)))
        metric_columns[1].metric("Relationships created", int(result.get("relationships_created", 0)))
        metric_columns[2].metric("Existing rels reused", int(result.get("existing_relationships", 0)))
        metric_columns[3].metric("Join tables collapsed", int(result.get("join_tables_collapsed", 0)))
        metric_columns[4].metric("Direct FK used", int(result.get("direct_fk_relationships_used", 0)))
        metric_columns[5].metric("Unmatched records", int(result.get("unmatched_records", 0)))
        warnings = list(result.get("warnings", []))
        metric_columns[6].metric("Warnings", len(warnings))

        unmatched_examples = list(result.get("unmatched_examples", []))
        if unmatched_examples:
            st.subheader("Unmatched Diagnostics")
            st.caption("These are sample rows where a relationship could not be created because one or both endpoint nodes were not found in Neo4j.")
            unmatched_df = pd.DataFrame(unmatched_examples)
            st.dataframe(unmatched_df, use_container_width=True, hide_index=True)

        existing_relationships = int(result.get("existing_relationships", 0))
        if existing_relationships:
            st.info(
                f"{existing_relationships} relationship row(s) matched existing Neo4j relationships. "
                "These were reused rather than created again."
            )

        report = str(result.get("report", ""))
        st.subheader("Conversion Report")
        st.text(report)
        if warnings:
            with st.expander("Warning details"):
                for warning in warnings:
                    st.warning(str(warning))

        st.download_button(
            label="Download conversion report",
            data=report,
            file_name="neo4j_conversion_report.txt",
            mime="text/plain",
        )

    if st.button("Convert", type="primary"):
        try:
            if source_info["kind"] == "postgres":
                stats = convert_postgres_to_neo4j(
                    postgres_config=source_info["config"],
                    schema=schema,
                    mapping_config=relationship_config,
                    neo4j_uri=neo4j_uri,
                    neo4j_user=neo4j_user,
                    neo4j_password=neo4j_password,
                    mode=mode,
                    graph_mapping=graph_mapping if mode == "semantic" else None,
                )
            else:
                stats = convert_sqlite_to_neo4j(
                    sqlite_db_path=source_info["db_path"],
                    schema=schema,
                    mapping_config=relationship_config,
                    neo4j_uri=neo4j_uri,
                    neo4j_user=neo4j_user,
                    neo4j_password=neo4j_password,
                    mode=mode,
                    graph_mapping=graph_mapping if mode == "semantic" else None,
                )
            report = stats.to_readable_report()
            st.session_state["neo4j_conversion_result"] = {
                "context_key": conversion_context_key,
                "mode": mode,
                "stats": {**asdict(stats), "report": report},
            }
            st.success("Conversion completed.")
        except Exception as exc:  # surfaced to user in app
            st.error(f"Conversion failed: {exc}")

    stored_result = st.session_state.get("neo4j_conversion_result")
    if stored_result and stored_result.get("context_key") == conversion_context_key:
        _render_conversion_result(dict(stored_result.get("stats", {})))


def _render_query_section(
    graph_mapping: dict[str, object],
    query_registry: dict[str, object] | None = None,
    source_schema: list[object] | None = None,
    source_foreign_keys: list[object] | None = None,
) -> None:
    st.subheader("Graph Query")
    neo4j_conn = st.session_state.get("neo4j_connection") or {}
    if not neo4j_conn.get("uri") or not neo4j_conn.get("user"):
        st.info("Enter Neo4j connection details in the conversion section first.")
        return

    example_groups = _load_example_questions()
    example_source = st.radio(
        "Example question set",
        options=["Teaching database", "Chinook database", "Other"],
        horizontal=True,
        key="graph_query_example_source",
    )
    previous_example_source = st.session_state.get("graph_query_example_source_previous")
    if previous_example_source != example_source:
        for source_name in ("Teaching database", "Chinook database"):
            st.session_state.pop(f"graph_query_example_select_{source_name}", None)
        if previous_example_source is not None:
            st.session_state["graph_query_question"] = ""
    st.session_state["graph_query_example_source_previous"] = example_source

    if example_source != "Other":
        examples = example_groups.get(example_source, [])
        if examples:
            selected_example = st.selectbox(
                "Example question",
                options=[""] + examples,
                format_func=lambda value: "Select an example question" if value == "" else value,
                key=f"graph_query_example_select_{example_source}",
            )
            if selected_example:
                st.session_state["graph_query_question"] = selected_example
        else:
            st.caption(f"No example questions were loaded for {example_source.lower()}.")

    question = st.text_input("Natural language question", key="graph_query_question")
    plan_clicked = st.button("Generate query", type="primary", key="graph_query_generate")

    if plan_clicked:
        ai_settings = load_ai_settings()
        if not ai_settings:
            st.session_state["graph_query_plan"] = {
                "status": "unsupported",
                "query_type": "unsupported",
                "cypher": "",
                "params": {},
                "explanation": "Graph query now requires AI configuration.",
                "planner": "ai_required",
                "debug": {},
            }
            st.session_state["graph_query_rows"] = None
        else:
            plan = plan_graph_query_full_ai(
                question,
                graph_mapping,
                ai_settings=ai_settings,
                query_registry=query_registry,
                source_schema=source_schema,
                source_foreign_keys=source_foreign_keys,
            )
            st.session_state["graph_query_plan"] = {
                "status": plan.status,
                "query_type": plan.query_type,
                "cypher": plan.cypher,
                "params": plan.params,
                "explanation": plan.explanation,
                "planner": plan.planner,
                "debug": plan.debug,
            }
            st.session_state["graph_query_rows"] = None

    plan = st.session_state.get("graph_query_plan")
    if not plan:
        return

    if plan.get("status") != "ok":
        st.warning(str(plan.get("explanation")))
        return

    st.markdown("**Generated Cypher**")
    st.code(str(plan.get("cypher")), language="cypher")

    if st.button("Run query", key="graph_query_run"):
        try:
            rows = run_graph_query(
                neo4j_uri=str(neo4j_conn["uri"]),
                neo4j_user=str(neo4j_conn["user"]),
                neo4j_password=str(neo4j_conn.get("password", "")),
                cypher=str(plan.get("cypher", "")),
                params=dict(plan.get("params", {})),
            )
            st.session_state["graph_query_rows"] = rows
        except Exception as exc:
            st.error(f"Query failed: {exc}")
            st.session_state["graph_query_rows"] = []

    rows = st.session_state.get("graph_query_rows")
    if rows is not None:
        st.markdown("**Query Results**")
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.info("No rows returned.")

source_info: dict[str, object] | None = None
tmp_path: Path | None = None

if source_type == "SQLite file" and uploaded_file is not None:
    payload = uploaded_file.getvalue()
    db_signature = _db_signature(uploaded_file.name, payload)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as tmp_file:
        tmp_file.write(payload)
        tmp_path = Path(tmp_file.name)
    source_info = {"kind": "sqlite", "db_path": tmp_path}
elif source_type == "PostgreSQL connection" and postgres_config:
    db_signature = _postgres_signature(postgres_config)
    source_info = {"kind": "postgres", "config": postgres_config}
else:
    db_signature = None

if source_info is not None and db_signature is not None:
    progress_placeholder = st.empty()
    try:
        _update_pipeline_progress(
            progress_placeholder,
            10,
            "Loading source schema",
            "Inspecting tables, columns, and foreign keys from the selected database.",
        )
        if source_info["kind"] == "sqlite":
            schema = introspect_sqlite_schema(source_info["db_path"])
            foreign_keys = load_foreign_keys(source_info["db_path"], [table.name for table in schema]) if schema else []
        else:
            schema = introspect_postgres_schema(**source_info["config"])
            foreign_keys = load_postgres_foreign_keys(
                **source_info["config"],
                table_names=[table.name for table in schema],
            ) if schema else []

        foreign_keys_by_table: dict[str, list[dict[str, object]]] = {}
        for fk in foreign_keys:
            foreign_keys_by_table.setdefault(fk.source_table, []).append(
                {
                    "source_columns": ", ".join(fk.source_columns),
                    "target_table": fk.target_table,
                    "target_columns": ", ".join(fk.target_columns),
                }
            )

        if not schema:
            progress_placeholder.empty()
            st.warning("No user-defined tables found in this database.")
        else:
            _update_pipeline_progress(
                progress_placeholder,
                28,
                "Rendering relational schema",
                "Preparing the schema diagram and detailed table metadata for the page.",
            )
            st.session_state["current_schema"] = schema
            st.session_state["current_schema_foreign_keys"] = foreign_keys_by_table
            st.success(f"Found {len(schema)} table(s).")
            _render_section_banner(
                "RDBMS Schema Inspection",
                "Review the original relational structure before graph mapping. Each table includes columns, primary keys, indexes, and foreign keys.",
                "schema",
            )
            total_foreign_keys = sum(len(items) for items in foreign_keys_by_table.values())
            st.subheader("Schema Diagram")
            st.caption("A compact ER-style view of the relational schema. Edges point from foreign-key holder tables to referenced tables.")
            _render_schema_graph(schema, foreign_keys)
            st.subheader("Schema Details")
            st.caption(f"{len(schema)} tables detected, with {total_foreign_keys} foreign key definitions. Expand a table only when you need its full details.")
            for table_schema in schema:
                _render_table_schema(table_schema, foreign_keys_by_table)

            try:
                if st.session_state.get("ai_graph_mapping_signature") != db_signature:
                    _update_pipeline_progress(
                        progress_placeholder,
                        45,
                        "Building graph mapping",
                        "Converting the relational structure into nodes, relationships, and join-table semantics.",
                    )
                    schema_sql = _build_schema_sql_from_metadata(schema, foreign_keys)
                    detected = detect_relationships(schema_sql)
                    graph_mapping = build_graph_mapping(schema=schema, foreign_keys=foreign_keys, structural_mapping=detected)
                    _update_pipeline_progress(
                        progress_placeholder,
                        62,
                        "Reviewing semantic relationships",
                        "Refining relationship naming, direction, and semantic interpretations.",
                    )
                    semantic_review = run_semantic_relationship_review(graph_mapping)
                    graph_mapping = apply_semantic_relationship_review(graph_mapping, semantic_review)
                    _update_pipeline_progress(
                        progress_placeholder,
                        78,
                        "Running AI review",
                        "Applying the optional AI pass to improve the graph mapping where configured.",
                    )
                    ai_review = run_ai_mapping_review(graph_mapping, review_relationships=False)
                    reviewed_graph_mapping = apply_ai_review(graph_mapping, ai_review)
                    st.session_state["ai_graph_mapping_signature"] = db_signature
                    st.session_state["ai_graph_mapping"] = reviewed_graph_mapping

                base_graph_mapping = st.session_state["ai_graph_mapping"]
                _ensure_relationship_state(db_signature, base_graph_mapping)
                candidate_graph_mapping = _graph_mapping_with_edits(base_graph_mapping)
                review_signature = f"{db_signature}:{int(st.session_state.get('relationship_revision', 0))}"
                if st.session_state.get("reviewed_graph_mapping_signature") != review_signature:
                    if int(st.session_state.get("relationship_revision", 0)) > 0:
                        _update_pipeline_progress(
                            progress_placeholder,
                            86,
                            "Applying manual relationship edits",
                            "Rebuilding the reviewed graph mapping with your latest relationship changes.",
                        )
                        graph_mapping = _graph_mapping_with_edits(base_graph_mapping)
                        graph_mapping["manual_edits_applied"] = True
                    else:
                        _update_pipeline_progress(
                            progress_placeholder,
                            86,
                            "Refreshing reviewed mapping",
                            "Preparing the final reviewed graph mapping for visualization and export.",
                        )
                        semantic_review = run_semantic_relationship_review(candidate_graph_mapping)
                        candidate_graph_mapping = apply_semantic_relationship_review(candidate_graph_mapping, semantic_review)
                        ai_review = run_ai_mapping_review(candidate_graph_mapping, review_relationships=False)
                        graph_mapping = apply_ai_review(candidate_graph_mapping, ai_review)
                    st.session_state["reviewed_graph_mapping_signature"] = review_signature
                    st.session_state["reviewed_graph_mapping"] = graph_mapping
                    _set_editable_relationships(list(graph_mapping.get("relationships", [])))
                graph_mapping = st.session_state["reviewed_graph_mapping"]
                st.session_state["relationship_config"] = _relationship_config_from_graph(graph_mapping)
            except Exception as exc:
                st.warning(f"AI review failed, showing deterministic result: {exc}")
                if st.session_state.get("ai_graph_mapping_signature") != db_signature:
                    _update_pipeline_progress(
                        progress_placeholder,
                        72,
                        "Falling back to deterministic mapping",
                        "AI review was not available, so the app is preparing the non-AI graph mapping.",
                    )
                    schema_sql = _build_schema_sql_from_metadata(schema, foreign_keys)
                    detected = detect_relationships(schema_sql)
                    graph_mapping = build_graph_mapping(schema=schema, foreign_keys=foreign_keys, structural_mapping=detected)
                    semantic_review = run_semantic_relationship_review(graph_mapping)
                    graph_mapping = apply_semantic_relationship_review(graph_mapping, semantic_review)
                    graph_mapping = apply_ai_review(graph_mapping, None)
                    st.session_state["ai_graph_mapping_signature"] = db_signature
                    st.session_state["ai_graph_mapping"] = graph_mapping
                base_graph_mapping = st.session_state["ai_graph_mapping"]
                _ensure_relationship_state(db_signature, base_graph_mapping)
                graph_mapping = _graph_mapping_with_edits(base_graph_mapping)
                if int(st.session_state.get("relationship_revision", 0)) > 0:
                    graph_mapping["manual_edits_applied"] = True
                graph_mapping["review_status"] = "deterministic_only"
                graph_mapping["review_provider"] = None
                graph_mapping["review_model"] = None
                st.session_state["reviewed_graph_mapping"] = graph_mapping
                st.session_state["reviewed_graph_mapping_signature"] = f"{db_signature}:fallback:{int(st.session_state.get('relationship_revision', 0))}"
                st.session_state["relationship_config"] = _relationship_config_from_graph(graph_mapping)

            _update_pipeline_progress(
                progress_placeholder,
                94,
                "Preparing results",
                "Loading the final graph mapping, conversion section, and query interface.",
            )
            _render_section_banner(
                "Graph Mapping Result",
                "Inspect the graph-oriented interpretation of the source schema, including node labels, relationship semantics, counts, and explainability details.",
                "graph",
            )
            query_registry = build_query_schema_registry(
                graph_mapping,
                source_schema=schema,
                source_foreign_keys=foreign_keys,
            )
            st.session_state["query_schema_registry"] = query_registry
            _render_graph_mapping_summary(graph_mapping, source_info, query_registry)
            _render_section_banner(
                "Neo4j Conversion",
                "Use the reviewed mapping to load data into Neo4j and generate a conversion report.",
                "convert",
            )
            conversion_context_key = (
                f"{db_signature}:"
                f"{st.session_state.get('reviewed_graph_mapping_signature', '')}:"
                f"{source_info['kind']}"
            )
            _render_convert_section(source_info, schema, graph_mapping, conversion_context_key)
            _render_section_banner(
                "Graph Query",
                "Query the final graph schema with a constrained natural-language interface that generates and runs read-only Cypher.",
                "convert",
            )
            _render_query_section(graph_mapping, query_registry, schema, foreign_keys)
            _update_pipeline_progress(
                progress_placeholder,
                100,
                "Ready",
                "The schema, graph mapping, conversion tools, and query module are now available below.",
            )
            progress_placeholder.empty()

    except Exception as exc:  # surfaced to user in app
        progress_placeholder.empty()
        st.error(_friendly_source_error(source_info, exc))
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
else:
    if source_type == "SQLite file":
        st.info("Waiting for a SQLite database upload.")
    else:
        st.info("Enter PostgreSQL connection details and click `Connect PostgreSQL`.")
