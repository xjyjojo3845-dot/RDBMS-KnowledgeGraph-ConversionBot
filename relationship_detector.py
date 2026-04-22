import json
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class Column:
    name: str
    data_type: str
    is_primary_key: bool = False


@dataclass
class ForeignKey:
    source_columns: List[str]
    target_table: str
    target_columns: List[str]


@dataclass
class Table:
    name: str
    columns: Dict[str, Column] = field(default_factory=dict)
    primary_keys: Set[str] = field(default_factory=set)
    unique_indexes: List[List[str]] = field(default_factory=list)
    indexes: List[List[str]] = field(default_factory=list)
    foreign_keys: List[ForeignKey] = field(default_factory=list)


def normalize_identifier(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def canonical_term(name: str) -> str:
    normalized = normalize_identifier(name)
    synonym_groups = {
        "zip": {"zip", "zipcode", "postalcode", "postcode", "zipcd"},
        "id": {"id", "identifier"},
    }
    for canonical, terms in synonym_groups.items():
        if normalized in terms:
            return canonical
    return normalized


def base_type(dtype: str) -> str:
    normalized = dtype.lower()
    if "int" in normalized:
        return "int"
    if any(token in normalized for token in ["char", "text", "clob", "varchar", "string"]):
        return "text"
    if any(token in normalized for token in ["float", "double", "real", "numeric", "decimal"]):
        return "float"
    if any(token in normalized for token in ["date", "time"]):
        return "datetime"
    return "other"


def _unquote_identifier(name: str) -> str:
    name = name.strip()
    if len(name) >= 2 and name[0] == name[-1] and name[0] in {'"', "`", "'"}:
        return name[1:-1]
    return name


def parse_schema(schema_sql: str) -> Dict[str, Table]:
    tables: Dict[str, Table] = {}

    identifier = r"(?:`[^`]+`|\"[^\"]+\"|'[^']+'|\w+)"
    create_table_pattern = re.compile(
        rf"CREATE\s+TABLE\s+({identifier})\s*\((.*?)\);",
        re.IGNORECASE | re.DOTALL,
    )

    for match in create_table_pattern.finditer(schema_sql):
        table_name, body = match.groups()
        table_name = _unquote_identifier(table_name)
        table = Table(name=table_name)

        parts = _split_definition_parts(body)
        for part in parts:
            stripped = part.strip()
            upper = stripped.upper()

            if upper.startswith("PRIMARY KEY"):
                cols = _extract_column_list(stripped)
                table.primary_keys.update(cols)
                continue

            if "FOREIGN KEY" in upper:
                fk_match = re.search(
                    rf"FOREIGN\s+KEY\s*\((.*?)\)\s+REFERENCES\s+({identifier})\s*\((.*?)\)",
                    stripped,
                    re.IGNORECASE,
                )
                if fk_match:
                    src_cols, target_table, target_cols = fk_match.groups()
                    table.foreign_keys.append(
                        ForeignKey(
                            source_columns=[_unquote_identifier(c) for c in src_cols.split(",")],
                            target_table=_unquote_identifier(target_table),
                            target_columns=[_unquote_identifier(c) for c in target_cols.split(",")],
                        )
                    )
                continue

            col_match = re.match(rf"^({identifier})\s+([^,]+)$", stripped)
            if col_match:
                col_name, rest = col_match.groups()
                col_name = _unquote_identifier(col_name)
                first_token = rest.split()[0]
                is_pk = "PRIMARY KEY" in upper
                table.columns[col_name] = Column(name=col_name, data_type=first_token, is_primary_key=is_pk)
                if is_pk:
                    table.primary_keys.add(col_name)

        for pk in table.primary_keys:
            if pk in table.columns:
                table.columns[pk].is_primary_key = True

        tables[table_name] = table

    index_pattern = re.compile(
        rf"CREATE\s+(UNIQUE\s+)?INDEX\s+{identifier}\s+ON\s+({identifier})\s*\((.*?)\);",
        re.IGNORECASE | re.DOTALL,
    )
    for match in index_pattern.finditer(schema_sql):
        is_unique, table_name, cols_raw = match.groups()
        table_name = _unquote_identifier(table_name)
        cols = [_unquote_identifier(c) for c in cols_raw.split(",")]
        table = tables.get(table_name)
        if not table:
            continue
        table.indexes.append(cols)
        if is_unique:
            table.unique_indexes.append(cols)

    return tables


def _split_definition_parts(body: str) -> List[str]:
    parts: List[str] = []
    current = []
    depth = 0
    for ch in body:
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1

        if ch == ',' and depth == 0:
            parts.append(''.join(current))
            current = []
        else:
            current.append(ch)
    if current:
        parts.append(''.join(current))
    return parts


def _extract_column_list(defn: str) -> List[str]:
    m = re.search(r"\((.*?)\)", defn)
    if not m:
        return []
    return [_unquote_identifier(c) for c in m.group(1).split(",")]


def detect_relationships(schema_sql: str) -> Dict[str, object]:
    tables = parse_schema(schema_sql)
    relationships: List[Dict[str, object]] = []

    # 1) explicit FKs
    for table in tables.values():
        for fk in table.foreign_keys:
            relationships.append(
                {
                    "source_table": table.name,
                    "source_columns": fk.source_columns,
                    "target_table": fk.target_table,
                    "target_columns": fk.target_columns,
                    "relationship_type": "explicit_fk",
                    "confidence": 1.0,
                    "signals": ["foreign_key_constraint"],
                }
            )

    # 2) heuristic inference
    inferred = _infer_relationships(tables)
    relationships.extend(inferred)

    mapping = _build_mapping(relationships)
    return {
        "relationships": relationships,
        "mapping": mapping,
    }


def build_review_defaults(
    relationships: List[Dict[str, object]],
    auto_accept_threshold: float = 0.9,
) -> List[Dict[str, object]]:
    review_items: List[Dict[str, object]] = []
    for rel in relationships:
        rel_type = rel.get("relationship_type", "")
        confidence = float(rel.get("confidence", 0.0))
        if rel_type == "explicit_fk":
            action = "accept"
        elif confidence >= auto_accept_threshold:
            action = "accept"
        else:
            action = "review"

        review_items.append(
            {
                "relationship": rel,
                "action": action,
                "edited_relationship": None,
            }
        )
    return review_items


def build_final_mapping_config(review_items: List[Dict[str, object]]) -> Dict[str, object]:
    finalized_relationships: List[Dict[str, object]] = []

    for item in review_items:
        action = item.get("action", "review")
        relationship = item.get("relationship")
        edited_relationship = item.get("edited_relationship")

        if action == "reject":
            continue
        if action == "edit" and edited_relationship:
            finalized_relationships.append(edited_relationship)
            continue
        if relationship:
            finalized_relationships.append(relationship)

    mapping = _build_mapping(finalized_relationships)
    return {
        "relationships": finalized_relationships,
        "mapping": mapping,
    }


def build_mapping_config(relationships: List[Dict[str, object]]) -> Dict[str, object]:
    mapping = _build_mapping(relationships)
    return {
        "relationships": relationships,
        "mapping": mapping,
    }


def build_auto_mapping_config(
    relationships: List[Dict[str, object]],
    inferred_confidence_threshold: float = 0.9,
) -> Dict[str, object]:
    finalized_relationships: List[Dict[str, object]] = []

    for rel in relationships:
        rel_type = str(rel.get("relationship_type", ""))
        if rel_type == "explicit_fk":
            finalized_relationships.append(rel)
            continue

        confidence = float(rel.get("confidence", 0.0))
        signals = set(rel.get("signals", []))
        is_strong_inference = (
            confidence >= inferred_confidence_threshold
            or (
                confidence >= 0.82
                and "name_canonical_match" in signals
                and "target_key_like" in signals
            )
        )
        if is_strong_inference:
            finalized_relationships.append(rel)

    return build_mapping_config(finalized_relationships)


def _infer_relationships(tables: Dict[str, Table]) -> List[Dict[str, object]]:
    inferred: List[Dict[str, object]] = []

    explicit_pairs = set()
    for table in tables.values():
        for fk in table.foreign_keys:
            explicit_pairs.add((table.name, tuple(fk.source_columns), fk.target_table, tuple(fk.target_columns)))

    for source_name, source_table in tables.items():
        for source_col in source_table.columns.values():
            source_term = canonical_term(source_col.name)
            if source_col.is_primary_key:
                continue

            for target_name, target_table in tables.items():
                if source_name == target_name:
                    continue

                for target_col in target_table.columns.values():
                    pair_sig = (source_name, (source_col.name,), target_name, (target_col.name,))
                    if pair_sig in explicit_pairs:
                        continue

                    score, signals = _score_candidate(
                        source_table,
                        source_col,
                        target_table,
                        target_col,
                    )

                    if score >= 0.60:
                        inferred.append(
                            {
                                "source_table": source_name,
                                "source_columns": [source_col.name],
                                "target_table": target_name,
                                "target_columns": [target_col.name],
                                "relationship_type": "inferred",
                                "confidence": round(min(score, 0.99), 2),
                                "signals": signals,
                            }
                        )

    # keep best inferred relationship per source table/column
    best: Dict[Tuple[str, str], Dict[str, object]] = {}
    for rel in inferred:
        key = (rel["source_table"], rel["source_columns"][0])
        existing = best.get(key)
        if not existing or rel["confidence"] > existing["confidence"]:
            best[key] = rel

    return list(best.values())


def _score_candidate(source_table: Table, source_col: Column, target_table: Table, target_col: Column) -> Tuple[float, List[str]]:
    score = 0.0
    signals: List[str] = []

    s_term = canonical_term(source_col.name)
    t_term = canonical_term(target_col.name)

    if s_term == t_term:
        score += 0.55
        signals.append("name_canonical_match")
    else:
        ratio = SequenceMatcher(None, s_term, t_term).ratio()
        if ratio >= 0.75:
            score += 0.35
            signals.append("name_similarity")

    if base_type(source_col.data_type) == base_type(target_col.data_type):
        score += 0.20
        signals.append("data_type_compatible")

    is_target_key_like = target_col.is_primary_key or any([target_col.name in idx for idx in target_table.unique_indexes])
    if is_target_key_like:
        score += 0.15
        signals.append("target_key_like")

    source_indexed = any([source_col.name in idx for idx in source_table.indexes])
    if source_indexed:
        score += 0.10
        signals.append("source_indexed")

    # avoid obvious audit field noise
    if s_term in {"createdat", "updatedat"} or t_term in {"createdat", "updatedat"}:
        score -= 0.40

    return score, signals


def _build_mapping(relationships: List[Dict[str, object]]) -> Dict[str, List[Dict[str, object]]]:
    mapping: Dict[str, List[Dict[str, object]]] = {}
    for rel in sorted(relationships, key=lambda r: (-r["confidence"], r["source_table"])):
        mapping.setdefault(rel["source_table"], []).append(
            {
                "to_table": rel["target_table"],
                "from_columns": rel["source_columns"],
                "to_columns": rel["target_columns"],
                "type": rel["relationship_type"],
                "confidence": rel["confidence"],
            }
        )
    return mapping


if __name__ == "__main__":
    import sys

    schema = sys.stdin.read()
    result = detect_relationships(schema)
    print(json.dumps(result, indent=2))
