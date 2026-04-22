from __future__ import annotations

import re
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any

from ai.client import AIClient
from ai.config import AISettings


ALLOWED_QUERY_TYPES = {
    "entity_lookup",
    "one_hop",
    "two_hop",
    "relationship_property",
    "fixed_multi_hop",
    "constrained_multi_hop",
}


@dataclass
class QueryPlan:
    query_type: str
    cypher: str
    params: dict[str, Any]
    explanation: str
    status: str = "ok"
    planner: str = "rules"
    debug: dict[str, Any] | None = None


@dataclass
class PreprocessedQuestion:
    raw_question: str
    normalized_question: str
    quoted_values: list[str]
    person_token: str | None
    year_values: list[int]
    semester_values: list[str]
    hop_limit: int | None
    node_mentions: list[dict[str, Any]]
    relationship_mentions: list[dict[str, Any]]


@dataclass
class QueryTypeDecision:
    query_type: str
    rationale: str


@dataclass
class ExtractedIntent:
    query_type: str
    intent_contract: Any | None = None
    anchor_label: str | None = None
    return_label: str | None = None
    anchor_value: str | None = None
    return_value: str | None = None
    answer_label: str | None = None
    target_label: str | None = None
    requested_relationship_property: str | None = None
    relationship_type: str | None = None
    answer_value: str | None = None
    target_value: str | None = None
    answer_property: str | None = None
    target_property: str | None = None
    relationship_filters: list[dict[str, Any]] | None = None
    hop_limit: int | None = None
    explicit_node_sequence: list[str] | None = None
    required_entity_labels: list[str] | None = None
    allowed_relationship_types: list[str] | None = None
    return_hint: str | None = None


@dataclass
class ResolvedPath:
    query_type: str
    node_sequence: list[dict[str, Any]]
    relationship_sequence: list[dict[str, Any]]
    answer_node: dict[str, Any] | None
    target_node: dict[str, Any] | None
    anchor_node: dict[str, Any] | None = None
    return_node: dict[str, Any] | None = None
    requested_relationship_property: str | None = None
    anchor_value: str | None = None
    return_value: str | None = None
    answer_property: str | None = None
    answer_value: str | None = None
    target_property: str | None = None
    target_value: str | None = None
    relationship_filters: list[dict[str, Any]] | None = None


@dataclass
class FilterSpec:
    scope: str
    property: str
    operator: str
    value: Any
    relationship_type: str | None = None
    from_label: str | None = None
    to_label: str | None = None
    segment_index: int | None = None


@dataclass
class EntityRef:
    label: str
    table: str


@dataclass
class RelationshipRef:
    type: str
    from_label: str
    to_label: str


@dataclass
class EntityLookupIntent:
    target_entity: EntityRef
    filters: list[FilterSpec]
    return_fields: list[dict[str, Any]]
    limit: int = 25


@dataclass
class OneHopIntent:
    source_entity: EntityRef
    target_entity: EntityRef
    relationship: RelationshipRef
    source_filters: list[FilterSpec]
    target_filters: list[FilterSpec]
    relationship_filters: list[FilterSpec]
    return_fields: list[dict[str, Any]]
    limit: int = 25


@dataclass
class TwoHopIntent:
    source_entity: EntityRef
    middle_entity: EntityRef
    target_entity: EntityRef
    path: list[RelationshipRef]
    source_filters: list[FilterSpec]
    middle_filters: list[FilterSpec]
    target_filters: list[FilterSpec]
    relationship_filters: list[FilterSpec]
    return_fields: list[dict[str, Any]]
    limit: int = 10


@dataclass
class RelationshipPropertyIntent:
    source_entity: EntityRef
    target_entity: EntityRef
    relationship: RelationshipRef
    requested_relationship_property: str | None
    source_filters: list[FilterSpec]
    target_filters: list[FilterSpec]
    relationship_filters: list[FilterSpec]
    limit: int = 25


@dataclass
class FixedMultiHopIntent:
    path_template_id: str
    entities: list[EntityRef]
    relationships: list[RelationshipRef]
    filters: list[FilterSpec]
    return_fields: list[dict[str, Any]]
    limit: int = 25


@dataclass
class ConstrainedMultiHopIntent:
    source_entity: EntityRef
    target_entity: EntityRef
    max_hops: int
    required_entities: list[EntityRef]
    allowed_relationship_types: list[str]
    filters: list[FilterSpec]
    return_fields: list[dict[str, Any]]
    limit: int = 10


@dataclass
class CandidateSet:
    query_type: str
    intent_contract_type: str | None = None
    candidate_anchor_label: str | None = None
    candidate_return_label: str | None = None
    candidate_answer_label: str | None = None
    candidate_target_label: str | None = None
    candidate_relationship_type: str | None = None
    candidate_relationship_property: str | None = None
    candidate_required_entity_labels: list[str] | None = None
    candidate_allowed_relationship_types: list[str] | None = None
    candidate_hop_limit: int | None = None


@dataclass
class BoundConstraints:
    clauses: list[str]
    params: dict[str, Any]
    note: str


@dataclass
class CypherBuildResult:
    query_type: str
    cypher: str
    explanation: str


@dataclass
class OutputField:
    scope: str
    property: str


@dataclass
class OutputShape:
    distinct: bool
    return_scopes: list[str]
    fields: list[OutputField]
    limit: int
    rationale: str = ""


def _debug_dict(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return {key: _debug_dict(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_debug_dict(item) for item in value]
    return value


def _make_entity_ref(label: str | None, query_schema: dict[str, Any]) -> EntityRef | None:
    node = _resolve_node(label, query_schema)
    if not node:
        return None
    return EntityRef(label=node["label"], table=node["table"])


def _make_relationship_ref(rel_type: str | None, query_schema: dict[str, Any]) -> RelationshipRef | None:
    rel = next((item for item in query_schema["relationships"] if item["type"] == rel_type), None)
    if not rel:
        return None
    return RelationshipRef(type=rel["type"], from_label=rel["from_label"], to_label=rel["to_label"])


def _filters_from_value(scope: str, prop: str | None, value: Any, *, operator: str = "=") -> list[FilterSpec]:
    if not prop or value in (None, "", []):
        return []
    return [FilterSpec(scope=scope, property=prop, operator=operator, value=value)]


def _filters_from_relationship_items(items: list[dict[str, Any]] | None) -> list[FilterSpec]:
    filters: list[FilterSpec] = []
    for item in items or []:
        prop = item.get("property")
        operator = item.get("operator", "=")
        value = item.get("value")
        if prop and value is not None:
            filters.append(
                FilterSpec(
                    scope="relationship",
                    property=str(prop),
                    operator=str(operator),
                    value=value,
                    relationship_type=str(item.get("relationship_type")) if item.get("relationship_type") else None,
                    from_label=str(item.get("from_label")) if item.get("from_label") else None,
                    to_label=str(item.get("to_label")) if item.get("to_label") else None,
                    segment_index=int(item.get("segment_index")) if item.get("segment_index") is not None else None,
                )
            )
    return filters


def _strip_code_fences(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def _normalize_full_ai_cypher(cypher: str) -> str:
    normalized = cypher
    normalized = re.sub(r"([=<>]\s*)'(-?\d+(?:\.\d+)?)'", r"\1\2", normalized)
    normalized = re.sub(r"(>=\s*)'(-?\d+(?:\.\d+)?)'", r"\1\2", normalized)
    normalized = re.sub(r"(<=\s*)'(-?\d+(?:\.\d+)?)'", r"\1\2", normalized)
    return normalized


def _extract_node_alias_label_map(cypher: str) -> dict[str, str]:
    alias_map: dict[str, str] = {}
    node_re = re.compile(r"\((?P<inner>[^)]*)\)")
    for match in node_re.finditer(cypher):
        inner = match.group("inner").strip()
        labeled = re.match(r"(?P<alias>[A-Za-z_][A-Za-z0-9_]*)\s*:\s*`?(?P<label>[A-Za-z0-9_]+)`?", inner)
        if labeled:
            alias_map[labeled.group("alias")] = labeled.group("label")
    return alias_map


def _parse_node_inner(inner: str, alias_map: dict[str, str]) -> dict[str, str | None]:
    inner = inner.strip()
    alias = None
    label = None
    labeled = re.match(r"(?P<alias>[A-Za-z_][A-Za-z0-9_]*)\s*:\s*`?(?P<label>[A-Za-z0-9_]+)`?", inner)
    if labeled:
        alias = labeled.group("alias")
        label = labeled.group("label")
    else:
        alias_only = re.match(r"(?P<alias>[A-Za-z_][A-Za-z0-9_]*)\b", inner)
        if alias_only:
            alias = alias_only.group("alias")
            label = alias_map.get(alias)
    return {"alias": alias, "label": label}


def _iter_full_ai_relationship_matches(cypher: str) -> list[re.Match[str]]:
    pattern_re = re.compile(
        r"\((?P<left>[^)]*)\)"
        r"(?P<left_ws>\s*)(?P<left_arrow><-|-)\[(?P<inside>[^\]]*)\](?P<right_arrow>->|-)(?P<right_ws>\s*)"
        r"\((?P<right>[^)]*)\)"
    )
    matches: list[re.Match[str]] = []
    start = 0
    while start < len(cypher):
        match = pattern_re.search(cypher, start)
        if not match:
            break
        matches.append(match)
        start = match.start() + 1
    return matches


def _extract_full_ai_relationship_patterns(cypher: str) -> list[dict[str, str]]:
    patterns: list[dict[str, str]] = []
    alias_map = _extract_node_alias_label_map(cypher)
    for match in _iter_full_ai_relationship_matches(cypher):
        rel_match = re.search(r":`?([A-Za-z0-9_]+)`?", match.group("inside") or "")
        if not rel_match:
            continue
        left_node = _parse_node_inner(match.group("left"), alias_map)
        right_node = _parse_node_inner(match.group("right"), alias_map)
        if not left_node["label"] or not right_node["label"]:
            continue
        left_arrow = match.group("left_arrow")
        right_arrow = match.group("right_arrow")
        if left_arrow == "-" and right_arrow == "->":
            direction = "forward"
        elif left_arrow == "<-" and right_arrow == "-":
            direction = "reverse"
        else:
            direction = "undirected"
        patterns.append(
            {
                "type": rel_match.group(1),
                "left_label": str(left_node["label"]),
                "right_label": str(right_node["label"]),
                "left_alias": str(left_node["alias"] or ""),
                "right_alias": str(right_node["alias"] or ""),
                "direction": direction,
            }
        )
    return patterns


def _rewrite_full_ai_relationship_directions(cypher: str, query_schema: dict[str, Any]) -> tuple[str, list[str]]:
    rewrites: list[str] = []
    alias_map = _extract_node_alias_label_map(cypher)

    def _same_label(actual: str, rel: dict[str, Any], side: str) -> bool:
        return actual in {str(rel[f"{side}_label"]), str(rel[f"{side}_neo4j_label"])}

    replacements: list[tuple[int, int, str]] = []
    for match in _iter_full_ai_relationship_matches(cypher):
        inside = match.group("inside") or ""
        rel_match = re.search(r":`?([A-Za-z0-9_]+)`?", inside)
        if not rel_match:
            continue
        rel_type = rel_match.group(1)
        rel_defs = [rel for rel in query_schema["relationships"] if rel["type"] == rel_type]
        if not rel_defs:
            continue
        left_node = _parse_node_inner(match.group("left"), alias_map)
        right_node = _parse_node_inner(match.group("right"), alias_map)
        left_label = left_node["label"]
        right_label = right_node["label"]
        if not left_label or not right_label:
            continue
        left_arrow = match.group("left_arrow")
        right_arrow = match.group("right_arrow")
        direction = "forward" if left_arrow == "-" and right_arrow == "->" else "reverse" if left_arrow == "<-" and right_arrow == "-" else "undirected"
        prev_non_ws = next((ch for ch in reversed(cypher[:match.start()]) if not ch.isspace()), "")
        next_non_ws = next((ch for ch in cypher[match.end():] if not ch.isspace()), "")
        chain_connected = prev_non_ws in {"-", "<", ">", "]"} or next_non_ws in {"-", "<", ">", "["}
        replacement = match.group(0)
        for rel in rel_defs:
            forward_match = _same_label(left_label, rel, "from") and _same_label(right_label, rel, "to")
            reverse_match = _same_label(left_label, rel, "to") and _same_label(right_label, rel, "from")
            if forward_match:
                canonical = f"({match.group('left')}){match.group('left_ws')}-[{inside}]->{match.group('right_ws')}({match.group('right')})"
                if canonical != match.group(0):
                    replacement = canonical
                    rewrites.append(f"{rel_type}: normalized to {rel['from_label']}->{rel['to_label']}")
                break
            if reverse_match:
                if chain_connected:
                    if direction == "forward":
                        replacement = (
                            f"({match.group('left')}){match.group('left_ws')}<-[{inside}]-{match.group('right_ws')}({match.group('right')})"
                        )
                    else:
                        replacement = match.group(0)
                    if replacement != match.group(0):
                        rewrites.append(f"{rel_type}: reversed arrow to preserve chain and match {rel['from_label']}->{rel['to_label']}")
                else:
                    replacement = (
                        f"({match.group('right')}){match.group('right_ws')}-[{inside}]->{match.group('left_ws')}({match.group('left')})"
                    )
                    rewrites.append(f"{rel_type}: reordered to {rel['from_label']}->{rel['to_label']}")
                break
        if replacement != match.group(0):
            replacements.append((match.start(), match.end(), replacement))

    rewritten = cypher
    for start, end, replacement in sorted(replacements, key=lambda item: item[0], reverse=True):
        rewritten = rewritten[:start] + replacement + rewritten[end:]
    return rewritten, rewrites


def _validate_full_ai_relationship_directions(cypher: str, query_schema: dict[str, Any]) -> str | None:
    patterns = _extract_full_ai_relationship_patterns(cypher)
    if not patterns:
        return None

    def _same_label(actual: str, rel: dict[str, Any], side: str) -> bool:
        return actual in {str(rel[f"{side}_label"]), str(rel[f"{side}_neo4j_label"])}

    for pattern in patterns:
        if pattern["direction"] == "undirected":
            continue
        rel_defs = [rel for rel in query_schema["relationships"] if rel["type"] == pattern["type"]]
        if not rel_defs:
            return f"AI generated unknown relationship type `{pattern['type']}` that is not present in the registry."
        for rel in rel_defs:
            forward_match = _same_label(pattern["left_label"], rel, "from") and _same_label(pattern["right_label"], rel, "to")
            reverse_match = _same_label(pattern["left_label"], rel, "to") and _same_label(pattern["right_label"], rel, "from")
            if pattern["direction"] == "forward" and forward_match:
                break
            if pattern["direction"] == "reverse" and reverse_match:
                break
            if pattern["direction"] == "forward" and reverse_match:
                return (
                    f"AI generated `{pattern['type']}` with the opposite direction. "
                    f"Registry expects {rel['from_label']} -> {rel['to_label']}."
                )
            if pattern["direction"] == "reverse" and forward_match:
                return (
                    f"AI generated `{pattern['type']}` with the opposite direction. "
                    f"Registry expects {rel['from_label']} -> {rel['to_label']}."
                )
    return None


def _detect_sql_like_syntax(cypher: str) -> str | None:
    normalized = " ".join(cypher.lower().split())
    sql_markers = [
        "select ",
        " join ",
        " having ",
        " group by ",
        " union ",
        " in (select ",
    ]
    for marker in sql_markers:
        if marker in normalized:
            return f"AI produced SQL-style syntax (`{marker.strip()}`) inside the Cypher query."
    return None


def _add_full_ai_context_alignment(cypher: str, query_schema: dict[str, Any]) -> tuple[str, list[str]]:
    patterns = _extract_full_ai_relationship_patterns(cypher)
    if len(patterns) < 2:
        return cypher, []
    rel_lookup = {rel["type"]: rel for rel in query_schema["relationships"]}
    contextual_props = {"semester", "year", "term", "quarter", "section", "session", "academic_year", "date", "start_date", "end_date"}
    added_clauses: list[str] = []
    for left, right in zip(patterns, patterns[1:]):
        if not left.get("right_alias") or left.get("right_alias") != right.get("left_alias"):
            continue
        left_rel = rel_lookup.get(left["type"])
        right_rel = rel_lookup.get(right["type"])
        if not left_rel or not right_rel:
            continue
        left_alias = _extract_relationship_alias(left["type"], cypher, left["left_alias"], left["right_alias"])
        right_alias = _extract_relationship_alias(right["type"], cypher, right["left_alias"], right["right_alias"])
        if not left_alias or not right_alias:
            continue
        shared_props = contextual_props.intersection(left_rel.get("properties", [])).intersection(right_rel.get("properties", []))
        for prop in sorted(shared_props):
            if prop in {"year", "academic_year"}:
                clause = f"toInteger({left_alias}.`{prop}`) = toInteger({right_alias}.`{prop}`)"
            else:
                clause = f"toLower(toString({left_alias}.`{prop}`)) = toLower(toString({right_alias}.`{prop}`))"
            if clause not in cypher:
                added_clauses.append(clause)
    if not added_clauses:
        return cypher, []
    upper = cypher.upper()
    return_idx = upper.rfind(" RETURN ")
    if return_idx == -1:
        return cypher, []
    where_idx = upper.find(" WHERE ")
    if where_idx != -1 and where_idx < return_idx:
        cypher = cypher[:return_idx] + " AND " + " AND ".join(added_clauses) + cypher[return_idx:]
    else:
        cypher = cypher[:return_idx] + " WHERE " + " AND ".join(added_clauses) + cypher[return_idx:]
    return cypher, added_clauses


def _extract_relationship_alias(rel_type: str, cypher: str, left_alias: str, right_alias: str) -> str | None:
    rel_re = re.compile(
        rf"\({re.escape(left_alias)}(?::[^\)]*)?\)\s*(?:<-|-)\[(?P<alias>[A-Za-z_][A-Za-z0-9_]*)?:`?{re.escape(rel_type)}`?[^\]]*\](?:->|-)\s*\({re.escape(right_alias)}(?::[^\)]*)?\)"
    )
    match = rel_re.search(cypher)
    if match and match.group("alias"):
        return match.group("alias")
    rel_re_rev = re.compile(
        rf"\({re.escape(right_alias)}(?::[^\)]*)?\)\s*(?:<-|-)\[(?P<alias>[A-Za-z_][A-Za-z0-9_]*)?:`?{re.escape(rel_type)}`?[^\]]*\](?:->|-)\s*\({re.escape(left_alias)}(?::[^\)]*)?\)"
    )
    match = rel_re_rev.search(cypher)
    if match and match.group("alias"):
        return match.group("alias")
    return None


def _infer_query_type_from_cypher(cypher: str, fallback: str) -> str:
    normalized = " ".join(cypher.lower().split())
    relationship_hops = len(re.findall(r"-\[[^\]]*\]-", normalized))
    if "*" in normalized and relationship_hops >= 1:
        return "constrained_multi_hop"
    if relationship_hops >= 3:
        return "fixed_multi_hop"
    if "match p=" in normalized or relationship_hops == 2:
        return "two_hop"
    if relationship_hops == 1:
        if "relationship_" in normalized:
            return "relationship_property"
        return "one_hop"
    if " where " in normalized:
        return "entity_lookup"
    if "match (" in normalized:
        return "entity_lookup"
    return fallback


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _pluralize(name: str) -> str:
    if name.endswith("y") and len(name) > 1 and name[-2] not in "aeiou":
        return f"{name[:-1]}ies"
    if name.endswith("s"):
        return name
    return f"{name}s"


def _humanize_rel(name: str) -> str:
    return name.replace("_", " ").lower()


def _neo4j_label_for_table(table: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z_]", "_", table)
    return cleaned or "X"


def _neo4j_label_for_node_label(label: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z_]", "_", label)
    return cleaned or "X"


def _relationship_synonyms(rel_type: str) -> set[str]:
    synonyms: dict[str, set[str]] = {
        "ENROLLED_IN": {"enroll", "enrolled", "enrolled in", "enrollment", "take", "takes", "took", "taken", "study", "studies"},
        "TEACHES": {"teach", "teaches", "taught", "taught by", "instruct", "instructs", "professor"},
        "OFFERED_BY": {"offer", "offers", "offered", "offered by", "belong", "belongs", "belongs to"},
        "WORKS_FOR": {"work", "works", "worked", "works for"},
        "REPORTS_TO": {"report", "reports", "reported"},
        "HAS": {"has", "have", "contains", "contain", "includes", "include", "belongs", "belong"},
        "CONTAINS": {"contains", "contain", "includes", "include", "belongs", "belong"},
        "BELONGS_TO": {"belongs", "belong", "belongs to", "part of", "owned by"},
    }
    return synonyms.get(rel_type, set())


def _relationship_display_aliases(rel: dict[str, Any]) -> set[str]:
    aliases: set[str] = set()
    display_text = _normalize(str(rel.get("display_text", "")))
    if display_text:
        aliases.add(display_text)
        from_label = _normalize(str(rel.get("from_label", "")))
        to_label = _normalize(str(rel.get("to_label", "")))
        phrase = display_text
        if from_label:
            phrase = re.sub(rf"^\b{re.escape(from_label)}\b\s*", "", phrase).strip()
        if to_label:
            phrase = re.sub(rf"\b{re.escape(to_label)}\b$", "", phrase).strip()
        if phrase:
            aliases.add(phrase)
    rel_name = _humanize_rel(str(rel.get("type", "")))
    if rel_name:
        aliases.add(_normalize(rel_name))
    return {alias for alias in aliases if alias}


def _extract_person_token(question: str) -> str | None:
    tokens = re.findall(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+|[A-Z][a-z]+)\b", question)
    stopwords = {"What", "Which", "Who", "Show", "Find", "List", "Professor", "Instructor", "Student", "Course", "Department", "Fall", "Spring", "Summer", "Winter"}
    candidates = [token for token in tokens if token.split()[0] not in stopwords]
    return candidates[0] if candidates else None


def _serialize_source_schema(source_schema: list[Any] | None, source_foreign_keys: list[Any] | None) -> dict[str, Any]:
    if not source_schema:
        return {"tables": [], "foreign_keys": []}

    fk_rows: list[dict[str, Any]] = []
    for fk in source_foreign_keys or []:
        fk_rows.append(
            {
                "source_table": getattr(fk, "source_table", None),
                "source_columns": list(getattr(fk, "source_columns", []) or []),
                "target_table": getattr(fk, "target_table", None),
                "target_columns": list(getattr(fk, "target_columns", []) or []),
            }
        )

    tables = []
    for table in source_schema:
        table_name = getattr(table, "name", "")
        table_fks = [row for row in fk_rows if row["source_table"] == table_name]
        tables.append(
            {
                "name": table_name,
                "columns": [
                    {
                        "name": getattr(col, "name", ""),
                        "data_type": getattr(col, "data_type", ""),
                        "not_null": bool(getattr(col, "not_null", False)),
                        "is_primary_key": bool(getattr(col, "is_primary_key", False)),
                    }
                    for col in getattr(table, "columns", []) or []
                ],
                "primary_keys": list(getattr(table, "primary_keys", []) or []),
                "indexes": [
                    {
                        "name": getattr(idx, "name", ""),
                        "unique": bool(getattr(idx, "unique", False)),
                        "columns": list(getattr(idx, "columns", []) or []),
                    }
                    for idx in getattr(table, "indexes", []) or []
                ],
                "foreign_keys": table_fks,
            }
        )

    return {"tables": tables, "foreign_keys": fk_rows}


def _serialize_graph_mapping_context(graph_mapping: dict[str, Any]) -> dict[str, Any]:
    return {
        "nodes": [
            {
                "table": node.get("table"),
                "label": node.get("label"),
                "primary_key": list(node.get("primary_key", []) or []),
                "properties": list(node.get("properties", []) or []),
                "classification": node.get("classification"),
            }
            for node in graph_mapping.get("nodes", []) or []
        ],
        "relationships": [
            {
                "from_table": rel.get("from_table"),
                "to_table": rel.get("to_table"),
                "type": rel.get("final_name") or rel.get("semantic_relationship_type"),
                "display_text": rel.get("semantic_display_text") or rel.get("display_text"),
                "properties": list(rel.get("properties", []) or []),
                "classification": rel.get("classification"),
                "via_table": rel.get("via_table"),
            }
            for rel in graph_mapping.get("relationships", []) or []
        ],
        "join_tables": [
            {
                "table": item.get("table"),
                "classification": item.get("classification"),
                "linked_tables": list(item.get("linked_tables", []) or []),
                "non_key_columns": list(item.get("non_key_columns", []) or []),
            }
            for item in graph_mapping.get("join_tables", []) or []
        ],
    }


def build_query_schema_registry(
    graph_mapping: dict[str, Any],
    *,
    source_schema: list[Any] | None = None,
    source_foreign_keys: list[Any] | None = None,
) -> dict[str, Any]:
    nodes = graph_mapping.get("nodes", [])
    relationships = graph_mapping.get("relationships", [])

    node_defs = []
    for node in nodes:
        label = str(node["label"])
        table = str(node["table"])
        aliases = {
            _normalize(label),
            _normalize(table),
            _normalize(_pluralize(label)),
            _normalize(_pluralize(table)),
        }
        payload = {
            "label": label,
            "table": table,
            "neo4j_label": _neo4j_label_for_node_label(label),
            "primary_key": list(node.get("primary_key", [])),
            "properties": list(node.get("properties", [])),
            "aliases": sorted(alias for alias in aliases if alias),
        }
        node_defs.append(payload)

    rel_defs = []
    for rel in relationships:
        rel_type = str(rel.get("final_name"))
        payload = {
            "type": rel_type,
            "from_table": str(rel.get("from_table")),
            "to_table": str(rel.get("to_table")),
            "from_label": next((n["label"] for n in node_defs if n["table"] == rel.get("from_table")), str(rel.get("from_table"))),
            "to_label": next((n["label"] for n in node_defs if n["table"] == rel.get("to_table")), str(rel.get("to_table"))),
            "from_neo4j_label": next((n["neo4j_label"] for n in node_defs if n["table"] == rel.get("from_table")), _neo4j_label_for_table(str(rel.get("from_table")))),
            "to_neo4j_label": next((n["neo4j_label"] for n in node_defs if n["table"] == rel.get("to_table")), _neo4j_label_for_table(str(rel.get("to_table")))),
            "properties": list(rel.get("properties", [])),
            "classification": rel.get("classification"),
            "display_text": str(rel.get("semantic_display_text") or rel.get("display_text") or ""),
            "aliases": [],
        }
        payload["aliases"] = sorted(
            {
                _normalize(rel_type),
                _normalize(_humanize_rel(rel_type)),
                *{_normalize(item) for item in _relationship_synonyms(rel_type)},
                *_relationship_display_aliases(payload),
            }
        )
        rel_defs.append(payload)

    return {
        "registry_type": "query_schema_registry",
        "registry_version": 1,
        "source_schema_context": _serialize_source_schema(source_schema, source_foreign_keys),
        "graph_mapping_context": _serialize_graph_mapping_context(graph_mapping),
        "nodes": node_defs,
        "relationships": rel_defs,
    }


def _materialize_query_schema(registry: dict[str, Any]) -> dict[str, Any]:
    node_defs = list(registry.get("nodes", []))
    rel_defs = list(registry.get("relationships", []))
    node_lookup: dict[str, dict[str, Any]] = {}
    for node in node_defs:
        for alias in node.get("aliases", []):
            if alias:
                node_lookup[str(alias)] = node
    rel_lookup: dict[str, dict[str, Any]] = {}
    for rel in rel_defs:
        for alias in rel.get("aliases", []):
            if alias:
                rel_lookup[str(alias)] = rel
    return {
        "registry_type": registry.get("registry_type", "query_schema_registry"),
        "registry_version": registry.get("registry_version", 1),
        "source_schema_context": registry.get("source_schema_context", {"tables": [], "foreign_keys": []}),
        "graph_mapping_context": registry.get("graph_mapping_context", {"nodes": [], "relationships": [], "join_tables": []}),
        "nodes": node_defs,
        "relationships": rel_defs,
        "node_lookup": node_lookup,
        "relationship_lookup": rel_lookup,
        "node_labels": sorted({n["label"] for n in node_defs}),
        "relationship_types": sorted({r["type"] for r in rel_defs}),
    }


def build_query_schema(
    graph_mapping_or_registry: dict[str, Any],
    *,
    source_schema: list[Any] | None = None,
    source_foreign_keys: list[Any] | None = None,
) -> dict[str, Any]:
    if graph_mapping_or_registry.get("registry_type") == "query_schema_registry":
        return _materialize_query_schema(graph_mapping_or_registry)
    registry = build_query_schema_registry(
        graph_mapping_or_registry,
        source_schema=source_schema,
        source_foreign_keys=source_foreign_keys,
    )
    return _materialize_query_schema(registry)


def _quoted_values(question: str) -> list[str]:
    values: list[str] = []
    for pattern in [r'"([^"]+)"', r"'([^']+)'"]:
        values.extend(match.strip() for match in re.findall(pattern, question))
    return [value for value in values if value]


def _match_nodes_with_positions(question: str, query_schema: dict[str, Any]) -> list[dict[str, Any]]:
    normalized = _normalize(question)
    matches: dict[str, tuple[int, dict[str, Any]]] = {}
    for alias, node in query_schema["node_lookup"].items():
        if not alias:
            continue
        match = re.search(rf"\b{re.escape(alias)}\b", normalized)
        if not match:
            continue
        current = matches.get(node["table"])
        if current is None or match.start() < current[0]:
            matches[node["table"]] = (match.start(), node)
    return [item[1] for item in sorted(matches.values(), key=lambda item: item[0])]


def _match_relationships_with_positions(question: str, query_schema: dict[str, Any]) -> list[dict[str, Any]]:
    normalized = _normalize(question)
    matches: dict[str, tuple[int, dict[str, Any]]] = {}
    for alias, rel in query_schema["relationship_lookup"].items():
        if not alias:
            continue
        match = re.search(rf"\b{re.escape(alias)}\b", normalized)
        if not match:
            continue
        current = matches.get(rel["type"])
        if current is None or match.start() < current[0]:
            matches[rel["type"]] = (match.start(), rel)
    return [item[1] for item in sorted(matches.values(), key=lambda item: item[0])]


def _extract_hop_limit(question: str) -> int | None:
    normalized = _normalize(question)
    for pattern in [
        r"\bwithin\s+(\d+)\s+hops?\b",
        r"\bup to\s+(\d+)\s+hops?\b",
        r"\bmax(?:imum)?\s+(\d+)\s+hops?\b",
        r"\b(\d+)\s+hops?\b",
    ]:
        match = re.search(pattern, normalized)
        if match:
            return int(match.group(1))
    return None


def _extract_years(question: str) -> list[int]:
    return [int(value) for value in re.findall(r"\b(?:19|20)\d{2}\b", question)]


def _extract_semesters(question: str) -> list[str]:
    normalized = _normalize(question)
    season_map = {
        "spring": "Spring",
        "summer": "Summer",
        "fall": "Fall",
        "autumn": "Fall",
        "winter": "Winter",
    }
    found: list[str] = []
    for token, canonical in season_map.items():
        if re.search(rf"\b{token}\b", normalized):
            found.append(canonical)
    return found


def _preprocess_question(question: str, query_schema: dict[str, Any]) -> PreprocessedQuestion:
    return PreprocessedQuestion(
        raw_question=question,
        normalized_question=_normalize(question),
        quoted_values=_quoted_values(question),
        person_token=_extract_person_token(question),
        year_values=_extract_years(question),
        semester_values=_extract_semesters(question),
        hop_limit=_extract_hop_limit(question),
        node_mentions=_match_nodes_with_positions(question, query_schema),
        relationship_mentions=_match_relationships_with_positions(question, query_schema),
    )


def _legacy_classify_query_type(pre: PreprocessedQuestion, query_schema: dict[str, Any]) -> QueryTypeDecision:
    from backend.graph_query_legacy import classify_query_type

    return classify_query_type(pre, query_schema)


def _person_like_node(node: dict[str, Any]) -> bool:
    props = set(_preferred_node_properties(node))
    return "first_name" in props or "last_name" in props


def _legacy_extract_intent(pre: PreprocessedQuestion, decision: QueryTypeDecision, query_schema: dict[str, Any]) -> ExtractedIntent:
    from backend.graph_query_legacy import extract_intent

    return extract_intent(pre, decision, query_schema)


def _resolve_node(label: str | None, query_schema: dict[str, Any]) -> dict[str, Any] | None:
    if not label:
        return None
    return next((node for node in query_schema["nodes"] if node["label"] == label), None)


def _graph_edges(query_schema: dict[str, Any]) -> list[dict[str, Any]]:
    edges: list[dict[str, Any]] = []
    for rel in query_schema["relationships"]:
        edges.append({"from_label": rel["from_label"], "to_label": rel["to_label"], "relationship": rel, "reversed": False})
        edges.append({"from_label": rel["to_label"], "to_label": rel["from_label"], "relationship": rel, "reversed": True})
    return edges


def _find_paths(
    source_label: str,
    target_label: str,
    query_schema: dict[str, Any],
    *,
    min_hops: int,
    max_hops: int,
    required_labels: set[str] | None = None,
    allowed_relationship_types: set[str] | None = None,
) -> list[list[dict[str, Any]]]:
    results: list[list[dict[str, Any]]] = []
    edges = _graph_edges(query_schema)

    def dfs(current: str, path: list[dict[str, Any]], visited: list[str]) -> None:
        hops = len(path)
        if hops > max_hops:
            return
        if hops >= min_hops and current == target_label:
            node_labels = {source_label, target_label, *[step["to_label"] for step in path]}
            if required_labels and not required_labels.issubset(node_labels):
                return
            results.append(list(path))
            return
        if hops == max_hops:
            return
        for edge in edges:
            if edge["from_label"] != current:
                continue
            rel_type = edge["relationship"]["type"]
            if allowed_relationship_types and rel_type not in allowed_relationship_types:
                continue
            next_label = edge["to_label"]
            if next_label in visited and next_label != target_label:
                continue
            dfs(next_label, path + [edge], visited + [next_label])

    dfs(source_label, [], [source_label])
    return results


def _resolve_path(intent: ExtractedIntent, query_schema: dict[str, Any]) -> ResolvedPath | None:
    if intent.query_type == "entity_lookup":
        answer_node = _resolve_node(intent.answer_label, query_schema)
        if not answer_node:
            return None
        return ResolvedPath(
            query_type=intent.query_type,
            node_sequence=[answer_node],
            relationship_sequence=[],
            answer_node=answer_node,
            target_node=None,
            answer_property=intent.answer_property,
            answer_value=intent.answer_value,
        )

    if intent.query_type in {"one_hop", "relationship_property"}:
        rel = next((item for item in query_schema["relationships"] if item["type"] == intent.relationship_type), None)
        if rel is None:
            if intent.answer_label and intent.target_label:
                rel = _relationship_between_nodes(
                    _resolve_node(intent.answer_label, query_schema) or {},
                    _resolve_node(intent.target_label, query_schema) or {},
                    query_schema,
                )
        if not rel:
            return None
        answer_node = _resolve_node(intent.answer_label or rel["from_label"], query_schema)
        target_node = _resolve_node(intent.target_label or rel["to_label"], query_schema)
        if not answer_node or not target_node:
            return None
        relationship_step = {
            "from_label": answer_node["label"],
            "to_label": target_node["label"],
            "relationship": rel,
            "reversed": not (rel["from_label"] == answer_node["label"] and rel["to_label"] == target_node["label"]),
        }
        return ResolvedPath(
            query_type=intent.query_type,
            node_sequence=[answer_node, target_node],
            relationship_sequence=[relationship_step],
            answer_node=answer_node,
            target_node=target_node,
            anchor_node=answer_node if intent.query_type == "one_hop" else None,
            return_node=target_node if intent.query_type == "one_hop" else None,
            requested_relationship_property=intent.requested_relationship_property,
            anchor_value=intent.anchor_value if intent.query_type == "one_hop" else None,
            return_value=intent.return_value if intent.query_type == "one_hop" else None,
            answer_value=intent.answer_value,
            target_value=intent.target_value,
            relationship_filters=intent.relationship_filters or [],
        )

    if intent.query_type == "two_hop":
        labels = intent.explicit_node_sequence or []
        if len(labels) >= 3:
            source_label, middle_label, target_label = labels[0], labels[1], labels[-1]
            first = next((edge for edge in _graph_edges(query_schema) if edge["from_label"] == source_label and edge["to_label"] == middle_label), None)
            second = next((edge for edge in _graph_edges(query_schema) if edge["from_label"] == middle_label and edge["to_label"] == target_label), None)
            path = [edge for edge in [first, second] if edge]
            if len(path) == 2:
                return ResolvedPath(
                    query_type=intent.query_type,
                    node_sequence=[_resolve_node(source_label, query_schema), _resolve_node(middle_label, query_schema), _resolve_node(target_label, query_schema)],
                    relationship_sequence=path,
                    answer_node=_resolve_node(source_label, query_schema),
                    target_node=_resolve_node(target_label, query_schema),
                    relationship_filters=intent.relationship_filters or [],
                )
        if len(labels) >= 2:
            paths = _find_paths(labels[0], labels[-1], query_schema, min_hops=2, max_hops=2)
            if paths:
                node_sequence = [_resolve_node(labels[0], query_schema)]
                for step in paths[0]:
                    node_sequence.append(_resolve_node(step["to_label"], query_schema))
                return ResolvedPath(
                    query_type=intent.query_type,
                    node_sequence=node_sequence,
                    relationship_sequence=paths[0],
                    answer_node=node_sequence[0],
                    target_node=node_sequence[-1],
                    relationship_filters=intent.relationship_filters or [],
                )
        return None

    if intent.query_type == "fixed_multi_hop":
        labels = intent.explicit_node_sequence or []
        if len(labels) < 3:
            return None
        path: list[dict[str, Any]] = []
        for left, right in zip(labels, labels[1:]):
            edge = next((item for item in _graph_edges(query_schema) if item["from_label"] == left and item["to_label"] == right), None)
            if edge is None:
                return None
            path.append(edge)
        node_sequence = [_resolve_node(label, query_schema) for label in labels]
        if any(node is None for node in node_sequence):
            return None
        return ResolvedPath(
            query_type=intent.query_type,
            node_sequence=[node for node in node_sequence if node],
            relationship_sequence=path,
            answer_node=node_sequence[0],
            target_node=node_sequence[-1],
            relationship_filters=intent.relationship_filters or [],
        )

    if intent.query_type == "constrained_multi_hop":
        labels = intent.explicit_node_sequence or []
        if len(labels) < 2:
            return None
        source_label, target_label = labels[0], labels[-1]
        paths = _find_paths(
            source_label,
            target_label,
            query_schema,
            min_hops=2,
            max_hops=intent.hop_limit or 4,
            required_labels=set(intent.required_entity_labels or []),
            allowed_relationship_types=set(intent.allowed_relationship_types or []) or None,
        )
        if not paths:
            return None
        best = paths[0]
        node_sequence = [_resolve_node(source_label, query_schema)]
        for step in best:
            node_sequence.append(_resolve_node(step["to_label"], query_schema))
        if any(node is None for node in node_sequence):
            return None
        return ResolvedPath(
            query_type=intent.query_type,
            node_sequence=[node for node in node_sequence if node],
            relationship_sequence=best,
            answer_node=node_sequence[0],
            target_node=node_sequence[-1],
            relationship_filters=intent.relationship_filters or [],
        )

    return None


def _property_clause(alias: str, prop: str, value: Any, *, key: str, exact: bool = False) -> tuple[str, dict[str, Any]]:
    if isinstance(value, int):
        return f"toInteger({alias}.`{prop}`) = ${key}", {key: value}
    if exact or _is_email_like(str(value)):
        return f"toLower(toString({alias}.`{prop}`)) = toLower(${key})", {key: value}
    return f"toLower(toString({alias}.`{prop}`)) CONTAINS toLower(${key})", {key: value}


def _bind_constraints(resolved: ResolvedPath, query_schema: dict[str, Any]) -> tuple[list[str], dict[str, Any], str]:
    clauses: list[str] = []
    params: dict[str, Any] = {}
    notes: list[str] = []

    if resolved.query_type == "entity_lookup" and resolved.answer_node and resolved.answer_value:
        prop = resolved.answer_property or _preferred_text_filter_property(resolved.answer_node)
        if prop:
            clause, bound = _property_clause("n", prop, resolved.answer_value, key="answer_value")
            clauses.append(clause)
            params.update(bound)
            notes.append(f"{resolved.answer_node['label']}.{prop}")
        return clauses, params, ", ".join(notes)

    if resolved.answer_node and resolved.answer_value:
        name_parts = _split_name(str(resolved.answer_value)) if _person_like_node(resolved.answer_node) else None
        if name_parts and {"first_name", "last_name"}.issubset(set(_preferred_node_properties(resolved.answer_node))):
            clauses.append("toLower(toString(n0.`first_name`)) = toLower($source_first_name)")
            clauses.append("toLower(toString(n0.`last_name`)) = toLower($source_last_name)")
            params["source_first_name"] = name_parts[0]
            params["source_last_name"] = name_parts[1]
            notes.append(f"{resolved.answer_node['label']}.first_name/last_name")
        else:
            prop = "first_name" if _person_like_node(resolved.answer_node) and not _is_email_like(resolved.answer_value) else _preferred_text_filter_property(resolved.answer_node)
            if prop:
                clause, bound = _property_clause("n0", prop, resolved.answer_value, key="source_value", exact=prop in {"first_name", "last_name", "email"})
                clauses.append(clause)
                params.update(bound)
                notes.append(f"{resolved.answer_node['label']}.{prop}")

    if resolved.target_node and resolved.target_value and len(resolved.node_sequence) >= 2:
        target_alias = f"n{len(resolved.node_sequence) - 1}"
        prop = _preferred_text_filter_property(resolved.target_node)
        if prop:
            clause, bound = _property_clause(target_alias, prop, resolved.target_value, key="target_value")
            clauses.append(clause)
            params.update(bound)
            notes.append(f"{resolved.target_node['label']}.{prop}")

    for filter_idx, item in enumerate(resolved.relationship_filters or []):
        matching_indices = _matching_relationship_indices(item, resolved)
        if len(matching_indices) != 1:
            continue
        idx = matching_indices[0]
        rel = resolved.relationship_sequence[idx]["relationship"]
        rel_clauses, rel_params, rel_notes = _build_relationship_filter_clauses([item], rel)
        renamed: dict[str, Any] = {}
        rename_map = {key: f"path_{idx}_{key}_{filter_idx}" for key in rel_params}
        for clause in rel_clauses:
            updated_clause = clause.replace("r.", f"r{idx}.")
            for old_key, new_key in rename_map.items():
                updated_clause = updated_clause.replace(f"${old_key}", f"${new_key}")
            clauses.append(updated_clause)
        for old_key, new_key in rename_map.items():
            renamed[new_key] = rel_params[old_key]
        params.update(renamed)
        notes.extend(rel_notes)

    # For multi-hop edge traversals, align shared contextual relationship properties
    # across adjacent segments so we match the same course offering / term / section.
    if resolved.query_type in {"two_hop", "fixed_multi_hop", "constrained_multi_hop"} and len(resolved.relationship_sequence) >= 2:
        contextual_props = {
            "semester",
            "year",
            "term",
            "quarter",
            "section",
            "session",
            "academic_year",
            "start_date",
            "end_date",
            "date",
        }
        for left_idx, right_idx in zip(range(len(resolved.relationship_sequence) - 1), range(1, len(resolved.relationship_sequence))):
            left_rel = resolved.relationship_sequence[left_idx]["relationship"]
            right_rel = resolved.relationship_sequence[right_idx]["relationship"]
            shared_props = contextual_props.intersection(left_rel.get("properties", [])).intersection(right_rel.get("properties", []))
            for prop in sorted(shared_props):
                if prop in {"year", "academic_year"}:
                    clause = f"toInteger(r{left_idx}.`{prop}`) = toInteger(r{right_idx}.`{prop}`)"
                else:
                    clause = f"toLower(toString(r{left_idx}.`{prop}`)) = toLower(toString(r{right_idx}.`{prop}`))"
                clauses.append(clause)
                notes.append(f"align r{left_idx}.{prop}=r{right_idx}.{prop}")

    return clauses, params, ", ".join(notes)


def _build_chain_match(resolved: ResolvedPath) -> str:
    pattern = f"(n0:`{resolved.node_sequence[0]['neo4j_label']}`)"
    for index, step in enumerate(resolved.relationship_sequence):
        next_node = resolved.node_sequence[index + 1]
        rel = step["relationship"]
        if step["reversed"]:
            pattern += f"<-[r{index}:`{rel['type']}`]-(n{index + 1}:`{next_node['neo4j_label']}`)"
        else:
            pattern += f"-[r{index}:`{rel['type']}`]->(n{index + 1}:`{next_node['neo4j_label']}`)"
    return pattern


def _build_standard_plan(resolved: ResolvedPath, clauses: list[str], params: dict[str, Any], note: str, output_shape: OutputShape) -> QueryPlan:
    if resolved.query_type == "entity_lookup":
        node = resolved.answer_node
        cypher = f"MATCH (n:`{node['neo4j_label']}`)"
        if clauses:
            cypher += f" WHERE {' AND '.join(clauses)}"
        cypher += f" RETURN {'DISTINCT ' if output_shape.distinct else ''}{_node_return_clause(node, 'n', node['label'].lower())} LIMIT {output_shape.limit}"
        return QueryPlan(
            query_type="entity_lookup",
            cypher=cypher,
            params=params,
            explanation=f"Registry-grounded entity lookup{f' with {note}' if note else ''}.",
            planner="registry_pipeline",
        )

    chain = _build_chain_match(resolved)
    cypher = f"MATCH {chain}"
    if clauses:
        cypher += f" WHERE {' AND '.join(clauses)}"

    if resolved.query_type == "relationship_property":
        cypher += _build_return_clause_from_output_shape(resolved, output_shape)
        return QueryPlan(
            query_type="relationship_property",
            cypher=cypher,
            params=params,
            explanation=f"Registry-grounded relationship property lookup{f' with {note}' if note else ''}.",
            planner="registry_pipeline",
        )

    if resolved.query_type == "one_hop":
        cypher += _build_return_clause_from_output_shape(resolved, output_shape)
    elif resolved.query_type == "two_hop":
        cypher += _build_return_clause_from_output_shape(resolved, output_shape)
    else:
        cypher += _build_return_clause_from_output_shape(resolved, output_shape)

    return QueryPlan(
        query_type=resolved.query_type,
        cypher=cypher,
        params=params,
        explanation=f"Registry-grounded {resolved.query_type.replace('_', ' ')} query{f' with {note}' if note else ''}.",
        planner="registry_pipeline",
    )


def _build_candidate_set(intent: ExtractedIntent) -> CandidateSet:
    return CandidateSet(
        query_type=intent.query_type,
        intent_contract_type=type(intent.intent_contract).__name__ if intent.intent_contract is not None else None,
        candidate_anchor_label=intent.anchor_label,
        candidate_return_label=intent.return_label,
        candidate_answer_label=intent.answer_label,
        candidate_target_label=intent.target_label,
        candidate_relationship_type=intent.relationship_type,
        candidate_relationship_property=intent.requested_relationship_property,
        candidate_required_entity_labels=intent.required_entity_labels or [],
        candidate_allowed_relationship_types=intent.allowed_relationship_types or [],
        candidate_hop_limit=intent.hop_limit,
    )


def _build_cypher_result(plan: QueryPlan) -> CypherBuildResult:
    return CypherBuildResult(
        query_type=plan.query_type,
        cypher=plan.cypher,
        explanation=plan.explanation,
    )


def _legacy_run_registry_pipeline(question: str, query_schema: dict[str, Any]) -> QueryPlan:
    from backend.graph_query_legacy import run_registry_pipeline

    return run_registry_pipeline(question, query_schema)


def _extract_quoted_value(question: str) -> str | None:
    for pattern in [r'"([^"]+)"', r"'([^']+)'"]:
        match = re.search(pattern, question)
        if match:
            return match.group(1).strip()
    return None


def _match_nodes(question: str, query_schema: dict[str, Any]) -> list[dict[str, Any]]:
    normalized = _normalize(question)
    matches = []
    for alias, node in query_schema["node_lookup"].items():
        if alias and re.search(rf"\b{re.escape(alias)}\b", normalized):
            matches.append(node)
    deduped: list[dict[str, Any]] = []
    seen = set()
    for node in matches:
        if node["table"] not in seen:
            deduped.append(node)
            seen.add(node["table"])
    return deduped


def _match_relationships(question: str, query_schema: dict[str, Any]) -> list[dict[str, Any]]:
    normalized = _normalize(question)
    matches = []
    for alias, rel in query_schema["relationship_lookup"].items():
        if alias and re.search(rf"\b{re.escape(alias)}\b", normalized):
            matches.append(rel)
    deduped: list[dict[str, Any]] = []
    seen = set()
    for rel in matches:
        if rel["type"] not in seen:
            deduped.append(rel)
            seen.add(rel["type"])
    return deduped


def _detect_property(question: str, candidates: list[str]) -> str | None:
    normalized = _normalize(question)
    for candidate in candidates:
        alias = _normalize(candidate)
        if alias and re.search(rf"\b{re.escape(alias)}\b", normalized):
            return candidate
    return None


def _is_email_like(value: str | None) -> bool:
    return bool(value and "@" in value and "." in value)


def _split_name(value: str | None) -> tuple[str, str] | None:
    if not value:
        return None
    parts = [part for part in value.strip().split() if part]
    if len(parts) < 2:
        return None
    return parts[0], " ".join(parts[1:])


def _preferred_node_properties(node: dict[str, Any]) -> list[str]:
    props = list(node.get("properties", []))
    primary_keys = list(node.get("primary_key", []))
    return props + primary_keys


def _display_properties(node: dict[str, Any]) -> list[str]:
    preferred_order = [
        "name",
        "title",
        "course_name",
        "department_name",
        "first_name",
        "last_name",
        "email",
        "course_code",
    ]
    props = _preferred_node_properties(node)
    ordered = [prop for prop in preferred_order if prop in props]
    remaining = [prop for prop in props if prop not in ordered]
    return (ordered + remaining)[:4]


def _node_return_clause(node: dict[str, Any], alias: str, prefix: str) -> str:
    cols: list[str] = []
    for prop in _display_properties(node):
        cols.append(f"{alias}.`{prop}` AS `{prefix}_{prop}`")
    if node.get("primary_key"):
        pk = str(node["primary_key"][0])
        if pk not in _display_properties(node):
            cols.insert(0, f"{alias}.`{pk}` AS `{prefix}_{pk}`")
    return ", ".join(cols) if cols else f"{alias} AS `{prefix}`"


def _node_field_clause(node: dict[str, Any], alias: str, prefix: str, properties: list[str]) -> str:
    cols: list[str] = []
    for prop in properties:
        if prop in _preferred_node_properties(node):
            cols.append(f"{alias}.`{prop}` AS `{prefix}_{prop}`")
    return ", ".join(cols) if cols else _node_return_clause(node, alias, prefix)


def _relationship_return_clause(rel: dict[str, Any], alias: str = "r") -> str:
    cols = [f"type({alias}) AS relationship_type"]
    for prop in list(rel.get("properties", []))[:4]:
        cols.append(f"{alias}.`{prop}` AS `relationship_{prop}`")
    return ", ".join(cols)


def _relationship_property_value_clause(prop: str, alias: str = "r") -> str:
    return f"{alias}.`{prop}` AS `relationship_{prop}`"


def _scope_alias(scope: str, resolved: ResolvedPath) -> tuple[str, dict[str, Any] | None]:
    if scope == "source":
        return "n0", resolved.node_sequence[0] if resolved.node_sequence else None
    if scope == "target":
        return f"n{len(resolved.node_sequence)-1}", resolved.node_sequence[-1] if resolved.node_sequence else None
    if scope == "middle" and len(resolved.node_sequence) > 2:
        return "n1", resolved.node_sequence[1]
    if scope == "relationship" and resolved.relationship_sequence:
        return "r0", resolved.relationship_sequence[0]["relationship"]
    return "", None


def _default_output_shape(resolved: ResolvedPath) -> OutputShape:
    if resolved.query_type == "entity_lookup":
        return OutputShape(distinct=False, return_scopes=["source"], fields=[], limit=25, rationale="Default entity lookup output.")
    if resolved.query_type == "relationship_property":
        return OutputShape(
            distinct=False,
            return_scopes=["source", "relationship", "target"],
            fields=([OutputField(scope="relationship", property=resolved.requested_relationship_property)] if resolved.requested_relationship_property else []),
            limit=25,
            rationale="Default relationship-property output.",
        )
    if resolved.query_type == "one_hop":
        if resolved.anchor_node and resolved.return_node and resolved.anchor_value and not resolved.return_value:
            return OutputShape(distinct=True, return_scopes=["target"], fields=[], limit=25, rationale="One-hop answer should focus on the return entity.")
        return OutputShape(distinct=False, return_scopes=["source", "relationship", "target"], fields=[], limit=25, rationale="Fallback one-hop output.")
    if resolved.query_type == "two_hop":
        return OutputShape(distinct=True, return_scopes=["target"], fields=[], limit=10, rationale="Two-hop answer should focus on destination entities.")
    return OutputShape(distinct=True, return_scopes=["target"], fields=[], limit=10, rationale="Multi-hop answer should focus on destination entities.")


def _build_return_clause_from_output_shape(resolved: ResolvedPath, output_shape: OutputShape) -> str:
    columns: list[str] = []
    for scope in output_shape.return_scopes:
        alias, ref = _scope_alias(scope, resolved)
        scoped_fields = [field.property for field in output_shape.fields if field.scope == scope]
        if scope == "relationship":
            if scoped_fields:
                for prop in scoped_fields:
                    columns.append(_relationship_property_value_clause(prop, alias))
            elif resolved.relationship_sequence:
                columns.append(_relationship_return_clause(resolved.relationship_sequence[0]["relationship"], alias))
            continue
        node = ref if isinstance(ref, dict) else None
        if not alias or not node:
            continue
        prefix = node["label"].lower()
        if scoped_fields:
            columns.append(_node_field_clause(node, alias, prefix, scoped_fields))
        else:
            columns.append(_node_return_clause(node, alias, prefix))
    distinct_prefix = "DISTINCT " if output_shape.distinct else ""
    return f" RETURN {distinct_prefix}{', '.join(columns)} LIMIT {output_shape.limit}"


def _ai_select_output_shape(
    question: str,
    query_schema: dict[str, Any],
    intent: ExtractedIntent,
    resolved: ResolvedPath,
    ai_settings: AISettings,
) -> OutputShape | None:
    try:
        client = AIClient(api_key=ai_settings.api_key, base_url=ai_settings.base_url, model=ai_settings.model)
        payload = {
            "task": "Choose a read-friendly output shape for a graph query. Return strict JSON only. Do not generate Cypher.",
            "question": question,
            "query_type": intent.query_type,
            "intent_contract": _debug_dict(intent.intent_contract),
            "resolved_path": {
                "nodes": [{"label": node["label"], "properties": node.get("properties", [])} for node in resolved.node_sequence],
                "relationships": [
                    {"type": step["relationship"]["type"], "properties": step["relationship"].get("properties", [])}
                    for step in resolved.relationship_sequence
                ],
                "anchor_label": intent.anchor_label,
                "return_label": intent.return_label,
            },
            "rules": [
                "Return strict JSON only.",
                "Choose whether DISTINCT is needed.",
                "Choose which scopes to return from: source, target, middle, relationship.",
                "Prefer concise, readable tabular outputs.",
                "When the question asks for the answer entity, prefer only that entity's fields.",
                "Only choose properties that exist on the selected node or relationship.",
            ],
            "response_contract": {
                "distinct": "boolean",
                "return_scopes": ["source|target|middle|relationship"],
                "fields": [{"scope": "source|target|middle|relationship", "property": "existing property"}],
                "limit": "integer",
                "rationale": "short explanation",
            },
        }
        _, parsed = client.complete_json(payload)
        if not isinstance(parsed, dict):
            return None
        scopes = [scope for scope in parsed.get("return_scopes", []) if scope in {"source", "target", "middle", "relationship"}]
        if not scopes:
            return None
        fields: list[OutputField] = []
        for item in parsed.get("fields", []) or []:
            scope = item.get("scope")
            prop = item.get("property")
            if scope in {"source", "target", "middle", "relationship"} and prop:
                fields.append(OutputField(scope=str(scope), property=str(prop)))
        limit = parsed.get("limit", 25)
        try:
            limit = int(limit)
        except Exception:
            limit = 25
        return OutputShape(
            distinct=bool(parsed.get("distinct")),
            return_scopes=scopes,
            fields=fields,
            limit=max(1, min(limit, 100)),
            rationale=str(parsed.get("rationale") or ""),
        )
    except Exception:
        return None


def _preferred_text_filter_property(node: dict[str, Any]) -> str | None:
    preferred = ["course_name", "department_name", "name", "title", "email", "first_name", "last_name"]
    props = _preferred_node_properties(node)
    for prop in preferred:
        if prop in props:
            return prop
    return props[0] if props else None


def _requested_relationship_property(question: str, rel: dict[str, Any]) -> str | None:
    normalized = _normalize(question)
    if not any(normalized.startswith(prefix) for prefix in ("what ", "which ", "who ")):
        return None
    return _detect_property(question, list(rel.get("properties", [])))


def _extract_trailing_target_phrase(question: str) -> str | None:
    for pattern in [
        r"\b(?:in|for|on|at|from)\s+(.+?)\s*\??$",
        r"\b(?:about|regarding)\s+(.+?)\s*\??$",
    ]:
        match = re.search(pattern, question, re.IGNORECASE)
        if not match:
            continue
        phrase = match.group(1).strip().strip("'\" ")
        normalized = _normalize(phrase)
        if not phrase or not normalized:
            continue
        if re.fullmatch(r"(spring|summer|fall|autumn|winter)( (19|20)\d{2})?", normalized):
            continue
        return phrase
    return None


def _strip_relation_tail(text: str) -> str:
    cleaned = text.strip().strip("?").strip()
    cleaned = re.sub(
        r"\s+in\s+(spring|summer|fall|autumn|winter)(\s+((?:19|20)\d{2}))?\s*$",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\s+in\s+((?:19|20)\d{2})\s*$", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^(the|a|an)\s+", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip().strip("?").strip()


def _extract_relation_object_phrase(question: str, rel: dict[str, Any]) -> str | None:
    patterns = sorted(
        {
            *_relationship_synonyms(rel["type"]),
            *_relationship_display_aliases(rel),
            _humanize_rel(rel["type"]),
        },
        key=len,
        reverse=True,
    )
    for token in patterns:
        if not token:
            continue
        match = re.search(rf"\b{re.escape(token)}\b\s+(.+)$", question, re.IGNORECASE)
        if not match:
            continue
        phrase = _strip_relation_tail(match.group(1))
        if phrase and _normalize(phrase) not in {"to", "in", "by", "for"}:
            return phrase
    return None


def _extract_relation_anchor_phrase(question: str, rel: dict[str, Any]) -> str | None:
    patterns = sorted(
        {
            *_relationship_synonyms(rel["type"]),
            *_relationship_display_aliases(rel),
            _humanize_rel(rel["type"]),
        },
        key=len,
        reverse=True,
    )
    for token in patterns:
        if not token:
            continue
        match = re.search(
            rf"^(?:what|which|who)\s+.+?\s+(?:does|do|did|is|are|was|were)\s+(.+?)\s+{re.escape(token)}\b",
            question.strip(),
            re.IGNORECASE,
        )
        if not match:
            continue
        phrase = _strip_relation_tail(match.group(1))
        if phrase:
            return phrase
    return None


def _extract_relationship_property_filters(question: str, rel: dict[str, Any]) -> list[dict[str, Any]]:
    normalized = _normalize(question)
    properties = set(rel.get("properties", []))
    filters: list[dict[str, Any]] = []

    if "year" in properties:
        between_match = re.search(r"\bbetween\s+((?:19|20)\d{2})\s+and\s+((?:19|20)\d{2})\b", normalized)
        if between_match:
            filters.append({"property": "year", "operator": "between", "value": [int(between_match.group(1)), int(between_match.group(2))]})
        else:
            compare_patterns = [
                (r"\b(?:after|since|later than)\s+((?:19|20)\d{2})\b", ">"),
                (r"\b(?:before|earlier than|until)\s+((?:19|20)\d{2})\b", "<"),
                (r"\b(?:from)\s+((?:19|20)\d{2})\b", "="),
                (r"\bin\s+((?:19|20)\d{2})\b", "="),
            ]
            matched = False
            for pattern, operator in compare_patterns:
                match = re.search(pattern, normalized)
                if match:
                    filters.append({"property": "year", "operator": operator, "value": int(match.group(1))})
                    matched = True
                    break
            if not matched:
                year_match = re.search(r"\b(19|20)\d{2}\b", question)
                if year_match:
                    filters.append({"property": "year", "operator": "=", "value": int(year_match.group(0))})

    if "semester" in properties:
        season_map = {
            "spring": "Spring",
            "summer": "Summer",
            "fall": "Fall",
            "autumn": "Fall",
            "winter": "Winter",
        }
        for token, canonical in season_map.items():
            if re.search(rf"\b{token}\b", normalized):
                filters.append({"property": "semester", "operator": "=", "value": canonical})
                break

    for prop in sorted(properties - {"year", "semester"}):
        alias = _normalize(prop)
        if prop == "grade":
            grade_match = re.search(r"\bgrade\s+([a-f][+-]?)\b", normalized, re.IGNORECASE)
            if grade_match:
                filters.append({"property": "grade", "operator": "=", "value": grade_match.group(1).upper()})
                continue
        if alias and re.search(rf"\b{re.escape(alias)}\b", normalized):
            quoted = _extract_quoted_value(question)
            if quoted:
                filters.append({"property": prop, "operator": "contains", "value": quoted})
                continue
            numeric_match = re.search(rf"\b{re.escape(alias)}\s+(?:>=|=>|at least)\s+(\d+)\b", normalized)
            if numeric_match:
                filters.append({"property": prop, "operator": ">=", "value": int(numeric_match.group(1))})
                continue
            numeric_match = re.search(rf"\b{re.escape(alias)}\s+(?:<=|=<|at most)\s+(\d+)\b", normalized)
            if numeric_match:
                filters.append({"property": prop, "operator": "<=", "value": int(numeric_match.group(1))})
                continue
            numeric_match = re.search(rf"\b{re.escape(alias)}\s+(?:>|more than|over)\s+(\d+)\b", normalized)
            if numeric_match:
                filters.append({"property": prop, "operator": ">", "value": int(numeric_match.group(1))})
                continue
            numeric_match = re.search(rf"\b{re.escape(alias)}\s+(?:<|less than|under)\s+(\d+)\b", normalized)
            if numeric_match:
                filters.append({"property": prop, "operator": "<", "value": int(numeric_match.group(1))})
                continue

    return filters


def _build_relationship_filter_clauses(filters: list[dict[str, Any]], rel: dict[str, Any]) -> tuple[list[str], dict[str, Any], list[str]]:
    clauses: list[str] = []
    params: dict[str, Any] = {}
    notes: list[str] = []
    rel_properties = set(rel.get("properties", []))
    for idx, item in enumerate(filters):
        prop = str(item.get("property"))
        operator = str(item.get("operator", "="))
        value = item.get("value")
        if prop not in rel_properties:
            continue
        if operator == "between" and isinstance(value, list) and len(value) == 2:
            low_key = f"rel_filter_{idx}_low"
            high_key = f"rel_filter_{idx}_high"
            clauses.append(f"toInteger(r.`{prop}`) >= ${low_key} AND toInteger(r.`{prop}`) <= ${high_key}")
            params[low_key] = int(value[0])
            params[high_key] = int(value[1])
            notes.append(f"{prop} between {value[0]} and {value[1]}")
        elif isinstance(value, int):
            key = f"rel_filter_{idx}"
            clauses.append(f"toInteger(r.`{prop}`) {operator} ${key}")
            params[key] = value
            notes.append(f"{prop}{operator}{value}")
        elif operator == "contains":
            key = f"rel_filter_{idx}"
            clauses.append(f"toLower(toString(r.`{prop}`)) CONTAINS toLower(${key})")
            params[key] = value
            notes.append(f"{prop} contains {value}")
        else:
            key = f"rel_filter_{idx}"
            clauses.append(f"toLower(toString(r.`{prop}`)) = toLower(${key})")
            params[key] = value
            notes.append(f"{prop}={value}")
    return clauses, params, notes


def _matching_relationship_indices(item: dict[str, Any], resolved: ResolvedPath) -> list[int]:
    explicit_segment = item.get("segment_index")
    if explicit_segment is not None:
        try:
            idx = int(explicit_segment)
        except Exception:
            return []
        return [idx] if 0 <= idx < len(resolved.relationship_sequence) else []

    prop = str(item.get("property"))
    rel_type = item.get("relationship_type")
    from_label = item.get("from_label")
    to_label = item.get("to_label")
    matches: list[int] = []
    for index, step in enumerate(resolved.relationship_sequence):
        rel = step["relationship"]
        if prop not in rel.get("properties", []):
            continue
        if rel_type and str(rel.get("type")) != str(rel_type):
            continue
        step_from = step["from_label"]
        step_to = step["to_label"]
        if from_label and to_label and not (step_from == from_label and step_to == to_label):
            continue
        matches.append(index)
    return matches


def _relationship_between_nodes(
    source: dict[str, Any],
    target: dict[str, Any],
    query_schema: dict[str, Any],
) -> dict[str, Any] | None:
    return next(
        (
            item
            for item in query_schema["relationships"]
            if item["from_table"] == source["table"] and item["to_table"] == target["table"]
        ),
        None,
    ) or next(
        (
            item
            for item in query_schema["relationships"]
            if item["from_table"] == target["table"] and item["to_table"] == source["table"]
        ),
        None,
    )


def _build_one_hop_plan(
    *,
    source: dict[str, Any],
    target: dict[str, Any],
    rel: dict[str, Any],
    anchor: dict[str, Any],
    other: dict[str, Any],
    anchor_clauses: list[str],
    anchor_params: dict[str, Any],
    other_clauses: list[str] | None = None,
    other_params: dict[str, Any] | None = None,
    rel_filter_clauses: list[str] | None = None,
    rel_filter_params: dict[str, Any] | None = None,
    explanation_suffix: str | None = None,
) -> QueryPlan | None:
    all_clauses = [*anchor_clauses, *(other_clauses or []), *(rel_filter_clauses or [])]
    if not all_clauses:
        return None
    params = dict(anchor_params)
    params.update(other_params or {})
    params.update(rel_filter_params or {})
    if rel["from_table"] == anchor["table"]:
        cypher = (
            f"MATCH (a:`{anchor['neo4j_label']}`)-[r:`{rel['type']}`]->(b:`{other['neo4j_label']}`) "
            f"WHERE {' AND '.join(all_clauses)} "
            f"RETURN {_node_return_clause(anchor, 'a', anchor['label'].lower())}, "
            f"{_relationship_return_clause(rel)}, "
            f"{_node_return_clause(other, 'b', other['label'].lower())} LIMIT 25"
        )
    else:
        cypher = (
            f"MATCH (a:`{anchor['neo4j_label']}`)<-[r:`{rel['type']}`]-(b:`{other['neo4j_label']}`) "
            f"WHERE {' AND '.join(all_clauses)} "
            f"RETURN {_node_return_clause(anchor, 'a', anchor['label'].lower())}, "
            f"{_relationship_return_clause(rel)}, "
            f"{_node_return_clause(other, 'b', other['label'].lower())} LIMIT 25"
        )
    explanation = f"Detected a one-hop traversal between `{source['label']}` and `{target['label']}`."
    if explanation_suffix:
        explanation = f"{explanation[:-1]} with {explanation_suffix}."
    return QueryPlan(
        query_type="one_hop_relation",
        cypher=cypher,
        params=params,
        explanation=explanation,
        planner="rules",
    )


def _build_relationship_property_result_plan(
    *,
    rel: dict[str, Any],
    from_node: dict[str, Any],
    to_node: dict[str, Any],
    requested_property: str,
    where_clauses: list[str],
    params: dict[str, Any],
    explanation_suffix: str | None = None,
    planner: str = "rules",
    debug: dict[str, Any] | None = None,
) -> QueryPlan | None:
    if not where_clauses:
        return None
    return QueryPlan(
        query_type="relationship_property_filter",
        cypher=(
            f"MATCH (a:`{rel['from_neo4j_label']}`)-[r:`{rel['type']}`]->(b:`{rel['to_neo4j_label']}`) "
            f"WHERE {' AND '.join(where_clauses)} "
            f"RETURN {_node_return_clause(from_node, 'a', rel['from_label'].lower())}, "
            f"{_relationship_property_value_clause(requested_property)}, "
            f"{_node_return_clause(to_node, 'b', rel['to_label'].lower())} LIMIT 25"
        ),
        params=params,
        explanation=(
            f"Detected a relationship-property lookup for `{rel['type']}.{requested_property}`"
            + (f" with {explanation_suffix}." if explanation_suffix else ".")
        ),
        planner=planner,
        debug=debug,
    )


def _ai_extract_plan(
    question: str,
    query_schema: dict[str, Any],
    ai_settings: AISettings,
    *,
    graph_mapping: dict[str, Any] | None = None,
    source_schema: list[Any] | None = None,
    source_foreign_keys: list[Any] | None = None,
) -> dict[str, Any]:
    client = AIClient(api_key=ai_settings.api_key, base_url=ai_settings.base_url, model=ai_settings.model)
    payload = {
        "task": "Parse the natural-language graph query into one supported query contract. Return strict JSON only. Do not generate Cypher.",
        "allowed_query_types": sorted(ALLOWED_QUERY_TYPES),
        "question": question,
        "source_schema_context": _serialize_source_schema(source_schema, source_foreign_keys),
        "graph_mapping_context": _serialize_graph_mapping_context(graph_mapping or {}),
        "query_schema_registry": {
            "nodes": [
                {"label": item["label"], "table": item["table"], "properties": item["properties"]}
                for item in query_schema["nodes"]
            ],
            "relationships": [
                {"type": item["type"], "from_label": item["from_label"], "to_label": item["to_label"], "properties": item["properties"]}
                for item in query_schema["relationships"]
            ],
        },
        "rules": [
            "Only choose among the allowed query types.",
            "Only return labels, relationship types, and properties that exist in the provided query schema registry.",
            "Use the source schema context and final graph mapping context to resolve ambiguity before guessing.",
            "If the question is outside the supported scope, return status=unsupported.",
            "Do not generate Cypher.",
            "You may normalize natural wording to the closest schema element when the intent is obvious.",
            "Examples of acceptable normalization: professor -> Instructor, teach/teaches/taught -> TEACHES, took/take -> ENROLLED_IN.",
            "All schema elements must be selected from the registry. Never invent labels, relationships, or properties.",
            "Return one of the six contract shapes in intent_contract and keep it schema-grounded.",
            "If a relationship filter belongs to a specific path segment, include relationship_type and when helpful from_label/to_label or segment_index.",
            "This applies to any query type that traverses relationships, including one_hop, two_hop, fixed_multi_hop, and constrained_multi_hop.",
            "For two-hop questions where both hops are edge traversals, include relationship filters for the correct hop instead of treating them as node filters.",
        ],
        "few_shot_examples": [
            {
                "question": "Show all students",
                "output": {
                    "status": "ok",
                    "query_type": "entity_lookup",
                    "intent_contract": {
                        "target_entity": {"label": "Student", "table": "students"},
                        "filters": [],
                        "return_fields": [],
                        "limit": 25,
                    },
                },
            },
            {
                "question": "What courses does Professor Smith teach?",
                "output": {
                    "status": "ok",
                    "query_type": "one_hop",
                    "intent_contract": {
                        "source_entity": {"label": "Instructor", "table": "instructors"},
                        "target_entity": {"label": "Course", "table": "courses"},
                        "relationship": {"type": "TEACHES", "from_label": "Instructor", "to_label": "Course"},
                        "source_filters": [{"scope": "source", "property": "last_name", "operator": "=", "value": "Smith"}],
                        "target_filters": [],
                        "relationship_filters": [],
                        "return_fields": [],
                        "limit": 25,
                    },
                },
            },
            {
                "question": 'Find students with email "alice@example.edu"',
                "output": {
                    "status": "ok",
                    "query_type": "entity_lookup",
                    "intent_contract": {
                        "target_entity": {"label": "Student", "table": "students"},
                        "filters": [{"scope": "source", "property": "email", "operator": "=", "value": "alice@example.edu"}],
                        "return_fields": [],
                        "limit": 25,
                    },
                },
            },
            {
                "question": "Which courses did Alice take in Fall 2024?",
                "output": {
                    "status": "ok",
                    "query_type": "one_hop",
                    "intent_contract": {
                        "source_entity": {"label": "Student", "table": "students"},
                        "target_entity": {"label": "Course", "table": "courses"},
                        "relationship": {"type": "ENROLLED_IN", "from_label": "Student", "to_label": "Course"},
                        "source_filters": [{"scope": "source", "property": "first_name", "operator": "=", "value": "Alice"}],
                        "target_filters": [],
                        "relationship_filters": [
                            {"scope": "relationship", "property": "semester", "operator": "=", "value": "Fall", "relationship_type": "ENROLLED_IN", "from_label": "Student", "to_label": "Course"},
                            {"scope": "relationship", "property": "year", "operator": "=", "value": 2024, "relationship_type": "ENROLLED_IN", "from_label": "Student", "to_label": "Course"}
                        ],
                        "return_fields": [],
                        "limit": 25,
                    },
                },
            },
            {
                "question": "Show two hop paths from students to departments",
                "output": {
                    "status": "ok",
                    "query_type": "two_hop",
                    "intent_contract": {
                        "source_entity": {"label": "Student", "table": "students"},
                        "middle_entity": {"label": "Course", "table": "courses"},
                        "target_entity": {"label": "Department", "table": "departments"},
                        "path": [
                            {"type": "ENROLLED_IN", "from_label": "Student", "to_label": "Course"},
                            {"type": "OFFERED_BY", "from_label": "Course", "to_label": "Department"}
                        ],
                        "source_filters": [],
                        "middle_filters": [],
                        "target_filters": [],
                        "relationship_filters": [
                            {"scope": "relationship", "property": "semester", "operator": "=", "value": "Spring", "relationship_type": "ENROLLED_IN", "segment_index": 0}
                        ],
                        "return_fields": [],
                        "limit": 10,
                    },
                },
            },
            {
                "question": "Which departments are reached from students through courses in Spring 2024 and offered in 2025?",
                "output": {
                    "status": "ok",
                    "query_type": "two_hop",
                    "intent_contract": {
                        "source_entity": {"label": "Student", "table": "students"},
                        "middle_entity": {"label": "Course", "table": "courses"},
                        "target_entity": {"label": "Department", "table": "departments"},
                        "path": [
                            {"type": "ENROLLED_IN", "from_label": "Student", "to_label": "Course"},
                            {"type": "OFFERED_BY", "from_label": "Course", "to_label": "Department"}
                        ],
                        "source_filters": [],
                        "middle_filters": [],
                        "target_filters": [],
                        "relationship_filters": [
                            {"scope": "relationship", "property": "semester", "operator": "=", "value": "Spring", "relationship_type": "ENROLLED_IN", "segment_index": 0},
                            {"scope": "relationship", "property": "year", "operator": "=", "value": 2024, "relationship_type": "ENROLLED_IN", "segment_index": 0},
                            {"scope": "relationship", "property": "year", "operator": "=", "value": 2025, "relationship_type": "OFFERED_BY", "segment_index": 1}
                        ],
                        "return_fields": [],
                        "limit": 10,
                    },
                },
            },
            {
                "question": "What grade did Alice get in Introduction to Programming?",
                "output": {
                    "status": "ok",
                    "query_type": "relationship_property",
                    "intent_contract": {
                        "source_entity": {"label": "Student", "table": "students"},
                        "target_entity": {"label": "Course", "table": "courses"},
                        "relationship": {"type": "ENROLLED_IN", "from_label": "Student", "to_label": "Course"},
                        "requested_relationship_property": "grade",
                        "source_filters": [{"scope": "source", "property": "first_name", "operator": "=", "value": "Alice"}],
                        "target_filters": [{"scope": "target", "property": "course_name", "operator": "=", "value": "Introduction to Programming"}],
                        "relationship_filters": [],
                        "limit": 25,
                    },
                },
            },
            {
                "question": "Show paths from students via courses instructors departments",
                "output": {
                    "status": "ok",
                    "query_type": "fixed_multi_hop",
                    "intent_contract": {
                        "path_template_id": "student_course_instructor_department",
                        "entities": [
                            {"label": "Student", "table": "students"},
                            {"label": "Course", "table": "courses"},
                            {"label": "Instructor", "table": "instructors"},
                            {"label": "Department", "table": "departments"}
                        ],
                        "relationships": [
                            {"type": "ENROLLED_IN", "from_label": "Student", "to_label": "Course"},
                            {"type": "TEACHES", "from_label": "Instructor", "to_label": "Course"},
                            {"type": "WORKS_FOR", "from_label": "Instructor", "to_label": "Department"}
                        ],
                        "filters": [],
                        "return_fields": [],
                        "limit": 25,
                    },
                },
            },
            {
                "question": "Find paths from students to departments within 3 hops through courses",
                "output": {
                    "status": "ok",
                    "query_type": "constrained_multi_hop",
                    "intent_contract": {
                        "source_entity": {"label": "Student", "table": "students"},
                        "target_entity": {"label": "Department", "table": "departments"},
                        "max_hops": 3,
                        "required_entities": [{"label": "Course", "table": "courses"}],
                        "allowed_relationship_types": ["ENROLLED_IN", "OFFERED_BY", "TEACHES", "WORKS_FOR"],
                        "filters": [],
                        "return_fields": [],
                        "limit": 10,
                    },
                },
            },
        ],
            "response_contract": {
                "status": "ok or unsupported",
                "query_type": "one of allowed_query_types or unsupported",
                "intent_contract": "exactly one contract matching the chosen query_type",
                "confidence": "0 to 1",
            "notes": "short grounding explanation",
        },
    }
    _, parsed = client.complete_json(payload)
    return parsed


def _ai_generate_full_cypher(
    question: str,
    query_schema: dict[str, Any],
    ai_settings: AISettings,
    *,
    graph_mapping: dict[str, Any] | None = None,
    source_schema: list[Any] | None = None,
    source_foreign_keys: list[Any] | None = None,
) -> dict[str, Any]:
    client = AIClient(api_key=ai_settings.api_key, base_url=ai_settings.base_url, model=ai_settings.model)
    payload = {
        "task": "Generate a single read-only Cypher query for the given natural-language question using the provided graph schema.",
        "question": question,
        "source_schema_context": _serialize_source_schema(source_schema, source_foreign_keys),
        "graph_mapping_context": _serialize_graph_mapping_context(graph_mapping or {}),
        "graph_schema": {
            "nodes": [
                {
                    "label": item["label"],
                    "neo4j_label": item["neo4j_label"],
                    "primary_key": item["primary_key"],
                    "properties": item["properties"],
                }
                for item in query_schema["nodes"]
            ],
            "relationships": [
                {
                    "type": item["type"],
                    "from_label": item["from_label"],
                    "to_label": item["to_label"],
                    "from_neo4j_label": item["from_neo4j_label"],
                    "to_neo4j_label": item["to_neo4j_label"],
                    "properties": item["properties"],
                }
                for item in query_schema["relationships"]
            ],
        },
        "rules": [
            "Return strict JSON only.",
            "Only generate read-only Cypher using MATCH, OPTIONAL MATCH, WHERE, WITH, RETURN, ORDER BY, LIMIT.",
            "Never use SQL syntax. Do not use SELECT, JOIN, HAVING, GROUP BY, UNION, subqueries inside IN (SELECT ...), or table aliases from SQL.",
            "Do not use CREATE, MERGE, DELETE, SET, REMOVE, DROP, CALL dbms, LOAD CSV, or APOC procedures.",
            "Use the provided neo4j_label values exactly.",
            "Use only relationship types that exist in the provided graph_schema.relationships. Never invent a relationship type.",
            "For every question, inspect every relationship segment on the candidate path and explicitly decide whether relationship properties are relevant. Do not ignore edge properties just because the question is phrased as an entity query.",
            "If a relationship has properties, check whether those properties are needed for filtering, aligning adjacent edges, disambiguating repeated offerings or events, constraining time or term context, defining aggregation scope, or avoiding incorrect matches.",
            "When two or more adjacent relationship segments can refer to the same contextual event or offering, use shared relationship properties such as semester, year, term, section, or date to keep the path aligned.",
            "For prepositional phrases such as 'from X' or 'in X', attach the phrase to the immediately preceding entity phrase by default unless the wording explicitly requires a different attachment.",
            "For every relationship segment, follow the registry direction exactly: from_label -> to_label. Before returning the final Cypher, double-check every segment against the registry and correct the arrow direction if needed.",
            "Prefer table-friendly RETURN columns over returning raw nodes, relationships, or paths.",
            "Use string literals directly in Cypher for this trial mode; params should usually be an empty object.",
            "If the question cannot be answered from the schema, return status=unsupported.",
        ],
        "few_shot_examples": [
            {
                "question": "Which instructors from Computer Science taught Database Systems in Fall 2024?",
                "graph_pattern": "Department <-[:WORKS_FOR]- Instructor -[:TEACHES]-> Course",
                "reasoning": "Attach 'from Computer Science' to the immediately preceding entity phrase 'instructors'.",
            },
            {
                "question": "Which instructors were employed by the Computer Science department?",
                "graph_pattern": "Department -[:EMPLOYS]-> Instructor",
                "reasoning": "Keep the registry direction Department -> Instructor even when the wording is passive.",
            },
        ],
        "response_contract": {
            "status": "ok or unsupported",
            "query_type": "short description such as one_hop_relation, relationship_property_filter, aggregation, or unsupported",
            "cypher": "single read-only Cypher query string",
            "params": "JSON object of query parameters, usually empty",
            "explanation": "short explanation of what the query is doing",
        },
    }
    _, parsed = client.complete_json(payload)
    return parsed


def _plan_from_extraction(extraction: dict[str, Any], query_schema: dict[str, Any]) -> QueryPlan | None:
    if extraction.get("status") != "ok":
        return None
    query_type = extraction.get("query_type")
    if query_type not in ALLOWED_QUERY_TYPES:
        return None
    anchor_label = extraction.get("anchor_label")
    anchor_property = extraction.get("anchor_property")
    anchor_value = extraction.get("anchor_value")
    target_label = extraction.get("target_label")
    target_property = extraction.get("target_property")
    target_value = extraction.get("target_value")
    relationship_type = extraction.get("relationship_type")
    relationship_property = extraction.get("relationship_property")
    relationship_property_filters = extraction.get("relationship_property_filters") or []
    anchor_node = next((item for item in query_schema["nodes"] if item["label"] == anchor_label), None)
    target_node = next((item for item in query_schema["nodes"] if item["label"] == target_label), None)

    if query_type == "entity_retrieval" and anchor_label:
        if not anchor_node:
            return None
        return QueryPlan(
            query_type=query_type,
            cypher=f"MATCH (n:`{anchor_node['neo4j_label']}`) RETURN {_node_return_clause(anchor_node, 'n', anchor_label.lower())} LIMIT 25",
            params={},
            explanation=f"Detected a simple entity retrieval for `{anchor_label}`.",
            planner="ai_assisted",
            debug=extraction,
        )
    if query_type == "attribute_filter" and anchor_label and anchor_property and anchor_value:
        if not anchor_node:
            return None
        return QueryPlan(
            query_type=query_type,
            cypher=(
                f"MATCH (n:`{anchor_node['neo4j_label']}`) "
                f"WHERE toLower(toString(n.`{anchor_property}`)) CONTAINS toLower($value) "
                f"RETURN {_node_return_clause(anchor_node, 'n', anchor_label.lower())} LIMIT 25"
            ),
            params={"value": anchor_value},
            explanation=f"Detected an attribute filter on `{anchor_label}.{anchor_property}`.",
            planner="ai_assisted",
            debug=extraction,
        )
    if query_type == "one_hop_relation" and anchor_label and target_label and relationship_type and anchor_property and anchor_value:
        rel = next((item for item in query_schema["relationships"] if item["type"] == relationship_type), None)
        if not rel or not anchor_node or not target_node:
            return None
        where_parts = [f"toLower(toString(a.`{anchor_property}`)) CONTAINS toLower($value)"]
        params = {"value": anchor_value}
        if target_property and target_value:
            where_parts.append(f"toLower(toString(b.`{target_property}`)) CONTAINS toLower($target_value)")
            params["target_value"] = target_value
        ai_rel_clauses, ai_rel_params, _ = _build_relationship_filter_clauses(relationship_property_filters, rel)
        where_parts.extend(ai_rel_clauses)
        params.update(ai_rel_params)
        if rel["from_label"] == anchor_label and rel["to_label"] == target_label:
            cypher = (
                f"MATCH (a:`{anchor_node['neo4j_label']}`)-[r:`{relationship_type}`]->(b:`{target_node['neo4j_label']}`) "
                f"WHERE {' AND '.join(where_parts)} "
                f"RETURN {_node_return_clause(anchor_node, 'a', anchor_label.lower())}, "
                f"{_relationship_return_clause(rel)}, "
                f"{_node_return_clause(target_node, 'b', target_label.lower())} LIMIT 25"
            )
        elif rel["to_label"] == anchor_label and rel["from_label"] == target_label:
            cypher = (
                f"MATCH (a:`{anchor_node['neo4j_label']}`)<-[r:`{relationship_type}`]-(b:`{target_node['neo4j_label']}`) "
                f"WHERE {' AND '.join(where_parts)} "
                f"RETURN {_node_return_clause(anchor_node, 'a', anchor_label.lower())}, "
                f"{_relationship_return_clause(rel)}, "
                f"{_node_return_clause(target_node, 'b', target_label.lower())} LIMIT 25"
            )
        else:
            return None
        return QueryPlan(
            query_type=query_type,
            cypher=cypher,
            params=params,
            explanation=f"Detected a one-hop traversal using `{relationship_type}`.",
            planner="ai_assisted",
            debug=extraction,
        )
    if query_type == "relationship_property_filter" and relationship_type and relationship_property and anchor_value:
        rel = next((item for item in query_schema["relationships"] if item["type"] == relationship_type), None)
        if not rel:
            return None
        from_node = next((n for n in query_schema["nodes"] if n["table"] == rel["from_table"]), None)
        to_node = next((n for n in query_schema["nodes"] if n["table"] == rel["to_table"]), None)
        if not from_node or not to_node:
            return None
        where_clauses = [f"toLower(toString(a.`{anchor_property or _preferred_text_filter_property(from_node)}`)) CONTAINS toLower($anchor_value)"]
        params = {"anchor_value": anchor_value}
        if target_property and target_value:
            where_clauses.append(f"toLower(toString(b.`{target_property}`)) CONTAINS toLower($target_value)")
            params["target_value"] = target_value
        return _build_relationship_property_result_plan(
            rel=rel,
            from_node=from_node,
            to_node=to_node,
            requested_property=relationship_property,
            where_clauses=where_clauses,
            params=params,
            explanation_suffix=f"{from_node['label']} and {to_node['label']} constraints",
            planner="ai_assisted",
            debug=extraction,
        )
    if query_type == "two_hop_traversal" and anchor_label and target_label:
        if not anchor_node or not target_node:
            return None
        return QueryPlan(
            query_type=query_type,
            cypher=(
                f"MATCH p=(a:`{anchor_node['neo4j_label']}`)-[r1]->(m)-[r2]->(b:`{target_node['neo4j_label']}`) "
                f"RETURN {_node_return_clause(anchor_node, 'a', anchor_label.lower())}, "
                "type(r1) AS first_relationship, "
                "labels(m) AS middle_labels, "
                "properties(m) AS middle_node, "
                "type(r2) AS second_relationship, "
                f"{_node_return_clause(target_node, 'b', target_label.lower())} LIMIT 10"
            ),
            params={},
            explanation=f"Detected a constrained two-hop traversal from `{anchor_label}` to `{target_label}`.",
            planner="ai_assisted",
            debug=extraction,
        )
    return None


def _relationship_filters_from_contract(items: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    filters: list[dict[str, Any]] = []
    for item in items or []:
        prop = item.get("property")
        value = item.get("value")
        if prop and value is not None:
            payload = {
                "property": str(prop),
                "operator": str(item.get("operator") or "="),
                "value": value,
            }
            if item.get("relationship_type"):
                payload["relationship_type"] = str(item.get("relationship_type"))
            if item.get("from_label"):
                payload["from_label"] = str(item.get("from_label"))
            if item.get("to_label"):
                payload["to_label"] = str(item.get("to_label"))
            if item.get("segment_index") is not None:
                payload["segment_index"] = int(item.get("segment_index"))
            filters.append(payload)
    return filters


def _first_filter_value(items: list[dict[str, Any]] | None) -> tuple[str | None, Any]:
    if not items:
        return None, None
    first = items[0]
    return (str(first.get("property")) if first.get("property") else None, first.get("value"))


def _intent_from_contract(extraction: dict[str, Any], query_schema: dict[str, Any]) -> ExtractedIntent | None:
    if extraction.get("status") != "ok":
        return None
    query_type = extraction.get("query_type")
    if query_type not in ALLOWED_QUERY_TYPES:
        return None
    contract = extraction.get("intent_contract") or {}
    if not isinstance(contract, dict):
        return None

    if query_type == "entity_lookup":
        target = contract.get("target_entity") or {}
        filters = contract.get("filters") or []
        answer_property, answer_value = _first_filter_value(filters)
        return ExtractedIntent(
            query_type=query_type,
            intent_contract=contract,
            answer_label=target.get("label"),
            answer_value=answer_value,
            answer_property=answer_property,
            relationship_filters=[],
            explicit_node_sequence=[target.get("label")] if target.get("label") else [],
            return_hint="entity",
        )

    if query_type in {"one_hop", "relationship_property"}:
        source = contract.get("source_entity") or {}
        target = contract.get("target_entity") or {}
        relationship = contract.get("relationship") or {}
        source_filters = contract.get("source_filters") or []
        target_filters = contract.get("target_filters") or []
        answer_property, answer_value = _first_filter_value(source_filters)
        target_property, target_value = _first_filter_value(target_filters)
        return ExtractedIntent(
            query_type=query_type,
            intent_contract=contract,
            answer_label=source.get("label"),
            target_label=target.get("label"),
            requested_relationship_property=contract.get("requested_relationship_property"),
            relationship_type=relationship.get("type"),
            answer_value=answer_value,
            target_value=target_value,
            answer_property=answer_property,
            target_property=target_property,
            relationship_filters=_relationship_filters_from_contract(contract.get("relationship_filters")),
            explicit_node_sequence=[label for label in [source.get("label"), target.get("label")] if label],
            required_entity_labels=[],
            allowed_relationship_types=[relationship.get("type")] if relationship.get("type") else [],
        )

    if query_type == "two_hop":
        source = contract.get("source_entity") or {}
        middle = contract.get("middle_entity") or {}
        target = contract.get("target_entity") or {}
        path = contract.get("path") or []
        return ExtractedIntent(
            query_type=query_type,
            intent_contract=contract,
            answer_label=source.get("label"),
            target_label=target.get("label"),
            relationship_type=path[0].get("type") if path else None,
            relationship_filters=_relationship_filters_from_contract(contract.get("relationship_filters")),
            explicit_node_sequence=[label for label in [source.get("label"), middle.get("label"), target.get("label")] if label],
            required_entity_labels=[middle.get("label")] if middle.get("label") else [],
            allowed_relationship_types=[item.get("type") for item in path if item.get("type")],
        )

    if query_type == "fixed_multi_hop":
        entities = contract.get("entities") or []
        relationships = contract.get("relationships") or []
        labels = [item.get("label") for item in entities if item.get("label")]
        return ExtractedIntent(
            query_type=query_type,
            intent_contract=contract,
            answer_label=labels[0] if labels else None,
            target_label=labels[-1] if labels else None,
            relationship_filters=_relationship_filters_from_contract(contract.get("filters")),
            explicit_node_sequence=labels,
            required_entity_labels=labels[1:-1] if len(labels) > 2 else [],
            allowed_relationship_types=[item.get("type") for item in relationships if item.get("type")],
        )

    if query_type == "constrained_multi_hop":
        source = contract.get("source_entity") or {}
        target = contract.get("target_entity") or {}
        required_entities = contract.get("required_entities") or []
        return ExtractedIntent(
            query_type=query_type,
            intent_contract=contract,
            answer_label=source.get("label"),
            target_label=target.get("label"),
            relationship_filters=_relationship_filters_from_contract(contract.get("filters")),
            hop_limit=contract.get("max_hops"),
            explicit_node_sequence=[label for label in [source.get("label"), *[item.get("label") for item in required_entities], target.get("label")] if label],
            required_entity_labels=[item.get("label") for item in required_entities if item.get("label")],
            allowed_relationship_types=[item for item in (contract.get("allowed_relationship_types") or []) if item],
        )

    return None


def _run_registry_pipeline_from_intent(
    question: str,
    query_schema: dict[str, Any],
    pre: PreprocessedQuestion,
    decision: QueryTypeDecision,
    intent: ExtractedIntent,
    *,
    planner_name: str,
    ai_settings: AISettings | None = None,
    extraction_debug: dict[str, Any] | None = None,
) -> QueryPlan:
    candidates = _build_candidate_set(intent)
    resolved = _resolve_path(intent, query_schema)
    if not resolved:
        return QueryPlan(
            query_type="unsupported",
            cypher="",
            params={},
            explanation="Could not resolve a schema-grounded path for this question.",
            status="unsupported",
            planner=planner_name,
            debug={
                "stage_1_preprocessor": _debug_dict(pre),
                "stage_2_query_type_classifier": _debug_dict(decision),
                "stage_3_entity_value_extractor": _debug_dict(intent),
                "stage_4_candidate_generator": _debug_dict(candidates),
                "ai_extraction": extraction_debug or {},
            },
        )
    clauses, params, note = _bind_constraints(resolved, query_schema)
    constraints = BoundConstraints(clauses=clauses, params=params, note=note)
    output_shape = _default_output_shape(resolved)
    if ai_settings:
        ai_output_shape = _ai_select_output_shape(question, query_schema, intent, resolved, ai_settings)
        if ai_output_shape:
            output_shape = ai_output_shape
    plan = _build_standard_plan(resolved, constraints.clauses, constraints.params, constraints.note, output_shape)
    plan.planner = planner_name
    cypher_result = _build_cypher_result(plan)
    plan.debug = {
        "stage_1_preprocessor": _debug_dict(pre),
        "stage_2_query_type_classifier": _debug_dict(decision),
        "stage_3_entity_value_extractor": {
            **_debug_dict(intent),
            "anchor_label": intent.anchor_label,
            "return_label": intent.return_label,
            "anchor_value": intent.anchor_value,
            "return_value": intent.return_value,
        },
        "stage_4_candidate_generator": _debug_dict(candidates),
        "stage_5_path_resolver": {
            "nodes": [node["label"] for node in resolved.node_sequence],
            "relationships": [step["relationship"]["type"] for step in resolved.relationship_sequence],
            "resolved_path": _debug_dict(resolved),
        },
        "stage_6_constraint_binder": _debug_dict(constraints),
        "stage_6b_output_selector": _debug_dict(output_shape),
        "stage_7_cypher_builder": _debug_dict(cypher_result),
        "ai_extraction": extraction_debug or {},
    }
    return plan


def plan_graph_query(
    question: str,
    graph_mapping: dict[str, Any],
    *,
    ai_settings: AISettings | None = None,
    query_registry: dict[str, Any] | None = None,
    source_schema: list[Any] | None = None,
    source_foreign_keys: list[Any] | None = None,
) -> QueryPlan:
    query_schema = build_query_schema(
        query_registry or graph_mapping,
        source_schema=source_schema,
        source_foreign_keys=source_foreign_keys,
    )
    if not ai_settings:
        return _legacy_run_registry_pipeline(question, query_schema)

    extraction = _ai_extract_plan(
        question,
        query_schema,
        ai_settings,
        graph_mapping=graph_mapping,
        source_schema=source_schema,
        source_foreign_keys=source_foreign_keys,
    )
    intent = _intent_from_contract(extraction, query_schema)
    if not intent:
        return QueryPlan(
            query_type="unsupported",
            cypher="",
            params={},
            explanation="AI query parsing did not return a valid schema-grounded contract.",
            status="unsupported",
            planner="ai_assisted",
            debug={"ai_extraction": extraction},
        )

    pre = _preprocess_question(question, query_schema)
    decision = QueryTypeDecision(
        query_type=intent.query_type,
        rationale="AI parser produced a schema-grounded contract that matches the supported planner types.",
    )
    ai_plan = _run_registry_pipeline_from_intent(
        question,
        query_schema,
        pre,
        decision,
        intent,
        planner_name="ai_assisted",
        ai_settings=ai_settings,
        extraction_debug=extraction,
    )
    if ai_plan.status == "ok":
        return ai_plan
    return QueryPlan(
        query_type="unsupported",
        cypher="",
        params={},
        explanation="AI query parsing succeeded, but the schema-grounded resolver could not build a valid path.",
        status="unsupported",
        planner="ai_assisted",
        debug={"ai_extraction": extraction, "resolver_plan": ai_plan.debug or {}},
    )


def plan_graph_query_full_ai(
    question: str,
    graph_mapping: dict[str, Any],
    *,
    ai_settings: AISettings,
    query_registry: dict[str, Any] | None = None,
    source_schema: list[Any] | None = None,
    source_foreign_keys: list[Any] | None = None,
) -> QueryPlan:
    query_schema = build_query_schema(
        query_registry or graph_mapping,
        source_schema=source_schema,
        source_foreign_keys=source_foreign_keys,
    )
    extraction = _ai_generate_full_cypher(
        question,
        query_schema,
        ai_settings,
        graph_mapping=graph_mapping,
        source_schema=source_schema,
        source_foreign_keys=source_foreign_keys,
    )
    if extraction.get("status") != "ok":
        return QueryPlan(
            query_type="unsupported",
            cypher="",
            params={},
            explanation=str(extraction.get("explanation") or "AI could not generate a supported query."),
            status="unsupported",
            planner="ai_full_cypher",
            debug=extraction,
        )

    cypher = _normalize_full_ai_cypher(_strip_code_fences(str(extraction.get("cypher") or "")))
    if not cypher:
        return QueryPlan(
            query_type="unsupported",
            cypher="",
            params={},
            explanation="AI returned an empty Cypher query.",
            status="unsupported",
            planner="ai_full_cypher",
            debug=extraction,
        )

    blocked = {"create ", "merge ", "delete ", "set ", "remove ", "drop ", "call dbms", "load csv", "apoc."}
    cypher_normalized = cypher.strip().lower()
    if any(token in cypher_normalized for token in blocked):
        return QueryPlan(
            query_type="unsupported",
            cypher=cypher,
            params={},
            explanation="AI produced a non-read-only Cypher query, so it was rejected.",
            status="unsupported",
            planner="ai_full_cypher",
            debug=extraction,
        )

    sql_like_error = _detect_sql_like_syntax(cypher)
    if sql_like_error:
        return QueryPlan(
            query_type="unsupported",
            cypher=cypher,
            params={},
            explanation=sql_like_error,
            status="unsupported",
            planner="ai_full_cypher",
            debug=extraction,
        )

    cypher, direction_rewrites = _rewrite_full_ai_relationship_directions(cypher, query_schema)
    cypher, alignment_clauses = _add_full_ai_context_alignment(cypher, query_schema)
    direction_error = _validate_full_ai_relationship_directions(cypher, query_schema)
    if direction_error:
        return QueryPlan(
            query_type="unsupported",
            cypher=cypher,
            params={},
            explanation=direction_error,
            status="unsupported",
            planner="ai_full_cypher",
            debug=extraction,
        )

    params = extraction.get("params")
    if not isinstance(params, dict):
        params = {}

    return QueryPlan(
        query_type=_infer_query_type_from_cypher(cypher, str(extraction.get("query_type") or "ai_generated")),
        cypher=cypher,
        params=params,
        explanation=(
            str(extraction.get("explanation") or "AI generated the Cypher directly.")
            + (f" Auto-corrected relationship directions: {', '.join(direction_rewrites)}." if direction_rewrites else "")
            + (f" Added edge alignment constraints: {', '.join(alignment_clauses)}." if alignment_clauses else "")
        ),
        planner="ai_full_cypher",
        debug=extraction,
    )


def run_graph_query(
    *,
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
    cypher: str,
    params: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if not cypher.strip():
        return []
    blocked = {"create ", "merge ", "delete ", "set ", "remove ", "drop ", "call dbms", "load csv"}
    cypher_normalized = cypher.strip().lower()
    if any(token in cypher_normalized for token in blocked):
        raise ValueError("Only read-only graph queries are allowed in the query UI.")

    from neo4j import GraphDatabase

    def _to_jsonable(value: Any) -> Any:
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, list):
            return [_to_jsonable(item) for item in value]
        if isinstance(value, dict):
            return {key: _to_jsonable(item) for key, item in value.items()}
        if hasattr(value, "items") and hasattr(value, "labels"):
            return {"labels": list(value.labels), "properties": {key: _to_jsonable(item) for key, item in dict(value).items()}}
        if hasattr(value, "type") and hasattr(value, "items"):
            return {"type": str(value.type), "properties": {key: _to_jsonable(item) for key, item in dict(value).items()}}
        if hasattr(value, "start_node") and hasattr(value, "end_node"):
            return str(value)
        return str(value)

    with GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password)) as driver:
        with driver.session() as session:
            result = session.run(cypher, **(params or {}))
            return [{key: _to_jsonable(value) for key, value in record.items()} for record in result]
