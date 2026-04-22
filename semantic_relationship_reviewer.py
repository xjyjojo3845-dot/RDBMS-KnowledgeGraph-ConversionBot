from __future__ import annotations

import json
import logging
from typing import Any

from ai.client import AIClient
from ai.config import load_ai_settings, semantic_review_enabled

logger = logging.getLogger(__name__)

SEMANTIC_RESULT_SCHEMA = {
    "type": "object",
    "required": [
        "relationship_id",
        "final_from_table",
        "final_to_table",
        "relationship_type",
        "direction",
        "explanation",
        "confidence",
    ],
}


def _upper_snake(value: str) -> str:
    return value.replace("-", "_").replace(" ", "_").upper()


def _humanize_relationship_name(name: str) -> str:
    return name.replace("_", " ").strip().lower()


def _singularize(name: str) -> str:
    cleaned = name.replace("_", " ").strip()
    if cleaned.endswith("s") and len(cleaned) > 1:
        cleaned = cleaned[:-1]
    return cleaned.title().replace(" ", "")


def _display_text(from_table: str, rel_name: str, to_table: str, props: list[str]) -> str:
    base = f"{_singularize(from_table)} {_humanize_relationship_name(rel_name)} {_singularize(to_table)}"
    if props:
        return f"{base} {{{', '.join(props)}}}"
    return base


def _relationship_candidate(rel: dict[str, Any]) -> dict[str, Any]:
    base_from = str(rel.get("raw_from_table") or rel["from_table"])
    base_to = str(rel.get("raw_to_table") or rel["to_table"])
    fk_columns = list(rel.get("raw_from_join_columns", rel.get("raw_from_columns", rel.get("from_join_columns", rel.get("from_columns", [])))))
    referenced_keys = list(rel.get("raw_to_join_columns", rel.get("raw_to_columns", rel.get("to_join_columns", rel.get("to_columns", [])))))
    base_direction = {
        "base_from_table": base_from,
        "base_to_table": base_to,
        "fk_holder_table": base_from,
        "referenced_table": base_to,
        "cardinality": rel.get("cardinality"),
        "classification": rel.get("raw_classification", rel.get("classification")),
        "via_table": rel.get("raw_via_table", rel.get("via_table")),
        "has_payload_columns": bool(rel.get("properties")),
        "payload_columns": list(rel.get("properties", [])),
        "signals": list(rel.get("signals", [])),
    }
    if len(fk_columns) == 1 and len(referenced_keys) == 1:
        base_direction["fk_column"] = fk_columns[0]
        base_direction["referenced_key"] = referenced_keys[0]
    else:
        base_direction["fk_columns"] = fk_columns
        base_direction["referenced_keys"] = referenced_keys

    return {
        "relationship_id": rel["id"],
        "base_direction": base_direction,
        "raw_inferred_relationship": {
            "base_from_table": base_from,
            "base_to_table": base_to,
            "relationship_type": rel.get("rule_based_name") or rel.get("final_name"),
            "cardinality": rel.get("cardinality"),
            "source_is_join_table": bool(rel.get("classification") == "collapsed_join_table"),
            "via_table": rel.get("raw_via_table", rel.get("via_table")),
            "payload_columns": list(rel.get("properties", [])),
            "has_payload_columns": bool(rel.get("properties")),
            "classification": rel.get("raw_classification", rel.get("classification")),
            "signals": list(rel.get("signals", [])),
        },
        "allowed_tables": [base_from, base_to],
        "modeling_principles": [
            "Use only schema evidence. Do not invent domain facts.",
            "The candidate has one canonical base direction derived from foreign-key storage: base_from_table is the FK holder and base_to_table is the referenced table.",
            "The result must preserve the same two endpoint tables, but you may reverse direction semantically.",
            "Choose a relationship name that reads naturally in a business sentence.",
            "Prefer concise property-graph relationship names such as CREATED, CONTAINS, PLACED, SUPPORTS, REPORTS_TO.",
            "Prefer active voice over passive voice. Avoid names such as CREATED_BY when the edge can be reversed to CREATED.",
            "If the evidence is weak, keep the deterministic direction and improve the name conservatively.",
        ],
        "response_contract": {
            "required_fields": SEMANTIC_RESULT_SCHEMA["required"],
            "direction_values": ["kept", "reversed"],
            "confidence_range": [0.0, 1.0],
        },
        "few_shot_examples": [
            {
                "input": {
                    "base_from_table": "Book",
                    "base_to_table": "Author",
                    "relationship_type": "BOOK_TO_AUTHOR",
                    "cardinality": "many_to_one",
                    "has_payload_columns": False,
                },
                "output": {
                    "final_from_table": "Author",
                    "final_to_table": "Book",
                    "relationship_type": "WROTE",
                    "direction": "reversed",
                },
            },
            {
                "input": {
                    "base_from_table": "Order",
                    "base_to_table": "Customer",
                    "relationship_type": "ORDER_TO_CUSTOMER",
                    "cardinality": "many_to_one",
                    "has_payload_columns": False,
                },
                "output": {
                    "final_from_table": "Customer",
                    "final_to_table": "Order",
                    "relationship_type": "PLACED",
                    "direction": "reversed",
                },
            },
            {
                "input": {
                    "base_from_table": "Library",
                    "base_to_table": "Book",
                    "relationship_type": "LIBRARY_TO_BOOK",
                    "cardinality": "one_to_many",
                    "has_payload_columns": False,
                },
                "output": {
                    "final_from_table": "Library",
                    "final_to_table": "Book",
                    "relationship_type": "CONTAINS",
                    "direction": "kept",
                },
            },
            {
                "input": {
                    "base_from_table": "Product",
                    "base_to_table": "Category",
                    "relationship_type": "PRODUCT_TO_CATEGORY",
                    "cardinality": "many_to_one",
                    "has_payload_columns": False,
                },
                "output": {
                    "final_from_table": "Product",
                    "final_to_table": "Category",
                    "relationship_type": "BELONGS_TO",
                    "direction": "kept",
                },
            },
            {
                "input": {
                    "base_from_table": "Employee",
                    "base_to_table": "Employee",
                    "relationship_type": "EMPLOYEE_TO_EMPLOYEE",
                    "cardinality": "many_to_one",
                    "signals": ["self_reference", "fk_name_reports_to"],
                },
                "output": {
                    "final_from_table": "Employee",
                    "final_to_table": "Employee",
                    "relationship_type": "REPORTS_TO",
                    "direction": "kept",
                },
            },
            {
                "input": {
                    "base_from_table": "Student",
                    "base_to_table": "Course",
                    "relationship_type": "STUDENT_COURSE",
                    "cardinality": "many_to_many",
                    "via_table": "Enrollment",
                    "has_payload_columns": False,
                },
                "output": {
                    "final_from_table": "Student",
                    "final_to_table": "Course",
                    "relationship_type": "ENROLLED_IN",
                    "direction": "kept",
                },
            },
            {
                "input": {
                    "base_from_table": "Order",
                    "base_to_table": "Product",
                    "relationship_type": "ORDER_ITEM",
                    "cardinality": "many_to_many",
                    "via_table": "OrderItem",
                    "has_payload_columns": True,
                    "payload_columns": ["quantity", "price"],
                },
                "output": {
                    "final_from_table": "Order",
                    "final_to_table": "Product",
                    "relationship_type": "CONTAINS",
                    "direction": "kept",
                },
            },
        ],
    }


def _normalize_semantic_payload(response: Any) -> dict[str, Any]:
    if isinstance(response, dict):
        for key in ("semantic_review", "review", "result", "candidate_review"):
            if isinstance(response.get(key), dict):
                return response[key]
        return dict(response)
    if isinstance(response, list) and response and isinstance(response[0], dict):
        return dict(response[0])
    return {}


def _normalize_direction(value: Any, *, from_table: str | None, to_table: str | None, expected_kept: tuple[str, str]) -> str:
    normalized = str(value or "").strip().lower()
    aliases = {
        "same": "kept",
        "as_inferred": "kept",
        "original": "kept",
        "reverse": "reversed",
        "flip": "reversed",
    }
    if normalized in aliases:
        normalized = aliases[normalized]
    if normalized in {"kept", "reversed"}:
        return normalized
    if from_table is not None and to_table is not None and (from_table, to_table) != expected_kept:
        return "reversed"
    return "kept"


def _normalize_confidence(value: Any) -> float:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.endswith("%"):
            return float(stripped[:-1]) / 100.0
        return float(stripped)
    return float(value)


def _to_active_voice(
    relationship_type: str,
    from_table: str,
    to_table: str,
    direction: str,
) -> tuple[str, str, str, str]:
    passive_map = {
        "CREATED_BY": "CREATED",
        "PLACED_BY": "PLACED",
        "SUPPORTED_BY": "SUPPORTS",
        "MANAGED_BY": "MANAGES",
        "OWNED_BY": "OWNS",
        "OFFERED_BY": "OFFERS",
        "TAUGHT_BY": "TEACHES",
    }
    if relationship_type in passive_map and from_table != to_table:
        return passive_map[relationship_type], to_table, from_table, "reversed"
    if relationship_type.endswith("_BY") and from_table != to_table:
        stem = relationship_type[: -len("_BY")]
        return stem or relationship_type, to_table, from_table, "reversed"
    return relationship_type, from_table, to_table, direction


def _coerce_semantic_result(candidate: dict[str, Any], response: dict[str, Any]) -> dict[str, Any]:
    base_direction = candidate.get("base_direction") or {
        "base_from_table": candidate.get("raw_inferred_relationship", {}).get("base_from_table")
        or candidate.get("raw_inferred_relationship", {}).get("source_table"),
        "base_to_table": candidate.get("raw_inferred_relationship", {}).get("base_to_table")
        or candidate.get("raw_inferred_relationship", {}).get("target_table"),
        "cardinality": candidate.get("raw_inferred_relationship", {}).get("cardinality"),
    }
    expected_from, expected_to = base_direction["base_from_table"], base_direction["base_to_table"]
    response = dict(response)
    response.setdefault("relationship_id", response.get("id") or candidate["relationship_id"])
    response.setdefault("final_from_table", response.get("from_table") or response.get("final_source") or response.get("source_table") or response.get("start_table"))
    response.setdefault("final_to_table", response.get("to_table") or response.get("final_target") or response.get("target_table") or response.get("end_table"))
    response.setdefault("relationship_type", response.get("final_name") or response.get("name") or response.get("type"))
    response.setdefault("explanation", response.get("reasoning_summary") or response.get("reason"))
    if "confidence" not in response and "score" in response:
        response["confidence"] = response["score"]
    response["direction"] = _normalize_direction(
        response.get("direction"),
        from_table=response.get("final_from_table"),
        to_table=response.get("final_to_table"),
        expected_kept=(expected_from, expected_to),
    )
    if not response.get("final_from_table") or not response.get("final_to_table"):
        if response["direction"] == "reversed" and len(set((expected_from, expected_to))) > 1:
            response["final_from_table"], response["final_to_table"] = expected_to, expected_from
        else:
            response["final_from_table"], response["final_to_table"] = expected_from, expected_to
    return response


def _deterministic_post_check(
    relationship_type: str,
    final_from_table: str,
    final_to_table: str,
    candidate: dict[str, Any],
) -> None:
    base_from = candidate["base_direction"]["base_from_table"]
    base_to = candidate["base_direction"]["base_to_table"]
    actor_event_verbs = {"PLACED", "CREATED", "WROTE"}
    category_verbs = {"BELONGS_TO"}
    hierarchy_verbs = {"REPORTS_TO"}
    collection_verbs = {"CONTAINS", "ENROLLED_IN"}

    if relationship_type in actor_event_verbs and final_from_table == base_from and final_to_table == base_to and base_from != base_to:
        raise ValueError(f"{relationship_type} is invalid in base FK direction; reverse it to actor->event.")
    if relationship_type in category_verbs and final_from_table != base_from and final_to_table != base_to and base_from != base_to:
        raise ValueError(f"{relationship_type} should remain in member->category direction.")
    if relationship_type in hierarchy_verbs and final_from_table != final_to_table:
        raise ValueError(f"{relationship_type} should remain a self-reference hierarchy when both endpoints are the same table.")
    if relationship_type in collection_verbs and candidate["base_direction"]["cardinality"] == "one_to_many" and final_from_table != base_from:
        raise ValueError(f"{relationship_type} should keep collection->member direction for one-to-many patterns.")


def _validate_semantic_result(candidate: dict[str, Any], response: dict[str, Any]) -> dict[str, Any]:
    response = _coerce_semantic_result(candidate, response)
    base_direction = candidate.get("base_direction") or {
        "base_from_table": candidate.get("raw_inferred_relationship", {}).get("base_from_table")
        or candidate.get("raw_inferred_relationship", {}).get("source_table"),
        "base_to_table": candidate.get("raw_inferred_relationship", {}).get("base_to_table")
        or candidate.get("raw_inferred_relationship", {}).get("target_table"),
        "cardinality": candidate.get("raw_inferred_relationship", {}).get("cardinality"),
    }
    for field in SEMANTIC_RESULT_SCHEMA["required"]:
        if field not in response:
            raise ValueError(f"Missing required semantic review field: {field}")

    relationship_id = str(response["relationship_id"])
    if relationship_id != candidate["relationship_id"]:
        raise ValueError("Relationship id did not match the reviewed candidate.")

    allowed_tables = set(candidate["allowed_tables"])
    from_table = str(response["final_from_table"])
    to_table = str(response["final_to_table"])
    if from_table not in allowed_tables or to_table not in allowed_tables:
        raise ValueError("Semantic review returned tables outside the allowed endpoints.")
    if from_table == to_table and len(allowed_tables) > 1:
        raise ValueError("Semantic review collapsed a non-self relationship into the same endpoint.")

    direction = _normalize_direction(
        response["direction"],
        from_table=from_table,
        to_table=to_table,
        expected_kept=(base_direction["base_from_table"], base_direction["base_to_table"]),
    )

    relationship_type = _upper_snake(str(response["relationship_type"]).strip())
    if not relationship_type:
        raise ValueError("Relationship type cannot be empty.")
    relationship_type, from_table, to_table, direction = _to_active_voice(
        relationship_type,
        from_table,
        to_table,
        direction,
    )

    confidence = _normalize_confidence(response["confidence"])
    if confidence > 1:
        confidence = confidence / 100.0
    if confidence < 0 or confidence > 1:
        raise ValueError("Confidence must be between 0 and 1.")

    explanation = str(response["explanation"]).strip()
    if not explanation:
        raise ValueError("Explanation cannot be empty.")

    expected_kept = (
        base_direction["base_from_table"],
        base_direction["base_to_table"],
    )
    actual = (from_table, to_table)
    if direction == "kept" and actual != expected_kept:
        raise ValueError("Direction said `kept` but the endpoint order changed.")
    if direction == "reversed" and actual == expected_kept and len(set(expected_kept)) > 1:
        raise ValueError("Direction said `reversed` but the endpoint order stayed the same.")
    candidate_for_check = dict(candidate)
    candidate_for_check["base_direction"] = base_direction
    _deterministic_post_check(relationship_type, from_table, to_table, candidate_for_check)

    return {
        "relationship_id": relationship_id,
        "from_table": from_table,
        "to_table": to_table,
        "final_from_table": from_table,
        "final_to_table": to_table,
        "relationship_type": relationship_type,
        "direction": direction,
        "explanation": explanation,
        "confidence": round(confidence, 2),
    }


def run_semantic_relationship_review(
    graph_mapping: dict[str, Any],
    *,
    model: str | None = None,
    enabled: bool | None = None,
) -> dict[str, Any] | None:
    relationships = list(graph_mapping.get("relationships", []))
    if enabled is None:
        enabled = semantic_review_enabled()
    if not enabled:
        return {
            "status": "disabled",
            "provider": None,
            "model": None,
            "semantic_review_enabled": False,
            "reviews": [],
            "errors": [],
            "debug": [
                {
                    "relationship_id": rel["id"],
                    "candidate_payload": _relationship_candidate(rel),
                    "raw_output": None,
                    "normalized_output": None,
                    "validated_output": None,
                    "fallback_reason": "Semantic review is disabled by feature flag.",
                }
                for rel in relationships
            ],
        }

    settings = load_ai_settings()
    if not settings:
        return {
            "status": "ai_unavailable",
            "provider": None,
            "model": None,
            "semantic_review_enabled": True,
            "reviews": [],
            "errors": [{"relationship_id": rel["id"], "error": "No AI settings were found for semantic review."} for rel in relationships],
            "debug": [
                {
                    "relationship_id": rel["id"],
                    "candidate_payload": _relationship_candidate(rel),
                    "raw_output": None,
                    "normalized_output": None,
                    "validated_output": None,
                    "fallback_reason": "No AI settings were found for semantic review.",
                }
                for rel in relationships
            ],
        }

    client = AIClient(
        api_key=settings.api_key,
        base_url=settings.base_url,
        model=model or settings.model,
    )
    reviews: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    debug: list[dict[str, Any]] = []
    for rel in relationships:
        if rel.get("semantic_review_source") in {"user_edited", "user_added"} or rel.get("origin") in {"user_edited", "user_added"}:
            debug.append(
                {
                    "relationship_id": rel["id"],
                    "candidate_payload": _relationship_candidate(rel),
                    "raw_output": None,
                    "normalized_output": None,
                    "validated_output": None,
                    "fallback_reason": "Skipped semantic review because this relationship was manually edited in the UI.",
                }
            )
            continue
        candidate = _relationship_candidate(rel)
        raw_output: str | None = None
        normalized: dict[str, Any] | None = None
        prompt_payload = {
            "task": "Review one rule-based graph relationship candidate and return a semantically improved relationship direction and type.",
            "candidate": candidate,
            "instructions": [
                "Return strict JSON only.",
                "Do not introduce new tables, new columns, or unsupported domain facts.",
                "Use only the evidence in the candidate payload.",
                "The candidate payload already defines one canonical base direction: base_from_table is the FK holder and base_to_table is the referenced table.",
                "You may keep that base direction or reverse it semantically.",
                "Prefer active voice. If the natural phrase would be passive, reverse the edge and use an active relationship name instead.",
                "Generalize from the abstract few-shot examples. Do not memorize or rely on specific table names from the examples.",
                "Keep the explanation short and concrete.",
            ],
            "output_example": {
                "relationship_id": candidate["relationship_id"],
                "final_from_table": candidate["base_direction"]["base_to_table"],
                "final_to_table": candidate["base_direction"]["base_from_table"],
                "relationship_type": "CREATED",
                "direction": "reversed",
                "explanation": "The foreign key is stored on the base_from_table side, but reversing to actor->artifact is the more natural graph relation.",
                "confidence": 0.91,
            },
        }
        logger.info("semantic reviewer input for %s: %s", rel["id"], json.dumps(prompt_payload, ensure_ascii=False))
        try:
            raw_output, parsed_output = client.complete_json(
                prompt_payload,
                system_prompt="You are reviewing one graph relationship candidate. Return strict JSON only. Do not include markdown or chain-of-thought.",
            )
            logger.info("semantic reviewer raw output for %s: %s", rel["id"], raw_output)
            normalized = _normalize_semantic_payload(parsed_output)
            validated = _validate_semantic_result(candidate, normalized)
            logger.info("semantic reviewer validated result for %s: %s", rel["id"], json.dumps(validated, ensure_ascii=False))
            reviews.append(validated)
            debug.append(
                {
                    "relationship_id": rel["id"],
                    "candidate_payload": candidate,
                    "raw_output": raw_output,
                    "normalized_output": normalized,
                    "validated_output": validated,
                    "fallback_reason": None,
                }
            )
        except Exception as exc:
            fallback_reason = f"Semantic review validation failed: {exc}"
            errors.append({"relationship_id": rel["id"], "error": fallback_reason})
            logger.warning("semantic reviewer failed for %s: %s", rel["id"], exc)
            debug.append(
                {
                    "relationship_id": rel["id"],
                    "candidate_payload": candidate,
                    "raw_output": raw_output,
                    "normalized_output": normalized,
                    "validated_output": None,
                    "fallback_reason": fallback_reason,
                }
            )

    return {
        "status": "completed",
        "provider": settings.provider,
        "model": model or settings.model,
        "semantic_review_enabled": True,
        "reviews": reviews,
        "errors": errors,
        "debug": debug,
    }


def apply_semantic_relationship_review(
    graph_mapping: dict[str, Any],
    semantic_review: dict[str, Any] | None,
) -> dict[str, Any]:
    updated = dict(graph_mapping)
    review_by_id = {item["relationship_id"]: item for item in (semantic_review or {}).get("reviews", [])}
    error_by_id = {item["relationship_id"]: item["error"] for item in (semantic_review or {}).get("errors", [])}
    debug_by_id = {item["relationship_id"]: item for item in (semantic_review or {}).get("debug", [])}

    ai_used = 0
    changed_direction = 0
    changed_name = 0
    fallback_count = 0
    updated_relationships: list[dict[str, Any]] = []

    for rel in graph_mapping.get("relationships", []):
        merged = dict(rel)
        raw_from = str(rel.get("raw_from_table") or rel["from_table"])
        raw_to = str(rel.get("raw_to_table") or rel["to_table"])
        raw_name = str(rel.get("raw_final_name") or rel.get("rule_based_name") or rel.get("final_name") or "RELATED_TO")
        raw_display = str(rel.get("raw_display_text") or _display_text(raw_from, raw_name, raw_to, list(rel.get("properties", []))))

        merged["raw_from_table"] = raw_from
        merged["raw_to_table"] = raw_to
        merged["raw_final_name"] = raw_name
        merged["raw_display_text"] = raw_display
        merged["raw_classification"] = rel.get("raw_classification", rel.get("classification"))
        merged["raw_via_table"] = rel.get("raw_via_table", rel.get("via_table"))
        if "raw_from_columns" in rel or "from_columns" in rel:
            merged["raw_from_columns"] = list(rel.get("raw_from_columns", rel.get("from_columns", [])))
        if "raw_to_columns" in rel or "to_columns" in rel:
            merged["raw_to_columns"] = list(rel.get("raw_to_columns", rel.get("to_columns", [])))
        if "raw_from_join_columns" in rel or "from_join_columns" in rel:
            merged["raw_from_join_columns"] = list(rel.get("raw_from_join_columns", rel.get("from_join_columns", [])))
        if "raw_to_join_columns" in rel or "to_join_columns" in rel:
            merged["raw_to_join_columns"] = list(rel.get("raw_to_join_columns", rel.get("to_join_columns", [])))
        merged["semantic_candidate_payload"] = debug_by_id.get(rel["id"], {}).get("candidate_payload")
        merged["semantic_raw_output"] = debug_by_id.get(rel["id"], {}).get("raw_output")
        merged["semantic_normalized_output"] = debug_by_id.get(rel["id"], {}).get("normalized_output")

        review = review_by_id.get(rel["id"])
        if review:
            semantic_from = review["from_table"]
            semantic_to = review["to_table"]
            semantic_name = review["relationship_type"]
            merged["from_table"] = semantic_from
            merged["to_table"] = semantic_to
            merged["final_name"] = semantic_name
            merged["display_text"] = _display_text(semantic_from, semantic_name, semantic_to, list(rel.get("properties", [])))
            merged["semantic_from_table"] = semantic_from
            merged["semantic_to_table"] = semantic_to
            merged["semantic_relationship_type"] = semantic_name
            merged["semantic_display_text"] = merged["display_text"]
            merged["semantic_direction"] = review["direction"]
            merged["semantic_explanation"] = review["explanation"]
            merged["semantic_confidence"] = review["confidence"]
            merged["confidence"] = review["confidence"]
            merged["semantic_review_source"] = "ai_review"
            merged["semantic_fallback_reason"] = None
            merged["origin"] = "ai_refined" if (semantic_name != raw_name or semantic_from != raw_from or semantic_to != raw_to) else "ai_accepted"
            ai_used += 1
            if semantic_name != raw_name:
                changed_name += 1
            if semantic_from != raw_from or semantic_to != raw_to:
                changed_direction += 1
        elif rel.get("semantic_review_source") in {"user_edited", "user_added"} or rel.get("origin") in {"user_edited", "user_added"}:
            semantic_from = str(rel.get("semantic_from_table") or rel.get("final_from_table") or rel.get("from_table"))
            semantic_to = str(rel.get("semantic_to_table") or rel.get("final_to_table") or rel.get("to_table"))
            semantic_name = str(rel.get("semantic_relationship_type") or rel.get("final_name") or raw_name)
            merged["from_table"] = semantic_from
            merged["to_table"] = semantic_to
            merged["final_name"] = semantic_name
            merged["display_text"] = _display_text(semantic_from, semantic_name, semantic_to, list(rel.get("properties", [])))
            merged["semantic_from_table"] = semantic_from
            merged["semantic_to_table"] = semantic_to
            merged["final_from_table"] = semantic_from
            merged["final_to_table"] = semantic_to
            merged["semantic_relationship_type"] = semantic_name
            merged["semantic_display_text"] = merged["display_text"]
            merged["semantic_direction"] = str(rel.get("semantic_direction") or ("reversed" if (semantic_from != raw_from or semantic_to != raw_to) else "kept"))
            merged["semantic_explanation"] = str(rel.get("semantic_explanation") or "Manually edited in the UI.")
            merged["semantic_confidence"] = rel.get("semantic_confidence", rel.get("confidence"))
            merged["semantic_review_source"] = str(rel.get("semantic_review_source") or rel.get("origin") or "user_edited")
            merged["semantic_fallback_reason"] = None
            merged["origin"] = str(rel.get("origin") or "user_edited")
        else:
            merged["semantic_from_table"] = raw_from
            merged["semantic_to_table"] = raw_to
            merged["semantic_relationship_type"] = raw_name
            merged["semantic_display_text"] = raw_display
            merged["semantic_direction"] = "kept"
            merged["semantic_fallback_reason"] = error_by_id.get(rel["id"]) or debug_by_id.get(rel["id"], {}).get("fallback_reason") or "Semantic review did not return a valid result."
            merged["semantic_explanation"] = merged["semantic_fallback_reason"]
            merged["semantic_confidence"] = rel.get("confidence")
            merged["semantic_review_source"] = "rule_based"
            merged["origin"] = merged.get("origin", merged.get("classification", "deterministic"))
            fallback_count += 1

        updated_relationships.append(merged)

    updated["relationships"] = updated_relationships
    updated["semantic_review_summary"] = {
        "enabled": bool((semantic_review or {}).get("semantic_review_enabled")),
        "status": (semantic_review or {}).get("status", "not_run"),
        "provider": (semantic_review or {}).get("provider"),
        "model": (semantic_review or {}).get("model"),
        "relationships_total": len(updated_relationships),
        "ai_reviewed_relationships": ai_used,
        "fallback_relationships": fallback_count,
        "renamed_relationships": changed_name,
        "reversed_relationships": changed_direction,
        "errors": list((semantic_review or {}).get("errors", [])),
        "debug": list((semantic_review or {}).get("debug", [])),
    }
    return updated
