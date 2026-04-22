from __future__ import annotations

from typing import Any

from ai.client import AIClient
from ai.config import load_ai_settings


def _normalize_reviews_payload(response: Any) -> list[dict[str, Any]]:
    """Normalize model output into a list of review objects.

    Some providers/models may return:
    - {"reviews": [...]} (preferred)
    - [...] (bare list)
    - {"data": [...]} / {"items": [...]} fallback wrappers
    """
    if isinstance(response, dict):
        if isinstance(response.get("reviews"), list):
            return [item for item in response["reviews"] if isinstance(item, dict)]
        for alt_key in ("data", "items"):
            value = response.get(alt_key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        return []

    if isinstance(response, list):
        return [item for item in response if isinstance(item, dict)]

    return []


def run_ai_mapping_review(
    graph_mapping: dict[str, Any],
    model: str | None = None,
    *,
    review_relationships: bool = True,
) -> dict[str, Any] | None:
    """Return AI-reviewed graph mapping or None when AI is unavailable."""
    settings = load_ai_settings()
    if not settings:
        return None
    client = AIClient(
        api_key=settings.api_key,
        base_url=settings.base_url,
        model=model or settings.model,
    )
    payload = {
        "task": "Review the full graph mapping and return the final approved nodes, relationships, join tables, and ambiguous cases",
        "requirements": {
            "keys": [
                "object_type",
                "object_id",
                "include",
                "label",
                "final_name",
                "from_table",
                "to_table",
                "from_columns",
                "to_columns",
                "properties",
                "classification",
                "confidence",
                "reasoning_summary",
            ],
            "instructions": [
                "Review nodes, relationships, join tables, and ambiguous cases together.",
                "For nodes, you may refine labels, properties, classification, and confidence.",
                "For relationships, you may refine direction, name, column mappings, properties, classification, and confidence."
                if review_relationships
                else "Relationship semantics are handled by a separate semantic reviewer in this pipeline; do not review relationships here.",
                "For join tables, you may refine linked tables, join columns, non-key columns, classification, and confidence.",
                "For ambiguous cases, you may keep, refine, or remove the case.",
                "Set `include` to false for objects that should be removed, except direct foreign-key relationships which should always be preserved.",
                "Keep column arrays aligned by position.",
            ],
        },
        "nodes": [
            {
                "object_type": "node",
                "object_id": f"node::{node['table']}",
                "table": node["table"],
                "label": node["label"],
                "primary_key": node.get("primary_key", []),
                "properties": node.get("properties", []),
                "classification": node.get("classification"),
                "confidence": node.get("confidence"),
                "signals": node.get("signals", []),
            }
            for node in graph_mapping.get("nodes", [])
        ],
        "relationships": [
            {
                "object_type": "relationship",
                "object_id": rel["id"],
                "from_table": rel["from_table"],
                "to_table": rel["to_table"],
                "via_table": rel.get("via_table"),
                "final_name": rel.get("final_name"),
                "from_columns": rel.get("from_join_columns", rel.get("from_columns", [])),
                "to_columns": rel.get("to_join_columns", rel.get("to_columns", [])),
                "properties": rel.get("properties", []),
                "classification": rel.get("classification"),
                "confidence": rel.get("confidence"),
                "signals": rel.get("signals", []),
            }
            for rel in graph_mapping.get("relationships", [])
        ] if review_relationships else [],
        "join_tables": [
            {
                "object_type": "join_table",
                "object_id": f"join::{item['table']}",
                **item,
            }
            for item in graph_mapping.get("join_tables", [])
        ],
        "ambiguous_cases": [
            {
                "object_type": "ambiguous_case",
                "object_id": item.get("object_id"),
                **item,
            }
            for item in graph_mapping.get("ambiguous_cases", [])
        ],
    }

    response = client.review_mapping(payload)
    reviews = _normalize_reviews_payload(response)
    return {"provider": settings.provider, "model": model or settings.model, "reviews": reviews}


def apply_ai_review(graph_mapping: dict[str, Any], ai_review: dict[str, Any] | None) -> dict[str, Any]:
    """Apply AI-reviewed graph mapping while preserving direct foreign keys."""
    if not ai_review:
        updated = dict(graph_mapping)
        updated["review_status"] = "deterministic_only"
        updated["review_provider"] = None
        updated["review_model"] = None
        updated["ai_review_summary"] = {
            "ai_reviewed": False,
            "nodes_reviewed": len(graph_mapping.get("nodes", [])),
            "join_tables_reviewed": len(graph_mapping.get("join_tables", [])),
            "relationships_reviewed": len(graph_mapping.get("relationships", [])),
            "nodes_in_scope": len(graph_mapping.get("nodes", [])),
            "join_tables_in_scope": len(graph_mapping.get("join_tables", [])),
            "relationships_in_scope": len(graph_mapping.get("relationships", [])),
            "node_refinements": 0,
            "join_table_refinements": 0,
            "relationship_refinements": 0,
            "renamed_relationships": 0,
            "ambiguous_cases_flagged": len(graph_mapping.get("ambiguous_cases", [])),
            "rule_based_names_accepted": len(graph_mapping.get("relationships", [])),
        }
        return updated

    review_by_id = {item.get("object_id"): item for item in ai_review.get("reviews", [])}
    review_counts = {
        "node": 0,
        "join_table": 0,
        "relationship": 0,
        "ambiguous_case": 0,
    }
    for item in ai_review.get("reviews", []):
        object_type = item.get("object_type")
        object_id = str(item.get("object_id", ""))
        if object_type not in review_counts:
            if object_id.startswith("node::"):
                object_type = "node"
            elif object_id.startswith("join::"):
                object_type = "join_table"
            elif object_id.startswith("ambiguous::"):
                object_type = "ambiguous_case"
            else:
                object_type = "relationship"
        review_counts[object_type] = review_counts.get(object_type, 0) + 1

    updated = dict(graph_mapping)
    updated_nodes: list[dict[str, Any]] = []
    updated_relationships: list[dict[str, Any]] = []
    updated_join_tables: list[dict[str, Any]] = []
    updated_ambiguous_cases: list[dict[str, Any]] = []
    renamed_relationships = 0
    rule_based_names_accepted = 0

    for node in graph_mapping.get("nodes", []):
        merged = dict(node)
        review = review_by_id.get(f"node::{node.get('table')}")
        if review:
            if review.get("include", True) is False:
                continue
            merged["label"] = review.get("label") or merged.get("label")
            if isinstance(review.get("properties"), list):
                merged["properties"] = review["properties"]
            merged["classification"] = review.get("classification") or merged.get("classification")
            merged["confidence"] = review.get("confidence", merged.get("confidence"))
            merged["ai_reasoning_summary"] = review.get("reasoning_summary")
            merged["origin"] = "ai_refined"
            merged["review_status"] = "ai_reviewed"
        updated_nodes.append(merged)

    for rel in graph_mapping.get("relationships", []):
        merged = dict(rel)
        review = review_by_id.get(rel.get("id"))
        if review:
            if review.get("include", True) is False and merged.get("classification") != "direct_fk_relationship":
                continue
            merged["final_name"] = review.get("final_name") or merged.get("final_name")
            merged["from_table"] = review.get("from_table") or merged.get("from_table")
            merged["to_table"] = review.get("to_table") or merged.get("to_table")
            review_from_columns = review.get("from_columns")
            review_to_columns = review.get("to_columns")
            if merged.get("classification") == "collapsed_join_table":
                if isinstance(review_from_columns, list):
                    merged["from_join_columns"] = review_from_columns
                if isinstance(review_to_columns, list):
                    merged["to_join_columns"] = review_to_columns
            else:
                if isinstance(review_from_columns, list):
                    merged["from_columns"] = review_from_columns
                if isinstance(review_to_columns, list):
                    merged["to_columns"] = review_to_columns
            merged["confidence"] = review.get("confidence", merged.get("confidence"))
            merged["ai_reasoning_summary"] = review.get("reasoning_summary")
            merged["display_text"] = (
                f"{merged['from_table'].rstrip('s').title()} {merged['final_name']} {merged['to_table'].rstrip('s').title()}"
            )
            if merged["final_name"] != rel.get("final_name"):
                renamed_relationships += 1
                merged["origin"] = "ai_refined"
            else:
                rule_based_names_accepted += 1
                merged["origin"] = merged.get("origin", "ai_accepted")
            merged["review_status"] = "ai_reviewed"
        else:
            merged["review_status"] = merged.get("review_status", "deterministic_only")
            merged["origin"] = merged.get("origin", merged.get("classification", "deterministic"))
        updated_relationships.append(merged)

    for item in graph_mapping.get("join_tables", []):
        merged = dict(item)
        review = review_by_id.get(f"join::{item.get('table')}")
        if review:
            if review.get("include", True) is False:
                continue
            if isinstance(review.get("linked_tables"), list):
                merged["linked_tables"] = review["linked_tables"]
            if isinstance(review.get("join_columns"), list):
                merged["join_columns"] = review["join_columns"]
            if isinstance(review.get("non_key_columns"), list):
                merged["non_key_columns"] = review["non_key_columns"]
            merged["classification"] = review.get("classification") or merged.get("classification", "join_table")
            merged["confidence"] = review.get("confidence", merged.get("confidence"))
            merged["ai_reasoning_summary"] = review.get("reasoning_summary")
            merged["origin"] = "ai_refined"
            merged["review_status"] = "ai_reviewed"
        updated_join_tables.append(merged)

    for item in graph_mapping.get("ambiguous_cases", []):
        merged = dict(item)
        review = review_by_id.get(item.get("object_id"))
        if review:
            if review.get("include", True) is False:
                continue
            merged["reason"] = review.get("reasoning_summary") or merged.get("reason")
            merged["origin"] = "ai_reviewed"
            merged["review_status"] = "ai_reviewed"
        updated_ambiguous_cases.append(merged)

    updated["nodes"] = updated_nodes
    updated["relationships"] = updated_relationships
    updated["join_tables"] = updated_join_tables
    updated["ambiguous_cases"] = updated_ambiguous_cases
    updated["review_status"] = "ai_reviewed"
    updated["review_provider"] = ai_review.get("provider")
    updated["review_model"] = ai_review.get("model")
    updated["ai_review_summary"] = {
        "ai_reviewed": True,
        "nodes_reviewed": len(updated_nodes),
        "join_tables_reviewed": len(updated_join_tables),
        "relationships_reviewed": len(updated_relationships),
        "nodes_in_scope": len(updated_nodes),
        "join_tables_in_scope": len(updated_join_tables),
        "relationships_in_scope": len(updated_relationships),
        "node_refinements": review_counts.get("node", 0),
        "join_table_refinements": review_counts.get("join_table", 0),
        "relationship_refinements": review_counts.get("relationship", 0),
        "renamed_relationships": renamed_relationships,
        "ambiguous_cases_flagged": len(updated_ambiguous_cases),
        "rule_based_names_accepted": rule_based_names_accepted,
    }
    return updated
