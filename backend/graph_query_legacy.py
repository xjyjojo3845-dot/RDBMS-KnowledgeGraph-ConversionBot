from __future__ import annotations

from typing import Any


def _gq():
    from backend import graph_query as gq

    return gq


def classify_query_type(pre, query_schema: dict[str, Any]):
    gq = _gq()
    normalized = pre.normalized_question
    requested_rel_prop = None
    for rel in pre.relationship_mentions or query_schema["relationships"]:
        requested_rel_prop = gq._requested_relationship_property(pre.raw_question, rel)
        if requested_rel_prop:
            break

    relationship_filter_match = False
    for rel in pre.relationship_mentions or query_schema["relationships"]:
        if gq._extract_relationship_property_filters(pre.raw_question, rel):
            relationship_filter_match = True
            break

    one_hop_signal = bool(
        pre.relationship_mentions
        or len(pre.node_mentions) >= 2
        or any(gq._extract_relation_object_phrase(pre.raw_question, rel) for rel in pre.relationship_mentions)
    )

    if requested_rel_prop:
        return gq.QueryTypeDecision("relationship_property", "Question asks for a relationship property value.")

    if pre.hop_limit and pre.hop_limit > 2:
        return gq.QueryTypeDecision("constrained_multi_hop", "Question explicitly constrains hop count.")

    if "two hop" in normalized or "2 hop" in normalized:
        return gq.QueryTypeDecision("two_hop", "Question explicitly asks for a two-hop traversal.")

    if ("through" in normalized or "via" in normalized) and len(pre.node_mentions) >= 4:
        return gq.QueryTypeDecision("fixed_multi_hop", "Question names an explicit multi-hop entity sequence.")

    if ("through" in normalized or "via" in normalized) and len(pre.node_mentions) >= 2:
        return gq.QueryTypeDecision("two_hop", "Question implies a constrained path through an intermediate entity.")

    if one_hop_signal:
        return gq.QueryTypeDecision("one_hop", "Question references a direct relationship pattern.")

    if relationship_filter_match:
        return gq.QueryTypeDecision("relationship_property", "Question filters on relationship properties.")

    return gq.QueryTypeDecision("entity_lookup", "Defaulting to entity lookup.")


def _infer_answer_label_from_question(pre, query_schema: dict[str, Any]) -> str | None:
    gq = _gq()
    normalized = pre.normalized_question
    if normalized.startswith("who "):
        for rel in pre.relationship_mentions:
            endpoints = [
                next((node for node in query_schema["nodes"] if node["label"] == rel["from_label"]), None),
                next((node for node in query_schema["nodes"] if node["label"] == rel["to_label"]), None),
            ]
            for endpoint in endpoints:
                if endpoint and gq._person_like_node(endpoint):
                    return endpoint["label"]
    return pre.node_mentions[0]["label"] if pre.node_mentions else None


def _build_intent_contract(
    decision,
    query_schema: dict[str, Any],
    *,
    answer_label: str | None,
    target_label: str | None,
    relationship_type: str | None,
    requested_rel_prop: str | None,
    answer_value: str | None,
    target_value: str | None,
    answer_property: str | None,
    target_property: str | None,
    relationship_filters: list[dict[str, Any]],
    explicit_node_sequence: list[str],
    hop_limit: int | None,
):
    gq = _gq()
    source_ref = gq._make_entity_ref(answer_label, query_schema)
    target_ref = gq._make_entity_ref(target_label, query_schema)
    relationship_ref = gq._make_relationship_ref(relationship_type, query_schema)
    source_filters = gq._filters_from_value("source", answer_property, answer_value)
    target_filters = gq._filters_from_value("target", target_property, target_value)
    rel_filters = gq._filters_from_relationship_items(relationship_filters)

    if decision.query_type == "entity_lookup" and source_ref:
        return gq.EntityLookupIntent(target_entity=source_ref, filters=source_filters, return_fields=[], limit=25)

    if decision.query_type == "one_hop" and source_ref and target_ref and relationship_ref:
        return gq.OneHopIntent(
            source_entity=source_ref,
            target_entity=target_ref,
            relationship=relationship_ref,
            source_filters=source_filters,
            target_filters=target_filters,
            relationship_filters=rel_filters,
            return_fields=[],
            limit=25,
        )

    if decision.query_type == "relationship_property" and source_ref and target_ref and relationship_ref:
        return gq.RelationshipPropertyIntent(
            source_entity=source_ref,
            target_entity=target_ref,
            relationship=relationship_ref,
            requested_relationship_property=requested_rel_prop,
            source_filters=source_filters,
            target_filters=target_filters,
            relationship_filters=rel_filters,
            limit=25,
        )

    if decision.query_type == "two_hop" and len(explicit_node_sequence) >= 3:
        source_entity = gq._make_entity_ref(explicit_node_sequence[0], query_schema)
        middle_entity = gq._make_entity_ref(explicit_node_sequence[1], query_schema)
        target_entity = gq._make_entity_ref(explicit_node_sequence[-1], query_schema)
        path_refs = []
        for left, right in zip(explicit_node_sequence, explicit_node_sequence[1:]):
            edge = next((item for item in gq._graph_edges(query_schema) if item["from_label"] == left and item["to_label"] == right), None)
            if edge:
                path_refs.append(gq.RelationshipRef(type=edge["relationship"]["type"], from_label=left, to_label=right))
        if source_entity and middle_entity and target_entity and len(path_refs) >= 2:
            return gq.TwoHopIntent(
                source_entity=source_entity,
                middle_entity=middle_entity,
                target_entity=target_entity,
                path=path_refs[:2],
                source_filters=source_filters,
                middle_filters=[],
                target_filters=target_filters,
                relationship_filters=rel_filters,
                return_fields=[],
                limit=10,
            )

    if decision.query_type == "fixed_multi_hop" and len(explicit_node_sequence) >= 3:
        entities = [ref for label in explicit_node_sequence if (ref := gq._make_entity_ref(label, query_schema))]
        relationships = []
        for left, right in zip(explicit_node_sequence, explicit_node_sequence[1:]):
            edge = next((item for item in gq._graph_edges(query_schema) if item["from_label"] == left and item["to_label"] == right), None)
            if edge:
                relationships.append(gq.RelationshipRef(type=edge["relationship"]["type"], from_label=left, to_label=right))
        if entities and relationships:
            return gq.FixedMultiHopIntent(
                path_template_id="_".join(label.lower() for label in explicit_node_sequence),
                entities=entities,
                relationships=relationships,
                filters=rel_filters,
                return_fields=[],
                limit=25,
            )

    if decision.query_type == "constrained_multi_hop" and len(explicit_node_sequence) >= 2:
        source_entity = gq._make_entity_ref(explicit_node_sequence[0], query_schema)
        target_entity = gq._make_entity_ref(explicit_node_sequence[-1], query_schema)
        required_entities = [ref for label in explicit_node_sequence[1:-1] if (ref := gq._make_entity_ref(label, query_schema))]
        if source_entity and target_entity:
            return gq.ConstrainedMultiHopIntent(
                source_entity=source_entity,
                target_entity=target_entity,
                max_hops=hop_limit or 4,
                required_entities=required_entities,
                allowed_relationship_types=[item for item in ([relationship_type] if relationship_type else []) if item],
                filters=rel_filters,
                return_fields=[],
                limit=10,
            )

    return None


def extract_intent(pre, decision, query_schema: dict[str, Any]):
    gq = _gq()
    rel = pre.relationship_mentions[0] if pre.relationship_mentions else None
    answer_label = _infer_answer_label_from_question(pre, query_schema)
    target_label = pre.node_mentions[-1]["label"] if len(pre.node_mentions) >= 2 else None
    answer_value = pre.quoted_values[0] if pre.quoted_values else pre.person_token
    target_value = None
    requested_rel_prop = None

    if rel:
        requested_rel_prop = gq._requested_relationship_property(pre.raw_question, rel)
        anchor_phrase = gq._extract_relation_anchor_phrase(pre.raw_question, rel)
        object_phrase = gq._extract_relation_object_phrase(pre.raw_question, rel)
        target_value = object_phrase or gq._extract_trailing_target_phrase(pre.raw_question)
        if decision.query_type == "one_hop":
            from_node = next((n for n in query_schema["nodes"] if n["label"] == rel["from_label"]), None)
            to_node = next((n for n in query_schema["nodes"] if n["label"] == rel["to_label"]), None)
            anchor_value = anchor_phrase or answer_value
            asked_label = pre.node_mentions[0]["label"] if pre.node_mentions else None
            if answer_value and from_node and to_node:
                if gq._person_like_node(from_node) and not gq._person_like_node(to_node):
                    answer_label = from_node["label"]
                    target_label = to_node["label"]
                elif gq._person_like_node(to_node) and not gq._person_like_node(from_node):
                    answer_label = to_node["label"]
                    target_label = from_node["label"]
            if normalized := pre.normalized_question:
                if normalized.startswith("who ") and gq._person_like_node(next((n for n in query_schema["nodes"] if n["label"] == rel["from_label"]), {"properties": []})):
                    answer_label = rel["from_label"]
                    target_label = rel["to_label"]
                elif answer_label == rel["to_label"]:
                    target_label = rel["from_label"]
                else:
                    answer_label = answer_label or rel["from_label"]
                    target_label = target_label or rel["to_label"]
            if anchor_phrase and from_node and to_node:
                if asked_label == from_node["label"]:
                    answer_label = to_node["label"]
                    target_label = from_node["label"]
                elif asked_label == to_node["label"]:
                    answer_label = from_node["label"]
                    target_label = to_node["label"]
                elif answer_label == to_node["label"]:
                    answer_label = to_node["label"]
                    target_label = from_node["label"]
                else:
                    answer_label = from_node["label"]
                    target_label = to_node["label"]
            if len(pre.node_mentions) == 1 and pre.quoted_values:
                only_node = pre.node_mentions[0]
                if only_node["label"] == rel["to_label"]:
                    answer_label = rel["from_label"]
                    target_label = rel["to_label"]
                elif only_node["label"] == rel["from_label"]:
                    answer_label = rel["from_label"]
                    target_label = rel["to_label"]
            if anchor_phrase:
                answer_value = anchor_phrase
                target_value = None
            else:
                answer_value = anchor_value
                if answer_value and target_value and gq._normalize(str(answer_value)) == gq._normalize(str(target_value)):
                    target_value = None
    elif decision.query_type == "one_hop" and len(pre.node_mentions) >= 2 and answer_value:
        left, right = pre.node_mentions[0], pre.node_mentions[1]
        if gq._person_like_node(left) and not gq._person_like_node(right):
            answer_label = left["label"]
            target_label = right["label"]
        elif gq._person_like_node(right) and not gq._person_like_node(left):
            answer_label = right["label"]
            target_label = left["label"]

    if decision.query_type == "entity_lookup":
        answer_label = answer_label or (pre.node_mentions[0]["label"] if pre.node_mentions else None)
        if answer_value and answer_label:
            node = next((item for item in query_schema["nodes"] if item["label"] == answer_label), None)
            if node:
                if gq._is_email_like(answer_value) and "email" in gq._preferred_node_properties(node):
                    answer_property = "email"
                elif pre.person_token and gq._person_like_node(node):
                    answer_property = "first_name"
                else:
                    answer_property = gq._preferred_text_filter_property(node)
            else:
                answer_property = None
        else:
            answer_property = None
        return gq.ExtractedIntent(
            query_type=decision.query_type,
            intent_contract=_build_intent_contract(
                decision,
                query_schema,
                answer_label=answer_label,
                target_label=None,
                relationship_type=None,
                requested_rel_prop=None,
                answer_value=answer_value,
                target_value=None,
                answer_property=answer_property,
                target_property=None,
                relationship_filters=[],
                explicit_node_sequence=[node["label"] for node in pre.node_mentions],
                hop_limit=pre.hop_limit,
            ),
            anchor_label=answer_label,
            return_label=target_label,
            anchor_value=answer_value,
            answer_label=answer_label,
            answer_value=answer_value,
            answer_property=answer_property,
            relationship_filters=[],
            explicit_node_sequence=[node["label"] for node in pre.node_mentions],
            return_hint="entity",
        )

    relationship_filters = []
    rel_pool = pre.relationship_mentions or query_schema["relationships"]
    for candidate_rel in rel_pool:
        filters = gq._extract_relationship_property_filters(pre.raw_question, candidate_rel)
        if filters:
            relationship_filters = filters
            if rel is None:
                rel = candidate_rel
            break

    if decision.query_type == "relationship_property":
        if not requested_rel_prop:
            for candidate_rel in rel_pool:
                requested_rel_prop = gq._requested_relationship_property(pre.raw_question, candidate_rel)
                if requested_rel_prop:
                    rel = candidate_rel
                    break
        target_value = target_value or gq._extract_trailing_target_phrase(pre.raw_question)
        if rel:
            answer_label = answer_label or rel["from_label"]
            target_label = target_label or rel["to_label"]
        elif requested_rel_prop:
            matching_rels = [candidate for candidate in query_schema["relationships"] if requested_rel_prop in candidate.get("properties", [])]
            if len(matching_rels) == 1:
                rel = matching_rels[0]
                answer_label = answer_label or rel["from_label"]
                target_label = target_label or rel["to_label"]
        elif relationship_filters:
            matching_rels = [
                candidate
                for candidate in query_schema["relationships"]
                if all(item.get("property") in candidate.get("properties", []) for item in relationship_filters)
            ]
            if len(matching_rels) == 1:
                rel = matching_rels[0]
                answer_label = answer_label or rel["from_label"]
                target_label = target_label or rel["to_label"]

    explicit_node_sequence = [node["label"] for node in pre.node_mentions]

    answer_property = None
    target_property = None
    if answer_value and answer_label:
        node = next((item for item in query_schema["nodes"] if item["label"] == answer_label), None)
        if node:
            if gq._is_email_like(answer_value) and "email" in gq._preferred_node_properties(node):
                answer_property = "email"
            elif pre.person_token and gq._person_like_node(node):
                answer_property = "first_name"
            else:
                answer_property = gq._preferred_text_filter_property(node)
    if target_value and target_label:
        target_node = next((item for item in query_schema["nodes"] if item["label"] == target_label), None)
        if target_node:
            target_property = gq._preferred_text_filter_property(target_node)

    return gq.ExtractedIntent(
        query_type=decision.query_type,
        intent_contract=_build_intent_contract(
            decision,
            query_schema,
            answer_label=answer_label,
            target_label=target_label,
            relationship_type=rel["type"] if rel else None,
            requested_rel_prop=requested_rel_prop,
            answer_value=answer_value,
            target_value=target_value,
            answer_property=answer_property,
            target_property=target_property,
            relationship_filters=relationship_filters,
            explicit_node_sequence=explicit_node_sequence,
            hop_limit=pre.hop_limit,
        ),
        anchor_label=answer_label if decision.query_type == "one_hop" else None,
        return_label=target_label if decision.query_type == "one_hop" else None,
        anchor_value=answer_value if decision.query_type == "one_hop" else None,
        return_value=target_value if decision.query_type == "one_hop" else None,
        answer_label=answer_label,
        target_label=target_label,
        requested_relationship_property=requested_rel_prop,
        relationship_type=rel["type"] if rel else None,
        answer_value=answer_value,
        target_value=target_value,
        answer_property=answer_property,
        target_property=target_property,
        relationship_filters=relationship_filters,
        hop_limit=pre.hop_limit,
        explicit_node_sequence=explicit_node_sequence,
        required_entity_labels=explicit_node_sequence[1:-1] if len(explicit_node_sequence) > 2 else [],
        allowed_relationship_types=[rel["type"] for rel in pre.relationship_mentions],
        return_hint="person" if pre.normalized_question.startswith("who ") else None,
    )


def run_registry_pipeline(question: str, query_schema: dict[str, Any]):
    gq = _gq()
    pre = gq._preprocess_question(question, query_schema)
    decision = classify_query_type(pre, query_schema)
    intent = extract_intent(pre, decision, query_schema)
    return gq._run_registry_pipeline_from_intent(
        question,
        query_schema,
        pre,
        decision,
        intent,
        planner_name="registry_pipeline",
        ai_settings=None,
    )
