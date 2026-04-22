from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from types import SimpleNamespace

from backend.graph_query import QueryPlan, _ai_generate_full_cypher, build_query_schema, build_query_schema_registry, plan_graph_query, plan_graph_query_full_ai


def _sample_graph_mapping():
    return {
        "nodes": [
            {"table": "students", "label": "Student", "primary_key": ["student_id"], "properties": ["first_name", "email"]},
            {"table": "courses", "label": "Course", "primary_key": ["course_id"], "properties": ["course_name"]},
            {"table": "instructors", "label": "Instructor", "primary_key": ["instructor_id"], "properties": ["first_name", "last_name", "email"]},
            {"table": "departments", "label": "Department", "primary_key": ["department_id"], "properties": ["department_name"]},
        ],
        "relationships": [
            {
                "from_table": "students",
                "to_table": "courses",
                "final_name": "ENROLLED_IN",
                "display_text": "Student enrolled in Course",
                "properties": ["semester", "year", "grade"],
            },
            {
                "from_table": "instructors",
                "to_table": "courses",
                "final_name": "TEACHES",
                "display_text": "Instructor teaches Course",
                "properties": ["semester"],
            },
            {
                "from_table": "courses",
                "to_table": "departments",
                "final_name": "OFFERED_BY",
                "display_text": "Course offered by Department",
                "properties": [],
            },
        ],
    }


def test_build_query_schema_exposes_labels_and_relationship_types():
    query_schema = build_query_schema(_sample_graph_mapping())
    assert "Student" in query_schema["node_labels"]
    assert "ENROLLED_IN" in query_schema["relationship_types"]


def test_build_query_schema_registry_is_separate_artifact():
    registry = build_query_schema_registry(_sample_graph_mapping())
    assert registry["registry_type"] == "query_schema_registry"
    assert registry["registry_version"] == 1
    assert len(registry["nodes"]) == 4
    assert len(registry["relationships"]) == 3
    materialized = build_query_schema(registry)
    assert "Student" in materialized["node_labels"]
    assert "ENROLLED_IN" in materialized["relationship_types"]


def test_plan_graph_query_entity_retrieval():
    registry = build_query_schema_registry(_sample_graph_mapping())
    plan = plan_graph_query("Show all students", _sample_graph_mapping(), query_registry=registry)
    assert plan.status == "ok"
    assert plan.query_type == "entity_lookup"
    assert "MATCH (n:`Student`)" in plan.cypher


def test_plan_graph_query_attribute_filter():
    plan = plan_graph_query('Find students with email "alice@example.edu"', _sample_graph_mapping())
    assert plan.status == "ok"
    assert plan.query_type == "entity_lookup"
    assert "MATCH (n:`Student`)" in plan.cypher
    assert "n.`email`" in plan.cypher
    assert "= toLower($answer_value)" in plan.cypher
    assert plan.params["answer_value"] == "alice@example.edu"


def test_plan_graph_query_debug_exposes_all_pipeline_stages():
    plan = plan_graph_query("Which students took Database Systems in Spring 2025?", _sample_graph_mapping())
    assert plan.status == "ok"
    assert plan.debug is not None
    required_stages = {
        "stage_1_preprocessor",
        "stage_2_query_type_classifier",
        "stage_3_entity_value_extractor",
        "stage_4_candidate_generator",
        "stage_5_path_resolver",
        "stage_6_constraint_binder",
        "stage_7_cypher_builder",
    }
    assert required_stages.issubset(set(plan.debug))
    assert plan.debug["stage_4_candidate_generator"]["candidate_relationship_type"] == "ENROLLED_IN"
    assert plan.debug["stage_4_candidate_generator"]["candidate_anchor_label"] == "Student"
    assert plan.debug["stage_4_candidate_generator"]["candidate_return_label"] == "Course"
    assert plan.debug["stage_3_entity_value_extractor"]["intent_contract"]["relationship"]["type"] == "ENROLLED_IN"
    assert plan.debug["stage_3_entity_value_extractor"]["anchor_label"] == "Student"
    assert plan.debug["stage_3_entity_value_extractor"]["return_label"] == "Course"
    assert plan.debug["stage_4_candidate_generator"]["intent_contract_type"] == "OneHopIntent"
    assert "clauses" in plan.debug["stage_6_constraint_binder"]
    assert plan.debug["stage_7_cypher_builder"]["cypher"] == plan.cypher


def test_plan_graph_query_relationship_property_filter():
    plan = plan_graph_query('Find enrollments with grade "A"', _sample_graph_mapping())
    assert plan.status == "ok"
    assert plan.query_type == "relationship_property"
    assert "MATCH (n0:`Student`)-[r0:`ENROLLED_IN`]->(n1:`Course`)" in plan.cypher
    assert "r0.`grade`" in plan.cypher


def test_plan_graph_query_two_hop_traversal():
    plan = plan_graph_query("Show two hop paths from students to departments", _sample_graph_mapping())
    assert plan.status == "ok"
    assert plan.query_type == "two_hop"
    assert "MATCH (n0:`Student`)-[r0:`ENROLLED_IN`]->(n1:`Course`)-[r1:`OFFERED_BY`]->(n2:`Department`)" in plan.cypher


def test_plan_graph_query_one_hop_full_name_filter():
    plan = plan_graph_query('Show courses for instructor "Ada Lovelace"', _sample_graph_mapping())
    assert plan.status == "ok"
    assert plan.query_type == "one_hop"
    assert "MATCH (n0:`Instructor`)-[r0:`TEACHES`]->(n1:`Course`)" in plan.cypher
    assert "n0.`first_name`" in plan.cypher
    assert "n0.`last_name`" in plan.cypher
    assert plan.params["source_first_name"] == "Ada"
    assert plan.params["source_last_name"] == "Lovelace"


def test_plan_graph_query_rule_based_one_hop_with_relationship_filters():
    plan = plan_graph_query("Which courses did Alice take in Fall 2024?", _sample_graph_mapping())
    assert plan.status == "ok"
    assert plan.query_type == "one_hop"
    assert "MATCH (n0:`Student`)-[r0:`ENROLLED_IN`]->(n1:`Course`)" in plan.cypher
    assert "r0.`semester`" in plan.cypher
    assert "r0.`year`" in plan.cypher


def test_plan_graph_query_relationship_property_range():
    plan = plan_graph_query('Find enrollments between 2023 and 2024 with grade "A"', _sample_graph_mapping())
    assert plan.status == "ok"
    assert plan.query_type == "relationship_property"
    assert "toInteger(r0.`year`) >=" in plan.cypher
    assert "toInteger(r0.`year`) <=" in plan.cypher
    assert "r0.`grade`" in plan.cypher


def test_plan_graph_query_relationship_property_numeric_comparison():
    mapping = _sample_graph_mapping()
    mapping["relationships"][0]["properties"] = ["score"]
    plan = plan_graph_query("Find enrollments with score over 90", mapping)
    assert plan.status == "ok"
    assert plan.query_type == "relationship_property"
    assert "toInteger(r0.`score`) >" in plan.cypher


def test_plan_graph_query_relationship_property_lookup_between_endpoints():
    mapping = _sample_graph_mapping()
    mapping["nodes"][1]["properties"] = ["course_name"]
    plan = plan_graph_query("What grade did Alice get in Introduction to Programming?", mapping)
    assert plan.status == "ok"
    assert plan.query_type == "relationship_property"
    assert "MATCH (n0:`Student`)-[r0:`ENROLLED_IN`]->(n1:`Course`)" in plan.cypher
    assert "n0.`first_name`" in plan.cypher
    assert "n1.`course_name`" in plan.cypher
    assert "r0.`grade` AS `relationship_grade`" in plan.cypher


def test_plan_graph_query_full_ai(monkeypatch):
    def fake_generate(question, query_schema, ai_settings, **kwargs):
        return {
            "status": "ok",
            "query_type": "one_hop_relation",
            "cypher": "```cypher\nMATCH (a:`Instructor`)-[r:`TEACHES`]->(b:`Course`) RETURN a.first_name AS instructor_first_name, b.course_name AS course_name LIMIT 25\n```",
            "params": {},
            "explanation": "AI generated a read-only one-hop query.",
        }

    monkeypatch.setattr("backend.graph_query._ai_generate_full_cypher", fake_generate)
    ai_settings = SimpleNamespace(api_key="x", base_url="x", model="qwen-plus")
    plan = plan_graph_query_full_ai("What courses does Professor Smith teach?", _sample_graph_mapping(), ai_settings=ai_settings)
    assert plan.status == "ok"
    assert plan.planner == "ai_full_cypher"
    assert plan.query_type == "one_hop"
    assert "```" not in plan.cypher
    assert "MATCH (a:`Instructor`)-[r:`TEACHES`]->(b:`Course`)" in plan.cypher


def test_full_ai_prompt_includes_multi_hop_ordering_rules(monkeypatch):
    captured = {}

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def complete_json(self, payload):
            captured["payload"] = payload
            return "", {"status": "unsupported", "explanation": "captured"}

    monkeypatch.setattr("backend.graph_query.AIClient", FakeClient)
    ai_settings = SimpleNamespace(api_key="x", base_url="x", model="qwen-plus")
    _ai_generate_full_cypher(
        "Which instructors teach courses that Alice Johnson is enrolled in?",
        build_query_schema(_sample_graph_mapping()),
        ai_settings,
        graph_mapping=_sample_graph_mapping(),
    )
    payload = captured["payload"]
    assert any("double-check every segment" in rule for rule in payload["rules"])
    assert any("immediately preceding entity phrase" in rule for rule in payload["rules"])
    assert any("inspect every relationship segment" in rule for rule in payload["rules"])
    assert any("shared relationship properties" in rule for rule in payload["rules"])
    assert any(example.get("question") == "Which instructors from Computer Science taught Database Systems in Fall 2024?" for example in payload["few_shot_examples"])
    assert any(example.get("question") == "Which instructors were employed by the Computer Science department?" for example in payload["few_shot_examples"])


def test_plan_graph_query_full_ai_normalizes_numeric_literals(monkeypatch):
    def fake_generate(question, query_schema, ai_settings, **kwargs):
        return {
            "status": "ok",
            "query_type": "one_hop_relation",
            "cypher": "MATCH (i:Instructor)-[t:TEACHES]->(c:Course) WHERE c.course_name = 'Database Systems' AND t.semester = 'Fall' AND t.year = '2024' RETURN i.first_name, i.last_name",
            "params": {},
            "explanation": "AI generated a read-only one-hop query.",
        }

    monkeypatch.setattr("backend.graph_query._ai_generate_full_cypher", fake_generate)
    ai_settings = SimpleNamespace(api_key="x", base_url="x", model="qwen-plus")
    plan = plan_graph_query_full_ai("Who taught Database Systems in Fall 2024?", _sample_graph_mapping(), ai_settings=ai_settings)
    assert plan.status == "ok"
    assert "t.year = 2024" in plan.cypher
    assert "t.semester = 'Fall'" in plan.cypher


def test_plan_graph_query_full_ai_infers_two_hop_type_from_cypher(monkeypatch):
    def fake_generate(question, query_schema, ai_settings, **kwargs):
        return {
            "status": "ok",
            "query_type": "one_hop_relation",
            "cypher": "MATCH p=(a:`Student`)-[r1]->(m)-[r2]->(b:`Department`) RETURN a.first_name AS student_first_name, type(r1) AS first_relationship, labels(m) AS middle_labels, type(r2) AS second_relationship, b.department_name AS department_name LIMIT 10",
            "params": {},
            "explanation": "AI generated a path query.",
        }

    monkeypatch.setattr("backend.graph_query._ai_generate_full_cypher", fake_generate)
    ai_settings = SimpleNamespace(api_key="x", base_url="x", model="qwen-plus")
    plan = plan_graph_query_full_ai("Show two-hop paths from students to departments", _sample_graph_mapping(), ai_settings=ai_settings)
    assert plan.status == "ok"
    assert plan.planner == "ai_full_cypher"
    assert plan.query_type == "two_hop"


def test_plan_graph_query_full_ai_rejects_write_query(monkeypatch):
    def fake_generate(question, query_schema, ai_settings, **kwargs):
        return {
            "status": "ok",
            "query_type": "one_hop_relation",
            "cypher": "MATCH (n) DELETE n",
            "params": {},
            "explanation": "bad",
        }

    monkeypatch.setattr("backend.graph_query._ai_generate_full_cypher", fake_generate)
    ai_settings = SimpleNamespace(api_key="x", base_url="x", model="qwen-plus")
    plan = plan_graph_query_full_ai("bad", _sample_graph_mapping(), ai_settings=ai_settings)
    assert plan.status == "unsupported"
    assert plan.planner == "ai_full_cypher"


def test_plan_graph_query_full_ai_rejects_sql_like_syntax(monkeypatch):
    def fake_generate(question, query_schema, ai_settings, **kwargs):
        return {
            "status": "ok",
            "query_type": "one_hop_relation",
            "cypher": "MATCH (d:Department)-[:EMPLOYS]->(i:Instructor)-[:TEACHES]->(c:Course) WHERE d.department_name = 'Computer Science' AND i.instructor_id IN (SELECT instructor_id FROM course_instructors ci JOIN courses co ON ci.course_id = co.course_id) RETURN i.first_name, i.last_name",
            "params": {},
            "explanation": "AI generated a read-only query.",
        }

    monkeypatch.setattr("backend.graph_query._ai_generate_full_cypher", fake_generate)
    ai_settings = SimpleNamespace(api_key="x", base_url="x", model="qwen-plus")
    plan = plan_graph_query_full_ai("Which instructors from Computer Science taught Database Systems in Fall 2024?", _sample_graph_mapping(), ai_settings=ai_settings)
    assert plan.status == "unsupported"
    assert "SQL-style syntax" in plan.explanation


def test_plan_graph_query_full_ai_rejects_unknown_relationship_type(monkeypatch):
    def fake_generate(question, query_schema, ai_settings, **kwargs):
        return {
            "status": "ok",
            "query_type": "one_hop_relation",
            "cypher": "MATCH (i:`Instructor`)-[:EMPLOYS]->(d:`Department`) RETURN i.first_name, d.department_name",
            "params": {},
            "explanation": "AI generated a read-only query.",
        }

    monkeypatch.setattr("backend.graph_query._ai_generate_full_cypher", fake_generate)
    ai_settings = SimpleNamespace(api_key="x", base_url="x", model="qwen-plus")
    plan = plan_graph_query_full_ai("Which department employs each instructor?", _sample_graph_mapping(), ai_settings=ai_settings)
    assert plan.status == "unsupported"
    assert "unknown relationship type" in plan.explanation


def test_plan_graph_query_full_ai_rewrites_wrong_relationship_direction(monkeypatch):
    def fake_generate(question, query_schema, ai_settings, **kwargs):
        return {
            "status": "ok",
            "query_type": "one_hop_relation",
            "cypher": "MATCH (a:`Course`)-[r:`TEACHES`]->(b:`Instructor`) RETURN a.course_name, b.first_name",
            "params": {},
            "explanation": "AI generated a read-only one-hop query.",
        }

    monkeypatch.setattr("backend.graph_query._ai_generate_full_cypher", fake_generate)
    ai_settings = SimpleNamespace(api_key="x", base_url="x", model="qwen-plus")
    plan = plan_graph_query_full_ai("What courses does Professor Smith teach?", _sample_graph_mapping(), ai_settings=ai_settings)
    assert plan.status == "ok"
    assert "MATCH (b:`Instructor`)-[r:`TEACHES`]->(a:`Course`)" in plan.cypher
    assert "Auto-corrected relationship directions" in plan.explanation


def test_plan_graph_query_full_ai_rewrites_wrong_two_hop_directions(monkeypatch):
    def fake_generate(question, query_schema, ai_settings, **kwargs):
        return {
            "status": "ok",
            "query_type": "two_hop_relation",
            "cypher": "MATCH (a:`Student`)-[r0:`ENROLLED_IN`]->(b:`Course`)-[r1:`TEACHES`]->(c:`Instructor`) RETURN a.first_name, c.first_name",
            "params": {},
            "explanation": "AI generated a two-hop query.",
        }

    monkeypatch.setattr("backend.graph_query._ai_generate_full_cypher", fake_generate)
    ai_settings = SimpleNamespace(api_key="x", base_url="x", model="qwen-plus")
    plan = plan_graph_query_full_ai("Which instructors teach courses that Alice Johnson is enrolled in?", _sample_graph_mapping(), ai_settings=ai_settings)
    assert plan.status == "ok"
    assert "MATCH (a:`Student`)-[r0:`ENROLLED_IN`]->(b:`Course`)<-[r1:`TEACHES`]-(c:`Instructor`)" in plan.cypher


def test_plan_graph_query_full_ai_rewrites_alias_only_directions_and_adds_alignment(monkeypatch):
    mapping = _sample_graph_mapping()
    mapping["nodes"][0]["properties"] = ["first_name", "last_name", "email"]
    mapping["relationships"][1]["properties"] = ["semester", "year"]
    mapping["relationships"].append(
        {
            "from_table": "departments",
            "to_table": "students",
            "final_name": "HAS",
            "display_text": "Department has Student",
            "properties": [],
        }
    )
    mapping["relationships"].append(
        {
            "from_table": "departments",
            "to_table": "instructors",
            "final_name": "EMPLOYS",
            "display_text": "Department employs Instructor",
            "properties": [],
        }
    )

    def fake_generate(question, query_schema, ai_settings, **kwargs):
        return {
            "status": "ok",
            "query_type": "two_hop_relation",
            "cypher": (
                "MATCH (s:Student)-[e:ENROLLED_IN]->(c:Course)<-[t:TEACHES]-(i:Instructor) "
                "MATCH (s)-[:HAS]->(studentDept:Department) "
                "MATCH (i)-[:EMPLOYS]->(instructorDept:Department) "
                "WHERE studentDept.department_id <> instructorDept.department_id "
                "RETURN DISTINCT s.student_id, s.first_name, s.last_name"
            ),
            "params": {},
            "explanation": "AI generated a read-only query.",
        }

    monkeypatch.setattr("backend.graph_query._ai_generate_full_cypher", fake_generate)
    ai_settings = SimpleNamespace(api_key="x", base_url="x", model="qwen-plus")
    plan = plan_graph_query_full_ai(
        "Which students were taught by instructors from a different department than their own?",
        mapping,
        ai_settings=ai_settings,
    )
    assert plan.status == "ok"
    assert "MATCH (studentDept:Department)-[:HAS]->(s)" in plan.cypher
    assert "MATCH (instructorDept:Department)-[:EMPLOYS]->(i)" in plan.cypher
    assert "toLower(toString(e.`semester`)) = toLower(toString(t.`semester`))" in plan.cypher
    assert "toInteger(e.`year`) = toInteger(t.`year`)" in plan.cypher


def test_plan_graph_query_uses_ai_contract_fallback(monkeypatch):
    def fake_ai_extract(question, query_schema, ai_settings, **kwargs):
        return {
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
            "confidence": 0.91,
            "notes": "Professor normalized to Instructor; teach normalized to TEACHES.",
        }

    monkeypatch.setattr("backend.graph_query._ai_extract_plan", fake_ai_extract)
    ai_settings = SimpleNamespace(api_key="x", base_url="x", model="qwen-plus")
    plan = plan_graph_query("What courses does Professor Smith teach?", _sample_graph_mapping(), ai_settings=ai_settings)
    assert plan.status == "ok"
    assert plan.planner == "ai_assisted"
    assert plan.query_type == "one_hop"
    assert "MATCH (n0:`Instructor`)-[r0:`TEACHES`]->(n1:`Course`)" in plan.cypher
    assert plan.debug["ai_extraction"]["query_type"] == "one_hop"


def test_plan_graph_query_prefers_ai_contract_when_ai_is_available(monkeypatch):
    def fake_ai_extract(question, query_schema, ai_settings, **kwargs):
        return {
            "status": "ok",
            "query_type": "one_hop",
            "intent_contract": {
                "source_entity": {"label": "Student", "table": "students"},
                "target_entity": {"label": "Course", "table": "courses"},
                "relationship": {"type": "ENROLLED_IN", "from_label": "Student", "to_label": "Course"},
                "source_filters": [{"scope": "source", "property": "first_name", "operator": "=", "value": "Alice"}],
                "target_filters": [],
                "relationship_filters": [],
                "return_fields": [],
                "limit": 25,
            },
            "confidence": 0.92,
            "notes": "AI parser classified the question as one-hop.",
        }

    monkeypatch.setattr("backend.graph_query._ai_extract_plan", fake_ai_extract)
    ai_settings = SimpleNamespace(api_key="x", base_url="x", model="qwen-plus")
    plan = plan_graph_query("Which courses is Alice enrolled in?", _sample_graph_mapping(), ai_settings=ai_settings)
    assert plan.status == "ok"
    assert plan.planner == "ai_assisted"
    assert plan.query_type == "one_hop"
    assert plan.debug["ai_extraction"]["query_type"] == "one_hop"


def test_plan_graph_query_prefers_ai_contract_for_entity_lookup(monkeypatch):
    def fake_ai_extract(question, query_schema, ai_settings, **kwargs):
        return {
            "status": "ok",
            "query_type": "entity_lookup",
            "intent_contract": {
                "target_entity": {"label": "Student", "table": "students"},
                "filters": [],
                "return_fields": [],
                "limit": 25,
            },
            "confidence": 0.95,
            "notes": "AI parser classified this as entity lookup.",
        }

    monkeypatch.setattr("backend.graph_query._ai_extract_plan", fake_ai_extract)
    ai_settings = SimpleNamespace(api_key="x", base_url="x", model="qwen-plus")
    plan = plan_graph_query("Show all students", _sample_graph_mapping(), ai_settings=ai_settings)
    assert plan.status == "ok"
    assert plan.planner == "ai_assisted"
    assert plan.query_type == "entity_lookup"


def test_plan_graph_query_prefers_ai_contract_for_relationship_property(monkeypatch):
    def fake_ai_extract(question, query_schema, ai_settings, **kwargs):
        return {
            "status": "ok",
            "query_type": "relationship_property",
            "intent_contract": {
                "source_entity": {"label": "Student", "table": "students"},
                "target_entity": {"label": "Course", "table": "courses"},
                "relationship": {"type": "ENROLLED_IN", "from_label": "Student", "to_label": "Course"},
                "requested_relationship_property": "grade",
                "source_filters": [{"scope": "source", "property": "first_name", "operator": "=", "value": "Alice"}],
                "target_filters": [{"scope": "target", "property": "course_name", "operator": "=", "value": "Database Systems"}],
                "relationship_filters": [],
                "limit": 25,
            },
            "confidence": 0.93,
            "notes": "AI parser classified this as relationship property lookup.",
        }

    monkeypatch.setattr("backend.graph_query._ai_extract_plan", fake_ai_extract)
    ai_settings = SimpleNamespace(api_key="x", base_url="x", model="qwen-plus")
    plan = plan_graph_query("What grade did Alice get in Database Systems?", _sample_graph_mapping(), ai_settings=ai_settings)
    assert plan.status == "ok"
    assert plan.planner == "ai_assisted"
    assert plan.query_type == "relationship_property"
    assert "relationship_grade" in plan.cypher


def test_plan_graph_query_prefers_ai_contract_for_two_hop(monkeypatch):
    def fake_ai_extract(question, query_schema, ai_settings, **kwargs):
        return {
            "status": "ok",
            "query_type": "two_hop",
            "intent_contract": {
                "source_entity": {"label": "Student", "table": "students"},
                "middle_entity": {"label": "Course", "table": "courses"},
                "target_entity": {"label": "Department", "table": "departments"},
                "path": [
                    {"type": "ENROLLED_IN", "from_label": "Student", "to_label": "Course"},
                    {"type": "OFFERED_BY", "from_label": "Course", "to_label": "Department"},
                ],
                "source_filters": [],
                "middle_filters": [],
                "target_filters": [],
                "relationship_filters": [],
                "return_fields": [],
                "limit": 10,
            },
            "confidence": 0.9,
            "notes": "AI parser classified this as two-hop traversal.",
        }

    monkeypatch.setattr("backend.graph_query._ai_extract_plan", fake_ai_extract)
    ai_settings = SimpleNamespace(api_key="x", base_url="x", model="qwen-plus")
    plan = plan_graph_query("Show two hop paths from students to departments", _sample_graph_mapping(), ai_settings=ai_settings)
    assert plan.status == "ok"
    assert plan.planner == "ai_assisted"
    assert plan.query_type == "two_hop"
    assert "MATCH (n0:`Student`)-[r0:`ENROLLED_IN`]->(n1:`Course`)-[r1:`OFFERED_BY`]->(n2:`Department`)" in plan.cypher


def test_plan_graph_query_aligns_relationship_filters_to_correct_path_segment(monkeypatch):
    mapping = _sample_graph_mapping()
    mapping["relationships"][0]["properties"] = ["year"]
    mapping["relationships"][2]["properties"] = ["year"]

    def fake_ai_extract(question, query_schema, ai_settings, **kwargs):
        return {
            "status": "ok",
            "query_type": "two_hop",
            "intent_contract": {
                "source_entity": {"label": "Student", "table": "students"},
                "middle_entity": {"label": "Course", "table": "courses"},
                "target_entity": {"label": "Department", "table": "departments"},
                "path": [
                    {"type": "ENROLLED_IN", "from_label": "Student", "to_label": "Course"},
                    {"type": "OFFERED_BY", "from_label": "Course", "to_label": "Department"},
                ],
                "source_filters": [],
                "middle_filters": [],
                "target_filters": [],
                "relationship_filters": [
                    {"scope": "relationship", "property": "year", "operator": "=", "value": 2024, "segment_index": 0},
                    {"scope": "relationship", "property": "year", "operator": "=", "value": 2025, "segment_index": 1},
                ],
                "return_fields": [],
                "limit": 10,
            },
            "confidence": 0.9,
            "notes": "Aligned year filters to both path segments.",
        }

    monkeypatch.setattr("backend.graph_query._ai_extract_plan", fake_ai_extract)
    monkeypatch.setattr("backend.graph_query._ai_select_output_shape", lambda *args, **kwargs: None)
    ai_settings = SimpleNamespace(api_key="x", base_url="x", model="qwen-plus")
    plan = plan_graph_query("Show two hop paths from students to departments with segment years", mapping, ai_settings=ai_settings)
    assert plan.status == "ok"
    assert "toInteger(r0.`year`) = $path_0_rel_filter_0_0" in plan.cypher
    assert "toInteger(r1.`year`) = $path_1_rel_filter_0_1" in plan.cypher
    assert plan.params["path_0_rel_filter_0_0"] == 2024
    assert plan.params["path_1_rel_filter_0_1"] == 2025


def test_plan_graph_query_two_hop_edge_routes_bind_edge_properties_on_both_hops(monkeypatch):
    mapping = _sample_graph_mapping()
    mapping["relationships"][0]["properties"] = ["semester", "year"]
    mapping["relationships"][2]["properties"] = ["year"]

    def fake_ai_extract(question, query_schema, ai_settings, **kwargs):
        return {
            "status": "ok",
            "query_type": "two_hop",
            "intent_contract": {
                "source_entity": {"label": "Student", "table": "students"},
                "middle_entity": {"label": "Course", "table": "courses"},
                "target_entity": {"label": "Department", "table": "departments"},
                "path": [
                    {"type": "ENROLLED_IN", "from_label": "Student", "to_label": "Course"},
                    {"type": "OFFERED_BY", "from_label": "Course", "to_label": "Department"},
                ],
                "source_filters": [],
                "middle_filters": [],
                "target_filters": [],
                "relationship_filters": [
                    {"scope": "relationship", "property": "semester", "operator": "=", "value": "Spring", "relationship_type": "ENROLLED_IN", "segment_index": 0},
                    {"scope": "relationship", "property": "year", "operator": "=", "value": 2024, "relationship_type": "ENROLLED_IN", "segment_index": 0},
                    {"scope": "relationship", "property": "year", "operator": "=", "value": 2025, "relationship_type": "OFFERED_BY", "segment_index": 1},
                ],
                "return_fields": [],
                "limit": 10,
            },
            "confidence": 0.94,
            "notes": "Bound each relationship filter to the correct hop.",
        }

    monkeypatch.setattr("backend.graph_query._ai_extract_plan", fake_ai_extract)
    monkeypatch.setattr("backend.graph_query._ai_select_output_shape", lambda *args, **kwargs: None)
    ai_settings = SimpleNamespace(api_key="x", base_url="x", model="qwen-plus")
    plan = plan_graph_query(
        "Which departments are reached from students through courses in Spring 2024 and offered in 2025?",
        mapping,
        ai_settings=ai_settings,
    )
    assert plan.status == "ok"
    assert plan.query_type == "two_hop"
    assert "toLower(toString(r0.`semester`)) = toLower($path_0_rel_filter_0_0)" in plan.cypher
    assert "toInteger(r0.`year`) = $path_0_rel_filter_0_1" in plan.cypher
    assert "toInteger(r1.`year`) = $path_1_rel_filter_0_2" in plan.cypher


def test_plan_graph_query_two_hop_auto_aligns_shared_edge_context(monkeypatch):
    mapping = _sample_graph_mapping()
    mapping["relationships"][1]["properties"] = ["semester", "year"]

    def fake_ai_extract(question, query_schema, ai_settings, **kwargs):
        return {
            "status": "ok",
            "query_type": "fixed_multi_hop",
            "intent_contract": {
                "path_template_id": "student_course_instructor",
                "entities": [
                    {"label": "Student", "table": "students"},
                    {"label": "Course", "table": "courses"},
                    {"label": "Instructor", "table": "instructors"},
                ],
                "relationships": [
                    {"type": "ENROLLED_IN", "from_label": "Student", "to_label": "Course"},
                    {"type": "TEACHES", "from_label": "Instructor", "to_label": "Course"},
                ],
                "filters": [],
                "return_fields": [],
                "limit": 25,
            },
            "confidence": 0.93,
            "notes": "Instructor should be aligned to the same course offering context.",
        }

    monkeypatch.setattr("backend.graph_query._ai_extract_plan", fake_ai_extract)
    monkeypatch.setattr("backend.graph_query._ai_select_output_shape", lambda *args, **kwargs: None)
    ai_settings = SimpleNamespace(api_key="x", base_url="x", model="qwen-plus")
    plan = plan_graph_query(
        "Which instructors teach courses that Alice Johnson is enrolled in?",
        mapping,
        ai_settings=ai_settings,
    )
    assert plan.status == "ok"
    assert plan.query_type == "fixed_multi_hop"
    assert "toLower(toString(r0.`semester`)) = toLower(toString(r1.`semester`))" in plan.cypher
    assert "toInteger(r0.`year`) = toInteger(r1.`year`)" in plan.cypher


def test_plan_graph_query_uses_ai_contract_for_inverse_relation_wording(monkeypatch):
    def fake_ai_extract(question, query_schema, ai_settings, **kwargs):
        return {
            "status": "ok",
            "query_type": "one_hop",
            "intent_contract": {
                "source_entity": {"label": "Course", "table": "courses"},
                "target_entity": {"label": "Department", "table": "departments"},
                "relationship": {"type": "OFFERED_BY", "from_label": "Course", "to_label": "Department"},
                "source_filters": [{"scope": "source", "property": "course_name", "operator": "=", "value": "Database Systems"}],
                "target_filters": [],
                "relationship_filters": [],
                "return_fields": [],
                "limit": 25,
            },
            "confidence": 0.88,
            "notes": "belongs to normalized to OFFERED_BY in reverse wording.",
        }

    monkeypatch.setattr("backend.graph_query._ai_extract_plan", fake_ai_extract)
    ai_settings = SimpleNamespace(api_key="x", base_url="x", model="qwen-plus")
    plan = plan_graph_query("Which department does Database Systems belong to?", _sample_graph_mapping(), ai_settings=ai_settings)
    assert plan.status == "ok"
    assert plan.planner == "ai_assisted"
    assert plan.query_type == "one_hop"
    assert "MATCH (n0:`Course`)-[r0:`OFFERED_BY`]->(n1:`Department`)" in plan.cypher
    assert "n0.`course_name`" in plan.cypher


def test_plan_graph_query_uses_ai_contract_even_when_rule_one_hop_looks_valid(monkeypatch):
    def fake_ai_extract(question, query_schema, ai_settings, **kwargs):
        return {
            "status": "ok",
            "query_type": "one_hop",
            "intent_contract": {
                "source_entity": {"label": "Course", "table": "courses"},
                "target_entity": {"label": "Department", "table": "departments"},
                "relationship": {"type": "OFFERED_BY", "from_label": "Course", "to_label": "Department"},
                "source_filters": [{"scope": "source", "property": "course_name", "operator": "=", "value": "Database Systems"}],
                "target_filters": [],
                "relationship_filters": [],
                "return_fields": [],
                "limit": 25,
            },
            "confidence": 0.9,
            "notes": "Passive wording resolved to course as anchor and department as return.",
        }

    monkeypatch.setattr("backend.graph_query._ai_extract_plan", fake_ai_extract)
    ai_settings = SimpleNamespace(api_key="x", base_url="x", model="qwen-plus")
    plan = plan_graph_query("Which department is Database Systems offered by?", _sample_graph_mapping(), ai_settings=ai_settings)
    assert plan.status == "ok"
    assert plan.planner == "ai_assisted"
    assert "MATCH (n0:`Course`)-[r0:`OFFERED_BY`]->(n1:`Department`)" in plan.cypher
    assert "n0.`course_name`" in plan.cypher


def test_plan_graph_query_does_not_run_rule_pipeline_first_when_ai_is_available(monkeypatch):
    def fake_ai_extract(question, query_schema, ai_settings, **kwargs):
        return {
            "status": "ok",
            "query_type": "one_hop",
            "intent_contract": {
                "source_entity": {"label": "Student", "table": "students"},
                "target_entity": {"label": "Course", "table": "courses"},
                "relationship": {"type": "ENROLLED_IN", "from_label": "Student", "to_label": "Course"},
                "source_filters": [{"scope": "source", "property": "first_name", "operator": "=", "value": "Alice"}],
                "target_filters": [],
                "relationship_filters": [],
                "return_fields": [],
                "limit": 25,
            },
            "confidence": 0.91,
            "notes": "AI-first contract parsing succeeded.",
        }

    monkeypatch.setattr("backend.graph_query._ai_extract_plan", fake_ai_extract)
    ai_settings = SimpleNamespace(api_key="x", base_url="x", model="qwen-plus")

    plan = plan_graph_query("Which courses is Alice enrolled in?", _sample_graph_mapping(), ai_settings=ai_settings)

    assert plan.status == "ok"
    assert plan.planner == "ai_assisted"
    assert plan.query_type == "one_hop"
    assert "MATCH (n0:`Student`)-[r0:`ENROLLED_IN`]->(n1:`Course`)" in plan.cypher


def test_plan_graph_query_one_hop_with_object_phrase_and_relationship_filters():
    mapping = _sample_graph_mapping()
    mapping["nodes"][0]["properties"] = ["first_name", "last_name", "email"]
    mapping["nodes"][1]["properties"] = ["course_name"]
    plan = plan_graph_query("Which students took Database Systems in Spring 2025?", mapping)
    assert plan.status == "ok"
    assert plan.query_type == "one_hop"
    assert "MATCH (n0:`Student`)-[r0:`ENROLLED_IN`]->(n1:`Course`)" in plan.cypher
    assert "n1.`course_name`" in plan.cypher
    assert "r0.`semester`" in plan.cypher
    assert "r0.`year`" in plan.cypher


def test_plan_graph_query_one_hop_who_with_object_phrase_and_relationship_filters():
    mapping = _sample_graph_mapping()
    mapping["nodes"][1]["properties"] = ["course_name"]
    plan = plan_graph_query("Who taught Database Systems in Fall 2024?", mapping)
    assert plan.status == "ok"
    assert plan.query_type == "one_hop"
    assert "MATCH (n0:`Instructor`)-[r0:`TEACHES`]->(n1:`Course`)" in plan.cypher
    assert "n1.`course_name`" in plan.cypher
    assert "r0.`semester`" in plan.cypher


def test_plan_graph_query_one_hop_full_name_returns_target_only():
    mapping = _sample_graph_mapping()
    mapping["nodes"][0]["properties"] = ["first_name", "last_name", "email"]
    plan = plan_graph_query("Which courses is Alice Johnson enrolled in?", mapping)
    assert plan.status == "ok"
    assert plan.query_type == "one_hop"
    assert "n0.`first_name`" in plan.cypher
    assert "n0.`last_name`" in plan.cypher
    assert "n1.`course_name` AS `course_course_name`" in plan.cypher
    assert "relationship_type" not in plan.cypher


def test_plan_graph_query_inverse_relation_wording_does_not_add_target_filter():
    mapping = _sample_graph_mapping()
    mapping["nodes"][0]["properties"] = ["first_name", "last_name", "email"]
    mapping["relationships"].append(
        {
            "from_table": "students",
            "to_table": "departments",
            "final_name": "BELONGS_TO",
            "display_text": "Student belongs to Department",
            "properties": [],
        }
    )
    plan = plan_graph_query("Which department does Alice Johnson belong to?", mapping)
    assert plan.status == "ok"
    assert plan.query_type == "one_hop"
    assert "n0.`first_name`" in plan.cypher
    assert "n0.`last_name`" in plan.cypher
    assert "target_value" not in plan.cypher
    assert "n1.`department_name` AS `department_department_name`" in plan.cypher


def test_plan_graph_query_anchor_return_semantics_for_offered_by():
    mapping = _sample_graph_mapping()
    mapping["nodes"][1]["properties"] = ["course_name"]
    plan = plan_graph_query("Which department is Database Systems offered by?", mapping)
    assert plan.status == "ok"
    assert plan.query_type == "one_hop"
    assert "n0.`course_name`" in plan.cypher
    assert "target_value" not in plan.cypher
    assert "n1.`department_name` AS `department_department_name`" in plan.cypher


def test_plan_graph_query_anchor_return_semantics_for_reports_to_style():
    mapping = _sample_graph_mapping()
    mapping["relationships"].append(
        {
            "from_table": "students",
            "to_table": "instructors",
            "final_name": "ADVISED_BY",
            "display_text": "Student advised by Instructor",
            "properties": [],
        }
    )
    mapping["nodes"][0]["properties"] = ["first_name", "last_name", "email"]
    plan = plan_graph_query("Which instructors is Alice Johnson advised by?", mapping)
    assert plan.status == "ok"
    assert plan.query_type == "one_hop"
    assert "n0.`first_name`" in plan.cypher
    assert "n0.`last_name`" in plan.cypher
    assert "n1.`first_name` AS `instructor_first_name`" in plan.cypher


def test_plan_graph_query_fixed_multi_hop():
    mapping = _sample_graph_mapping()
    mapping["relationships"].append(
        {
            "from_table": "instructors",
            "to_table": "departments",
            "final_name": "WORKS_FOR",
            "display_text": "Instructor works for Department",
            "properties": [],
        }
    )
    plan = plan_graph_query("Show paths from students via courses instructors departments", mapping)
    assert plan.status == "ok"
    assert plan.query_type == "fixed_multi_hop"
    assert "MATCH (n0:`Student`)-[r0:`ENROLLED_IN`]->(n1:`Course`)<-[r1:`TEACHES`]-(n2:`Instructor`)-[r2:`WORKS_FOR`]->(n3:`Department`)" in plan.cypher


def test_plan_graph_query_constrained_multi_hop():
    mapping = _sample_graph_mapping()
    plan = plan_graph_query("Find paths from students to departments within 3 hops through courses", mapping)
    assert plan.status == "ok"
    assert plan.query_type == "constrained_multi_hop"
    assert "MATCH (n0:`Student`)-[r0:`ENROLLED_IN`]->(n1:`Course`)-[r1:`OFFERED_BY`]->(n2:`Department`)" in plan.cypher
