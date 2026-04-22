from pathlib import Path
import json
import os
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from ai.config import load_ai_settings


def test_load_ai_settings_prefers_qwen_from_secrets(tmp_path: Path):
    secrets = tmp_path / "secrets.json"
    secrets.write_text(
        json.dumps(
            {
                "AI_PROVIDER": "qwen",
                "QWEN_API_KEY": "qwen-test-key",
                "QWEN_BASE_URL": "https://dashscope.aliyuncs.com/compatible-mode/v1/",
                "QWEN_MODEL": "qwen-max",
            }
        )
    )

    settings = load_ai_settings(secrets)
    assert settings is not None
    assert settings.provider == "qwen"
    assert settings.api_key == "qwen-test-key"
    assert settings.base_url == "https://dashscope.aliyuncs.com/compatible-mode/v1"
    assert settings.model == "qwen-max"


def test_load_ai_settings_env_fallback_openai(monkeypatch):
    monkeypatch.setenv("AI_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-test-key")

    settings = load_ai_settings("/nonexistent/secrets.json")
    assert settings is not None
    assert settings.provider == "openai"
    assert settings.api_key == "openai-test-key"


from ai.reviewer import _normalize_reviews_payload, apply_ai_review


def test_normalize_reviews_payload_accepts_bare_list():
    payload = [{"object_id": "a->b", "ai_suggested_name": "REL"}]
    normalized = _normalize_reviews_payload(payload)
    assert len(normalized) == 1
    assert normalized[0]["object_id"] == "a->b"


def test_normalize_reviews_payload_accepts_dict_reviews_key():
    payload = {"reviews": [{"object_id": "x->y"}], "extra": 1}
    normalized = _normalize_reviews_payload(payload)
    assert len(normalized) == 1
    assert normalized[0]["object_id"] == "x->y"


def test_apply_ai_review_preserves_direct_fk_even_if_ai_excludes_it():
    graph_mapping = {
        "relationships": [
            {
                "id": "students->departments",
                "from_table": "students",
                "to_table": "departments",
                "from_columns": ["department_id"],
                "to_columns": ["department_id"],
                "classification": "direct_fk_relationship",
                "final_name": "MAJORS_IN",
            }
        ]
    }
    ai_review = {
        "provider": "qwen",
        "model": "qwen-plus",
        "reviews": [
            {"object_id": "students->departments", "include": False}
        ],
    }

    reviewed = apply_ai_review(graph_mapping, ai_review)
    assert len(reviewed["relationships"]) == 1
    assert reviewed["relationships"][0]["classification"] == "direct_fk_relationship"


def test_apply_ai_review_none_returns_deterministic_status():
    graph_mapping = {"relationships": []}
    reviewed = apply_ai_review(graph_mapping, None)
    assert reviewed["review_status"] == "deterministic_only"
    assert reviewed["review_provider"] is None


def test_apply_ai_review_updates_node_and_join_table():
    graph_mapping = {
        "nodes": [
            {
                "table": "students",
                "label": "Students",
                "properties": ["name"],
                "classification": "entity_table",
                "confidence": 0.85,
            }
        ],
        "relationships": [],
        "join_tables": [
            {
                "table": "enrollments",
                "linked_tables": ["students", "courses"],
                "join_columns": ["student_id", "course_id"],
                "non_key_columns": ["semester"],
                "confidence": 0.9,
            }
        ],
        "ambiguous_cases": [],
    }
    ai_review = {
        "provider": "qwen",
        "model": "qwen-plus",
        "reviews": [
            {
                "object_id": "node::students",
                "include": True,
                "label": "Student",
                "properties": ["name", "email"],
                "classification": "entity_table",
                "confidence": 0.95,
                "reasoning_summary": "Singular label is cleaner.",
            },
            {
                "object_id": "join::enrollments",
                "include": True,
                "linked_tables": ["students", "courses"],
                "join_columns": ["student_id", "course_id"],
                "non_key_columns": ["semester", "grade"],
                "classification": "join_table",
                "confidence": 0.97,
                "reasoning_summary": "Classic enrollment bridge table.",
            },
        ],
    }

    reviewed = apply_ai_review(graph_mapping, ai_review)
    assert reviewed["nodes"][0]["label"] == "Student"
    assert reviewed["nodes"][0]["properties"] == ["name", "email"]
    assert reviewed["join_tables"][0]["non_key_columns"] == ["semester", "grade"]
    assert reviewed["ai_review_summary"]["nodes_reviewed"] == 1
    assert reviewed["ai_review_summary"]["join_tables_reviewed"] == 1
