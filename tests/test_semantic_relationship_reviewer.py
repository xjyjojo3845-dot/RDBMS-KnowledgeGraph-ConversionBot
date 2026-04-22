from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import semantic_relationship_reviewer as semantic


def _graph_mapping_with_relationships():
    return {
        "relationships": [
            {
                "id": "albums->artists",
                "from_table": "albums",
                "to_table": "artists",
                "from_columns": ["ArtistId"],
                "to_columns": ["ArtistId"],
                "properties": [],
                "classification": "direct_fk_relationship",
                "cardinality": "many_to_one",
                "rule_based_name": "ALBUM_TO_ARTIST",
                "final_name": "ALBUM_TO_ARTIST",
                "display_text": "Album album to artist Artist",
                "confidence": 0.95,
                "signals": ["explicit_foreign_key"],
            },
            {
                "id": "tracks->albums",
                "from_table": "tracks",
                "to_table": "albums",
                "from_columns": ["AlbumId"],
                "to_columns": ["AlbumId"],
                "properties": [],
                "classification": "direct_fk_relationship",
                "cardinality": "many_to_one",
                "rule_based_name": "TRACK_TO_ALBUM",
                "final_name": "TRACK_TO_ALBUM",
                "display_text": "Track track to album Album",
                "confidence": 0.95,
                "signals": ["explicit_foreign_key"],
            },
            {
                "id": "playlists->tracks::via::playlist_track",
                "from_table": "playlists",
                "to_table": "tracks",
                "via_table": "playlist_track",
                "from_join_columns": ["PlaylistId"],
                "to_join_columns": ["TrackId"],
                "from_columns": ["PlaylistId"],
                "to_columns": ["TrackId"],
                "properties": [],
                "classification": "collapsed_join_table",
                "cardinality": "many_to_many",
                "rule_based_name": "PLAYLIST_TRACK",
                "final_name": "PLAYLIST_TRACK",
                "display_text": "Playlist playlist track Track",
                "confidence": 0.91,
                "signals": ["two_or_more_foreign_keys"],
            },
            {
                "id": "employees->employees",
                "from_table": "employees",
                "to_table": "employees",
                "from_columns": ["ReportsTo"],
                "to_columns": ["EmployeeId"],
                "properties": [],
                "classification": "direct_fk_relationship",
                "cardinality": "many_to_one",
                "rule_based_name": "EMPLOYEE_TO_EMPLOYEE",
                "final_name": "EMPLOYEE_TO_EMPLOYEE",
                "display_text": "Employee employee to employee Employee",
                "confidence": 0.95,
                "signals": ["explicit_foreign_key", "self_reference"],
            },
        ]
    }


def test_semantic_review_pipeline_rewrites_chinook_relationships(monkeypatch):
    graph_mapping = _graph_mapping_with_relationships()

    class FakeSettings:
        provider = "qwen"
        api_key = "test"
        base_url = "https://example.com"
        model = "qwen-plus"

    responses = {
        "albums->artists": {
            "semantic_review": {
                "relationship_id": "albums->artists",
                "from_table": "artists",
                "to_table": "albums",
                "relationship_type": "CREATED",
                "direction": "reversed",
                "explanation": "Album stores the artist FK, but Artist created Album is the clearer graph relation.",
                "confidence": 0.91,
            }
        },
        "tracks->albums": {
            "semantic_review": {
                "relationship_id": "tracks->albums",
                "from_table": "albums",
                "to_table": "tracks",
                "relationship_type": "CONTAINS",
                "direction": "reversed",
                "explanation": "Album contains Track is the natural containment direction.",
                "confidence": 0.93,
            }
        },
        "employees->employees": {
            "semantic_review": {
                "relationship_id": "employees->employees",
                "from_table": "employees",
                "to_table": "employees",
                "relationship_type": "REPORTS_TO",
                "direction": "kept",
                "explanation": "The employee self-reference indicates a reporting hierarchy.",
                "confidence": 0.96,
            }
        },
        "playlists->tracks::via::playlist_track": {
            "semantic_review": {
                "relationship_id": "playlists->tracks::via::playlist_track",
                "from_table": "playlists",
                "to_table": "tracks",
                "relationship_type": "CONTAINS",
                "direction": "kept",
                "explanation": "A playlist naturally contains tracks.",
                "confidence": 0.94,
            }
        },
    }

    class FakeClient:
        def __init__(self, api_key: str, base_url: str, model: str) -> None:
            del api_key, base_url, model

        def complete_json(self, prompt_payload, *, system_prompt):
            del system_prompt
            relationship_id = prompt_payload["candidate"]["relationship_id"]
            payload = responses[relationship_id]
            return str(payload), payload

    monkeypatch.setattr(semantic, "semantic_review_enabled", lambda: True)
    monkeypatch.setattr(semantic, "load_ai_settings", lambda: FakeSettings())
    monkeypatch.setattr(semantic, "AIClient", FakeClient)

    semantic_review = semantic.run_semantic_relationship_review(graph_mapping)
    reviewed = semantic.apply_semantic_relationship_review(graph_mapping, semantic_review)

    album_rel = next(rel for rel in reviewed["relationships"] if rel["id"] == "albums->artists")
    assert album_rel["raw_final_name"] == "ALBUM_TO_ARTIST"
    assert album_rel["raw_from_table"] == "albums"
    assert album_rel["raw_from_columns"] == ["ArtistId"]
    assert album_rel["raw_to_columns"] == ["ArtistId"]
    assert album_rel["semantic_relationship_type"] == "CREATED"
    assert album_rel["from_table"] == "artists"
    assert album_rel["to_table"] == "albums"
    assert album_rel["semantic_review_source"] == "ai_review"

    track_rel = next(rel for rel in reviewed["relationships"] if rel["id"] == "tracks->albums")
    assert track_rel["final_name"] == "CONTAINS"
    assert track_rel["semantic_direction"] == "reversed"

    employee_rel = next(rel for rel in reviewed["relationships"] if rel["id"] == "employees->employees")
    assert employee_rel["final_name"] == "REPORTS_TO"
    assert employee_rel["semantic_review_source"] == "ai_review"

    summary = reviewed["semantic_review_summary"]
    assert summary["ai_reviewed_relationships"] == 4
    assert summary["renamed_relationships"] == 4
    assert summary["reversed_relationships"] == 2


def test_apply_semantic_relationship_review_falls_back_when_ai_result_missing():
    graph_mapping = _graph_mapping_with_relationships()

    reviewed = semantic.apply_semantic_relationship_review(graph_mapping, None)

    playlist_rel = next(rel for rel in reviewed["relationships"] if rel["id"] == "playlists->tracks::via::playlist_track")
    assert playlist_rel["semantic_review_source"] == "rule_based"
    assert playlist_rel["semantic_relationship_type"] == "PLAYLIST_TRACK"
    assert playlist_rel["final_name"] == "PLAYLIST_TRACK"
    assert playlist_rel["semantic_fallback_reason"] == "Semantic review did not return a valid result."
    assert reviewed["semantic_review_summary"]["fallback_relationships"] == 4


def test_run_semantic_relationship_review_rejects_invalid_ai_payload(monkeypatch):
    graph_mapping = {
        "relationships": [
            {
                "id": "albums->artists",
                "from_table": "albums",
                "to_table": "artists",
                "from_columns": ["ArtistId"],
                "to_columns": ["ArtistId"],
                "properties": [],
                "classification": "direct_fk_relationship",
                "cardinality": "many_to_one",
                "rule_based_name": "ALBUM_TO_ARTIST",
                "final_name": "ALBUM_TO_ARTIST",
                "confidence": 0.95,
                "signals": ["explicit_foreign_key"],
            }
        ]
    }

    class FakeSettings:
        provider = "qwen"
        api_key = "test"
        base_url = "https://example.com"
        model = "qwen-plus"

    class FakeClient:
        def __init__(self, api_key: str, base_url: str, model: str) -> None:
            del api_key, base_url, model

        def complete_json(self, prompt_payload, *, system_prompt):
            del prompt_payload, system_prompt
            return "{}", {"relationship_id": "albums->artists", "relationship_type": "CREATED"}

    monkeypatch.setattr(semantic, "semantic_review_enabled", lambda: True)
    monkeypatch.setattr(semantic, "load_ai_settings", lambda: FakeSettings())
    monkeypatch.setattr(semantic, "AIClient", FakeClient)

    result = semantic.run_semantic_relationship_review(graph_mapping)
    assert result is not None
    assert result["reviews"] == []
    assert len(result["errors"]) == 1
    assert result["errors"][0]["relationship_id"] == "albums->artists"
    assert result["debug"][0]["fallback_reason"].startswith("Semantic review validation failed:")


def test_apply_semantic_relationship_review_preserves_raw_basis_across_multiple_passes():
    graph_mapping = _graph_mapping_with_relationships()
    semantic_review = {
        "semantic_review_enabled": True,
        "provider": "qwen",
        "model": "qwen-plus",
        "reviews": [
            {
                "relationship_id": "albums->artists",
                "from_table": "artists",
                "to_table": "albums",
                "relationship_type": "CREATED",
                "direction": "reversed",
                "explanation": "Artist created Album is the clearer graph relation.",
                "confidence": 0.91,
            }
        ],
        "errors": [],
        "debug": [],
    }

    first_pass = semantic.apply_semantic_relationship_review(graph_mapping, semantic_review)
    second_pass = semantic.apply_semantic_relationship_review(first_pass, semantic_review)

    album_rel = next(rel for rel in second_pass["relationships"] if rel["id"] == "albums->artists")
    assert album_rel["from_table"] == "artists"
    assert album_rel["to_table"] == "albums"
    assert album_rel["raw_from_table"] == "albums"
    assert album_rel["raw_to_table"] == "artists"
    assert album_rel["raw_from_columns"] == ["ArtistId"]


def test_validate_semantic_result_prefers_active_voice_for_passive_by_names():
    candidate = {
        "relationship_id": "albums->artists",
        "allowed_tables": ["albums", "artists"],
        "raw_inferred_relationship": {
            "source_table": "albums",
            "target_table": "artists",
        },
    }

    validated = semantic._validate_semantic_result(
        candidate,
        {
            "relationship_id": "albums->artists",
            "from_table": "albums",
            "to_table": "artists",
            "relationship_type": "CREATED_BY",
            "direction": "kept",
            "explanation": "Album is created by Artist in relational storage.",
            "confidence": 0.88,
        },
    )

    assert validated["relationship_type"] == "CREATED"
    assert validated["from_table"] == "artists"
    assert validated["to_table"] == "albums"
    assert validated["direction"] == "reversed"


def test_relationship_candidate_uses_canonical_base_direction():
    rel = {
        "id": "invoices->customers",
        "from_table": "customers",
        "to_table": "invoices",
        "raw_from_table": "invoices",
        "raw_to_table": "customers",
        "raw_from_columns": ["customer_id"],
        "raw_to_columns": ["customer_id"],
        "rule_based_name": "INVOICE_TO_CUSTOMER",
        "final_name": "PLACED",
        "classification": "direct_fk_relationship",
        "raw_classification": "direct_fk_relationship",
        "cardinality": "many_to_one",
        "signals": ["explicit_foreign_key"],
    }

    candidate = semantic._relationship_candidate(rel)

    assert candidate["base_direction"]["base_from_table"] == "invoices"
    assert candidate["base_direction"]["base_to_table"] == "customers"
    assert candidate["base_direction"]["fk_holder_table"] == "invoices"
    assert candidate["base_direction"]["referenced_table"] == "customers"
    assert candidate["base_direction"]["fk_column"] == "customer_id"
    assert candidate["base_direction"]["referenced_key"] == "customer_id"
    assert "source_columns" not in candidate["raw_inferred_relationship"]
    assert "target_columns" not in candidate["raw_inferred_relationship"]


def test_validate_semantic_result_rejects_invalid_placed_direction():
    candidate = {
        "relationship_id": "invoices->customers",
        "allowed_tables": ["invoices", "customers"],
        "base_direction": {
            "base_from_table": "invoices",
            "base_to_table": "customers",
            "cardinality": "many_to_one",
        },
        "raw_inferred_relationship": {
            "base_from_table": "invoices",
            "base_to_table": "customers",
        },
    }

    try:
        semantic._validate_semantic_result(
            candidate,
            {
                "relationship_id": "invoices->customers",
                "final_from_table": "invoices",
                "final_to_table": "customers",
                "relationship_type": "PLACED",
                "direction": "kept",
                "explanation": "Invoice placed customer.",
                "confidence": 0.82,
            },
        )
        raise AssertionError("Expected PLACED in base FK direction to be rejected.")
    except ValueError as exc:
        assert "reverse it to actor->event" in str(exc)
