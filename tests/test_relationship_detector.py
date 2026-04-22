from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from relationship_detector import (
    build_auto_mapping_config,
    build_final_mapping_config,
    build_review_defaults,
    detect_relationships,
)

SCHEMA = '''
CREATE TABLE zip_codes (
    id INTEGER NOT NULL,
    zip VARCHAR(5) NOT NULL,
    created_at DATETIME DEFAULT (CURRENT_TIMESTAMP),
    updated_at DATETIME,
    PRIMARY KEY (id)
);
CREATE INDEX ix_zip_codes_id ON zip_codes (id);
CREATE UNIQUE INDEX ix_zip_codes_zip ON zip_codes (zip);
CREATE TABLE housing_metrics (
    id INTEGER NOT NULL,
    zip VARCHAR(5) NOT NULL,
    name VARCHAR(100),
    median_rent FLOAT,
    created_at DATETIME DEFAULT (CURRENT_TIMESTAMP),
    updated_at DATETIME,
    PRIMARY KEY (id)
);
CREATE INDEX ix_housing_metrics_zip ON housing_metrics (zip);
CREATE TABLE building_info (
    id INTEGER NOT NULL,
    bbl VARCHAR(20) NOT NULL,
    zipcode VARCHAR(5),
    PRIMARY KEY (id)
);
CREATE INDEX ix_building_info_zipcode ON building_info (zipcode);
CREATE UNIQUE INDEX ix_building_info_bbl ON building_info (bbl);
CREATE TABLE building_stats (
    id INTEGER NOT NULL,
    zip VARCHAR(5) NOT NULL,
    PRIMARY KEY (id)
);
CREATE UNIQUE INDEX ix_building_stats_zip ON building_stats (zip);
'''


def test_detects_zip_relationships_with_confidence():
    result = detect_relationships(SCHEMA)
    rels = result["relationships"]

    assert any(
        r["source_table"] == "housing_metrics"
        and r["target_table"] == "zip_codes"
        and r["source_columns"] == ["zip"]
        and r["target_columns"] == ["zip"]
        and r["relationship_type"] == "inferred"
        and r["confidence"] >= 0.7
        for r in rels
    )

    assert any(
        r["source_table"] == "building_info"
        and r["target_table"] == "zip_codes"
        and r["source_columns"] == ["zipcode"]
        and r["target_columns"] == ["zip"]
        and r["relationship_type"] == "inferred"
        and r["confidence"] >= 0.7
        for r in rels
    )


def test_mapping_is_structured_by_source_table():
    result = detect_relationships(SCHEMA)
    mapping = result["mapping"]
    assert "housing_metrics" in mapping
    assert isinstance(mapping["housing_metrics"], list)
    assert {"to_table", "from_columns", "to_columns", "type", "confidence"}.issubset(mapping["housing_metrics"][0].keys())


def test_high_confidence_relationships_are_auto_accepted():
    relationships = [
        {
            "source_table": "housing_metrics",
            "source_columns": ["zip"],
            "target_table": "zip_codes",
            "target_columns": ["zip"],
            "relationship_type": "inferred",
            "confidence": 0.95,
            "signals": ["name_canonical_match"],
        }
    ]

    review_items = build_review_defaults(relationships, auto_accept_threshold=0.9)
    assert review_items[0]["action"] == "accept"


def test_rejected_and_edited_relationships_in_final_config():
    base = {
        "source_table": "building_info",
        "source_columns": ["zipcode"],
        "target_table": "zip_codes",
        "target_columns": ["zip"],
        "relationship_type": "inferred",
        "confidence": 0.82,
        "signals": ["name_canonical_match"],
    }
    review_items = [
        {"relationship": base, "action": "reject", "edited_relationship": None},
        {
            "relationship": base,
            "action": "edit",
            "edited_relationship": {
                **base,
                "target_table": "building_stats",
                "target_columns": ["zip"],
                "relationship_type": "edited",
            },
        },
    ]

    config = build_final_mapping_config(review_items)
    assert len(config["relationships"]) == 1
    assert config["relationships"][0]["target_table"] == "building_stats"
    assert config["mapping"]["building_info"][0]["type"] == "edited"


def test_detects_explicit_fk_with_quoted_identifiers():
    schema = '''
    CREATE TABLE "user" (
        "id" INTEGER PRIMARY KEY
    );
    CREATE TABLE "order" (
        "id" INTEGER PRIMARY KEY,
        "user_id" INTEGER,
        FOREIGN KEY ("user_id") REFERENCES "user" ("id")
    );
    '''
    result = detect_relationships(schema)
    assert any(
        r["source_table"] == "order"
        and r["target_table"] == "user"
        and r["source_columns"] == ["user_id"]
        and r["target_columns"] == ["id"]
        and r["relationship_type"] == "explicit_fk"
        for r in result["relationships"]
    )


def test_auto_mapping_config_filters_weak_inferences():
    relationships = [
        {
            "source_table": "housing_metrics",
            "source_columns": ["zip"],
            "target_table": "zip_codes",
            "target_columns": ["zip"],
            "relationship_type": "inferred",
            "confidence": 0.88,
            "signals": ["name_canonical_match", "target_key_like"],
        },
        {
            "source_table": "housing_metrics",
            "source_columns": ["name"],
            "target_table": "building_info",
            "target_columns": ["bbl"],
            "relationship_type": "inferred",
            "confidence": 0.7,
            "signals": ["data_type_compatible"],
        },
    ]

    config = build_auto_mapping_config(relationships)
    assert len(config["relationships"]) == 1
    assert config["relationships"][0]["target_table"] == "zip_codes"
