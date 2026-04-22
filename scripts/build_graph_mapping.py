from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from ai.reviewer import apply_ai_review, run_ai_mapping_review
from backend.layer2_mapping import build_graph_mapping, load_foreign_keys, load_postgres_foreign_keys, load_schema
from backend.schema_introspection import introspect_postgres_schema
from semantic_relationship_reviewer import apply_semantic_relationship_review, run_semantic_relationship_review


def _load_json(path: Path | None) -> dict[str, Any] | None:
    if not path:
        return None
    if not path.exists():
        raise FileNotFoundError(f"JSON input not found: {path}")
    return json.loads(path.read_text())


def _postgres_config_from_url(url: str) -> dict[str, Any]:
    parsed = urlparse(url)
    if parsed.scheme not in {"postgres", "postgresql"}:
        raise SystemExit("`--pg-url` must start with `postgres://` or `postgresql://`.")
    query = parse_qs(parsed.query)
    schema_name = query.get("schema", ["public"])[0]
    return {
        "host": parsed.hostname or "",
        "port": parsed.port or 5432,
        "database": (parsed.path or "").lstrip("/"),
        "user": parsed.username or "",
        "password": parsed.password or "",
        "schema_name": schema_name,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build deterministic Layer 2 graph mapping with optional AI review.")
    parser.add_argument("--db", help="Path to SQLite DB")
    parser.add_argument("--pg-url", help="PostgreSQL URL, e.g. postgresql://user:password@host:5432/dbname?schema=public")
    parser.add_argument("--pg-host", help="PostgreSQL host")
    parser.add_argument("--pg-port", type=int, default=5432, help="PostgreSQL port")
    parser.add_argument("--pg-database", help="PostgreSQL database name")
    parser.add_argument("--pg-user", help="PostgreSQL username")
    parser.add_argument("--pg-password", help="PostgreSQL password")
    parser.add_argument("--pg-schema", default="public", help="PostgreSQL schema name")
    parser.add_argument("--schema", help="Optional schema extraction JSON")
    parser.add_argument("--structural-mapping", help="Optional structural mapping JSON")
    parser.add_argument("--out", required=True, help="Output graph mapping JSON path")
    parser.add_argument("--ai-review-out", help="Optional AI review output path")
    parser.add_argument("--ai-model", default=None, help="Optional override model for AI review")
    args = parser.parse_args()

    schema_path = Path(args.schema) if args.schema else None
    structural_path = Path(args.structural_mapping) if args.structural_mapping else None

    using_sqlite = bool(args.db)
    using_postgres = bool(args.pg_url) or any([args.pg_host, args.pg_database, args.pg_user, args.pg_password])

    if using_sqlite and using_postgres:
        raise SystemExit("Use either SQLite input (`--db`) or PostgreSQL input (`--pg-*`), not both.")
    if not using_sqlite and not using_postgres:
        raise SystemExit("Provide either `--db` for SQLite or the required `--pg-*` PostgreSQL connection arguments.")

    if using_sqlite:
        db_path = Path(args.db)
        schema = load_schema(db_path, schema_path)
        foreign_keys = load_foreign_keys(db_path, [table.name for table in schema])
    else:
        if args.pg_url:
            postgres_config = _postgres_config_from_url(args.pg_url)
            if args.pg_schema and args.pg_schema != "public":
                postgres_config["schema_name"] = args.pg_schema
        else:
            required_pg_fields = {
                "--pg-host": args.pg_host,
                "--pg-database": args.pg_database,
                "--pg-user": args.pg_user,
                "--pg-password": args.pg_password,
            }
            missing = [flag for flag, value in required_pg_fields.items() if not value]
            if missing:
                raise SystemExit(f"Missing required PostgreSQL arguments: {', '.join(missing)}")

            postgres_config = {
                "host": args.pg_host,
                "port": args.pg_port,
                "database": args.pg_database,
                "user": args.pg_user,
                "password": args.pg_password,
                "schema_name": args.pg_schema,
            }
        schema = introspect_postgres_schema(**postgres_config)
        foreign_keys = load_postgres_foreign_keys(
            **postgres_config,
            table_names=[table.name for table in schema],
        )

    structural = _load_json(structural_path)

    graph_mapping = build_graph_mapping(schema=schema, foreign_keys=foreign_keys, structural_mapping=structural)

    semantic_review = run_semantic_relationship_review(graph_mapping, model=args.ai_model)
    graph_mapping = apply_semantic_relationship_review(graph_mapping, semantic_review)
    ai_review = run_ai_mapping_review(graph_mapping, model=args.ai_model, review_relationships=False)
    enriched = apply_ai_review(graph_mapping, ai_review)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(enriched, indent=2))
    print(f"[layer2] Wrote graph mapping: {out_path}")

    if args.ai_review_out:
        review_path = Path(args.ai_review_out)
        review_path.parent.mkdir(parents=True, exist_ok=True)
        payload = ai_review or {"reviews": [], "status": "ai_disabled"}
        review_path.write_text(json.dumps(payload, indent=2))
        if ai_review:
            print(f"[layer2] Wrote AI review: {review_path}")
        else:
            print("[layer2] AI review skipped (no local AI key found; configure QWEN_API_KEY or OPENAI_API_KEY).")


if __name__ == "__main__":
    main()
