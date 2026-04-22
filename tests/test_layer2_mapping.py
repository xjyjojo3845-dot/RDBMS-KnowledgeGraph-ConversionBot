from pathlib import Path
import sqlite3
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from backend.layer2_mapping import build_graph_mapping, load_foreign_keys
from backend.schema_introspection import introspect_sqlite_schema


def _build_demo_db(path: Path) -> None:
    with sqlite3.connect(path) as conn:
        conn.executescript(
            '''
            CREATE TABLE students (
              student_id INTEGER PRIMARY KEY,
              first_name TEXT,
              department_id INTEGER,
              FOREIGN KEY (department_id) REFERENCES departments(department_id)
            );
            CREATE TABLE courses (
              course_id INTEGER PRIMARY KEY,
              department_id INTEGER,
              FOREIGN KEY (department_id) REFERENCES departments(department_id)
            );
            CREATE TABLE instructors (
              instructor_id INTEGER PRIMARY KEY,
              department_id INTEGER,
              FOREIGN KEY (department_id) REFERENCES departments(department_id)
            );
            CREATE TABLE departments (
              department_id INTEGER PRIMARY KEY,
              name TEXT
            );
            CREATE TABLE enrollments (
              student_id INTEGER,
              course_id INTEGER,
              semester TEXT,
              year INTEGER,
              grade TEXT,
              FOREIGN KEY (student_id) REFERENCES students(student_id),
              FOREIGN KEY (course_id) REFERENCES courses(course_id)
            );
            '''
        )


def test_layer2_detects_join_tables_and_semantic_relationships(tmp_path: Path):
    db_path = tmp_path / "demo.db"
    _build_demo_db(db_path)

    schema = introspect_sqlite_schema(db_path)
    fks = load_foreign_keys(db_path, [t.name for t in schema])
    result = build_graph_mapping(schema, fks)

    join_table_names = {item["table"] for item in result["join_tables"]}
    assert "enrollments" in join_table_names

    rel_names = {rel["final_name"] for rel in result["relationships"]}
    assert "ENROLLED_IN" in rel_names
    assert "OFFERED_BY" in rel_names
    assert "WORKS_FOR" in rel_names
    assert "MAJORS_IN" in rel_names

    enrollment_rel = next(rel for rel in result["relationships"] if rel.get("via_table") == "enrollments")
    assert enrollment_rel["from_table"] == "students"
    assert enrollment_rel["to_table"] == "courses"
    assert set(enrollment_rel["properties"]) == {"semester", "year", "grade"}
    assert enrollment_rel["classification"] == "collapsed_join_table"
