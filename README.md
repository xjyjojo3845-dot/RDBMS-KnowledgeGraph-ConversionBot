# RDBMS to Graph Explorer

Streamlit app for:
- inspecting a relational database
- building a graph mapping from the schema
- optionally refining the mapping with AI
- converting the data into Neo4j
- querying the final graph with natural language

Supported sources:
- `SQLite`
- `PostgreSQL`

For the best AI-assisted experience, configure the project with your own AI API key after downloading it locally.

## Quick Start

1. Create and activate a virtual environment, then install dependencies.
2. Copy `config/secrets.example.json` to `config/secrets.json`.
3. Replace the placeholder settings with your own AI API key.
4. Start the app with `streamlit run app.py`.
5. Open `http://localhost:8501`.
6. Choose `SQLite file` and upload `data/mock.db` for the fastest first run.

## What the App Does

Typical workflow:

1. Load a SQLite file or connect to PostgreSQL.
2. Inspect tables, columns, keys, and foreign keys.
3. Build a graph mapping with nodes, relationships, and join-table transformations.
4. Optionally apply semantic review and AI review.
5. Convert the source data into Neo4j.
6. Ask a natural-language question and run the generated read-only Cypher.

## Known Limitations

- Natural-language query generation currently requires AI configuration.
- Only read-only Cypher is generated and allowed in the query module.
- Automatic graph mapping may still need manual adjustment in ambiguous cases.
- The included Chinook example is a PostgreSQL SQL script and must be imported into PostgreSQL before use.

## Example Databases

The repository includes two example datasets in `data/`.

### Teaching database

This is the smaller database created for this project.

File:
- `data/mock.db`

Use it directly in the app through the `SQLite file` option.

### Chinook database

This is based on the public Chinook sample database:
- https://github.com/lerocha/chinook-database

Included file:
- `data/Chinook_PostgreSql_SerialPKs.sql`

This is a PostgreSQL SQL script, not a SQLite file. You need to create your own PostgreSQL database and import this script before using it in the app.

Example:

```bash
createdb chinook
psql -U postgres -d chinook -f data/Chinook_PostgreSql_SerialPKs.sql
```

Then connect to it in the app with:
- host
- port
- database
- username
- password
- schema, usually `public`

## Requirements

- Python `3.10+`
- `pip`
- browser for Streamlit

Optional:
- Neo4j for conversion and graph queries
- AI credentials for AI review and query generation
- PostgreSQL if you want to use the Chinook dataset or any Postgres source

## Installation

```bash
git clone <your-repository-url>
cd RDBMS-GRAPH-TOOL-main
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If the project is already on your machine:

```bash
cd <project-folder>
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

To get better results from the AI-assisted features, update the project to use your own AI API key after downloading it locally.

## Run the App

```bash
streamlit run app.py
```

If needed:

```bash
python -m streamlit run app.py
```

Default local URL:

```text
http://localhost:8501
```

## Using the App

When the app starts, choose one source type:
- `SQLite file`
- `PostgreSQL connection`

While the app is processing a database, a centered progress card shows the current stage.

## SQLite Workflow

1. Choose `SQLite file`.
2. Upload a `.db`, `.sqlite`, or `.sqlite3` file.
3. Wait for the app to load the schema and graph mapping.

For the included teaching database, upload:

```text
data/mock.db
```

## PostgreSQL Workflow

1. Choose `PostgreSQL connection`.
2. Pick one input mode:
   - `Connection fields`
   - `Connection string`
3. Click `Connect PostgreSQL`.

### Connection fields

Fill in:
- `Host`
- `Port`
- `Schema`
- `Database`
- `Username`
- `Password`

Example:

```text
Host: localhost
Port: 5432
Schema: public
Database: chinook
Username: postgres
Password: your_password
```

### Connection string

Example:

```text
postgresql://postgres:your_password@localhost:5432/chinook?schema=public
```

You can also override the schema separately in the UI.

## UI Sections

## 1. RDBMS Schema Inspection

This section shows the original relational schema.

It includes:
- `Schema Diagram`: compact ER-style view of tables and foreign keys
- `Schema Details`: per-table details including columns, primary keys, indexes, and foreign keys

Use it to verify the source database was loaded correctly.

## 2. Graph Mapping Result

This section shows the graph interpretation of the relational schema.

It includes:
- node summary
- relationship summary
- join-table transformations
- ambiguous cases
- graph preview
- downloadable JSON artifacts

The `Relationship Summary` section is editable. You can:
- add a relationship manually
- edit an existing relationship
- adjust the mapping if the automatic result is not what you want

## 3. Neo4j Conversion

This section converts the source data into Neo4j.

You need:
- `Neo4j URI`
- `Username`
- `Password`

Typical local URI:

```text
bolt://localhost:7687
```

Available modes:
- `structural`
- `semantic`

Use `semantic` if you want the conversion to follow the reviewed graph mapping.

Basic flow:

1. Enter Neo4j connection details.
2. Choose conversion mode.
3. Click `Convert`.
4. Review the conversion report and diagnostics.

## 4. Graph Query

This section generates read-only Cypher from natural language and runs it against Neo4j.

Requirements:
- Neo4j connection details must already be entered
- AI configuration must be available

Flow:

1. Choose an example question set:
   - `Teaching database`
   - `Chinook database`
   - `Other`
2. If you choose one of the first two, select an example question from the dropdown.
3. Or type your own question in `Natural language question`.
4. Click `Generate query`.
5. Review the generated Cypher.
6. Click `Run query`.
7. Inspect the returned rows.

Only read-only Cypher is allowed.

## AI Configuration

AI is optional for mapping review, but required for the current natural-language query generator.

After downloading the project, make sure you replace the sample or placeholder configuration with your own AI API key. Using your own valid API key will generally give better and more reliable results for AI-assisted review and query generation.

You can configure AI with either `config/secrets.json` or environment variables.

The configuration always follows the same pattern:
- set `AI_PROVIDER`
- provide the matching API key
- optionally set that provider's base URL and model name

The project currently supports provider-specific settings for:
- `qwen`
- `openai`

Start from:

```bash
cp config/secrets.example.json config/secrets.json
```

General template:

```json
{
  "AI_PROVIDER": "<provider>",
  "..._API_KEY": "<your-key>",
  "..._BASE_URL": "<provider-base-url>",
  "..._MODEL": "<model-name>"
}
```

Example for `openai`:

```json
{
  "AI_PROVIDER": "openai",
  "OPENAI_API_KEY": "<your-key>",
  "OPENAI_BASE_URL": "https://api.openai.com/v1",
  "OPENAI_MODEL": "gpt-4.1-mini"
}
```

Example for `qwen`:

```json
{
  "AI_PROVIDER": "qwen",
  "QWEN_API_KEY": "<your-key>",
  "QWEN_BASE_URL": "https://dashscope.aliyuncs.com/compatible-mode/v1",
  "QWEN_MODEL": "qwen-plus"
}
```

Or with environment variables:

```bash
# OpenAI example
export AI_PROVIDER=openai
export OPENAI_API_KEY="<your-key>"
export OPENAI_BASE_URL="https://api.openai.com/v1"
export OPENAI_MODEL="gpt-4.1-mini"

# Qwen example
export AI_PROVIDER=qwen
export QWEN_API_KEY="<your-key>"
export QWEN_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
export QWEN_MODEL="qwen-plus"
```

Do not commit `config/secrets.json`.

## CLI

The repository also includes a CLI for generating graph mappings.

### SQLite

```bash
python scripts/build_graph_mapping.py \
  --db path/to/mock.db \
  --schema artifacts/schema_extraction.json \
  --structural-mapping artifacts/structural_mapping.json \
  --out artifacts/graph_mapping.json \
  --ai-review-out artifacts/ai_mapping_review.json
```

### PostgreSQL

With a URL:

```bash
python scripts/build_graph_mapping.py \
  --pg-url "postgresql://myuser:mypassword@localhost:5432/mydb?schema=public" \
  --out artifacts/graph_mapping.json \
  --ai-review-out artifacts/ai_mapping_review.json
```

With explicit fields:

```bash
python scripts/build_graph_mapping.py \
  --pg-host localhost \
  --pg-port 5432 \
  --pg-database mydb \
  --pg-user myuser \
  --pg-password mypassword \
  --pg-schema public \
  --out artifacts/graph_mapping.json \
  --ai-review-out artifacts/ai_mapping_review.json
```

## Project Structure

- `app.py` - Streamlit UI
- `backend/schema_introspection.py` - schema loading
- `relationship_detector.py` - structural relationship detection
- `backend/layer2_mapping.py` - graph mapping
- `semantic_relationship_reviewer.py` - semantic review
- `ai/config.py` - AI config loader
- `ai/client.py` - AI client
- `ai/reviewer.py` - AI review merge logic
- `backend/neo4j_converter.py` - Neo4j conversion
- `backend/graph_query.py` - query generation and execution
- `scripts/build_graph_mapping.py` - CLI

## Troubleshooting

### `streamlit: command not found`

Use:

```bash
python -m streamlit run app.py
```

### PostgreSQL connection fails

Check:
- host
- port
- database
- username
- password
- schema
- permissions

### Chinook does not appear in the app

Make sure you imported `data/Chinook_PostgreSql_SerialPKs.sql` into your own PostgreSQL database first.

### Query generation says AI is required

The query module currently requires AI configuration.

### Neo4j conversion or query fails

Check:
- Neo4j is running
- URI is correct, for example `bolt://localhost:7687`
- credentials are correct
