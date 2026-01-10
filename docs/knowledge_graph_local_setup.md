# Knowledge Graph Local Setup

This benchmark requires a local SPARQL endpoint with a **Freebase-style RDF dump** loaded.
By default the code expects:

```
http://127.0.0.1:3001/kb/sparql
```

## Prerequisites

- Docker (required for the default local SPARQL server)
- A Freebase RDF dump in N-Triples or Turtle format:
  - `.nt`, `.nt.gz`, `.ttl`, or `.ttl.gz`

> Note: The KG dump is **not included** in this repo. You must provide it yourself.

## Start the SPARQL server

Set the dump path and start Fuseki:

```bash
export LIFELONG_KG_DUMP_PATH="/path/to/freebase-dump.nt.gz"
python scripts/kg_sparql_server.py start
```

If your Docker host is ARM (Apple Silicon), the script will default to `linux/amd64` for the Fuseki image.
You can override it with:

```bash
export LIFELONG_KG_DOCKER_PLATFORM=linux/arm64
```

If the Fuseki image does not include the loader, the script will fall back to
`apache/jena` to run `tdb2.tdbloader`. You can override it with:

```bash
export LIFELONG_KG_LOADER_IMAGE=apache/jena
```

Optional env vars:

- `LIFELONG_KG_DATA_DIR` (defaults to `data/knowledge_graph/sparql/fuseki`)
- `LIFELONG_KG_CONTAINER_NAME` (defaults to `lifelong_fuseki`)

## Health check

```bash
python scripts/kg_sparql_server.py health
```

This runs a tiny `ASK { ?s ?p ?o }` query to verify the endpoint is reachable and the KB is loaded.

## Stop the SPARQL server

```bash
python scripts/kg_sparql_server.py stop
```

## Run the benchmark

```bash
python scripts/run_all_with_servers.py
```

The runner will start the SPARQL server automatically for KG configs and refuse to proceed
if the dump is missing or the KB is empty.

## If you need a dump

The repo does not ship the Freebase RDF dump. You must obtain a compatible Freebase dump
from an external source and point `LIFELONG_KG_DUMP_PATH` to it. The benchmark expects
Freebase namespace IRIs (e.g., `http://rdf.freebase.com/ns/`).
