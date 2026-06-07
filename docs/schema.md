# GraphRAG Sample Schema

This project intentionally keeps the current dataset in sample/prototype scope. Files ending in `_sample` are stable sample artifacts, not full-data production outputs. The field contracts below mirror the intended production structure so the sample pipeline can harden schemas before full indexing.

## Artifacts

- `law_articles_sample.jsonl`: parsed current criminal-law article records.
- `cases_sample.jsonl`: raw normalized sample case records extracted from immutable `raw/` files.
- `cases_enriched_sample.jsonl`: case records with rule-inferred metadata.
- `case_chunks_sample.jsonl`: field-preserving case chunks.
- `law_chunks_sample.jsonl`: article-level law chunks.
- `retrieval_chunks_sample.jsonl`: unified retrieval corpus for lexical, vector, graph, and hybrid retrieval.
- `graph_nodes_sample.jsonl`: typed graph nodes.
- `graph_edges_sample.jsonl`: typed graph edges.
- `graph_sample.jsonl`: compatibility alias containing graph edges for existing retrievers.

## Required Record Contracts

### Law Article

Required fields: `article_no`, `title`, `content`, `source_file`, `text_hash` when available.

Article numbers use normalized Arabic form such as `234` or `133.1`. The current sample law corpus is `law_version=current`.

### Case Raw / Enriched

Required fields: `case_id`, `title`, `source_file`, `fact_text`, `reasoning_text`, `judgment_text`.

Enriched records add rule-inferred fields such as `crimes`, `cited_articles`, `applied_articles`, `applied_laws`, `legal_basis_text`, `sentencing`, `amount_texts`, `court_name`, and `judgment_date_text`. Original extracted text remains separate from inferred metadata.

### Case Chunk

Required fields: `chunk_id`, `case_id`, `title`, `field`, `list_index`, `chunk_index`, `text`, `text_length`, `char_start`, `char_end`, `text_hash`, `source_file`, `extraction_method`, `quality_flags`, `case_metadata`.

Chunk IDs are deterministic from case, source, field, list index, and chunk index. Chunks preserve field provenance and carry quality flags instead of silently hiding extraction issues.

### Law Chunk

Required fields: `chunk_id`, `source_type`, `article_no`, `title`, `text`, `text_length`, `text_hash`, `source_file`, `extraction_method`, `quality_flags`, `law_metadata`.

`law_metadata` must include `law_name`, `article_no`, `title`, and `law_version`.

### Retrieval Chunk

Required fields: `retrieval_id`, `source_type`, `chunk_id`, `text`, `text_length`, `text_hash`, `title`, `source_file`, `field`, `case_id`, `case_metadata`, `law_metadata`, `article_no`, `law_name`, `law_version`, `extraction_method`, `quality_flags`.

`retrieval_id` is prefixed with `case:` or `law:` and is the only ID LLM citations may reference.

### Graph Node

Required fields: `node_id`, `node_type`.

Supported node types: `case`, `article`, `crime`, `chunk`, `unknown`.

### Graph Edge

Required fields: `type`, `source`, `target`, `evidence_text`, `confidence`, `extraction_method`.

Article citation edges also carry `raw_citation`, `normalized_citation`, `article_no`, `law_name`, `law_version`, `law_family`, `citation_role`, and `resolution_status`.

Supported edge types:

- `CASE_OF_CRIME`
- `CASE_HAS_CHUNK`
- `ARTICLE_HAS_CHUNK`
- `CASE_APPLIES_ARTICLE`
- `CASE_CITES_ARTICLE`

Law versions are `current`, `1979`, `judicial_interpretation`, `other`, or `unknown`. Historical or non-current law must not be silently mapped to current law chunks.
