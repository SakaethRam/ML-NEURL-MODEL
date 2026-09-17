# Data Pipeline: Ingestion Through Relationship Discovery

## Supported input formats

- CSV
- Excel (`.xlsx`, `.xls`, via `openpyxl`)
- JSON (records, list, or dict orientations)
- JSONL (newline-delimited JSON)
- Parquet (Apache Arrow columnar)
- XML (auto-parsed to tabular form)
- SQL exports (treated as CSV)
- ERP exports, CRM exports, business logs

A single universal loader (Cell 2, `DataIngestionEngine`) handles all of these, doing type inference and schema-metadata extraction regardless of source format, so downstream cells (feature engineering onward) work against one consistent internal representation rather than needing format-specific logic themselves.

## Data validation and cleaning (part of Cell 2's pipeline)

Per the "Data Processing Pipeline" diagram: schema checks, format detection, and a null audit happen before cleaning (dropping constant columns, handling missing values, deduplication) and type inference (numeric, categorical, datetime, text, identifier detection). This ordering matters: type inference running on already-cleaned data is more reliable than trying to infer types around missing values or constant/degenerate columns that add no signal.

## Feature engineering (Cell 3)

Per-column feature vectors combining:

- **Statistical features** — presumably distributional properties (cardinality, null rate, value distribution shape).
- **Linguistic features** — TF-IDF over column names, which is what lets `customer_name` and `client_full_name` be recognized as similar even without identical strings.
- **Structural features** — value overlap between columns (a strong signal for foreign-key relationships), temporal and frequency features.

Two specific outputs feed directly into later stages:

- **TF-IDF similarity matrix** — column-name-level similarity, used by Node Intelligence (Cell 4) for semantic grouping.
- **Value overlap matrix** — used by Relationship Intelligence (Cell 5) as foreign-key evidence: two columns whose value sets substantially overlap are a strong signal of a relationship, independent of whether their names are similar at all.

## Node Intelligence (Cell 4)

Hierarchical clustering over the column feature vectors from Cell 3, grouping columns into candidate entity schemas. Each discovered entity comes with a confidence score, not just a hard yes/no grouping, which means downstream consumers (the BI report, the graph construction step) can treat low-confidence entity discoveries differently from high-confidence ones rather than the pipeline silently committing to every clustering decision with equal certainty.

**Worked example from the README:** `Customer Name`, `Email`, `Phone` cluster into a `Customer` entity; `Invoice Number`, `Date`, `Amount` cluster into an `Invoice` entity. Note this happens without any hardcoded knowledge of what a "customer" or "invoice" is: the clustering is driven by the TF-IDF name similarity and statistical co-variation from Cell 3, not a lookup table of business concepts.

## Relationship Intelligence (Cell 5)

Directed relationship discovery between the entity schemas Cell 4 produced, using four kinds of evidence:

- Foreign-key patterns (structural, from the value overlap matrix)
- Naming conventions (e.g. a column named `customer_id` inside an `Invoice`-clustered group pointing at a `Customer` entity)
- Numeric correlation
- Temporal ordering (which entity's records tend to precede the other's, a useful signal for cause/effect or parent/child relationships)

Output: a `RelationshipSchema` list with cardinality labels (1:1, 1:N, N:N), which is what lets the eventual knowledge graph express not just "these entities are connected" but what kind of connection it is.

## Why this ordering (features → nodes → relationships) and not the reverse

Relationship discovery in Cell 5 explicitly takes the entity schemas from Cell 4 as an input, not raw columns. This means relationships are discovered *between already-identified entities*, not between arbitrary column pairs. That ordering keeps the relationship search space bounded to meaningful entity-to-entity connections rather than needing to consider every possible column-pair combination, which would grow quadratically with the number of raw columns and produce a much noisier relationship candidate set.
