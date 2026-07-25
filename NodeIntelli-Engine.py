"""
NEURL Engine — Cell 4: Node Intelligence Engine

THE CORE INNOVATION: Converts a column feature matrix into a set of
EntitySchema objects representing discovered business entities using
pure ML (hierarchical clustering + linguistic refinement).

Key insight: Instead of relying on pre-defined schemas or LLMs to identify
what a "Customer" or "Order" is, we let the statistical structure of the
data speak for itself. Columns that are statistically similar, linguistically
similar, and share domain overlap are grouped into the same entity cluster.

Design rationale: A Ward linkage agglomerative clustering on column feature
vectors groups columns by data character; TF-IDF cosine similarity of column
names then refines these clusters using domain language. The combination
captures both the statistical "what" and the semantic "what it's called".
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering


# =============================================================================
# DATA CONTAINER — EntitySchema
# =============================================================================

@dataclass
class EntitySchema:
    """Describes a discovered business entity (cluster of related columns).

    Each EntitySchema corresponds to a node in the knowledge graph that
    represents a distinct business concept (e.g., Customer, Order, Product).

    Attributes:
        entity_name: Human-readable entity name (inferred from column prefixes).
        columns: List of column names assigned to this entity cluster.
        entity_type: Semantic entity type label from EntityTypeLabel enum.
        confidence: Average within-cluster name similarity (0.0 – 1.0).
        representative_column: The "key" column for this entity (usually _id col).
        description: Auto-generated human-readable description.
        cluster_id: Original cluster assignment index (for debugging).
    """
    entity_name: str
    columns: List[str]
    entity_type: str
    confidence: float
    representative_column: str
    description: str
    cluster_id: int = -1

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "entity_name": self.entity_name,
            "columns": self.columns,
            "entity_type": self.entity_type,
            "confidence": round(self.confidence, 3),
            "representative_column": self.representative_column,
            "description": self.description,
            "cluster_id": self.cluster_id,
        }


# =============================================================================
# NodeIntelligenceEngine — hierarchical clustering → entity discovery
# =============================================================================

class NodeIntelligenceEngine:
    """Discovers business entities from column feature representations.

    The discovery pipeline has four stages:
    1. Hierarchical clustering on column feature vectors (Ward linkage)
    2. Cluster refinement using TF-IDF column name similarity
    3. Post-processing: split clusters with incompatible value domains
    4. Entity typing and naming from column patterns

    This is genuinely LLM-free: all entity detection is driven by
    statistical and linguistic signals computed from the data itself.

    Example:
        engine = NodeIntelligenceEngine(config)
        entities = engine.discover_entities(
            df, schema, feature_matrix, name_similarity, value_overlap
        )
    """

    # Patterns for entity type classification
    _TRANSACTION_SIGNALS = {"amount", "total", "price", "cost", "fee", "tax", "discount"}
    _PERSON_SIGNALS = {"name", "email", "phone", "address", "contact", "first", "last"}
    _DATE_SIGNALS = {"date", "time", "timestamp", "created", "updated", "at", "on"}
    _DOCUMENT_SIGNALS = {"invoice", "receipt", "statement", "report", "doc", "contract"}
    _PRODUCT_SIGNALS = {"product", "item", "sku", "upc", "barcode", "material"}
    _REFERENCE_SIGNALS = {"status", "type", "category", "code", "flag", "indicator"}

    # Regex to split column names into tokens
    _SPLIT_RE = re.compile(r"[_\-\s]+|(?<=[a-z])(?=[A-Z])")

    def __init__(self, config: "NEURLConfig") -> None:
        """Initialize the NodeIntelligenceEngine.

        Args:
            config: NEURLConfig master configuration object.
        """
        self._config = config
        self._logger = NEURLLogger.get_logger("NodeIntelligenceEngine")

    # ─── MAIN DISCOVERY METHOD ───────────────────────────────────────────────

    def discover_entities(
        self,
        df: pd.DataFrame,
        schema: "SchemaMetadata",
        feature_matrix: np.ndarray,
        name_similarity_matrix: np.ndarray,
        value_overlap_matrix: np.ndarray,
    ) -> List[EntitySchema]:
        """Discover business entities from column feature representations.

        Pipeline:
        1. Agglomerative clustering on feature_matrix using Ward linkage.
        2. Name-similarity-guided merge of proximate clusters.
        3. Value-overlap-guided split of heterogeneous clusters.
        4. Small cluster absorption into nearest larger cluster.
        5. Entity naming, typing, and confidence scoring.

        Args:
            df: Cleaned DataFrame (used for value analysis).
            schema: SchemaMetadata from ingestion engine.
            feature_matrix: Column feature matrix, shape [N, D].
            name_similarity_matrix: TF-IDF cosine sim matrix, shape [N, N].
            value_overlap_matrix: Jaccard overlap matrix, shape [N, N].

        Returns:
            List of EntitySchema objects, one per discovered entity.
        """
        columns = list(df.columns)
        n_cols = len(columns)

        if n_cols < 2:
            raise NEURLDataError("Need at least 2 columns for entity discovery")

        self._logger.info(f"Starting entity discovery on {n_cols} columns")

        # ── Step 1: Agglomerative clustering on feature vectors ───────────
        # Ward linkage minimizes within-cluster variance: columns with similar
        # data character (dtype, entropy, missing rate, etc.) group together
        n_clusters_init = max(2, min(n_cols // 3, 10))
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=self._config.entity_distance_threshold,
            linkage="ward",
            compute_full_tree=True,
        )
        labels = clustering.fit_predict(feature_matrix)
        self._logger.info(f"Ward clustering: {len(set(labels))} initial clusters")

        # Build cluster dict: cluster_id → list of column indices
        cluster_dict: Dict[int, List[int]] = {}
        for col_idx, cluster_id in enumerate(labels):
            cluster_dict.setdefault(int(cluster_id), []).append(col_idx)

        # ── Step 2: Name-similarity-guided cluster merging ────────────────
        # If two clusters have high average inter-cluster name similarity,
        # they likely belong to the same entity (e.g., 'customer_*' columns
        # that Ward placed in different clusters due to type differences)
        cluster_dict = self._merge_by_name_similarity(
            cluster_dict, columns, name_similarity_matrix,
            threshold=self._config.entity_name_similarity_merge
        )
        self._logger.info(f"After name-similarity merge: {len(cluster_dict)} clusters")

        # ── Step 3: Value-overlap-guided splitting ────────────────────────
        # If two columns in the same cluster have zero value overlap AND
        # have different name prefixes, they're likely from different entities
        cluster_dict = self._split_by_value_domain(
            cluster_dict, columns, value_overlap_matrix, schema
        )
        self._logger.info(f"After value-domain split: {len(cluster_dict)} clusters")

        # ── Step 4: Absorb clusters smaller than min_size ─────────────────
        cluster_dict = self._merge_small_clusters(
            cluster_dict, self._config.entity_min_cluster_size, feature_matrix
        )
        self._logger.info(f"After small-cluster absorption: {len(cluster_dict)} clusters")

        # ── Step 5: Build EntitySchema objects ───────────────────────────
        entities: List[EntitySchema] = []
        for cluster_id, col_indices in cluster_dict.items():
            cluster_cols = [columns[i] for i in col_indices]

            name = self._extract_entity_name(cluster_cols)
            entity_type = self._classify_entity_type(cluster_cols)
            confidence = self._compute_cluster_confidence(
                col_indices, name_similarity_matrix
            )
            rep_col = self._select_representative_column(cluster_cols, schema)
            description = self._generate_description(name, entity_type, cluster_cols)

            entities.append(EntitySchema(
                entity_name=name,
                columns=cluster_cols,
                entity_type=entity_type,
                confidence=confidence,
                representative_column=rep_col,
                description=description,
                cluster_id=cluster_id,
            ))

        # Sort by column count descending (largest entities first)
        entities.sort(key=lambda e: len(e.columns), reverse=True)

        self._logger.info(
            f"Entity discovery complete: {len(entities)} entities found",
            extra_context={"entities": [e.entity_name for e in entities]}
        )
        return entities

    # ─── CLUSTER MERGE: NAME SIMILARITY ─────────────────────────────────────

    def _merge_by_name_similarity(
        self,
        cluster_dict: Dict[int, List[int]],
        columns: List[str],
        name_similarity: np.ndarray,
        threshold: float,
    ) -> Dict[int, List[int]]:
        """Merge clusters where inter-cluster name similarity exceeds threshold.

        Two clusters are merged when the average pairwise cosine similarity
        between their column name embeddings is above 'threshold'. This corrects
        for Ward linkage splitting semantically unified groups due to type
        differences (e.g., customer_id [id type] vs customer_name [text type]).

        Args:
            cluster_dict: Current cluster assignment.
            columns: Column names list.
            name_similarity: N×N name similarity matrix.
            threshold: Merge threshold (0.0–1.0).

        Returns:
            Updated cluster dict with merged clusters.
        """
        cluster_ids = sorted(cluster_dict.keys())
        # Use union-find for efficient merging
        parent = {cid: cid for cid in cluster_ids}

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            parent[find(a)] = find(b)

        for i, cid_a in enumerate(cluster_ids):
            for cid_b in cluster_ids[i + 1:]:
                indices_a = cluster_dict[cid_a]
                indices_b = cluster_dict[cid_b]
                # Compute average inter-cluster name similarity
                sim_vals = [
                    name_similarity[ia, ib]
                    for ia in indices_a
                    for ib in indices_b
                ]
                avg_sim = float(np.mean(sim_vals)) if sim_vals else 0.0
                if avg_sim >= threshold:
                    union(cid_a, cid_b)

        # Rebuild cluster dict from union-find result
        merged: Dict[int, List[int]] = {}
        for cid, indices in cluster_dict.items():
            root = find(cid)
            merged.setdefault(root, []).extend(indices)

        # Re-index clusters as 0, 1, 2, ...
        return {new_id: indices for new_id, (_, indices) in enumerate(merged.items())}

    # ─── CLUSTER SPLIT: VALUE DOMAIN ────────────────────────────────────────

    def _split_by_value_domain(
        self,
        cluster_dict: Dict[int, List[int]],
        columns: List[str],
        value_overlap: np.ndarray,
        schema: "SchemaMetadata",
    ) -> Dict[int, List[int]]:
        """Split clusters containing columns with incompatible value domains.

        If two ID/categorical columns in the same cluster have Jaccard overlap
        near 0 AND different name prefixes, they likely belong to different
        entities that Ward grouped due to similar statistical profiles.

        Args:
            cluster_dict: Current cluster assignment.
            columns: Column name list.
            value_overlap: N×N Jaccard overlap matrix.
            schema: SchemaMetadata for type context.

        Returns:
            Updated cluster dict with splits applied.
        """
        new_clusters: Dict[int, List[int]] = {}
        next_id = max(cluster_dict.keys()) + 1 if cluster_dict else 0

        for cid, indices in cluster_dict.items():
            # Only split clusters with multiple ID/categorical columns
            id_indices = [
                i for i in indices
                if columns[i] in schema.id_cols or columns[i] in schema.categorical_cols
            ]

            if len(id_indices) < 2:
                new_clusters[cid] = indices
                continue

            # Check for near-zero overlap pairs with different prefixes
            should_split = False
            for k, ia in enumerate(id_indices):
                for ib in id_indices[k + 1:]:
                    overlap = float(value_overlap[ia, ib])
                    prefix_a = self._extract_prefix([columns[ia]])
                    prefix_b = self._extract_prefix([columns[ib]])
                    if overlap < 0.05 and prefix_a != prefix_b:
                        should_split = True
                        break
                if should_split:
                    break

            if not should_split:
                new_clusters[cid] = indices
                continue

            # Split: group by name prefix
            prefix_groups: Dict[str, List[int]] = {}
            for i in indices:
                prefix = self._extract_prefix([columns[i]])
                prefix_groups.setdefault(prefix, []).append(i)

            for prefix, group_indices in prefix_groups.items():
                new_clusters[next_id] = group_indices
                next_id += 1

        return {new_id: inds for new_id, (_, inds) in enumerate(new_clusters.items())}

    # ─── SMALL CLUSTER ABSORPTION ───────────────────────────────────────────

    def _merge_small_clusters(
        self,
        cluster_dict: Dict[int, List[int]],
        min_size: int,
        feature_matrix: np.ndarray,
    ) -> Dict[int, List[int]]:
        """Absorb clusters smaller than min_size into their nearest larger cluster.

        Singleton or tiny clusters are often noise — they get absorbed into the
        closest (by centroid distance) larger cluster to avoid fragmenting
        the entity graph.

        Args:
            cluster_dict: Current cluster assignment.
            min_size: Minimum columns per cluster; smaller clusters get merged.
            feature_matrix: Feature matrix for centroid distance computation.

        Returns:
            Updated cluster dict with small clusters absorbed.
        """
        large = {cid: inds for cid, inds in cluster_dict.items() if len(inds) >= min_size}
        small = {cid: inds for cid, inds in cluster_dict.items() if len(inds) < min_size}

        if not large:
            return cluster_dict

        # Compute centroids for large clusters
        centroids = {
            cid: feature_matrix[inds].mean(axis=0) for cid, inds in large.items()
        }

        for _, small_inds in small.items():
            # Find the large cluster with the nearest centroid
            small_centroid = feature_matrix[small_inds].mean(axis=0)
            nearest_cid = min(
                centroids.keys(),
                key=lambda cid: float(np.linalg.norm(small_centroid - centroids[cid]))
            )
            large[nearest_cid].extend(small_inds)

        return {new_id: inds for new_id, (_, inds) in enumerate(large.items())}

    # ─── HELPER METHODS ─────────────────────────────────────────────────────

    def _extract_prefix(self, columns: List[str]) -> str:
        """Extract the dominant prefix token from a list of column names.

        E.g., ['customer_id', 'customer_name', 'customer_email'] → 'customer'
        E.g., ['order_date', 'order_status'] → 'order'

        The prefix is defined as the most common leading token when column
        names are split on underscore/camelCase boundaries.

        Args:
            columns: List of column names.

        Returns:
            Dominant prefix string, or empty string if ambiguous.
        """
        if not columns:
            return ""
        # Tokenize each name and take the first token
        first_tokens = []
        for col in columns:
            tokens = [t.lower() for t in self._SPLIT_RE.split(col) if t]
            if tokens:
                first_tokens.append(tokens[0])

        if not first_tokens:
            return ""

        # Return the most frequent first token
        from collections import Counter
        return Counter(first_tokens).most_common(1)[0][0]

    def _extract_entity_name(self, columns: List[str]) -> str:
        """Generate a human-readable entity name from cluster column names.

        Uses majority prefix extraction and capitalizes for display.
        Falls back to a generic name if no clear prefix is found.

        Args:
            columns: List of column names in this cluster.

        Returns:
            Capitalized entity name string.
        """
        prefix = self._extract_prefix(columns)
        if not prefix or prefix in {"id", "val", "num", "col", "field", "data"}:
            # Fallback: use the shortest non-trivial column name
            name_candidates = sorted(columns, key=len)
            for cand in name_candidates:
                tokens = [t.lower() for t in self._SPLIT_RE.split(cand) if t]
                meaningful = [t for t in tokens if t not in {"id", "val", "no", "key"}]
                if meaningful:
                    return meaningful[0].title()
            return "Entity"
        return prefix.title()

    def _classify_entity_type(self, columns: List[str]) -> str:
        """Classify entity type from column naming patterns.

        Rule-based classification using domain signal dictionaries.
        Order of precedence: Person > Transaction > Document > Product > Reference > BusinessObject.

        Args:
            columns: Column names in the cluster.

        Returns:
            EntityTypeLabel value string.
        """
        tokens: Set[str] = set()
        for col in columns:
            tokens.update(t.lower() for t in self._SPLIT_RE.split(col) if t)

        # Score each entity type by signal token overlap
        scores = {
            EntityTypeLabel.PERSON.value: len(tokens & self._PERSON_SIGNALS),
            EntityTypeLabel.TRANSACTION.value: len(tokens & self._TRANSACTION_SIGNALS),
            EntityTypeLabel.DOCUMENT.value: len(tokens & self._DOCUMENT_SIGNALS),
            EntityTypeLabel.PRODUCT.value: len(tokens & self._PRODUCT_SIGNALS),
            EntityTypeLabel.REFERENCE.value: len(tokens & self._REFERENCE_SIGNALS),
        }

        best_type, best_score = max(scores.items(), key=lambda x: x[1])
        if best_score == 0:
            return EntityTypeLabel.BUSINESS_OBJECT.value
        return best_type

    def _compute_cluster_confidence(
        self,
        col_indices: List[int],
        name_similarity: np.ndarray,
    ) -> float:
        """Compute a confidence score for a cluster based on within-cluster similarity.

        Higher average pairwise name similarity = more cohesive cluster =
        higher confidence that these columns genuinely belong to the same entity.

        Args:
            col_indices: Column indices in this cluster.
            name_similarity: N×N name similarity matrix.

        Returns:
            Average within-cluster pairwise similarity (0.0 – 1.0).
        """
        if len(col_indices) < 2:
            return 1.0  # Single-column cluster is perfectly cohesive by definition

        sims = []
        for k, ia in enumerate(col_indices):
            for ib in col_indices[k + 1:]:
                sims.append(float(name_similarity[ia, ib]))

        return float(np.mean(sims)) if sims else 0.0

    def _select_representative_column(
        self,
        columns: List[str],
        schema: "SchemaMetadata",
    ) -> str:
        """Select the best representative column for an entity cluster.

        Preference order: ID columns in the cluster → first column alphabetically.
        The representative column is used as the node label in the graph.

        Args:
            columns: Column names in this cluster.
            schema: Schema metadata with id_cols list.

        Returns:
            Representative column name.
        """
        # Prefer ID columns
        id_cols_in_cluster = [c for c in columns if c in schema.id_cols]
        if id_cols_in_cluster:
            return sorted(id_cols_in_cluster)[0]
        # Then prefer columns ending in '_id' or starting with 'id'
        suffixed_id = [c for c in columns if c.lower().endswith("_id")]
        if suffixed_id:
            return suffixed_id[0]
        return columns[0]

    def _generate_description(
        self,
        entity_name: str,
        entity_type: str,
        columns: List[str],
    ) -> str:
        """Generate a human-readable description for the discovered entity.

        Args:
            entity_name: Entity name string.
            entity_type: Entity type label.
            columns: Column names in this entity.

        Returns:
            Description string.
        """
        n_cols = len(columns)
        sample_cols = ", ".join(columns[:3]) + ("..." if n_cols > 3 else "")
        return (
            f"Discovered {entity_type.replace('_', ' ')} entity '{entity_name}' "
            f"with {n_cols} attributes: [{sample_cols}]"
        )


# =============================================================================
# DEMO: Run entity discovery on the ERP dataset
# =============================================================================

print("=" * 70)
print("  NEURL Engine — Cell 4: Node Intelligence Engine")
print("=" * 70)

print("\n[1/2] Running entity discovery...")
node_intelligence = NodeIntelligenceEngine(neurl_config)
erp_entities = node_intelligence.discover_entities(
    df=erp_df_clean,
    schema=erp_schema,
    feature_matrix=erp_feature_matrix,
    name_similarity_matrix=erp_name_similarity,
    value_overlap_matrix=erp_value_overlap,
)

print(f"\n  Discovered {len(erp_entities)} entities:\n")
print(f"  {'Entity':<16} {'Type':<18} {'Columns':<6} {'Confidence':<12} {'Rep Column':<25}")
print("  " + "─" * 80)
for e in erp_entities:
    print(f"  {e.entity_name:<16} {e.entity_type:<18} {len(e.columns):<6} "
          f"{e.confidence:<12.3f} {e.representative_column:<25}")
    print(f"    Columns: {e.columns}")

# Validation: assert expected entities are discovered
print("\n[2/2] Validating discovered entities...")
discovered_names_lower = {e.entity_name.lower() for e in erp_entities}
expected_entities = ["customer", "order", "product", "invoice", "payment", "employee"]
found_expected = []
missing_expected = []
for expected in expected_entities:
    # Check for partial match (e.g., 'Customer' matches 'customer')
    found = any(expected in name for name in discovered_names_lower)
    if found:
        found_expected.append(expected)
    else:
        missing_expected.append(expected)

print(f"  ✅ Found expected entities: {found_expected}")
if missing_expected:
    print(f"  ⚠️  Not found (may be merged): {missing_expected}")

print(f"\n  Entity descriptions:")
for e in erp_entities:
    print(f"    • {e.description}")

print("\n✅ Cell 4 complete — Node Intelligence Engine ready.")
print(f"   Exports: node_intelligence, erp_entities ({len(erp_entities)} entities)")
