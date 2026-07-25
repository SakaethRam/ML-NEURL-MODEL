"""
NEURL Engine — Cell 5: Relationship Intelligence

Discovers directed relationships between EntitySchema objects using four
complementary evidence signals: foreign key patterns, prefix-based inference,
numeric correlation, and temporal ordering.

Design rationale: No single signal is sufficient — a pure FK detector misses
correlations, and correlation alone generates many false positives. The
confidence-weighted fusion of all four signals gives robust relationship
discovery that generalizes across diverse dataset structures.
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd


# =============================================================================
# DATA CONTAINER — RelationshipSchema
# =============================================================================

@dataclass
class RelationshipSchema:
    """Describes a discovered directed relationship between two entities.

    Each RelationshipSchema corresponds to a directed edge in the knowledge
    graph connecting two entity nodes.

    Attributes:
        source_entity: Name of the originating entity.
        target_entity: Name of the destination entity.
        relationship_type: Semantic type from RelationshipTypeLabel.
        relationship_name: Verb phrase describing the relationship (e.g., 'PLACES').
        confidence: Weighted evidence confidence score (0.0 – 1.0).
        supporting_evidence: List of evidence description strings.
        cardinality: '1:1', '1:N', or 'N:N'.
        direction: 'forward', 'backward', or 'bidirectional'.
        linking_columns: Tuple of (source_col, target_col) that evidence the link.
    """
    source_entity: str
    target_entity: str
    relationship_type: str
    relationship_name: str
    confidence: float
    supporting_evidence: List[str]
    cardinality: str
    direction: str
    linking_columns: Tuple[str, str] = ("", "")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "source": self.source_entity,
            "target": self.target_entity,
            "type": self.relationship_type,
            "name": self.relationship_name,
            "confidence": round(self.confidence, 3),
            "cardinality": self.cardinality,
            "direction": self.direction,
            "evidence": self.supporting_evidence,
        }


# =============================================================================
# RelationshipIntelligenceEngine
# =============================================================================

class RelationshipIntelligenceEngine:
    """Discovers relationships between business entities using statistical signals.

    Evidence signals and their base confidence contributions:
    1. FK-pattern (value subset): +0.5
    2. Prefix-based inference (entity_B_id in entity_A): +0.4
    3. Pearson correlation (numeric columns): +0.3 × |correlation|
    4. Temporal ordering (earlier entity → later entity): +0.2

    Total confidence is clipped to [0, 1] and relationships below the
    configured threshold are pruned.

    Example:
        engine = RelationshipIntelligenceEngine(config)
        relationships = engine.discover_relationships(df, entities, schema)
    """

    # Semantic relationship verb map: (source_type, target_type) → verb
    _VERB_MAP = {
        ("person", "transaction"): "PLACES",
        ("business_object", "transaction"): "PLACES",
        ("transaction", "document"): "GENERATES",
        ("document", "transaction"): "SETTLES",
        ("business_object", "product"): "ORDERS",
        ("person", "document"): "RECEIVES",
        ("transaction", "product"): "CONTAINS",
        ("reference", "transaction"): "CLASSIFIES",
        ("person", "business_object"): "MANAGES",
    }

    _SPLIT_RE = re.compile(r"[_\-\s]+|(?<=[a-z])(?=[A-Z])")

    def __init__(self, config: "NEURLConfig") -> None:
        """Initialize with global config.

        Args:
            config: NEURLConfig master configuration object.
        """
        self._config = config
        self._logger = NEURLLogger.get_logger("RelationshipIntelligenceEngine")

    # ─── MAIN DISCOVERY METHOD ───────────────────────────────────────────────

    def discover_relationships(
        self,
        df: pd.DataFrame,
        entities: List["EntitySchema"],
        schema: "SchemaMetadata",
    ) -> List["RelationshipSchema"]:
        """Discover directed relationships between entity pairs.

        Runs all four evidence signals over all entity pairs and fuses
        their confidence contributions. Results are deduplicated and
        pruned below the configured threshold.

        Args:
            df: Cleaned DataFrame.
            entities: Discovered EntitySchema list from Cell 4.
            schema: SchemaMetadata from Cell 2.

        Returns:
            List of RelationshipSchema objects sorted by confidence descending.
        """
        self._logger.info(
            f"Discovering relationships among {len(entities)} entities"
        )

        # Build entity column lookup for quick access
        entity_col_map: Dict[str, List[str]] = {
            e.entity_name: e.columns for e in entities
        }
        entity_type_map: Dict[str, str] = {
            e.entity_name: e.entity_type for e in entities
        }

        # Accumulate evidence for each entity pair
        # Key: (source_name, target_name) → evidence accumulator dict
        evidence_acc: Dict[Tuple[str, str], Dict[str, Any]] = {}

        def _get_or_init(src: str, tgt: str) -> Dict[str, Any]:
            key = (src, tgt)
            if key not in evidence_acc:
                evidence_acc[key] = {
                    "confidence": 0.0,
                    "evidence": [],
                    "cardinality": CardinalityType.ONE_TO_MANY.value,
                    "linking_cols": ("", ""),
                }
            return evidence_acc[key]

        entity_names = [e.entity_name for e in entities]
        n_entities = len(entity_names)

        for i in range(n_entities):
            ent_a = entity_names[i]
            cols_a = entity_col_map[ent_a]

            for j in range(n_entities):
                if i == j:
                    continue
                ent_b = entity_names[j]
                cols_b = entity_col_map[ent_b]

                # ── Signal 1: FK-pattern detection ───────────────────────
                fk_conf, fk_evidence, fk_link = self._detect_fk_pattern(
                    df, cols_a, cols_b, ent_a, ent_b, schema
                )
                if fk_conf > 0:
                    rec = _get_or_init(ent_a, ent_b)
                    rec["confidence"] += fk_conf
                    rec["evidence"].extend(fk_evidence)
                    if fk_link[0]:
                        rec["linking_cols"] = fk_link
                        rec["cardinality"] = self._detect_cardinality(
                            df, fk_link[0], fk_link[1]
                        )

                # ── Signal 2: Prefix-based inference ─────────────────────
                prefix_conf, prefix_evidence = self._detect_prefix_pattern(
                    cols_a, ent_b, ent_a
                )
                if prefix_conf > 0:
                    rec = _get_or_init(ent_a, ent_b)
                    rec["confidence"] += prefix_conf
                    rec["evidence"].extend(prefix_evidence)

                # ── Signal 3: Numeric correlation ─────────────────────────
                # Only check one direction to avoid double-counting
                if i < j:
                    corr_conf, corr_evidence = self._detect_correlation(
                        df, cols_a, cols_b, ent_a, ent_b, schema
                    )
                    if corr_conf > 0:
                        rec = _get_or_init(ent_a, ent_b)
                        rec["confidence"] += corr_conf
                        rec["evidence"].extend(corr_evidence)

                # ── Signal 4: Temporal ordering ───────────────────────────
                temp_conf, temp_evidence, temp_dir = self._detect_temporal_ordering(
                    df, cols_a, cols_b, ent_a, ent_b, schema
                )
                if temp_conf > 0:
                    rec = _get_or_init(ent_a, ent_b)
                    rec["confidence"] += temp_conf
                    rec["evidence"].extend(temp_evidence)

        # Build RelationshipSchema objects, prune, and deduplicate
        relationships: List[RelationshipSchema] = []
        seen: Set[Tuple[str, str]] = set()

        for (src, tgt), acc in evidence_acc.items():
            conf = min(acc["confidence"], 1.0)
            if conf < self._config.relationship_confidence_threshold:
                continue

            # Deduplicate bidirectional pairs (keep higher confidence direction)
            reverse_key = (tgt, src)
            if reverse_key in seen:
                continue
            seen.add((src, tgt))

            rel_type = self._classify_relationship_type(
                entity_type_map.get(src, "unknown"),
                entity_type_map.get(tgt, "unknown"),
                acc["cardinality"]
            )
            rel_name = self._infer_relationship_name(
                entity_type_map.get(src, "unknown"),
                entity_type_map.get(tgt, "unknown"),
                acc["evidence"]
            )

            relationships.append(RelationshipSchema(
                source_entity=src,
                target_entity=tgt,
                relationship_type=rel_type,
                relationship_name=rel_name,
                confidence=conf,
                supporting_evidence=acc["evidence"][:5],  # top-5 evidence strings
                cardinality=acc["cardinality"],
                direction="forward",
                linking_columns=acc["linking_cols"],
            ))

        relationships.sort(key=lambda r: r.confidence, reverse=True)

        self._logger.info(
            f"Relationship discovery complete: {len(relationships)} relationships",
            extra_context={"relationships": [(r.source_entity, r.target_entity)
                                              for r in relationships]}
        )
        return relationships

    # ─── SIGNAL 1: FK-PATTERN DETECTION ─────────────────────────────────────

    def _detect_fk_pattern(
        self,
        df: pd.DataFrame,
        cols_a: List[str],
        cols_b: List[str],
        ent_a: str,
        ent_b: str,
        schema: "SchemaMetadata",
    ) -> Tuple[float, List[str], Tuple[str, str]]:
        """Detect FK relationships by value-subset analysis.

        If values of col_x (from entity A) are > fk_threshold% contained
        in col_y (from entity B), then A references B via col_x → col_y.

        Args:
            df: Cleaned DataFrame.
            cols_a: Columns belonging to entity A.
            cols_b: Columns belonging to entity B.
            ent_a: Entity A name.
            ent_b: Entity B name.
            schema: Schema metadata.

        Returns:
            Tuple of (confidence, evidence_list, (link_col_a, link_col_b)).
        """
        threshold = self._config.fk_subset_threshold
        id_a = [c for c in cols_a if c in schema.id_cols]
        id_b = [c for c in cols_b if c in schema.id_cols]

        for col_a in id_a:
            vals_a = set(df[col_a].dropna().astype(str).unique())
            if len(vals_a) < 2:
                continue
            for col_b in id_b:
                vals_b = set(df[col_b].dropna().astype(str).unique())
                if len(vals_b) < 2:
                    continue
                overlap = len(vals_a & vals_b) / len(vals_a)
                if overlap >= threshold:
                    evidence = [
                        f"FK: {col_a} ⊆ {col_b} ({overlap:.0%} overlap)",
                    ]
                    return 0.5, evidence, (col_a, col_b)

        return 0.0, [], ("", "")

    # ─── SIGNAL 2: PREFIX-BASED INFERENCE ───────────────────────────────────

    def _detect_prefix_pattern(
        self,
        cols_a: List[str],
        entity_b_name: str,
        entity_a_name: str,
    ) -> Tuple[float, List[str]]:
        """Detect relationships where entity A contains a column named '{entity_B}_id'.

        This is the most common pattern in denormalized data: if an Order table
        has a 'customer_id' column, Order references Customer.

        Args:
            cols_a: Columns of entity A.
            entity_b_name: Name of entity B.
            entity_a_name: Name of entity A (for evidence string).

        Returns:
            Tuple of (confidence, evidence_list).
        """
        b_name_lower = entity_b_name.lower()
        for col in cols_a:
            col_lower = col.lower()
            tokens = [t.lower() for t in self._SPLIT_RE.split(col) if t]
            # Check if entity B's name appears in column tokens followed by 'id'/'key'
            for k, tok in enumerate(tokens):
                if tok == b_name_lower or tok.startswith(b_name_lower[:4]):
                    if k + 1 < len(tokens) and tokens[k + 1] in {"id", "key", "no", "num"}:
                        evidence = [
                            f"Prefix: {entity_a_name}.{col} references {entity_b_name}"
                        ]
                        return 0.4, evidence

        return 0.0, []

    # ─── SIGNAL 3: NUMERIC CORRELATION ──────────────────────────────────────

    def _detect_correlation(
        self,
        df: pd.DataFrame,
        cols_a: List[str],
        cols_b: List[str],
        ent_a: str,
        ent_b: str,
        schema: "SchemaMetadata",
    ) -> Tuple[float, List[str]]:
        """Detect dependency between entities via numeric column correlation.

        High Pearson correlation between numeric columns of two different entities
        suggests they're driven by the same underlying process (e.g., order_amount
        and invoice_amount are highly correlated because they're from the same transaction).

        Args:
            df: Cleaned DataFrame.
            cols_a: Numeric columns of entity A.
            cols_b: Numeric columns of entity B.
            ent_a: Entity A name.
            ent_b: Entity B name.
            schema: Schema metadata.

        Returns:
            Tuple of (confidence, evidence_list).
        """
        num_a = [c for c in cols_a if c in schema.numeric_cols]
        num_b = [c for c in cols_b if c in schema.numeric_cols]

        max_corr = 0.0
        best_pair = ("", "")
        for ca in num_a:
            for cb in num_b:
                try:
                    corr = abs(float(df[ca].corr(df[cb])))
                    if np.isnan(corr):
                        continue
                    if corr > max_corr:
                        max_corr = corr
                        best_pair = (ca, cb)
                except Exception:
                    continue

        if max_corr >= self._config.correlation_relationship_threshold:
            conf = 0.3 * max_corr
            evidence = [
                f"Correlation: {best_pair[0]} ↔ {best_pair[1]} (r={max_corr:.2f})"
            ]
            return conf, evidence

        return 0.0, []

    # ─── SIGNAL 4: TEMPORAL ORDERING ─────────────────────────────────────────

    def _detect_temporal_ordering(
        self,
        df: pd.DataFrame,
        cols_a: List[str],
        cols_b: List[str],
        ent_a: str,
        ent_b: str,
        schema: "SchemaMetadata",
    ) -> Tuple[float, List[str], str]:
        """Detect entity ordering based on mean datetime values.

        If entity A's dates consistently precede entity B's dates, this
        suggests A→B direction (e.g., Order precedes Invoice which precedes Payment).

        Args:
            df: Cleaned DataFrame.
            cols_a: Columns of entity A.
            cols_b: Columns of entity B.
            ent_a: Entity A name.
            ent_b: Entity B name.
            schema: Schema metadata.

        Returns:
            Tuple of (confidence, evidence_list, direction_string).
        """
        date_a = [c for c in cols_a if c in schema.datetime_cols]
        date_b = [c for c in cols_b if c in schema.datetime_cols]

        if not date_a or not date_b:
            return 0.0, [], "unknown"

        try:
            mean_a = pd.to_datetime(df[date_a[0]], errors="coerce").dropna().mean()
            mean_b = pd.to_datetime(df[date_b[0]], errors="coerce").dropna().mean()
            if pd.isna(mean_a) or pd.isna(mean_b):
                return 0.0, [], "unknown"
            direction = "forward" if mean_a <= mean_b else "backward"
            days_diff = abs((mean_b - mean_a).days)
            evidence = [
                f"Temporal: {ent_a}.{date_a[0]} precedes {ent_b}.{date_b[0]} "
                f"by ~{days_diff} days on avg"
            ]
            return 0.2, evidence, direction
        except Exception:
            return 0.0, [], "unknown"

    # ─── CARDINALITY DETECTION ───────────────────────────────────────────────

    def _detect_cardinality(
        self,
        df: pd.DataFrame,
        col_a: str,
        col_b: str,
    ) -> str:
        """Estimate relationship cardinality from linking column distributions.

        Logic: if col_a has many rows sharing the same col_b value,
        it's N:1 (many A per B, i.e., 1:N from B's perspective).

        Args:
            df: DataFrame containing both columns.
            col_a: Source (child) column.
            col_b: Target (parent) column.

        Returns:
            CardinalityType value string.
        """
        if col_a not in df.columns or col_b not in df.columns:
            return CardinalityType.ONE_TO_MANY.value
        try:
            # Average number of col_a values per unique col_b value
            avg_a_per_b = df.groupby(col_b)[col_a].nunique().mean()
            avg_b_per_a = df.groupby(col_a)[col_b].nunique().mean()

            if avg_a_per_b <= 1.1 and avg_b_per_a <= 1.1:
                return CardinalityType.ONE_TO_ONE.value
            elif avg_b_per_a > 1.5:
                return CardinalityType.MANY_TO_MANY.value
            else:
                return CardinalityType.ONE_TO_MANY.value
        except Exception:
            return CardinalityType.ONE_TO_MANY.value

    # ─── SEMANTIC NAMING ─────────────────────────────────────────────────────

    def _infer_relationship_name(
        self,
        source_type: str,
        target_type: str,
        evidence: List[str],
    ) -> str:
        """Generate a semantic verb phrase for a relationship.

        Uses a lookup table of (source_type, target_type) → verb.
        Falls back to RELATED_TO for unknown type combinations.

        Args:
            source_type: Entity type of the source.
            target_type: Entity type of the target.
            evidence: List of evidence strings (used for context).

        Returns:
            Uppercase verb phrase string.
        """
        key = (source_type, target_type)
        # Try exact match, then partial match
        if key in self._VERB_MAP:
            return self._VERB_MAP[key]
        for (s, t), verb in self._VERB_MAP.items():
            if s in source_type or t in target_type:
                return verb
        return RelationshipTypeLabel.RELATED_TO.value

    def _classify_relationship_type(
        self,
        source_type: str,
        target_type: str,
        cardinality: str,
    ) -> str:
        """Classify relationship into a semantic type label.

        Args:
            source_type: Entity type of the source.
            target_type: Entity type of the target.
            cardinality: '1:1', '1:N', or 'N:N'.

        Returns:
            RelationshipTypeLabel value string.
        """
        if target_type == EntityTypeLabel.TRANSACTION.value:
            return RelationshipTypeLabel.GENERATES.value
        if target_type == EntityTypeLabel.DOCUMENT.value:
            return RelationshipTypeLabel.GENERATES.value
        if source_type == EntityTypeLabel.TRANSACTION.value:
            return RelationshipTypeLabel.PROCESSES.value
        if cardinality == CardinalityType.ONE_TO_ONE.value:
            return RelationshipTypeLabel.HAS.value
        if cardinality == CardinalityType.MANY_TO_MANY.value:
            return RelationshipTypeLabel.REFERENCES.value
        return RelationshipTypeLabel.HAS.value


# =============================================================================
# DEMO: Run relationship discovery on the ERP dataset
# =============================================================================

print("=" * 70)
print("  NEURL Engine — Cell 5: Relationship Intelligence")
print("=" * 70)

print("\n[1/2] Running relationship discovery...")
relationship_engine = RelationshipIntelligenceEngine(neurl_config)
erp_relationships = relationship_engine.discover_relationships(
    df=erp_df_clean,
    entities=erp_entities,
    schema=erp_schema,
)

print(f"\n  Discovered {len(erp_relationships)} relationships:\n")
print(f"  {'Source':<16} {'→'} {'Target':<16} {'Type':<14} {'Card':<6} {'Conf':<8}")
print("  " + "─" * 70)
for r in erp_relationships:
    print(f"  {r.source_entity:<16} → {r.target_entity:<16} "
          f"{r.relationship_name:<14} {r.cardinality:<6} {r.confidence:.3f}")
    for ev in r.supporting_evidence[:2]:
        print(f"    ↳ {ev}")

print(f"\n[2/2] Relationship JSON sample:")
if erp_relationships:
    import json
    print(json.dumps(erp_relationships[0].to_dict(), indent=4))

print("\n✅ Cell 5 complete — Relationship Intelligence Engine ready.")
print(f"   Exports: relationship_engine, erp_relationships ({len(erp_relationships)} relationships)")
