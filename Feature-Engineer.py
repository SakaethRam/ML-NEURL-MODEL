"""
NEURL Engine — Cell 3: Feature Engineering

Converts raw DataFrames into a rich numeric feature representation where
each COLUMN (not row) becomes a feature vector. This column-as-node
representation is the key insight: we're building a graph of schema
structure, not a graph of data records.

Design rationale: By computing statistical, linguistic, and structural
features for each column, we enable ML-based entity discovery (Cell 4)
that learns which columns belong to the same business entity cluster.
"""

from __future__ import annotations
import re
from typing import Dict, List, Optional, Tuple, Any
from collections import Counter

import numpy as np
import pandas as pd


# =============================================================================
# Pure-numpy fallback implementations for optional ML deps
# =============================================================================

def _tfidf_fallback(texts: List[str]) -> np.ndarray:
    """Simple token-overlap TF matrix when sklearn is unavailable."""
    vocab = sorted(set(t for text in texts for t in text.split()))
    vocab_idx = {w: i for i, w in enumerate(vocab)}
    mat = np.zeros((len(texts), max(len(vocab), 1)), dtype=float)
    for i, text in enumerate(texts):
        for t in text.split():
            if t in vocab_idx:
                mat[i, vocab_idx[t]] += 1
    # TF normalization
    row_sums = mat.sum(axis=1, keepdims=True) + 1e-8
    return mat / row_sums


def _cosine_sim(A: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity matrix when sklearn is unavailable."""
    norms = np.linalg.norm(A, axis=1, keepdims=True) + 1e-8
    A_norm = A / norms
    return A_norm @ A_norm.T


def _entropy_fallback(probs: np.ndarray) -> float:
    """Shannon entropy from probability array when scipy is unavailable."""
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log(probs + 1e-10)))


# =============================================================================
# FeatureEngineer — column-level feature computation
# =============================================================================

class FeatureEngineer:
    """Computes multi-modal feature representations for DataFrame columns.

    Each column in the input DataFrame is converted to a numeric feature
    vector capturing statistical properties, linguistic patterns (column name),
    and structural relationships (value overlap between columns).

    This is the bridge between raw tabular data and the GNN's input space:
    the feature matrix X ∈ R^{N × D} where N = number of columns and
    D = feature vector dimension.

    Example:
        fe = FeatureEngineer(config)
        feat_matrix, feat_names, col_names = fe.build_column_feature_matrix(df, schema)
    """

    # Column name split pattern: split on underscores and camelCase transitions
    _NAME_SPLIT_PATTERN = re.compile(r"[_\-\s]+|(?<=[a-z])(?=[A-Z])")

    def __init__(self, config: "NEURLConfig") -> None:
        """Initialize the FeatureEngineer with global config.

        Args:
            config: NEURLConfig master configuration object.
        """
        self._config = config
        self._logger = NEURLLogger.get_logger("FeatureEngineer")

    # ─── COLUMN METADATA FEATURES ───────────────────────────────────────────

    def compute_column_metadata(
        self,
        df: pd.DataFrame,
        schema: "SchemaMetadata"
    ) -> Dict[str, np.ndarray]:
        """Compute a ~15-dim numeric feature vector for every column.

        Features capture: dtype category, missingness, uniqueness, basic
        statistics, text richness, ID-likeness, date-likeness, information
        content (entropy), and top-K frequency distribution. Together these
        encode the 'character' of each column in a form the GNN can process.

        Args:
            df: Cleaned DataFrame.
            schema: Inferred schema metadata for type context.

        Returns:
            Dict mapping column name → numpy feature vector (shape: [n_features]).
        """
        col_features: Dict[str, np.ndarray] = {}

        # Type → integer code mapping (matches ArchitectureType ordering)
        TYPE_CODES = {
            "numeric": 0, "categorical": 1, "datetime": 2,
            "boolean": 3, "text": 4, "id": 5, "unknown": 6
        }

        for col in df.columns:
            series = df[col]
            n = len(series)
            dtype_label = schema.column_types.get(col, "unknown")

            # --- Structural features ---
            dtype_code = float(TYPE_CODES.get(dtype_label, 6))
            null_rate = float(series.isna().mean())
            n_unique = int(series.nunique(dropna=True))
            unique_ratio = float(n_unique / max(n, 1))

            # --- Numeric statistics (0.0 if not numeric) ---
            if dtype_label == "numeric":
                numeric_vals = pd.to_numeric(series, errors="coerce").dropna()
                feat_mean = float(numeric_vals.mean()) if len(numeric_vals) > 0 else 0.0
                feat_std = float(numeric_vals.std()) if len(numeric_vals) > 0 else 0.0
                feat_min = float(numeric_vals.min()) if len(numeric_vals) > 0 else 0.0
                feat_max = float(numeric_vals.max()) if len(numeric_vals) > 0 else 0.0
                # Normalize to avoid scale dominance
                feat_range = feat_max - feat_min
                feat_mean_norm = feat_mean / (feat_range + 1e-9) if feat_range > 0 else 0.0
                feat_cv = feat_std / (abs(feat_mean) + 1e-9)  # coefficient of variation
            else:
                feat_mean = feat_std = feat_min = feat_max = 0.0
                feat_mean_norm = feat_cv = 0.0

            # --- Text richness ---
            if series.dtype == object or dtype_label in ("text", "categorical", "id"):
                avg_token_count = float(
                    series.dropna().astype(str).str.split().apply(len).mean()
                ) if series.dropna().shape[0] > 0 else 0.0
            else:
                avg_token_count = 0.0

            # --- ID-like and date-like flags ---
            is_id_like = float(col in schema.id_cols or dtype_label == "id")
            is_date_like = float(col in schema.datetime_cols or dtype_label == "datetime")

            # --- Information entropy (Shannon entropy of value distribution) ---
            entropy_val = self._compute_entropy(series, n_bins=self._config.features.entropy_bins)

            # --- Top-K value frequencies ---
            top_k_freq = self._compute_top_k_frequencies(
                series, k=self._config.features.top_k_frequency
            )

            # --- Assemble feature vector (15 features) ---
            feature_vec = np.array([
                dtype_code,          # [0]  Type category code
                null_rate,           # [1]  Missing rate
                unique_ratio,        # [2]  Unique value ratio
                feat_mean_norm,      # [3]  Normalized mean (numeric)
                feat_cv,             # [4]  Coefficient of variation (numeric)
                feat_min,            # [5]  Min value (numeric)
                feat_max,            # [6]  Max value (numeric)
                avg_token_count,     # [7]  Average token count (text)
                is_id_like,          # [8]  ID-like flag
                is_date_like,        # [9]  Date-like flag
                entropy_val,         # [10] Shannon entropy
                top_k_freq[0],       # [11] Freq of most common value
                top_k_freq[1],       # [12] Freq of 2nd most common
                top_k_freq[2],       # [13] Freq of 3rd most common
                float(n_unique),     # [14] Raw unique count
            ], dtype=np.float32)

            # Replace any NaN/inf that slipped through
            feature_vec = np.nan_to_num(feature_vec, nan=0.0, posinf=1.0, neginf=-1.0)
            col_features[col] = feature_vec

        return col_features

    def _compute_entropy(self, series: pd.Series, n_bins: int = 20) -> float:
        """Compute the Shannon entropy of a column's value distribution.

        Args:
            series: Column data.
            n_bins: Number of bins for continuous histogram estimation.

        Returns:
            Shannon entropy value (nats).
        """
        vals = series.dropna()
        if len(vals) == 0:
            return 0.0
        try:
            if pd.api.types.is_numeric_dtype(vals):
                counts, _ = np.histogram(vals, bins=min(n_bins, vals.nunique()))
            else:
                counts = vals.value_counts().values
            # Normalize to probability distribution before entropy
            probs = counts / (counts.sum() + 1e-10)
            probs = probs[probs > 0]
            # Use scipy if available, else numpy fallback
            if SCIPY_AVAILABLE:
                from scipy.stats import entropy as scipy_entropy
                return float(scipy_entropy(probs))
            else:
                return _entropy_fallback(probs)
        except Exception:
            return 0.0

    def _compute_top_k_frequencies(
        self, series: pd.Series, k: int = 3
    ) -> List[float]:
        """Compute the frequency of the top-K most common values.

        Args:
            series: Column data.
            k: Number of top values to retrieve.

        Returns:
            List of k frequency values (padded with 0.0 if fewer than k unique values).
        """
        n = len(series)
        if n == 0:
            return [0.0] * k
        counts = series.value_counts(normalize=True)
        freqs = counts.values[:k].tolist()
        # Pad with zeros if fewer than k unique values
        freqs += [0.0] * (k - len(freqs))
        return freqs[:k]

    # ─── COLUMN NAME EMBEDDINGS (TF-IDF) ────────────────────────────────────

    def compute_column_name_embeddings(
        self, columns: List[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute TF-IDF embeddings of column names and their pairwise similarity.

        Args:
            columns: List of column name strings.

        Returns:
            Tuple of (tfidf_matrix, similarity_matrix):
                - tfidf_matrix: shape [N, vocab_size]
                - similarity_matrix: shape [N, N], values in [0, 1]
        """
        if len(columns) < 2:
            n = len(columns)
            return np.zeros((n, 1)), np.eye(n)

        # Tokenize column names into space-separated word tokens
        tokenized = [
            " ".join(self._NAME_SPLIT_PATTERN.split(col)).lower()
            for col in columns
        ]

        if SKLEARN_AVAILABLE:
            vectorizer = TfidfVectorizer(
                max_features=self._config.features.tfidf_max_features,
                ngram_range=self._config.features.tfidf_ngram_range,
                analyzer="word",
                sublinear_tf=True,  # log-scaling of term frequencies
            )
            try:
                tfidf_matrix = vectorizer.fit_transform(tokenized).toarray()
            except ValueError:
                # Fallback if vocabulary is empty (e.g., single-char column names)
                tfidf_matrix = np.eye(len(columns))

            # Compute pairwise cosine similarity
            similarity_matrix = cosine_similarity(tfidf_matrix)
        else:
            # Pure-numpy fallback
            tfidf_matrix = _tfidf_fallback(tokenized)
            similarity_matrix = _cosine_sim(tfidf_matrix)

        np.fill_diagonal(similarity_matrix, 1.0)  # Self-similarity = 1

        return tfidf_matrix, similarity_matrix

    # ─── VALUE OVERLAP MATRIX (Jaccard) ─────────────────────────────────────

    def compute_value_overlap_matrix(
        self,
        df: pd.DataFrame,
        schema: "SchemaMetadata"
    ) -> np.ndarray:
        """Compute Jaccard similarity of value sets between column pairs.

        Args:
            df: Cleaned DataFrame.
            schema: Schema metadata for column type filtering.

        Returns:
            Square matrix of shape [N, N] where N = len(df.columns).
        """
        columns = list(df.columns)
        n = len(columns)
        overlap_matrix = np.zeros((n, n), dtype=np.float32)
        np.fill_diagonal(overlap_matrix, 1.0)

        # Only compute for categorical, id, boolean columns (value-set comparison)
        target_cols = schema.categorical_cols + schema.id_cols + schema.boolean_cols
        target_indices = [i for i, col in enumerate(columns) if col in target_cols]

        if len(target_indices) < 2:
            return overlap_matrix

        # Sample to limit computation
        sample_size = min(len(df), self._config.features.value_overlap_sample)
        df_sample = df.sample(n=sample_size, random_state=self._config.random_seed) \
            if len(df) > sample_size else df

        # Precompute value sets for all target columns
        value_sets: Dict[int, set] = {}
        for idx in target_indices:
            col = columns[idx]
            value_sets[idx] = set(df_sample[col].dropna().astype(str).unique())

        # Compute pairwise Jaccard similarity
        for i_pos, i in enumerate(target_indices):
            for j in target_indices[i_pos + 1:]:
                set_i = value_sets[i]
                set_j = value_sets[j]
                union_size = len(set_i | set_j)
                if union_size > 0:
                    jaccard = len(set_i & set_j) / union_size
                    overlap_matrix[i, j] = jaccard
                    overlap_matrix[j, i] = jaccard

        return overlap_matrix

    # ─── COMBINED FEATURE MATRIX ─────────────────────────────────────────────

    def build_column_feature_matrix(
        self,
        df: pd.DataFrame,
        schema: "SchemaMetadata",
    ) -> Tuple[np.ndarray, List[str], List[str]]:
        """Build the combined column feature matrix for GNN input.

        Concatenates metadata features with TF-IDF name embeddings to
        create a rich D-dimensional feature vector per column.

        Args:
            df: Cleaned DataFrame.
            schema: Inferred schema metadata.

        Returns:
            Tuple of:
                - feature_matrix: shape [N_cols, D_features]
                - feature_names: list of feature dimension names
                - column_names: list of column identifiers
        """
        columns = list(df.columns)
        n_cols = len(columns)

        self._logger.info(
            f"Building feature matrix for {n_cols} columns",
            extra_context={"n_cols": n_cols}
        )

        # 1. Compute per-column metadata features
        col_metadata = self.compute_column_metadata(df, schema)
        metadata_matrix = np.stack(
            [col_metadata[col] for col in columns], axis=0
        )  # [N, 15]

        # 2. Compute TF-IDF name embeddings
        tfidf_matrix, _ = self.compute_column_name_embeddings(columns)  # [N, V]

        # Truncate TF-IDF to keep feature matrix manageable (PCA-like truncation)
        max_tfidf_dims = 20
        if tfidf_matrix.shape[1] > max_tfidf_dims:
            # Reduce via SVD (numpy-based PCA) when sklearn is unavailable or for efficiency
            if SKLEARN_AVAILABLE:
                pca = PCA(n_components=max_tfidf_dims, random_state=self._config.random_seed)
                try:
                    tfidf_matrix = pca.fit_transform(tfidf_matrix)
                except Exception:
                    # If PCA fails (e.g., too few samples), truncate directly
                    tfidf_matrix = tfidf_matrix[:, :max_tfidf_dims]
            else:
                # Numpy SVD fallback
                try:
                    X_centered = tfidf_matrix - tfidf_matrix.mean(0)
                    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
                    tfidf_matrix = (X_centered @ Vt[:max_tfidf_dims].T)
                except Exception:
                    tfidf_matrix = tfidf_matrix[:, :max_tfidf_dims]

        # 3. Normalize metadata features
        if SKLEARN_AVAILABLE:
            scaler = StandardScaler()
            metadata_matrix = scaler.fit_transform(metadata_matrix)
        else:
            # Pure-numpy z-score normalization
            metadata_matrix = (metadata_matrix - metadata_matrix.mean(0)) / (
                metadata_matrix.std(0) + 1e-8
            )

        # Replace NaN/inf from normalization
        metadata_matrix = np.nan_to_num(metadata_matrix, nan=0.0, posinf=1.0, neginf=-1.0)

        # 4. Concatenate: [N, 15 + tfidf_dims]
        feature_matrix = np.concatenate([metadata_matrix, tfidf_matrix], axis=1).astype(np.float32)

        # Feature names for interpretability
        metadata_feature_names = [
            "dtype_code", "null_rate", "unique_ratio", "mean_norm", "coeff_var",
            "min_val", "max_val", "avg_tokens", "is_id_like", "is_date_like",
            "entropy", "top1_freq", "top2_freq", "top3_freq", "n_unique",
        ]
        tfidf_feature_names = [f"tfidf_{i}" for i in range(tfidf_matrix.shape[1])]
        feature_names = metadata_feature_names + tfidf_feature_names

        self._logger.info(
            f"Feature matrix built: shape {feature_matrix.shape}",
            extra_context={"shape": list(feature_matrix.shape)}
        )

        return feature_matrix, feature_names, columns


# =============================================================================
# CELL 3 ENTRY POINT
# =============================================================================

feature_engineer = FeatureEngineer(config)

# Build feature matrix
feature_matrix, feature_names, column_names = feature_engineer.build_column_feature_matrix(
    df_clean, schema
)

# Compute column name similarity matrix
_, name_similarity_matrix = feature_engineer.compute_column_name_embeddings(column_names)

# Compute value overlap matrix
value_overlap_matrix = feature_engineer.compute_value_overlap_matrix(df_clean, schema)

print(f"✓ Feature matrix shape     : {feature_matrix.shape}")
print(f"✓ Name similarity matrix   : {name_similarity_matrix.shape}")
print(f"✓ Value overlap matrix     : {value_overlap_matrix.shape}")
print(f"✓ Feature names ({len(feature_names)}): {feature_names[:5]} ...")
print(f"✓ Columns analyzed         : {column_names}")
print(f"✓ sklearn available        : {SKLEARN_AVAILABLE}")
print(f"✓ scipy available          : {SCIPY_AVAILABLE}")
