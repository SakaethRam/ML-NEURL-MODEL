"""
NEURL Engine — Cell 2: Universal Data Ingestion & Validation

Implements the DataIngestionEngine that can load any structured tabular source,
compute a DataQualityReport, clean the data, and infer a SchemaMetadata object.
The schema inference (especially FK-pair detection) feeds directly into
graph construction in later cells.

Design rationale: Format-agnostic loading with automatic type inference means
the engine works on any enterprise dataset without manual configuration.
"""

import xml.etree.ElementTree as ET
import io
import uuid
import random
import string
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from pathlib import Path
import re


# =============================================================================
# DATA CONTAINERS — DataQualityReport and SchemaMetadata
# =============================================================================

@dataclass
class DataQualityReport:
    """Structured summary of data quality metrics for a DataFrame.

    Produced by DataIngestionEngine.validate() and passed to clean()
    so that cleaning decisions are based on measured data quality.

    Attributes:
        total_rows: Number of rows in the raw DataFrame.
        total_cols: Number of columns in the raw DataFrame.
        missing_rate: Per-column missing value rate (0.0 – 1.0).
        duplicate_row_count: Number of duplicate rows.
        type_inference: Per-column inferred type string.
        constant_columns: Columns with ≤1 unique non-null value.
        high_cardinality_cols: Columns where unique ratio > 95%.
        potential_id_cols: Columns likely to be primary/foreign keys.
        columns_to_drop: Columns recommended for removal.
    """
    total_rows: int = 0
    total_cols: int = 0
    missing_rate: Dict[str, float] = field(default_factory=dict)
    duplicate_row_count: int = 0
    type_inference: Dict[str, str] = field(default_factory=dict)
    constant_columns: List[str] = field(default_factory=list)
    high_cardinality_cols: List[str] = field(default_factory=list)
    potential_id_cols: List[str] = field(default_factory=list)
    columns_to_drop: List[str] = field(default_factory=list)

    def summary_str(self) -> str:
        """Human-readable quality summary string."""
        lines = [
            f"  Rows        : {self.total_rows:,}",
            f"  Columns     : {self.total_cols}",
            f"  Duplicates  : {self.duplicate_row_count}",
            f"  Constant cols: {self.constant_columns}",
            f"  High-card   : {self.high_cardinality_cols}",
            f"  Potential IDs: {self.potential_id_cols}",
        ]
        # Show cols with missing > 5%
        missing_notable = {k: v for k, v in self.missing_rate.items() if v > 0.05}
        if missing_notable:
            lines.append(f"  Missing>5%  : {missing_notable}")
        return "\n".join(lines)


@dataclass
class SchemaMetadata:
    """Inferred schema metadata for a cleaned DataFrame.

    Produced by DataIngestionEngine.infer_schema() and used by the
    feature engineering and entity discovery cells.

    Attributes:
        column_types: Column name → inferred type label.
        potential_keys: Columns that are likely primary keys.
        potential_fk_pairs: (col_a, col_b) pairs where col_a values
            are a subset of col_b values — suggesting a FK relationship.
        datetime_cols: Columns with datetime-compatible values.
        numeric_cols: Numeric (int/float) columns.
        categorical_cols: Low-cardinality categorical columns.
        text_cols: High-token-count string columns (prose/descriptions).
        id_cols: Columns identified as ID-like (keys).
        boolean_cols: Boolean or binary flag columns.
    """
    column_types: Dict[str, str] = field(default_factory=dict)
    potential_keys: List[str] = field(default_factory=list)
    potential_fk_pairs: List[Tuple[str, str]] = field(default_factory=list)
    datetime_cols: List[str] = field(default_factory=list)
    numeric_cols: List[str] = field(default_factory=list)
    categorical_cols: List[str] = field(default_factory=list)
    text_cols: List[str] = field(default_factory=list)
    id_cols: List[str] = field(default_factory=list)
    boolean_cols: List[str] = field(default_factory=list)

    def summary_str(self) -> str:
        """Human-readable schema summary string."""
        return (
            f"  Numeric     : {self.numeric_cols}\n"
            f"  Categorical : {self.categorical_cols}\n"
            f"  Datetime    : {self.datetime_cols}\n"
            f"  Text        : {self.text_cols}\n"
            f"  ID cols     : {self.id_cols}\n"
            f"  Boolean     : {self.boolean_cols}\n"
            f"  Potential PKs: {self.potential_keys}\n"
            f"  FK pairs    : {self.potential_fk_pairs}"
        )


# =============================================================================
# DataIngestionEngine — format-agnostic loader + validator + cleaner
# =============================================================================

class DataIngestionEngine:
    """Universal data ingestion engine for structured tabular datasets.

    Supports CSV, Excel, JSON (records/dict), JSONL, Parquet, XML, and
    in-memory DataFrames or raw dicts. Provides data quality assessment,
    automated cleaning, and schema inference.

    The schema inference is a key enabler for the downstream entity and
    relationship discovery — it provides the column type information that
    guides the feature engineering and clustering steps.

    Example:
        engine = DataIngestionEngine(config)
        df = engine.load("data/erp_data.csv")
        report = engine.validate(df)
        df_clean = engine.clean(df, report)
        schema = engine.infer_schema(df_clean)
    """

    # Mapping from inferred type label to a list of matching dtype patterns
    _NUMERIC_DTYPES = {"int64", "float64", "int32", "float32", "int16", "uint8"}
    _BOOL_VALUES = {0, 1, True, False, "true", "false", "yes", "no", "0", "1"}
    _UUID_PATTERN = re.compile(
        r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
        re.IGNORECASE
    )

    def __init__(self, config: "NEURLConfig") -> None:
        """Initialize the ingestion engine with global config.

        Args:
            config: NEURLConfig master configuration object.
        """
        self._config = config
        self._logger = NEURLLogger.get_logger("DataIngestionEngine")

    # ─── LOADING ────────────────────────────────────────────────────────────

    def load(
        self,
        path_or_data: Union[str, Path, Dict, pd.DataFrame]
    ) -> pd.DataFrame:
        """Load a dataset from a file path, dict, or existing DataFrame.

        Auto-detects format from file extension. Applies row cap from config
        to prevent OOM on very large files.

        Args:
            path_or_data: File path (str/Path), raw dict, or DataFrame.

        Returns:
            Loaded DataFrame (may be row-capped).

        Raises:
            NEURLDataError: If file format is unsupported or loading fails.
        """
        if isinstance(path_or_data, pd.DataFrame):
            self._logger.info("Received in-memory DataFrame",
                              extra_context={"shape": list(path_or_data.shape)})
            return path_or_data.copy()

        if isinstance(path_or_data, dict):
            self._logger.info("Converting dict to DataFrame")
            return pd.DataFrame([path_or_data]) if not isinstance(
                next(iter(path_or_data.values()), None), list
            ) else pd.DataFrame(path_or_data)

        path = Path(path_or_data)
        if not path.exists():
            raise NEURLDataError(f"File not found: {path}", {"path": str(path)})

        suffix = path.suffix.lower()
        loaders = {
            ".csv": self._load_csv,
            ".tsv": self._load_tsv,
            ".xlsx": self._load_excel,
            ".xls": self._load_excel,
            ".json": self._load_json,
            ".jsonl": self._load_jsonl,
            ".parquet": self._load_parquet,
            ".xml": self._load_xml,
        }

        if suffix not in loaders:
            raise NEURLDataError(
                f"Unsupported file format: '{suffix}'",
                {"supported": list(loaders.keys())}
            )

        try:
            df = loaders[suffix](path)
        except Exception as exc:
            raise NEURLDataError(f"Failed to load {path}: {exc}") from exc

        # Apply row cap to prevent OOM
        if len(df) > self._config.data.max_rows:
            self._logger.warning(
                f"Capping dataset from {len(df):,} to {self._config.data.max_rows:,} rows"
            )
            df = df.head(self._config.data.max_rows)

        self._logger.info(f"Loaded: {path.name}",
                          extra_context={"shape": list(df.shape), "suffix": suffix})
        return df

    def _load_csv(self, path: Path) -> pd.DataFrame:
        return pd.read_csv(path, low_memory=False)

    def _load_tsv(self, path: Path) -> pd.DataFrame:
        return pd.read_csv(path, sep="\t", low_memory=False)

    def _load_excel(self, path: Path) -> pd.DataFrame:
        return pd.read_excel(path, engine="openpyxl")

    def _load_json(self, path: Path) -> pd.DataFrame:
        """Load JSON, handling both records (list) and dict-of-columns formats."""
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, list):
            return pd.DataFrame(data)
        elif isinstance(data, dict):
            # Try dict-of-columns first, fall back to single-record
            try:
                return pd.DataFrame(data)
            except ValueError:
                return pd.DataFrame([data])
        raise NEURLDataError("JSON must be a list of records or dict-of-columns")

    def _load_jsonl(self, path: Path) -> pd.DataFrame:
        """Load JSONL (newline-delimited JSON) format."""
        records = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return pd.DataFrame(records)

    def _load_parquet(self, path: Path) -> pd.DataFrame:
        return pd.read_parquet(path)

    def _load_xml(self, path: Path) -> pd.DataFrame:
        """Load XML — assumes flat record structure (each child of root is a row)."""
        tree = ET.parse(path)
        root = tree.getroot()
        records = []
        for child in root:
            record = {subelem.tag: subelem.text for subelem in child}
            record.update(child.attrib)
            records.append(record)
        if not records:
            raise NEURLDataError("XML file contains no parseable records")
        return pd.DataFrame(records)

    # ─── VALIDATION ─────────────────────────────────────────────────────────

    def validate(self, df: pd.DataFrame) -> DataQualityReport:
        """Compute a comprehensive data quality report for a DataFrame.

        Inspects each column for missing values, cardinality, inferred type,
        and ID-like patterns. The report drives automated cleaning decisions.

        Args:
            df: Raw input DataFrame.

        Returns:
            DataQualityReport with quality metrics for every column.
        """
        report = DataQualityReport()
        report.total_rows = len(df)
        report.total_cols = len(df.columns)
        report.duplicate_row_count = int(df.duplicated().sum())

        for col in df.columns:
            n_null = df[col].isna().sum()
            report.missing_rate[col] = float(n_null / max(len(df), 1))

            n_unique = df[col].nunique(dropna=True)
            unique_ratio = n_unique / max(len(df), 1)

            # Infer column type
            report.type_inference[col] = self._infer_column_type(
                df[col], unique_ratio
            )

            # Flag constant columns (no information content)
            if n_unique <= self._config.data.constant_col_n_unique:
                report.constant_columns.append(col)

            # Flag high-cardinality columns (potential IDs)
            if unique_ratio > self._config.data.id_uniqueness_threshold:
                report.high_cardinality_cols.append(col)

            # Identify likely ID columns
            if self._is_id_column(df[col], col, unique_ratio):
                report.potential_id_cols.append(col)

        # Recommend columns to drop (constant + >threshold missing)
        report.columns_to_drop = list(set(
            report.constant_columns +
            [c for c, rate in report.missing_rate.items()
             if rate > self._config.data.missing_drop_threshold]
        ))

        self._logger.info("Validation complete", extra_context={
            "rows": report.total_rows,
            "duplicates": report.duplicate_row_count,
            "constant_cols": len(report.constant_columns),
        })
        return report

    def _infer_column_type(self, series: pd.Series, unique_ratio: float) -> str:
        """Infer the semantic type of a column from its values.

        Args:
            series: Column data.
            unique_ratio: Fraction of unique values.

        Returns:
            Type label: 'numeric'|'categorical'|'datetime'|'boolean'|'text'|'id'.
        """
        # Check boolean first (before numeric, since 0/1 are numeric)
        sample = series.dropna().head(100)
        if set(sample.astype(str).str.lower().unique()).issubset(
            {"0", "1", "true", "false", "yes", "no"}
        ):
            return "boolean"

        # Numeric types
        if pd.api.types.is_numeric_dtype(series):
            return "numeric"

        # Datetime — try parsing
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                parsed = pd.to_datetime(sample, errors="coerce", infer_datetime_format=True)
            if parsed.notna().mean() > 0.8:
                return "datetime"
        except Exception:
            pass

        # High uniqueness + looks like an ID → id
        if unique_ratio > self._config.data.id_uniqueness_threshold:
            return "id"

        # High avg token count → text (prose / descriptions)
        if series.dtype == object:
            avg_tokens = series.dropna().astype(str).str.split().apply(len).mean()
            if avg_tokens > self._config.data.text_min_avg_tokens:
                return "text"

        # Low cardinality string → categorical
        return "categorical"

    def _is_id_column(
        self, series: pd.Series, col_name: str, unique_ratio: float
    ) -> bool:
        """Heuristically determine if a column is an ID column.

        Checks: name contains '_id'/'id_'/'_key', unique ratio > threshold,
        sequential integer pattern, or UUID pattern.

        Args:
            series: Column data.
            col_name: Column name.
            unique_ratio: Fraction of unique values.

        Returns:
            True if column is likely a key/ID.
        """
        name_lower = col_name.lower()
        has_id_name = any(kw in name_lower for kw in ["_id", "id_", "_key", "_code", "_num"])

        if not (unique_ratio > self._config.data.id_uniqueness_threshold or has_id_name):
            return False

        sample = series.dropna().head(50)

        # UUID pattern
        if sample.dtype == object:
            uuid_match = sample.astype(str).str.match(self._UUID_PATTERN).mean()
            if uuid_match > 0.8:
                return True

        # Sequential integer
        try:
            as_int = pd.to_numeric(sample, errors="coerce")
            if as_int.notna().all():
                diffs = as_int.sort_values().diff().dropna()
                if (diffs == 1).mean() > 0.6:
                    return True
        except Exception:
            pass

        # Name-based heuristic alone
        return has_id_name and unique_ratio > 0.5

    # ─── CLEANING ───────────────────────────────────────────────────────────

    def clean(self, df: pd.DataFrame, report: DataQualityReport) -> pd.DataFrame:
        """Clean a DataFrame based on the quality report.

        Operations:
        1. Drop constant columns and high-missing columns.
        2. Strip whitespace from object columns.
        3. Convert type-inferred dtypes (numeric coercion, datetime parsing).
        4. Fill missing values (numeric→median, categorical/text→mode/"").

        Args:
            df: Raw or partially cleaned DataFrame.
            report: DataQualityReport from validate().

        Returns:
            Cleaned DataFrame.
        """
        df = df.copy()

        # 1. Drop recommended columns
        cols_to_drop = [c for c in report.columns_to_drop if c in df.columns]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
            self._logger.info(f"Dropped {len(cols_to_drop)} columns",
                              extra_context={"dropped": cols_to_drop})

        # 2. Strip whitespace from string columns
        for col in df.select_dtypes(include="object").columns:
            df[col] = df[col].astype(str).str.strip().replace("nan", np.nan)

        # 3. Type coercion based on inferred types
        for col, dtype in report.type_inference.items():
            if col not in df.columns:
                continue
            if dtype == "numeric":
                df[col] = pd.to_numeric(df[col], errors="coerce")
            elif dtype == "datetime":
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    df[col] = pd.to_datetime(df[col], errors="coerce", infer_datetime_format=True)
            elif dtype == "boolean":
                df[col] = df[col].map(
                    lambda x: True if str(x).lower() in {"1", "true", "yes"} else (
                        False if str(x).lower() in {"0", "false", "no"} else np.nan
                    )
                )

        # 4. Fill missing values
        for col in df.columns:
            if df[col].isna().sum() == 0:
                continue
            dtype_label = report.type_inference.get(col, "categorical")
            if dtype_label == "numeric":
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
            elif dtype_label in ("categorical", "id"):
                mode_vals = df[col].mode()
                fill_val = mode_vals.iloc[0] if len(mode_vals) > 0 else "UNKNOWN"
                df[col] = df[col].fillna(fill_val)
            elif dtype_label == "text":
                df[col] = df[col].fillna("")
            elif dtype_label == "datetime":
                # Forward-fill datetimes rather than imputing with mean
                df[col] = df[col].ffill().bfill()

        self._logger.info("Cleaning complete",
                          extra_context={"shape_after": list(df.shape)})
        return df

    # ─── SCHEMA INFERENCE ───────────────────────────────────────────────────

    def infer_schema(self, df: pd.DataFrame) -> SchemaMetadata:
        """Infer structural schema metadata from a cleaned DataFrame.

        FK-pair detection: for each pair of id/categorical columns (col_a, col_b),
        check if the value set of col_a is a subset of col_b's value set.
        This mirrors how a database schema analyst identifies foreign key relationships,
        but does it automatically from data patterns.

        Args:
            df: Cleaned DataFrame (output of clean()).

        Returns:
            SchemaMetadata with categorized columns and FK-pair candidates.
        """
        schema = SchemaMetadata()

        for col in df.columns:
            col_type = self._infer_column_type(
                df[col], df[col].nunique() / max(len(df), 1)
            )
            schema.column_types[col] = col_type

            if col_type == "numeric":
                schema.numeric_cols.append(col)
            elif col_type == "categorical":
                schema.categorical_cols.append(col)
            elif col_type == "datetime":
                schema.datetime_cols.append(col)
            elif col_type == "text":
                schema.text_cols.append(col)
            elif col_type == "id":
                schema.id_cols.append(col)
            elif col_type == "boolean":
                schema.boolean_cols.append(col)

        # Primary key detection — high uniqueness id-like columns
        schema.potential_keys = [
            col for col in df.columns
            if self._is_id_column(df[col], col, df[col].nunique() / max(len(df), 1))
            and df[col].nunique() > len(df) * 0.9
        ]

        # FK pair detection — value-subset heuristic
        schema.potential_fk_pairs = self._detect_fk_pairs(df, schema)

        self._logger.info("Schema inferred", extra_context={
            "numeric_cols": len(schema.numeric_cols),
            "categorical_cols": len(schema.categorical_cols),
            "datetime_cols": len(schema.datetime_cols),
            "id_cols": len(schema.id_cols),
            "fk_pairs": len(schema.potential_fk_pairs),
        })
        return schema

    def _detect_fk_pairs(
        self, df: pd.DataFrame, schema: SchemaMetadata
    ) -> List[Tuple[str, str]]:
        """Detect foreign key pairs by value-subset analysis.

        For each pair (col_a, col_b) among id/categorical columns,
        check if values of col_a are largely a subset of col_b values.
        This is the statistical equivalent of a FK constraint check.

        Args:
            df: Cleaned DataFrame.
            schema: Partially filled SchemaMetadata.

        Returns:
            List of (child_col, parent_col) FK candidate pairs.
        """
        candidate_cols = schema.id_cols + schema.categorical_cols
        # Limit to avoid O(n^2) on large column sets
        candidate_cols = candidate_cols[:30]

        fk_pairs = []
        for i, col_a in enumerate(candidate_cols):
            vals_a = set(df[col_a].dropna().astype(str).unique())
            if len(vals_a) < 2:
                continue
            for col_b in candidate_cols[i + 1:]:
                if col_a == col_b:
                    continue
                vals_b = set(df[col_b].dropna().astype(str).unique())
                if len(vals_b) < 2:
                    continue
                # Check directional subset: a ⊆ b means a is FK referencing b
                overlap_ab = len(vals_a & vals_b) / len(vals_a)
                overlap_ba = len(vals_a & vals_b) / len(vals_b)
                if overlap_ab > self._config.fk_subset_threshold and len(vals_a) < len(vals_b):
                    fk_pairs.append((col_a, col_b))
                elif overlap_ba > self._config.fk_subset_threshold and len(vals_b) < len(vals_a):
                    fk_pairs.append((col_b, col_a))

        return fk_pairs


# =============================================================================
# SYNTHETIC ERP DATASET GENERATOR
# =============================================================================

def generate_synthetic_erp_dataset(
    n_rows: int = 500,
    random_seed: int = 42
) -> pd.DataFrame:
    """Generate a realistic synthetic ERP (Enterprise Resource Planning) dataset.

    Creates a denormalized table with entities from multiple business domains:
    Customer, Order, Product, Invoice, Payment, and Employee. This mimics the
    kind of flat-file export that many ERP systems produce, which is then
    fed into NEURL Engine for automated knowledge graph construction.

    Args:
        n_rows: Number of rows to generate.
        random_seed: Random seed for reproducibility.

    Returns:
        DataFrame with 23 ERP columns across 6+ business entity domains.
    """
    rng = np.random.default_rng(random_seed)

    # Reference pools for realistic values
    first_names = ["Alice", "Bob", "Carol", "David", "Eva", "Frank", "Grace",
                   "Henry", "Irene", "Jack", "Karen", "Leo", "Mary", "Nate",
                   "Olivia", "Paul", "Quinn", "Rachel", "Sam", "Tina"]
    last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia",
                  "Miller", "Davis", "Wilson", "Moore", "Taylor", "Anderson"]
    domains = ["gmail.com", "yahoo.com", "outlook.com", "company.com", "corp.io"]
    order_statuses = ["Pending", "Shipped", "Delivered", "Cancelled", "Returned"]
    product_categories = ["Electronics", "Furniture", "Clothing", "Books",
                          "Sports", "Home & Garden", "Toys"]
    payment_methods = ["Credit Card", "Bank Transfer", "PayPal", "Invoice", "Cash"]
    departments = ["Sales", "Engineering", "Marketing", "Finance", "HR", "Operations"]
    regions = ["North", "South", "East", "West", "Central"]

    # Generate base IDs — these form the FK backbone of the dataset
    n_customers = n_rows // 5    # ~100 unique customers
    n_products = n_rows // 10    # ~50 unique products
    n_employees = n_rows // 25   # ~20 unique employees

    customer_ids = [f"CUST-{i:04d}" for i in range(1, n_customers + 1)]
    product_ids = [f"PROD-{i:04d}" for i in range(1, n_products + 1)]
    employee_ids = [f"EMP-{i:03d}" for i in range(1, n_employees + 1)]

    rows = []
    for i in range(n_rows):
        cust_idx = rng.integers(0, n_customers)
        cust_id = customer_ids[cust_idx]
        cust_fname = first_names[cust_idx % len(first_names)]
        cust_lname = last_names[cust_idx % len(last_names)]
        cust_name = f"{cust_fname} {cust_lname}"
        cust_email = f"{cust_fname.lower()}.{cust_lname.lower()}@{domains[cust_idx % len(domains)]}"
        cust_phone = f"+1-{rng.integers(200, 999)}-{rng.integers(100, 999)}-{rng.integers(1000, 9999)}"
        cust_addr = f"{rng.integers(1, 9999)} {last_names[rng.integers(0, len(last_names))]} St"

        order_id = f"ORD-{i + 1:05d}"
        base_date = pd.Timestamp("2022-01-01")
        order_date = base_date + pd.Timedelta(days=int(rng.integers(0, 730)))
        order_amount = round(float(rng.uniform(10.0, 5000.0)), 2)
        order_status = order_statuses[rng.integers(0, len(order_statuses))]

        prod_idx = rng.integers(0, n_products)
        product_id = product_ids[prod_idx]
        product_name = f"Product-{product_id}"
        product_category = product_categories[prod_idx % len(product_categories)]
        product_price = round(float(rng.uniform(5.0, 2000.0)), 2)

        invoice_id = f"INV-{i + 1:05d}"
        invoice_date = order_date + pd.Timedelta(days=int(rng.integers(1, 7)))
        invoice_amount = round(order_amount * rng.uniform(0.95, 1.05), 2)

        payment_id = f"PAY-{i + 1:05d}"
        payment_date = invoice_date + pd.Timedelta(days=int(rng.integers(0, 30)))
        payment_method = payment_methods[rng.integers(0, len(payment_methods))]

        emp_idx = rng.integers(0, n_employees)
        employee_id = employee_ids[emp_idx]
        employee_name = f"{first_names[emp_idx % len(first_names)]} {last_names[emp_idx % len(last_names)]}"
        department = departments[emp_idx % len(departments)]
        region = regions[emp_idx % len(regions)]

        rows.append({
            # Customer entity columns
            "customer_id": cust_id,
            "customer_name": cust_name,
            "customer_email": cust_email,
            "customer_phone": cust_phone,
            "customer_address": cust_addr,
            # Order entity columns
            "order_id": order_id,
            "order_date": order_date.strftime("%Y-%m-%d"),
            "order_amount": order_amount,
            "order_status": order_status,
            # Product entity columns
            "product_id": product_id,
            "product_name": product_name,
            "product_category": product_category,
            "product_price": product_price,
            # Invoice entity columns
            "invoice_id": invoice_id,
            "invoice_date": invoice_date.strftime("%Y-%m-%d"),
            "invoice_amount": invoice_amount,
            # Payment entity columns
            "payment_id": payment_id,
            "payment_date": payment_date.strftime("%Y-%m-%d"),
            "payment_method": payment_method,
            # Employee entity columns
            "employee_id": employee_id,
            "employee_name": employee_name,
            "department": department,
            "region": region,
        })

    df = pd.DataFrame(rows)
    return df


# =============================================================================
# DEMO: Run the full ingestion pipeline on the synthetic ERP dataset
# =============================================================================

print("=" * 70)
print("  NEURL Engine — Cell 2: Universal Data Ingestion & Validation")
print("=" * 70)

# 1. Generate synthetic ERP dataset
print("\n[1/4] Generating synthetic ERP dataset...")
raw_erp_df = generate_synthetic_erp_dataset(
    n_rows=neurl_config.data.sample_size,
    random_seed=neurl_config.random_seed
)
print(f"      Generated: {raw_erp_df.shape[0]} rows × {raw_erp_df.shape[1]} columns")
print(f"      Columns  : {list(raw_erp_df.columns)}")

# 2. Initialize engine and run validation
print("\n[2/4] Validating data quality...")
ingestion_engine = DataIngestionEngine(neurl_config)
erp_quality_report = ingestion_engine.validate(raw_erp_df)

print("\n  ── Data Quality Report ──────────────────────────────────────────")
print(erp_quality_report.summary_str())
print(f"\n  Type inference:")
for col, typ in erp_quality_report.type_inference.items():
    print(f"    {col:<25} → {typ}")

# 3. Clean the data
print("\n[3/4] Cleaning dataset...")
erp_df_clean = ingestion_engine.clean(raw_erp_df, erp_quality_report)
print(f"      Shape after cleaning: {erp_df_clean.shape}")

# 4. Infer schema
print("\n[4/4] Inferring schema metadata...")
erp_schema = ingestion_engine.infer_schema(erp_df_clean)
print("\n  ── Schema Metadata ──────────────────────────────────────────────")
print(erp_schema.summary_str())

print("\n✅ Cell 2 complete — Data ingestion pipeline ready.")
print(f"   Exports: raw_erp_df, erp_df_clean, erp_quality_report, erp_schema")
