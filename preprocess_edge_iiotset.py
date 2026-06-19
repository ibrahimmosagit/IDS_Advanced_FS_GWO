from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import kagglehub
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler


RAW_RELATIVE_PATH = r"Edge-IIoTset dataset\Selected dataset for ML and DL\ML-EdgeIIoT-dataset.csv"


DROP_OBJECT_COLUMNS = {
    "ip.src_host",
    "ip.dst_host",
    "arp.dst.proto_ipv4",
    "arp.src.proto_ipv4",
    "http.file_data",
    "http.request.uri.query",
    "http.referer",
    "http.request.full_uri",
    "tcp.payload",
    "dns.qry.name",
}


@dataclass
class PreprocessSummary:
    input_rows: int
    output_rows: int
    input_cols: int
    output_cols: int
    dropped_duplicate_rows: int
    dropped_identifier_cols: list[str]
    dropped_high_cardinality_cols: list[str]
    encoded_categorical_cols: list[str]
    converted_numeric_cols: list[str]
    processed_continuous_cols: list[str]
    dropped_constant_cols: list[str]


def slugify_text(value: str) -> str:
    text = str(value).strip().lower()
    text = "".join(ch if ch.isalnum() else "_" for ch in text)
    while "__" in text:
        text = text.replace("__", "_")
    return text.strip("_") or "unknown"


def make_safe_feature_names(columns: list[str]) -> list[str]:
    safe_names: list[str] = []
    used: dict[str, int] = {}

    for col in columns:
        base = slugify_text(col)
        suffix = used.get(base, 0)
        used[base] = suffix + 1
        safe_names.append(base if suffix == 0 else f"{base}_{suffix}")

    return safe_names


def resolve_raw_path(explicit_path: str | None) -> str:
    if explicit_path:
        return explicit_path

    dataset_path = kagglehub.dataset_download(
        "mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot"
    )
    return os.path.join(dataset_path, RAW_RELATIVE_PATH)


def is_binary_like(series: pd.Series) -> bool:
    values = pd.Series(series).dropna().unique().tolist()
    return set(values).issubset({0, 1})


def preprocess_edge_dataset_aligned(
    raw_path: str,
    output_path: str,
    min_numeric_ratio: float = 0.80,
    categorical_max_unique: int = 20,
    drop_duplicates: bool = False,
) -> PreprocessSummary:
    df = pd.read_csv(raw_path, low_memory=False)
    input_rows, input_cols = df.shape

    df.columns = [str(col).strip() for col in df.columns]

    required = {"Attack_type", "Attack_label"}
    missing_required = required.difference(df.columns)
    if missing_required:
        raise ValueError(f"Missing required columns: {sorted(missing_required)}")

    dropped_duplicate_rows = 0
    if drop_duplicates:
        before_dedup = len(df)
        df = df.drop_duplicates().reset_index(drop=True)
        dropped_duplicate_rows = before_dedup - len(df)

    label2 = (
        df["Attack_type"]
        .astype("string")
        .fillna("Unknown")
        .str.strip()
        .replace("", "Unknown")
    )
    label1 = pd.Series(
        np.where(label2.str.lower().eq("normal"), "normal", "attack"),
        index=df.index,
    )
    label_full = label2.map(lambda x: f"edgeiiot_{slugify_text(x)}")

    feature_df = df.drop(columns=["Attack_type", "Attack_label"], errors="ignore").copy()

    dropped_identifier_cols = [col for col in DROP_OBJECT_COLUMNS if col in feature_df.columns]
    feature_df = feature_df.drop(columns=dropped_identifier_cols, errors="ignore")

    numeric_features: dict[str, pd.Series] = {}
    converted_numeric_cols: list[str] = []
    encoded_categorical_cols: list[str] = []
    dropped_high_cardinality_cols: list[str] = []
    categorical_frames: list[pd.DataFrame] = []

    for col in feature_df.columns:
        series = feature_df[col]
        if pd.api.types.is_numeric_dtype(series):
            numeric_features[col] = pd.to_numeric(series, errors="coerce")
            continue

        normalized = (
            series.astype("string")
            .fillna("-")
            .str.strip()
            .replace("", "-")
        )

        converted = pd.to_numeric(normalized, errors="coerce")
        if float(converted.notna().mean()) >= float(min_numeric_ratio):
            numeric_features[col] = converted
            converted_numeric_cols.append(col)
            continue

        nunique = int(normalized.nunique(dropna=False))
        if nunique > int(categorical_max_unique):
            dropped_high_cardinality_cols.append(col)
            continue

        encoded_categorical_cols.append(col)
        dummies = pd.get_dummies(normalized, prefix=col, prefix_sep="__", dtype="int8")
        categorical_frames.append(dummies)

    X_df = pd.DataFrame(numeric_features)
    if categorical_frames:
        categorical_df = pd.concat(categorical_frames, axis=1)
        X_df = pd.concat([X_df, categorical_df], axis=1)

    X_df.replace([np.inf, -np.inf], np.nan, inplace=True)

    continuous_cols = [col for col in X_df.columns if not is_binary_like(X_df[col])]

    for col in continuous_cols:
        med = pd.to_numeric(X_df[col], errors="coerce").median()
        if pd.isna(med):
            med = 0.0
        X_df[col] = pd.to_numeric(X_df[col], errors="coerce").fillna(med)

    for col in X_df.columns:
        if col in continuous_cols:
            continue
        fill_val = pd.to_numeric(X_df[col], errors="coerce").median()
        if pd.isna(fill_val):
            fill_val = 0
        X_df[col] = pd.to_numeric(X_df[col], errors="coerce").fillna(fill_val)

    for col in continuous_cols:
        q1 = X_df[col].quantile(0.25)
        q3 = X_df[col].quantile(0.75)
        iqr = q3 - q1
        if pd.isna(iqr) or float(iqr) == 0.0:
            continue
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        X_df[col] = np.where(X_df[col] < lower, lower, np.where(X_df[col] > upper, upper, X_df[col]))

    if continuous_cols:
        scaler = RobustScaler()
        X_df[continuous_cols] = scaler.fit_transform(X_df[continuous_cols])

    for col in X_df.columns:
        if X_df[col].isna().any():
            fill_val = 0 if is_binary_like(X_df[col]) else float(pd.to_numeric(X_df[col], errors="coerce").median())
            if pd.isna(fill_val):
                fill_val = 0.0
            X_df[col] = pd.to_numeric(X_df[col], errors="coerce").fillna(fill_val)

    constant_mask = X_df.nunique(dropna=False) <= 1
    dropped_constant_cols = constant_mask[constant_mask].index.tolist()
    X_df = X_df.loc[:, ~constant_mask].copy()
    X_df.columns = make_safe_feature_names(X_df.columns.tolist())

    cleaned = pd.concat(
        [
            pd.DataFrame(
                {
                    "label_full": label_full,
                    "label1": label1,
                    "label2": label2,
                }
            ),
            X_df,
        ],
        axis=1,
    )

    cleaned.to_csv(output_path, index=False)

    return PreprocessSummary(
        input_rows=input_rows,
        output_rows=len(cleaned),
        input_cols=input_cols,
        output_cols=cleaned.shape[1],
        dropped_duplicate_rows=dropped_duplicate_rows,
        dropped_identifier_cols=sorted(dropped_identifier_cols),
        dropped_high_cardinality_cols=sorted(dropped_high_cardinality_cols),
        encoded_categorical_cols=sorted(encoded_categorical_cols),
        converted_numeric_cols=sorted(converted_numeric_cols),
        processed_continuous_cols=sorted(continuous_cols),
        dropped_constant_cols=sorted(dropped_constant_cols),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess Edge-IIoTset with a pipeline aligned to Summary_Analysis_and_Preprocessing."
    )
    parser.add_argument("--raw-path", default=None, help="Optional explicit path to ML-EdgeIIoT-dataset.csv")
    parser.add_argument(
        "--output-path",
        default=os.path.join(os.path.dirname(__file__), "cleaned_Edge-IIoTset_aligned.csv"),
        help="Output CSV path",
    )
    parser.add_argument("--min-numeric-ratio", type=float, default=0.80)
    parser.add_argument("--categorical-max-unique", type=int, default=20)
    parser.add_argument("--drop-duplicates", action="store_true")
    args = parser.parse_args()

    raw_path = resolve_raw_path(args.raw_path)
    summary = preprocess_edge_dataset_aligned(
        raw_path=raw_path,
        output_path=args.output_path,
        min_numeric_ratio=args.min_numeric_ratio,
        categorical_max_unique=args.categorical_max_unique,
        drop_duplicates=args.drop_duplicates,
    )

    print("Preprocessing completed.")
    print(f"Raw file: {raw_path}")
    print(f"Output file: {args.output_path}")
    print(f"Rows: {summary.input_rows} -> {summary.output_rows}")
    print(f"Columns: {summary.input_cols} -> {summary.output_cols}")
    print(f"Dropped duplicate rows: {summary.dropped_duplicate_rows}")
    print(f"Dropped identifier cols ({len(summary.dropped_identifier_cols)}): {summary.dropped_identifier_cols}")
    print(
        f"Dropped high-cardinality cols ({len(summary.dropped_high_cardinality_cols)}): "
        f"{summary.dropped_high_cardinality_cols}"
    )
    print(f"Encoded categorical cols ({len(summary.encoded_categorical_cols)}): {summary.encoded_categorical_cols}")
    print(f"Converted numeric-like cols ({len(summary.converted_numeric_cols)}): {summary.converted_numeric_cols}")
    print(
        f"Processed continuous cols with median/IQR/RobustScaler "
        f"({len(summary.processed_continuous_cols)}): {summary.processed_continuous_cols}"
    )
    print(f"Dropped constant cols ({len(summary.dropped_constant_cols)}): {summary.dropped_constant_cols}")


if __name__ == "__main__":
    main()
