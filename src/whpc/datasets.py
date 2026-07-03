from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


NSL_KDD_COLUMNS = [
    "duration",
    "protocol_type",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
    "attack_name",
    "difficulty",
]

CIC_IDS2017_DROP_COLUMNS = [
    "Flow ID",
    "Source IP",
    "Source Port",
    "Destination IP",
    "Destination Port",
    "Timestamp",
]


@dataclass(frozen=True)
class LoadedDataset:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    metadata: dict[str, Any]


def load_unsw_nb15(
    raw_dir: str | Path = "data/raw",
    target: str = "binary",
) -> LoadedDataset:
    raw_dir = Path(raw_dir)
    train_path = raw_dir / "UNSW_NB15_training-set.csv"
    test_path = raw_dir / "UNSW_NB15_testing-set.csv"

    train_df = pd.read_csv(train_path, encoding="utf-8-sig")
    test_df = pd.read_csv(test_path, encoding="utf-8-sig")

    return _build_unsw_dataset(train_df, test_df, target=target)


def load_nsl_kdd(
    raw_dir: str | Path = "data/raw/archive",
    target: str = "binary",
) -> LoadedDataset:
    raw_dir = Path(raw_dir)
    train_path = raw_dir / "KDDTrain+.txt"
    test_path = raw_dir / "KDDTest+.txt"

    train_df = pd.read_csv(train_path, header=None, names=NSL_KDD_COLUMNS)
    test_df = pd.read_csv(test_path, header=None, names=NSL_KDD_COLUMNS)

    return _build_nsl_kdd_dataset(train_df, test_df, target=target)


def load_cic_ids2017(
    raw_dir: str | Path = "data/raw/GeneratedLabelledFlows/TrafficLabelling",
    target: str = "binary",
    test_size: float = 0.3,
    random_state: int = 42,
    max_rows: int | None = None,
) -> LoadedDataset:
    raw_dir = Path(raw_dir)
    files = sorted(raw_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CIC-IDS2017 CSV files found in {raw_dir}.")

    frames: list[pd.DataFrame] = []
    rows_per_file = None
    if max_rows is not None:
        rows_per_file = max(1, math.ceil(max_rows / len(files)))

    for path in files:
        frame = pd.read_csv(
            path,
            nrows=rows_per_file,
            low_memory=False,
            encoding="utf-8",
            encoding_errors="replace",
        )
        frames.append(frame)

    df = pd.concat(frames, ignore_index=True)
    if max_rows is not None and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=random_state).reset_index(drop=True)

    return _build_cic_ids2017_dataset(
        df,
        target=target,
        test_size=test_size,
        random_state=random_state,
    )


def _build_unsw_dataset(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target: str,
) -> LoadedDataset:
    if target not in {"binary", "attack_cat"}:
        raise ValueError("target must be 'binary' or 'attack_cat'.")

    feature_columns = [column for column in train_df.columns if column not in {"id", "label", "attack_cat"}]
    X_train = train_df.loc[:, feature_columns].copy()
    X_test = test_df.loc[:, feature_columns].copy()

    if target == "binary":
        y_train = train_df["label"].astype(int).copy()
        y_test = test_df["label"].astype(int).copy()
    else:
        y_train = train_df["attack_cat"].copy()
        y_test = test_df["attack_cat"].copy()

    metadata = {
        "dataset_name": "UNSW-NB15",
        "feature_columns": feature_columns,
        "target_mode": target,
        "binary_label_column": "label",
        "attack_category_column": "attack_cat",
        "dropped_columns": ["id"],
    }

    return LoadedDataset(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        metadata=metadata,
    )


def _build_nsl_kdd_dataset(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target: str,
) -> LoadedDataset:
    if target not in {"binary", "attack_name"}:
        raise ValueError("target must be 'binary' or 'attack_name'.")

    feature_columns = [column for column in NSL_KDD_COLUMNS if column not in {"attack_name", "difficulty"}]
    X_train = train_df.loc[:, feature_columns].copy()
    X_test = test_df.loc[:, feature_columns].copy()

    if target == "binary":
        y_train = _to_binary_attack_label(train_df["attack_name"])
        y_test = _to_binary_attack_label(test_df["attack_name"])
    else:
        y_train = train_df["attack_name"].copy()
        y_test = test_df["attack_name"].copy()

    metadata = {
        "dataset_name": "NSL-KDD",
        "feature_columns": feature_columns,
        "target_mode": target,
        "attack_label_column": "attack_name",
        "difficulty_column": "difficulty",
        "dropped_columns": ["difficulty"],
    }

    return LoadedDataset(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        metadata=metadata,
    )


def _build_cic_ids2017_dataset(
    df: pd.DataFrame,
    target: str,
    test_size: float,
    random_state: int,
) -> LoadedDataset:
    if target not in {"binary", "label"}:
        raise ValueError("target must be 'binary' or 'label'.")

    cleaned_df, dropped_empty_rows = _clean_cic_ids2017_frame(df)
    feature_columns = [column for column in cleaned_df.columns if column != "Label"]

    X = cleaned_df.loc[:, feature_columns].copy()
    if target == "binary":
        y = _to_binary_attack_label(cleaned_df["Label"])
    else:
        y = cleaned_df["Label"].copy()

    stratify = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    metadata = {
        "dataset_name": "CIC-IDS2017",
        "feature_columns": feature_columns,
        "target_mode": target,
        "label_column": "Label",
        "dropped_columns": CIC_IDS2017_DROP_COLUMNS,
        "dropped_empty_rows": dropped_empty_rows,
        "stratified_split": stratify is not None,
    }

    return LoadedDataset(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        metadata=metadata,
    )


def _clean_cic_ids2017_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    cleaned_df = df.copy()
    cleaned_df.columns = [str(column).strip() for column in cleaned_df.columns]
    cleaned_df = cleaned_df.replace(r"^\s*$", pd.NA, regex=True)

    rows_before = len(cleaned_df)
    cleaned_df = cleaned_df.dropna(how="all")
    if "Label" not in cleaned_df.columns:
        raise ValueError("CIC-IDS2017 data must contain a 'Label' column.")

    cleaned_df = cleaned_df.dropna(subset=["Label"]).copy()
    dropped_empty_rows = rows_before - len(cleaned_df)

    cleaned_df["Label"] = _normalize_cic_ids2017_label(cleaned_df["Label"])
    drop_columns = [column for column in CIC_IDS2017_DROP_COLUMNS if column in cleaned_df.columns]
    feature_df = cleaned_df.drop(columns=drop_columns)
    feature_columns = [column for column in feature_df.columns if column != "Label"]
    numeric_df = feature_df.loc[:, feature_columns].apply(pd.to_numeric, errors="coerce")
    numeric_df = numeric_df.mask(~np.isfinite(numeric_df), np.nan)
    cleaned_feature_df = pd.concat([numeric_df, feature_df.loc[:, ["Label"]]], axis=1)
    return cleaned_feature_df, dropped_empty_rows


def _to_binary_attack_label(attack_name: pd.Series) -> pd.Series:
    normalized = attack_name.astype(str).str.strip().str.lower().str.rstrip(".")
    return (~normalized.isin({"normal", "benign"})).astype(int)


def _normalize_cic_ids2017_label(labels: pd.Series) -> pd.Series:
    return (
        labels.astype(str)
        .str.strip()
        .str.replace("\ufffd", " ", regex=False)
        .str.replace(r"\s+", " ", regex=True)
    )
