from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from .datasets import NSL_KDD_COLUMNS, _clean_cic_ids2017_frame


UNSW_NB15_SEEN_ATTACK_FAMILIES = ("Generic", "Exploits", "Fuzzers", "DoS")
UNSW_NB15_UNSEEN_ATTACK_FAMILIES = ("Reconnaissance", "Analysis", "Backdoor", "Shellcode")
UNSW_NB15_AUXILIARY_ATTACK_FAMILIES = ("Worms",)
NSL_KDD_NORMAL_LABEL = "normal"
CIC_IDS2017_SEEN_ATTACK_FAMILIES = (
    "DDoS",
    "DoS Hulk",
    "DoS GoldenEye",
    "DoS slowloris",
    "DoS Slowhttptest",
    "PortScan",
    "FTP-Patator",
    "SSH-Patator",
)
CIC_IDS2017_UNSEEN_ATTACK_FAMILIES = ("Bot", "Web Attack Brute Force", "Web Attack XSS")
CIC_IDS2017_AUXILIARY_ATTACK_FAMILIES = ("Infiltration", "Web Attack Sql Injection", "Heartbleed")
CIC_IDS2017_REQUIRED_PART3_FILES = {
    "Benign-Monday-no-metadata": "Benign-Monday-no-metadata.parquet",
    "Botnet-Friday-no-metadata": "Botnet-Friday-no-metadata.parquet",
    "Bruteforce-Tuesday-no-metadata": "Bruteforce-Tuesday-no-metadata.parquet",
    "DDoS-Friday-no-metadata": "DDoS-Friday-no-metadata.parquet",
    "DoS-Wednesday-no-metadata": "DoS-Wednesday-no-metadata.parquet",
    "Infiltration-Thursday-no-metadata": "Infiltration-Thursday-no-metadata.parquet",
    "Portscan-Friday-no-metadata": "Portscan-Friday-no-metadata.parquet",
    "WebAttacks-Thursday-no-metadata": "WebAttacks-Thursday-no-metadata.parquet",
}
CIC_IDS2017_TEMPORAL_BLOCK_SOURCES = (
    ("tuesday-bruteforce", 1, "Tuesday", "Bruteforce-Tuesday-no-metadata"),
    ("wednesday-dos", 2, "Wednesday", "DoS-Wednesday-no-metadata"),
    ("thursday-webattacks", 3, "Thursday", "WebAttacks-Thursday-no-metadata"),
    ("thursday-infiltration", 4, "Thursday", "Infiltration-Thursday-no-metadata"),
    ("friday-ddos", 5, "Friday", "DDoS-Friday-no-metadata"),
    ("friday-portscan", 6, "Friday", "Portscan-Friday-no-metadata"),
    ("friday-botnet", 7, "Friday", "Botnet-Friday-no-metadata"),
)


@dataclass(frozen=True)
class OneClassDatasetSplit:
    X_train_normal: pd.DataFrame
    X_val: pd.DataFrame
    y_val: np.ndarray
    val_attack_families: pd.Series
    X_test_standard: pd.DataFrame
    y_test_standard: np.ndarray
    test_standard_attack_families: pd.Series
    X_test_unseen: pd.DataFrame
    y_test_unseen: np.ndarray
    test_unseen_attack_families: pd.Series
    metadata: dict[str, Any]


@dataclass(frozen=True)
class StreamBatch:
    X: pd.DataFrame
    y: np.ndarray
    attack_family: pd.Series
    block_id: str
    timestamp_or_order: int
    source_key: str
    day: str
    role: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class CICTemporalStreamSplit:
    X_train_normal: pd.DataFrame
    X_val: pd.DataFrame
    y_val: np.ndarray
    val_attack_families: pd.Series
    stream_batches: tuple[StreamBatch, ...]
    metadata: dict[str, Any]


def load_unsw_nb15_raw_frames(raw_dir: str | Path = "data/raw") -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_dir = Path(raw_dir)
    train_path = raw_dir / "UNSW_NB15_training-set.csv"
    test_path = raw_dir / "UNSW_NB15_testing-set.csv"

    train_df = pd.read_csv(train_path, encoding="utf-8-sig")
    test_df = pd.read_csv(test_path, encoding="utf-8-sig")
    return train_df, test_df


def load_nsl_kdd_raw_frames(raw_dir: str | Path = "data/raw/archive") -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_dir = Path(raw_dir)
    train_path = raw_dir / "KDDTrain+.txt"
    test_path = raw_dir / "KDDTest+.txt"

    train_df = pd.read_csv(train_path, header=None, names=NSL_KDD_COLUMNS)
    test_df = pd.read_csv(test_path, header=None, names=NSL_KDD_COLUMNS)
    return train_df, test_df


def load_cic_ids2017_part3_frames(
    raw_dir: str | Path = "data/raw/CIC-IDS2017",
    max_rows_per_source: int | None = None,
) -> dict[str, pd.DataFrame]:
    raw_dir = Path(raw_dir)
    frames: dict[str, pd.DataFrame] = {}
    for family_key, filename in CIC_IDS2017_REQUIRED_PART3_FILES.items():
        path = raw_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Required CIC-IDS2017 Part 3 file not found: {path}")
        frame = pd.read_parquet(path)
        if max_rows_per_source is not None:
            if max_rows_per_source <= 0:
                raise ValueError("max_rows_per_source must be positive when provided.")
            frame = frame.head(max_rows_per_source).copy()
        frames[family_key] = frame
    return frames


def build_unsw_nb15_one_class_split(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    validation_size: float = 0.2,
    random_state: int = 42,
    seen_attack_families: tuple[str, ...] = UNSW_NB15_SEEN_ATTACK_FAMILIES,
    unseen_attack_families: tuple[str, ...] = UNSW_NB15_UNSEEN_ATTACK_FAMILIES,
    auxiliary_attack_families: tuple[str, ...] = UNSW_NB15_AUXILIARY_ATTACK_FAMILIES,
) -> OneClassDatasetSplit:
    if validation_size <= 0.0 or validation_size >= 1.0:
        raise ValueError("validation_size must be between 0 and 1.")

    _validate_unsw_columns(train_df, frame_name="train_df")
    _validate_unsw_columns(test_df, frame_name="test_df")

    normalized_train = _normalize_unsw_attack_categories(train_df)
    normalized_test = _normalize_unsw_attack_categories(test_df)
    feature_columns = [column for column in normalized_train.columns if column not in {"id", "label", "attack_cat"}]

    train_normal = normalized_train.loc[normalized_train["label"] == 0].copy()
    train_attacks = normalized_train.loc[normalized_train["label"] == 1].copy()
    test_normal = normalized_test.loc[normalized_test["label"] == 0].copy()
    test_attacks = normalized_test.loc[normalized_test["label"] == 1].copy()

    if train_normal.empty:
        raise ValueError("UNSW-NB15 Part 3 split requires at least one normal training sample.")
    if train_attacks.empty:
        raise ValueError("UNSW-NB15 Part 3 split requires at least one attack training sample for validation.")
    if test_normal.empty:
        raise ValueError("UNSW-NB15 Part 3 split requires at least one normal test sample.")
    if test_attacks.empty:
        raise ValueError("UNSW-NB15 Part 3 split requires at least one attack test sample.")

    train_normal_fit, val_normal = train_test_split(
        train_normal,
        test_size=validation_size,
        random_state=random_state,
    )
    if train_normal_fit.empty or val_normal.empty:
        raise ValueError("validation_size produced an empty normal fit or validation split.")

    seen_mask = train_attacks["attack_cat"].isin(seen_attack_families)
    val_seen_attacks = train_attacks.loc[seen_mask].copy()
    if val_seen_attacks.empty:
        raise ValueError("No seen-family training attacks matched the configured validation families.")

    unseen_mask = test_attacks["attack_cat"].isin(unseen_attack_families)
    test_unseen_attacks = test_attacks.loc[unseen_mask].copy()
    if test_unseen_attacks.empty:
        raise ValueError("No unseen-family test attacks matched the configured unseen families.")

    X_train_normal = train_normal_fit.loc[:, feature_columns].reset_index(drop=True)

    val_frame = pd.concat([val_normal, val_seen_attacks], axis=0, ignore_index=True)
    y_val = val_frame["label"].astype(int).to_numpy()
    val_attack_families = val_frame["attack_cat"].reset_index(drop=True)
    X_val = val_frame.loc[:, feature_columns].reset_index(drop=True)

    standard_test_frame = pd.concat([test_normal, test_attacks], axis=0, ignore_index=True)
    y_test_standard = standard_test_frame["label"].astype(int).to_numpy()
    test_standard_attack_families = standard_test_frame["attack_cat"].reset_index(drop=True)
    X_test_standard = standard_test_frame.loc[:, feature_columns].reset_index(drop=True)

    unseen_test_frame = pd.concat([test_normal, test_unseen_attacks], axis=0, ignore_index=True)
    y_test_unseen = unseen_test_frame["label"].astype(int).to_numpy()
    test_unseen_attack_families = unseen_test_frame["attack_cat"].reset_index(drop=True)
    X_test_unseen = unseen_test_frame.loc[:, feature_columns].reset_index(drop=True)

    metadata = {
        "feature_columns": feature_columns,
        "validation_size": validation_size,
        "seen_attack_families": list(seen_attack_families),
        "unseen_attack_families": list(unseen_attack_families),
        "auxiliary_attack_families": list(auxiliary_attack_families),
        "n_train_normal_fit": int(X_train_normal.shape[0]),
        "n_val_normal": int((y_val == 0).sum()),
        "n_val_seen_anomalies": int((y_val == 1).sum()),
        "n_test_standard_normal": int((y_test_standard == 0).sum()),
        "n_test_standard_anomalies": int((y_test_standard == 1).sum()),
        "n_test_unseen_normal": int((y_test_unseen == 0).sum()),
        "n_test_unseen_anomalies": int((y_test_unseen == 1).sum()),
    }

    return OneClassDatasetSplit(
        X_train_normal=X_train_normal,
        X_val=X_val,
        y_val=y_val,
        val_attack_families=val_attack_families,
        X_test_standard=X_test_standard,
        y_test_standard=y_test_standard,
        test_standard_attack_families=test_standard_attack_families,
        X_test_unseen=X_test_unseen,
        y_test_unseen=y_test_unseen,
        test_unseen_attack_families=test_unseen_attack_families,
        metadata=metadata,
    )


def build_nsl_kdd_one_class_split(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    validation_size: float = 0.2,
    random_state: int = 42,
) -> OneClassDatasetSplit:
    if validation_size <= 0.0 or validation_size >= 1.0:
        raise ValueError("validation_size must be between 0 and 1.")

    _validate_nsl_columns(train_df, frame_name="train_df")
    _validate_nsl_columns(test_df, frame_name="test_df")

    normalized_train = _normalize_nsl_attack_names(train_df)
    normalized_test = _normalize_nsl_attack_names(test_df)
    feature_columns = [column for column in normalized_train.columns if column not in {"attack_name", "difficulty"}]

    train_normal = normalized_train.loc[normalized_train["attack_name"] == NSL_KDD_NORMAL_LABEL].copy()
    train_attacks = normalized_train.loc[normalized_train["attack_name"] != NSL_KDD_NORMAL_LABEL].copy()
    test_normal = normalized_test.loc[normalized_test["attack_name"] == NSL_KDD_NORMAL_LABEL].copy()
    test_attacks = normalized_test.loc[normalized_test["attack_name"] != NSL_KDD_NORMAL_LABEL].copy()

    if train_normal.empty or train_attacks.empty or test_normal.empty or test_attacks.empty:
        raise ValueError("NSL-KDD Part 3 split requires normal and attack samples in both train and test.")

    train_normal_fit, val_normal = train_test_split(
        train_normal,
        test_size=validation_size,
        random_state=random_state,
    )
    if train_normal_fit.empty or val_normal.empty:
        raise ValueError("validation_size produced an empty NSL-KDD normal fit or validation split.")

    seen_attack_families = tuple(sorted(train_attacks["attack_name"].unique().tolist()))
    unseen_attack_families = tuple(
        sorted(set(test_attacks["attack_name"].unique().tolist()) - set(seen_attack_families))
    )
    if not unseen_attack_families:
        raise ValueError("NSL-KDD Part 3 split found no naturally unseen test attack families.")

    X_train_normal = train_normal_fit.loc[:, feature_columns].reset_index(drop=True)

    val_frame = pd.concat([val_normal, train_attacks], axis=0, ignore_index=True)
    y_val = (val_frame["attack_name"] != NSL_KDD_NORMAL_LABEL).astype(int).to_numpy()
    val_attack_families = val_frame["attack_name"].reset_index(drop=True)
    X_val = val_frame.loc[:, feature_columns].reset_index(drop=True)

    standard_test_frame = pd.concat([test_normal, test_attacks], axis=0, ignore_index=True)
    y_test_standard = (standard_test_frame["attack_name"] != NSL_KDD_NORMAL_LABEL).astype(int).to_numpy()
    test_standard_attack_families = standard_test_frame["attack_name"].reset_index(drop=True)
    X_test_standard = standard_test_frame.loc[:, feature_columns].reset_index(drop=True)

    test_unseen_attacks = test_attacks.loc[test_attacks["attack_name"].isin(unseen_attack_families)].copy()
    unseen_test_frame = pd.concat([test_normal, test_unseen_attacks], axis=0, ignore_index=True)
    y_test_unseen = (unseen_test_frame["attack_name"] != NSL_KDD_NORMAL_LABEL).astype(int).to_numpy()
    test_unseen_attack_families = unseen_test_frame["attack_name"].reset_index(drop=True)
    X_test_unseen = unseen_test_frame.loc[:, feature_columns].reset_index(drop=True)

    metadata = {
        "feature_columns": feature_columns,
        "validation_size": validation_size,
        "seen_attack_families": list(seen_attack_families),
        "unseen_attack_families": list(unseen_attack_families),
        "auxiliary_attack_families": [],
        "n_train_normal_fit": int(X_train_normal.shape[0]),
        "n_val_normal": int((y_val == 0).sum()),
        "n_val_seen_anomalies": int((y_val == 1).sum()),
        "n_test_standard_normal": int((y_test_standard == 0).sum()),
        "n_test_standard_anomalies": int((y_test_standard == 1).sum()),
        "n_test_unseen_normal": int((y_test_unseen == 0).sum()),
        "n_test_unseen_anomalies": int((y_test_unseen == 1).sum()),
    }

    return OneClassDatasetSplit(
        X_train_normal=X_train_normal,
        X_val=X_val,
        y_val=y_val,
        val_attack_families=val_attack_families,
        X_test_standard=X_test_standard,
        y_test_standard=y_test_standard,
        test_standard_attack_families=test_standard_attack_families,
        X_test_unseen=X_test_unseen,
        y_test_unseen=y_test_unseen,
        test_unseen_attack_families=test_unseen_attack_families,
        metadata=metadata,
    )


def build_cic_ids2017_one_class_split(
    frames_by_source: dict[str, pd.DataFrame],
    validation_size: float = 0.2,
    test_normal_size: float | None = None,
    random_state: int = 42,
    seen_attack_families: tuple[str, ...] = CIC_IDS2017_SEEN_ATTACK_FAMILIES,
    unseen_attack_families: tuple[str, ...] = CIC_IDS2017_UNSEEN_ATTACK_FAMILIES,
    auxiliary_attack_families: tuple[str, ...] = CIC_IDS2017_AUXILIARY_ATTACK_FAMILIES,
) -> OneClassDatasetSplit:
    if validation_size <= 0.0 or validation_size >= 1.0:
        raise ValueError("validation_size must be between 0 and 1.")
    if test_normal_size is None:
        test_normal_size = validation_size
    if test_normal_size <= 0.0 or test_normal_size >= 1.0:
        raise ValueError("test_normal_size must be between 0 and 1.")

    missing_sources = set(CIC_IDS2017_REQUIRED_PART3_FILES).difference(frames_by_source)
    if missing_sources:
        raise ValueError(f"Missing CIC-IDS2017 Part 3 sources: {sorted(missing_sources)}")

    cleaned_frames = {key: _clean_cic_part3_frame(frame) for key, frame in frames_by_source.items()}
    benign = cleaned_frames["Benign-Monday-no-metadata"]
    if benign.empty:
        raise ValueError("CIC-IDS2017 Part 3 benign frame must not be empty.")

    holdout_fraction = validation_size + test_normal_size
    if holdout_fraction >= 1.0:
        raise ValueError("validation_size + test_normal_size must be < 1.")

    train_normal_fit, normal_holdout = train_test_split(
        benign,
        test_size=holdout_fraction,
        random_state=random_state,
    )
    holdout_val_fraction = validation_size / holdout_fraction
    val_normal, test_normal = train_test_split(
        normal_holdout,
        test_size=1.0 - holdout_val_fraction,
        random_state=random_state,
    )

    seen_validation_frames = []
    seen_test_frames = []
    for source_key in (
        "Bruteforce-Tuesday-no-metadata",
        "DDoS-Friday-no-metadata",
        "DoS-Wednesday-no-metadata",
        "Portscan-Friday-no-metadata",
    ):
        source_frame = cleaned_frames[source_key]
        source_frame = source_frame.loc[source_frame["Label"].isin(seen_attack_families)].copy()
        if source_frame.empty:
            continue
        val_frame, test_frame = _split_attack_frame(source_frame, validation_size, random_state)
        seen_validation_frames.append(val_frame)
        seen_test_frames.append(test_frame)

    if not seen_validation_frames:
        raise ValueError("CIC-IDS2017 Part 3 split found no seen-family anomalies for validation.")

    unseen_test_frames = []
    for source_key in ("Botnet-Friday-no-metadata", "WebAttacks-Thursday-no-metadata"):
        source_frame = cleaned_frames[source_key]
        source_frame = source_frame.loc[source_frame["Label"].isin(unseen_attack_families)].copy()
        if not source_frame.empty:
            unseen_test_frames.append(source_frame)
    if not unseen_test_frames:
        raise ValueError("CIC-IDS2017 Part 3 split found no unseen-family anomalies for test.")

    feature_columns = [column for column in benign.columns if column != "Label"]
    X_train_normal = train_normal_fit.loc[:, feature_columns].reset_index(drop=True)

    val_seen_anomalies = pd.concat(seen_validation_frames, axis=0, ignore_index=True)
    val_frame = pd.concat([val_normal, val_seen_anomalies], axis=0, ignore_index=True)
    y_val = (~_is_cic_normal_label(val_frame["Label"])).astype(int).to_numpy()
    val_attack_families = val_frame["Label"].reset_index(drop=True)
    X_val = val_frame.loc[:, feature_columns].reset_index(drop=True)

    standard_anomalies = pd.concat(seen_test_frames + unseen_test_frames, axis=0, ignore_index=True)
    standard_test_frame = pd.concat([test_normal, standard_anomalies], axis=0, ignore_index=True)
    y_test_standard = (~_is_cic_normal_label(standard_test_frame["Label"])).astype(int).to_numpy()
    test_standard_attack_families = standard_test_frame["Label"].reset_index(drop=True)
    X_test_standard = standard_test_frame.loc[:, feature_columns].reset_index(drop=True)

    unseen_anomalies = pd.concat(unseen_test_frames, axis=0, ignore_index=True)
    unseen_test_frame = pd.concat([test_normal, unseen_anomalies], axis=0, ignore_index=True)
    y_test_unseen = (~_is_cic_normal_label(unseen_test_frame["Label"])).astype(int).to_numpy()
    test_unseen_attack_families = unseen_test_frame["Label"].reset_index(drop=True)
    X_test_unseen = unseen_test_frame.loc[:, feature_columns].reset_index(drop=True)

    metadata = {
        "feature_columns": feature_columns,
        "validation_size": validation_size,
        "test_normal_size": test_normal_size,
        "seen_attack_families": list(seen_attack_families),
        "unseen_attack_families": list(unseen_attack_families),
        "auxiliary_attack_families": list(auxiliary_attack_families),
        "n_train_normal_fit": int(X_train_normal.shape[0]),
        "n_val_normal": int((y_val == 0).sum()),
        "n_val_seen_anomalies": int((y_val == 1).sum()),
        "n_test_standard_normal": int((y_test_standard == 0).sum()),
        "n_test_standard_anomalies": int((y_test_standard == 1).sum()),
        "n_test_unseen_normal": int((y_test_unseen == 0).sum()),
        "n_test_unseen_anomalies": int((y_test_unseen == 1).sum()),
    }

    return OneClassDatasetSplit(
        X_train_normal=X_train_normal,
        X_val=X_val,
        y_val=y_val,
        val_attack_families=val_attack_families,
        X_test_standard=X_test_standard,
        y_test_standard=y_test_standard,
        test_standard_attack_families=test_standard_attack_families,
        X_test_unseen=X_test_unseen,
        y_test_unseen=y_test_unseen,
        test_unseen_attack_families=test_unseen_attack_families,
        metadata=metadata,
    )


def build_cic_ids2017_temporal_stream_split(
    frames_by_source: dict[str, pd.DataFrame],
    validation_size: float = 0.2,
    random_state: int = 42,
    include_auxiliary: bool = False,
    seen_attack_families: tuple[str, ...] = CIC_IDS2017_SEEN_ATTACK_FAMILIES,
    unseen_attack_families: tuple[str, ...] = CIC_IDS2017_UNSEEN_ATTACK_FAMILIES,
    auxiliary_attack_families: tuple[str, ...] = CIC_IDS2017_AUXILIARY_ATTACK_FAMILIES,
) -> CICTemporalStreamSplit:
    if validation_size <= 0.0 or validation_size >= 1.0:
        raise ValueError("validation_size must be between 0 and 1.")

    missing_sources = set(CIC_IDS2017_REQUIRED_PART3_FILES).difference(frames_by_source)
    if missing_sources:
        raise ValueError(f"Missing CIC-IDS2017 temporal stream sources: {sorted(missing_sources)}")

    cleaned_frames = {key: _clean_cic_part3_frame(frame) for key, frame in frames_by_source.items()}
    benign = cleaned_frames["Benign-Monday-no-metadata"]
    if benign.empty:
        raise ValueError("CIC-IDS2017 temporal stream benign frame must not be empty.")

    train_normal_fit, val_normal = train_test_split(
        benign,
        test_size=validation_size,
        random_state=random_state,
    )
    feature_columns = [column for column in benign.columns if column != "Label"]

    validation_frames: list[pd.DataFrame] = []
    stream_frames_by_source: dict[str, pd.DataFrame] = {}
    for source_key in (
        "Bruteforce-Tuesday-no-metadata",
        "DDoS-Friday-no-metadata",
        "DoS-Wednesday-no-metadata",
        "Portscan-Friday-no-metadata",
    ):
        source_frame = cleaned_frames[source_key]
        seen_source_frame = source_frame.loc[source_frame["Label"].isin(seen_attack_families)].copy()
        if seen_source_frame.empty:
            continue
        val_frame, stream_frame = _split_attack_frame(seen_source_frame, validation_size, random_state)
        validation_frames.append(val_frame)
        if include_auxiliary:
            auxiliary_source_frame = source_frame.loc[source_frame["Label"].isin(auxiliary_attack_families)].copy()
            if not auxiliary_source_frame.empty:
                stream_frame = pd.concat([stream_frame, auxiliary_source_frame], axis=0, ignore_index=True)
        stream_frames_by_source[source_key] = stream_frame

    if not validation_frames:
        raise ValueError("CIC-IDS2017 temporal stream found no seen-family anomalies for validation.")

    for source_key in (
        "Botnet-Friday-no-metadata",
        "WebAttacks-Thursday-no-metadata",
        "Infiltration-Thursday-no-metadata",
    ):
        stream_frames_by_source[source_key] = _filter_cic_stream_frame(
            cleaned_frames[source_key],
            seen_attack_families=seen_attack_families,
            unseen_attack_families=unseen_attack_families,
            auxiliary_attack_families=auxiliary_attack_families,
            include_auxiliary=include_auxiliary,
        )

    X_train_normal = train_normal_fit.loc[:, feature_columns].reset_index(drop=True)
    val_seen_anomalies = pd.concat(validation_frames, axis=0, ignore_index=True)
    val_frame = pd.concat([val_normal, val_seen_anomalies], axis=0, ignore_index=True)
    y_val = (~_is_cic_normal_label(val_frame["Label"])).astype(int).to_numpy()
    val_attack_families = val_frame["Label"].reset_index(drop=True)
    X_val = val_frame.loc[:, feature_columns].reset_index(drop=True)

    stream_batches: list[StreamBatch] = []
    for block_id, timestamp_or_order, day, source_key in CIC_IDS2017_TEMPORAL_BLOCK_SOURCES:
        block_frame = stream_frames_by_source.get(source_key)
        if block_frame is None or block_frame.empty:
            continue
        stream_batches.append(
            _build_cic_stream_batch(
                block_frame,
                block_id=block_id,
                timestamp_or_order=timestamp_or_order,
                source_key=source_key,
                day=day,
                feature_columns=feature_columns,
                seen_attack_families=seen_attack_families,
                unseen_attack_families=unseen_attack_families,
                auxiliary_attack_families=auxiliary_attack_families,
            )
        )

    if not stream_batches:
        raise ValueError("CIC-IDS2017 temporal stream produced no downstream blocks.")

    metadata = {
        "feature_columns": feature_columns,
        "validation_size": validation_size,
        "random_state": random_state,
        "include_auxiliary": include_auxiliary,
        "seen_attack_families": list(seen_attack_families),
        "unseen_attack_families": list(unseen_attack_families),
        "auxiliary_attack_families": list(auxiliary_attack_families),
        "n_train_normal_fit": int(X_train_normal.shape[0]),
        "n_val_normal": int((y_val == 0).sum()),
        "n_val_seen_anomalies": int((y_val == 1).sum()),
        "n_stream_blocks": len(stream_batches),
        "n_stream_rows": int(sum(batch.y.shape[0] for batch in stream_batches)),
        "temporal_block_ids": [batch.block_id for batch in stream_batches],
    }

    return CICTemporalStreamSplit(
        X_train_normal=X_train_normal,
        X_val=X_val,
        y_val=y_val,
        val_attack_families=val_attack_families,
        stream_batches=tuple(stream_batches),
        metadata=metadata,
    )


def compute_one_class_metrics(
    y_true: np.ndarray,
    normality_scores: np.ndarray,
    y_pred_anomalous: np.ndarray,
    attack_families: pd.Series | None = None,
    unseen_attack_families: tuple[str, ...] | None = None,
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    normality_scores = np.asarray(normality_scores, dtype=float)
    y_pred_anomalous = np.asarray(y_pred_anomalous, dtype=int)
    if y_true.ndim != 1 or normality_scores.ndim != 1 or y_pred_anomalous.ndim != 1:
        raise ValueError("y_true, normality_scores, and y_pred_anomalous must be 1D arrays.")
    if not (y_true.shape[0] == normality_scores.shape[0] == y_pred_anomalous.shape[0]):
        raise ValueError("All metric inputs must have the same number of samples.")

    anomaly_scores = -normality_scores
    metrics = {
        "auroc": float(roc_auc_score(y_true, anomaly_scores)),
        "auprc": float(average_precision_score(y_true, anomaly_scores)),
        "f1_anomalous": float(f1_score(y_true, y_pred_anomalous, pos_label=1, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred_anomalous)),
        "normal_acceptance_rate": float(np.mean(y_pred_anomalous[y_true == 0] == 0)),
        "anomaly_recall": float(recall_score(y_true, y_pred_anomalous, pos_label=1, zero_division=0)),
    }

    unseen_detection_rate = np.nan
    if attack_families is not None and unseen_attack_families is not None:
        normalized_families = attack_families.astype(str).str.strip()
        unseen_mask = (y_true == 1) & normalized_families.isin(unseen_attack_families).to_numpy()
        if np.any(unseen_mask):
            unseen_detection_rate = float(np.mean(y_pred_anomalous[unseen_mask] == 1))
    metrics["unseen_family_detection_rate"] = unseen_detection_rate
    return metrics


def select_best_one_class_row(rows: list[dict[str, object]]) -> dict[str, object]:
    if not rows:
        raise ValueError("rows must not be empty.")

    return max(
        rows,
        key=lambda row: (
            float(row["f1_anomalous"]),
            float(row["auprc"]),
            float(row["auroc"]),
            -int(row["n_normal_representatives"]),
        ),
    )


def _normalize_unsw_attack_categories(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    normalized["attack_cat"] = (
        normalized["attack_cat"]
        .fillna("Normal")
        .astype(str)
        .str.strip()
        .replace({"": "Normal"})
    )
    return normalized


def _validate_unsw_columns(df: pd.DataFrame, frame_name: str) -> None:
    required_columns = {"label", "attack_cat"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(f"{frame_name} is missing required UNSW-NB15 columns: {sorted(missing_columns)}")


def _normalize_nsl_attack_names(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    normalized["attack_name"] = (
        normalized["attack_name"]
        .fillna(NSL_KDD_NORMAL_LABEL)
        .astype(str)
        .str.strip()
        .str.lower()
        .str.rstrip(".")
        .replace({"": NSL_KDD_NORMAL_LABEL})
    )
    return normalized


def _validate_nsl_columns(df: pd.DataFrame, frame_name: str) -> None:
    required_columns = {"attack_name", "difficulty"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(f"{frame_name} is missing required NSL-KDD columns: {sorted(missing_columns)}")


def _clean_cic_part3_frame(df: pd.DataFrame) -> pd.DataFrame:
    cleaned_df, _ = _clean_cic_ids2017_frame(df)
    return cleaned_df.reset_index(drop=True)


def _is_cic_normal_label(labels: pd.Series) -> pd.Series:
    normalized = labels.astype(str).str.strip().str.lower()
    return normalized == "benign"


def _filter_cic_stream_frame(
    frame: pd.DataFrame,
    *,
    seen_attack_families: tuple[str, ...],
    unseen_attack_families: tuple[str, ...],
    auxiliary_attack_families: tuple[str, ...],
    include_auxiliary: bool,
) -> pd.DataFrame:
    allowed_families = set(seen_attack_families) | set(unseen_attack_families)
    if include_auxiliary:
        allowed_families.update(auxiliary_attack_families)
    return frame.loc[frame["Label"].isin(allowed_families)].copy()


def _build_cic_stream_batch(
    frame: pd.DataFrame,
    *,
    block_id: str,
    timestamp_or_order: int,
    source_key: str,
    day: str,
    feature_columns: list[str],
    seen_attack_families: tuple[str, ...],
    unseen_attack_families: tuple[str, ...],
    auxiliary_attack_families: tuple[str, ...],
) -> StreamBatch:
    labels = frame["Label"].reset_index(drop=True)
    y = (~_is_cic_normal_label(labels)).astype(int).to_numpy()
    role = _resolve_cic_block_role(
        labels,
        seen_attack_families=seen_attack_families,
        unseen_attack_families=unseen_attack_families,
        auxiliary_attack_families=auxiliary_attack_families,
    )
    label_counts = labels.value_counts().sort_index()
    metadata = {
        "n_rows": int(frame.shape[0]),
        "n_normal": int((y == 0).sum()),
        "n_anomalous": int((y == 1).sum()),
        "label_counts": {str(label): int(count) for label, count in label_counts.items()},
    }
    return StreamBatch(
        X=frame.loc[:, feature_columns].reset_index(drop=True),
        y=y,
        attack_family=labels,
        block_id=block_id,
        timestamp_or_order=timestamp_or_order,
        source_key=source_key,
        day=day,
        role=role,
        metadata=metadata,
    )


def _resolve_cic_block_role(
    labels: pd.Series,
    *,
    seen_attack_families: tuple[str, ...],
    unseen_attack_families: tuple[str, ...],
    auxiliary_attack_families: tuple[str, ...],
) -> str:
    label_set = set(labels.astype(str).str.strip().tolist())
    roles = set()
    if label_set.intersection(seen_attack_families):
        roles.add("seen")
    if label_set.intersection(unseen_attack_families):
        roles.add("unseen")
    if label_set.intersection(auxiliary_attack_families):
        roles.add("auxiliary")
    if not roles:
        return "normal"
    if len(roles) == 1:
        return next(iter(roles))
    return "mixed"


def _split_attack_frame(
    frame: pd.DataFrame,
    validation_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if frame.shape[0] < 2:
        raise ValueError("Attack frame must contain at least two samples for validation/test split.")
    label_counts = frame["Label"].value_counts()
    n_samples = frame.shape[0]
    n_val = int(round(validation_size * n_samples))
    n_val = min(max(n_val, 1), n_samples - 1)
    n_test = n_samples - n_val
    n_classes = int(label_counts.shape[0])
    stratify = None
    if label_counts.min() >= 2 and n_classes > 1 and n_val >= n_classes and n_test >= n_classes:
        stratify = frame["Label"]
    val_frame, test_frame = train_test_split(
        frame,
        train_size=n_val,
        random_state=random_state,
        stratify=stratify,
    )
    return val_frame.reset_index(drop=True), test_frame.reset_index(drop=True)
