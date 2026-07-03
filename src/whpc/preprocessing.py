from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


UNSW_CATEGORICAL_COLUMNS = ["proto", "service", "state"]
NSL_KDD_CATEGORICAL_COLUMNS = ["protocol_type", "service", "flag"]
CIC_IDS2017_CATEGORICAL_COLUMNS: list[str] = []


@dataclass(frozen=True)
class PreprocessingSpec:
    categorical_columns: list[str]
    numeric_columns: list[str]
    transformer: ColumnTransformer


def make_unsw_preprocessor(X: pd.DataFrame) -> PreprocessingSpec:
    return _build_preprocessor(X, categorical_columns=UNSW_CATEGORICAL_COLUMNS)


def make_nsl_kdd_preprocessor(X: pd.DataFrame) -> PreprocessingSpec:
    return _build_preprocessor(X, categorical_columns=NSL_KDD_CATEGORICAL_COLUMNS)


def make_cic_ids2017_preprocessor(X: pd.DataFrame) -> PreprocessingSpec:
    return _build_preprocessor(
        X,
        categorical_columns=CIC_IDS2017_CATEGORICAL_COLUMNS,
        numeric_imputer_strategy="median",
    )


def _build_preprocessor(
    X: pd.DataFrame,
    categorical_columns: list[str],
    numeric_imputer_strategy: str | None = None,
) -> PreprocessingSpec:
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame.")

    missing_columns = [column for column in categorical_columns if column not in X.columns]
    if missing_columns:
        raise ValueError(f"Missing expected categorical columns: {missing_columns}")

    numeric_columns = [column for column in X.columns if column not in categorical_columns]

    transformers = []
    if categorical_columns:
        transformers.append(
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_columns,
            )
        )

    numeric_steps = []
    if numeric_imputer_strategy is not None:
        numeric_steps.append(("imputer", SimpleImputer(strategy=numeric_imputer_strategy)))
    numeric_steps.append(("scaler", StandardScaler()))

    if numeric_columns:
        transformers.append(
            (
                "numeric",
                Pipeline(steps=numeric_steps),
                numeric_columns,
            )
        )

    if not transformers:
        raise ValueError("X must contain at least one feature column.")

    transformer = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return PreprocessingSpec(
        categorical_columns=categorical_columns,
        numeric_columns=numeric_columns,
        transformer=transformer,
    )
