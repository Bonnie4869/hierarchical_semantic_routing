#!/usr/bin/env python3
"""Create leakage-safe DL1/DL2/DL3 training data and a DL3 test set."""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd


def arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="./train_stratified_DL1_DL2_DL3.xlsx")
    p.add_argument("--train-output", default="./trainingset.xlsx")
    p.add_argument("--test-output", default="./testset.xlsx")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dl1-target", type=int, default=500)
    p.add_argument("--dl2-target", type=int, default=250)
    p.add_argument("--dl3-min", type=int, default=20)
    p.add_argument("--dl3-train", type=int, default=15)
    p.add_argument("--dl3-test", type=int, default=5)
    p.add_argument(
        "--allow-extra-dl3-for-dl2",
        action="store_true",
        help="Use additional non-test DL3 rows to fill DL2 shortages.",
    )
    p.add_argument(
        "--keep-provenance",
        action="store_true",
        help="Add source_DL and source_DDC columns.",
    )
    return p.parse_args()


def find_column(df: pd.DataFrame, name: str) -> str:
    mapping = {str(c).strip().casefold(): c for c in df.columns}
    if name.casefold() not in mapping:
        raise ValueError(f"Missing column {name!r}; columns={list(df.columns)!r}")
    return mapping[name.casefold()]


def only_digits(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if re.fullmatch(r"\d+\.0", text):
        text = text[:-2]
    return "".join(re.findall(r"\d", text))


def normalize_ddc(value: object, level: int) -> str:
    digits = only_digits(value)
    if not digits:
        raise ValueError(f"Blank DDC in DL{level}")
    if level == 1:
        return digits[0] + "00"
    if level == 2:
        if len(digits) == 1:
            digits = digits.zfill(2)
        return digits[:2] + "0"
    return digits.zfill(3)[-3:]


def clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return " ".join(str(value).casefold().split())


def add_doc_key(df: pd.DataFrame) -> pd.DataFrame:
    title = find_column(df, "Title")
    description = find_column(df, "description")
    out = df.copy()

    def fingerprint(row: pd.Series) -> str:
        text = clean_text(row[title]) + "\x00" + clean_text(row[description])
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    out["_doc_key"] = out.apply(fingerprint, axis=1)
    return out


def read_level(
    path: Path, sheet: str, level: int, rng: np.random.Generator
) -> tuple[pd.DataFrame, list[str], str]:
    df = pd.read_excel(path, sheet_name=sheet, dtype=object).dropna(how="all")
    original_columns = list(df.columns)
    ddc_col = find_column(df, "DDC")
    df = df.copy()
    df[ddc_col] = df[ddc_col].map(lambda x: normalize_ddc(x, level))
    df = add_doc_key(df)
    df["_source_DL"] = sheet
    df["_source_DDC"] = df[ddc_col]
    df["_rand"] = rng.random(len(df))
    return df, original_columns, ddc_col


def take(pool: pd.DataFrame, count: int) -> pd.DataFrame:
    if count <= 0 or pool.empty:
        return pool.iloc[0:0].copy()
    return pool.sort_values("_rand", kind="stable").head(count).copy()


def set_label(df: pd.DataFrame, ddc_col: str, ddc: str) -> pd.DataFrame:
    out = df.copy()
    out[ddc_col] = ddc
    return out


def select_dl3(
    dl3: pd.DataFrame,
    ddc_col: str,
    minimum: int,
    train_count: int,
    test_count: int,
) -> tuple[pd.DataFrame, pd.DataFrame, set[str], list[str], set[str]]:
    """Select 15 train + 5 test; skip a node if either side is unavailable.

    Test documents are globally disjoint from all selected DL3 training
    documents. Training documents may repeat in different training levels.
    """
    unique_counts = dl3.groupby(ddc_col)["_doc_key"].nunique()
    eligible = set(unique_counts[unique_counts >= minimum].index)
    source = dl3[dl3[ddc_col].isin(eligible)]

    test_parts: list[pd.DataFrame] = []
    train_parts: list[pd.DataFrame] = []
    test_keys: set[str] = set()
    train_keys: set[str] = set()
    kept: set[str] = set()
    skipped: list[str] = []

    for ddc, group in source.groupby(ddc_col, sort=True):
        candidates = (
            group.sort_values("_rand", kind="stable")
            .drop_duplicates("_doc_key")
        )

        # A test document cannot already exist in an accepted training sample.
        possible_test = candidates[
            ~candidates["_doc_key"].isin(test_keys | train_keys)
        ]
        chosen_test = possible_test.head(test_count).copy()
        if len(chosen_test) < test_count:
            skipped.append(ddc)
            continue

        proposed_test_keys = set(chosen_test["_doc_key"])
        # Training-training overlap is allowed. Only test keys are forbidden.
        possible_train = candidates[
            ~candidates["_doc_key"].isin(test_keys | proposed_test_keys)
        ]
        chosen_train = possible_train.head(train_count).copy()
        if len(chosen_train) < train_count:
            skipped.append(ddc)
            continue

        test_parts.append(chosen_test)
        train_parts.append(chosen_train)
        test_keys.update(proposed_test_keys)
        train_keys.update(chosen_train["_doc_key"])
        kept.add(ddc)

    if not kept:
        raise RuntimeError("No DL3 node can provide both 5 test and 15 train rows")

    return (
        pd.concat(train_parts, ignore_index=True),
        pd.concat(test_parts, ignore_index=True),
        kept,
        skipped,
        test_keys,
    )


def build_dl2(
    dl2: pd.DataFrame,
    dl3_train: pd.DataFrame,
    dl3_extra: pd.DataFrame,
    ddc_col: str,
    target: int,
    allow_extra: bool,
) -> tuple[pd.DataFrame, list[tuple[str, int]]]:
    results: list[pd.DataFrame] = []
    shortages: list[tuple[str, int]] = []

    for ddc in sorted(dl2[ddc_col].unique()):
        own = take(dl2[dl2[ddc_col] == ddc], target)
        need = target - len(own)

        child_train = dl3_train[dl3_train[ddc_col].str[:2] == ddc[:2]]
        added = take(child_train, need)
        need -= len(added)
        parts = [set_label(own, ddc_col, ddc), set_label(added, ddc_col, ddc)]

        if need > 0 and allow_extra:
            # Do not choose the same document twice inside this DL2 node.
            used = set(added["_doc_key"])
            pool = dl3_extra[
                (dl3_extra[ddc_col].str[:2] == ddc[:2])
                & (~dl3_extra["_doc_key"].isin(used))
            ]
            extra = take(pool, need)
            parts.append(set_label(extra, ddc_col, ddc))
            need -= len(extra)

        node = pd.concat(parts, ignore_index=True)
        results.append(node)
        if need > 0:
            shortages.append((ddc, len(node)))

    return pd.concat(results, ignore_index=True), shortages


def build_dl1(
    dl1: pd.DataFrame,
    dl2_training: pd.DataFrame,
    ddc_col: str,
    target: int,
) -> tuple[pd.DataFrame, list[tuple[str, int]]]:
    results: list[pd.DataFrame] = []
    shortages: list[tuple[str, int]] = []

    for ddc in sorted(dl1[ddc_col].unique()):
        own = take(dl1[dl1[ddc_col] == ddc], target)
        need = target - len(own)
        child = dl2_training[dl2_training[ddc_col].str[0] == ddc[0]]
        added = take(child, need)
        node = pd.concat(
            [set_label(own, ddc_col, ddc), set_label(added, ddc_col, ddc)],
            ignore_index=True,
        )
        results.append(node)
        if len(node) < target:
            shortages.append((ddc, len(node)))

    return pd.concat(results, ignore_index=True), shortages


def final_columns(
    df: pd.DataFrame, columns: list[str], keep_provenance: bool
) -> pd.DataFrame:
    out = df.copy()
    selected = list(columns)
    if keep_provenance:
        out["source_DL"] = out["_source_DL"]
        out["source_DDC"] = out["_source_DDC"]
        selected += ["source_DL", "source_DDC"]
    return out.loc[:, selected]


def main() -> None:
    args = arguments()
    source = Path(args.input)
    train_output = Path(args.train_output)
    test_output = Path(args.test_output)
    if train_output.resolve() == test_output.resolve():
        raise ValueError("train-output and test-output must be different")
    if args.dl3_train + args.dl3_test > args.dl3_min:
        raise ValueError("dl3-train + dl3-test cannot exceed dl3-min")

    rng = np.random.default_rng(args.seed)
    dl1, cols1, ddc1 = read_level(source, "DL1", 1, rng)
    dl2, cols2, ddc2 = read_level(source, "DL2", 2, rng)
    dl3, cols3, ddc3 = read_level(source, "DL3", 3, rng)
    if not (ddc1 == ddc2 == ddc3):
        raise ValueError("DDC column name differs between sheets")
    ddc_col = ddc1

    dl3_training, dl3_test, kept, skipped, test_keys = select_dl3(
        dl3,
        ddc_col,
        args.dl3_min,
        args.dl3_train,
        args.dl3_test,
    )

    # Every occurrence of a test document is removed from every training source.
    dl1_safe = dl1[~dl1["_doc_key"].isin(test_keys)].copy()
    dl2_safe = dl2[~dl2["_doc_key"].isin(test_keys)].copy()
    dl3_safe = dl3[
        dl3[ddc_col].isin(kept) & ~dl3["_doc_key"].isin(test_keys)
    ].copy()
    dl3_train_keys = set(dl3_training["_doc_key"])
    dl3_extra = dl3_safe[~dl3_safe["_doc_key"].isin(dl3_train_keys)].copy()

    dl2_training, dl2_shortages = build_dl2(
        dl2_safe,
        dl3_training,
        dl3_extra,
        ddc_col,
        args.dl2_target,
        args.allow_extra_dl3_for_dl2,
    )
    dl1_training, dl1_shortages = build_dl1(
        dl1_safe, dl2_training, ddc_col, args.dl1_target
    )

    # Final train/test leakage check. DL1, DL2 and DL3 may overlap each other.
    all_train_keys = set(dl1_training["_doc_key"])
    all_train_keys.update(dl2_training["_doc_key"])
    all_train_keys.update(dl3_training["_doc_key"])
    overlap = all_train_keys & test_keys
    if overlap:
        raise AssertionError(f"Train/test leakage detected: {len(overlap)} documents")

    train_output.parent.mkdir(parents=True, exist_ok=True)
    test_output.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(train_output, engine="openpyxl") as writer:
        final_columns(dl1_training, cols1, args.keep_provenance).to_excel(
            writer, sheet_name="DL1", index=False
        )
        final_columns(dl2_training, cols2, args.keep_provenance).to_excel(
            writer, sheet_name="DL2", index=False
        )
        final_columns(dl3_training, cols3, args.keep_provenance).to_excel(
            writer, sheet_name="DL3", index=False
        )
    with pd.ExcelWriter(test_output, engine="openpyxl") as writer:
        final_columns(dl3_test, cols3, args.keep_provenance).to_excel(
            writer, sheet_name="DL3", index=False
        )

    print(f"DL3 nodes kept: {len(kept)}")
    print(f"DL3 nodes skipped: {len(skipped)}")
    if skipped:
        print("Skipped DL3 DDC:", ", ".join(skipped))
    print(
        f"Training rows: DL1={len(dl1_training)}, "
        f"DL2={len(dl2_training)}, DL3={len(dl3_training)}"
    )
    print(f"Test rows: DL3={len(dl3_test)}")
    print(f"Training workbook: {train_output.resolve()}")
    print(f"Test workbook: {test_output.resolve()}")
    if dl1_shortages:
        print("WARNING - DL1 below target:", dl1_shortages)
    if dl2_shortages:
        print("WARNING - DL2 below target:", dl2_shortages)
        if not args.allow_extra_dl3_for_dl2:
            print("Add --allow-extra-dl3-for-dl2 if DL2 must reach 250 rows.")


if __name__ == "__main__":
    main()
