#!/usr/bin/env python3
"""Randomly sample exactly K rows per DL3 DDC node from an Excel test set."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="./testset.xlsx")
    parser.add_argument("--output", default="./testset_k_per_dl3.xlsx")
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sheet",
        help="Input sheet. If omitted, prefer DL3, then Sheet1, then the first sheet.",
    )
    parser.add_argument("--output-sheet", default=None)
    parser.add_argument(
        "--ddc-column",
        help="Grouping column. If omitted, prefer DL3 and then DDC.",
    )
    parser.add_argument(
        "--shortfall",
        choices=["skip", "error", "all"],
        default="skip",
        help=(
            "What to do when a DL3 node has fewer than K rows: "
            "skip it, raise an error, or retain all available rows."
        ),
    )
    return parser.parse_args()


def normalize_dl3(value: object) -> str:
    if pd.isna(value):
        raise ValueError("empty DL3 DDC")
    text = str(value).strip()
    if re.fullmatch(r"\d+\.0", text):
        text = text[:-2]
    if not re.fullmatch(r"\d{1,3}", text):
        raise ValueError(f"invalid DL3 DDC: {value!r}")
    return text.zfill(3)


def choose_sheet(path: Path, requested: str | None) -> str:
    sheets = pd.ExcelFile(path).sheet_names
    if requested:
        if requested not in sheets:
            raise ValueError(f"Sheet {requested!r} not found; available: {sheets}")
        return requested
    if "DL3" in sheets:
        return "DL3"
    if "Sheet1" in sheets:
        return "Sheet1"
    if not sheets:
        raise ValueError("Workbook has no sheets")
    return sheets[0]


def main() -> None:
    args = parse_args()
    if args.k < 1:
        raise ValueError("--k must be at least 1")

    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output paths must be different")

    sheet = choose_sheet(input_path, args.sheet)
    frame = pd.read_excel(input_path, sheet_name=sheet, dtype=object)
    frame = frame.dropna(how="all").copy()

    if args.ddc_column:
        ddc_column = args.ddc_column
    elif "DL3" in frame.columns:
        ddc_column = "DL3"
    elif "DDC" in frame.columns:
        ddc_column = "DDC"
    else:
        raise ValueError(
            f"No DL3 or DDC column found; columns: {list(frame.columns)}"
        )
    if ddc_column not in frame.columns:
        raise ValueError(f"Column {ddc_column!r} not found")

    normalized_codes: list[str] = []
    invalid_rows: list[tuple[int, object, str]] = []
    for excel_row, value in enumerate(frame[ddc_column], start=2):
        try:
            normalized_codes.append(normalize_dl3(value))
        except ValueError as exc:
            invalid_rows.append((excel_row, value, str(exc)))
    if invalid_rows:
        raise ValueError(f"Invalid DDC rows (up to 10): {invalid_rows[:10]}")

    frame[ddc_column] = normalized_codes
    frame["_source_order"] = np.arange(len(frame))
    rng = np.random.default_rng(args.seed)
    frame["_random"] = rng.random(len(frame))

    selected: list[pd.DataFrame] = []
    skipped: list[tuple[str, int]] = []
    short_kept: list[tuple[str, int]] = []

    for ddc, group in frame.groupby(ddc_column, sort=True):
        available = len(group)
        if available < args.k:
            if args.shortfall == "error":
                raise RuntimeError(
                    f"DL3 {ddc} has only {available} rows, fewer than K={args.k}"
                )
            if args.shortfall == "skip":
                skipped.append((ddc, available))
                continue
            short_kept.append((ddc, available))
            chosen = group
        else:
            chosen = group.sort_values("_random", kind="stable").head(args.k)
        selected.append(chosen)

    if not selected:
        raise RuntimeError("No rows selected")

    result = pd.concat(selected, ignore_index=True)
    result = result.sort_values(
        [ddc_column, "_source_order"], kind="stable"
    ).drop(columns=["_source_order", "_random"])

    output_sheet = args.output_sheet or sheet
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        result.to_excel(writer, sheet_name=output_sheet, index=False)

    counts = result.groupby(ddc_column).size()
    exact_nodes = int((counts == args.k).sum())
    print(f"Grouping column: {ddc_column}")
    print(f"K per DL3 node: {args.k}")
    print(f"DL3 nodes with exactly K rows: {exact_nodes}")
    print(f"Output rows: {len(result)}")
    print(f"Output: {output_path.resolve()}")
    if skipped:
        print(f"Skipped nodes with fewer than K rows: {len(skipped)}")
        print("Skipped DL3 DDC:", ", ".join(code for code, _ in skipped))
    if short_kept:
        print(f"Nodes retained with fewer than K rows: {len(short_kept)}")


if __name__ == "__main__":
    main()
