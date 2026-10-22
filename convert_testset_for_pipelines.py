#!/usr/bin/env python3
"""Convert a DL3-only test workbook to the input schema used by pipelines.py.

Input workbook:
  - sheet: DL3
  - required columns: DDC, description
  - DDC is a three-digit leaf code, for example 139 or 005

Output workbook:
  - sheet: Sheet1
  - adds DL1, DL2, DL3 and clean_description

Example:
  DDC 139 -> DL1 100, DL2 130, DL3 139
  DDC 005 -> DL1 000, DL2 000, DL3 005
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="./testset.xlsx")
    parser.add_argument("--output", default="./testset_for_pipelines.xlsx")
    parser.add_argument("--input-sheet", default="DL3")
    parser.add_argument("--output-sheet", default="Sheet1")
    parser.add_argument("--ddc-column", default="DDC")
    parser.add_argument("--text-column", default="description")
    return parser.parse_args()


def normalize_leaf_ddc(value: object) -> str:
    """Return one three-digit DL3 code while preserving leading zeroes."""
    if pd.isna(value):
        raise ValueError("DDC is empty")

    text = str(value).strip()
    # Excel may expose a numeric code as 5.0 rather than 005.
    if re.fullmatch(r"\d+\.0", text):
        text = text[:-2]

    if not re.fullmatch(r"\d{1,3}", text):
        raise ValueError(f"Invalid DDC value: {value!r}")

    return text.zfill(3)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(input_path)
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output paths must be different")

    available_sheets = pd.ExcelFile(input_path).sheet_names
    if args.input_sheet not in available_sheets:
        raise ValueError(
            f"Missing sheet {args.input_sheet!r}; available sheets: "
            f"{available_sheets}"
        )

    frame = pd.read_excel(
        input_path,
        sheet_name=args.input_sheet,
        dtype={args.ddc_column: str},
        keep_default_na=False,
    )

    required = [args.ddc_column, args.text_column]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(
            f"Missing required columns {missing}; found {list(frame.columns)}"
        )

    empty_rows = frame.astype(str).apply(lambda col: col.str.strip()).eq("").all(axis=1)
    frame = frame.loc[~empty_rows].copy()

    codes: list[str] = []
    invalid: list[tuple[int, object, str]] = []
    for excel_row, value in enumerate(frame[args.ddc_column], start=2):
        try:
            codes.append(normalize_leaf_ddc(value))
        except ValueError as exc:
            invalid.append((excel_row, value, str(exc)))

    if invalid:
        examples = invalid[:10]
        raise ValueError(f"Invalid DDC rows (up to 10 shown): {examples}")

    frame[args.ddc_column] = codes
    frame[args.text_column] = frame[args.text_column].astype(str).str.strip()

    empty_text = frame[args.text_column].eq("")
    if empty_text.any():
        print(f"Warning: dropping {int(empty_text.sum())} rows with empty text")
        frame = frame.loc[~empty_text].copy()

    code = frame[args.ddc_column]
    frame["DL1"] = code.str[0] + "00"
    frame["DL2"] = code.str[:2] + "0"
    frame["DL3"] = code
    # This matches pipelines.py's default --text-column clean_description.
    frame["clean_description"] = frame[args.text_column]

    # Put the fields consumed by pipelines.py first and retain the source fields.
    first = ["DL1", "DL2", "DL3", "clean_description"]
    remaining = [column for column in frame.columns if column not in first]
    frame = frame[first + remaining]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        frame.to_excel(writer, sheet_name=args.output_sheet, index=False)

    print(f"Converted rows: {len(frame)}")
    print(f"Output: {output_path.resolve()}")
    print("Example labels:")
    print(frame[["DL1", "DL2", "DL3"]].head(5).to_string(index=False))


if __name__ == "__main__":
    main()
