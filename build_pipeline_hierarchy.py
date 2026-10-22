#!/usr/bin/env python3
"""Build the DL1/DL2/DL3 hierarchy workbook required by pipelines.py."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


def normalize(value: object) -> str:
    if pd.isna(value):
        raise ValueError("empty DDC")
    text = str(value).strip()
    if re.fullmatch(r"\d+\.0", text):
        text = text[:-2]
    if not re.fullmatch(r"\d{1,3}", text):
        raise ValueError(f"invalid DDC: {value!r}")
    return text.zfill(3)


def read_codes(path: Path, sheet: str) -> set[str]:
    frame = pd.read_excel(path, sheet_name=sheet, usecols=["DDC"], dtype=object)
    return {normalize(value) for value in frame["DDC"].dropna()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="./trainingset.xlsx")
    parser.add_argument("--output", default="./pipeline_hierarchy.xlsx")
    args = parser.parse_args()

    source = Path(args.input)
    output = Path(args.output)
    if not source.exists():
        raise FileNotFoundError(source)
    if source.resolve() == output.resolve():
        raise ValueError("Input and output paths must be different")

    dl1 = read_codes(source, "DL1")
    dl2 = read_codes(source, "DL2")
    dl3 = read_codes(source, "DL3")

    missing_dl1 = sorted({code[0] + "00" for code in dl2} - dl1)
    missing_dl2 = sorted({code[:2] + "0" for code in dl3} - dl2)
    if missing_dl1 or missing_dl2:
        raise ValueError(
            f"Missing parents: DL1={missing_dl1[:20]}, DL2={missing_dl2[:20]}"
        )

    rows: list[dict[str, object]] = []
    # Include every DL2 node, including nodes that currently have no DL3 child.
    for code in sorted(dl2):
        rows.append({"DL1": code[0] + "00", "DL2": code, "DL3": pd.NA})
    # Include every retained DL3 node and its complete ancestor path.
    for code in sorted(dl3):
        rows.append(
            {"DL1": code[0] + "00", "DL2": code[:2] + "0", "DL3": code}
        )

    hierarchy = pd.DataFrame(rows).drop_duplicates()
    output.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        hierarchy.to_excel(writer, sheet_name="Hierarchy", index=False)

    print(f"DL1 nodes: {len(dl1)}")
    print(f"DL2 nodes: {len(dl2)}")
    print(f"DL3 nodes: {len(dl3)}")
    print(f"Output: {output.resolve()}")


if __name__ == "__main__":
    main()
