"""Count independent dataset rows for every DL1/DL2/DL3 DDC node."""

import argparse
from pathlib import Path

import pandas as pd


def parse_args():
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=base_dir / "train_stratified.xlsx",
        help="Input XLSX containing a DDC column (default: beside this script)",
    )
    parser.add_argument("--sheet", default="Sheet1", help="Input sheet name")
    parser.add_argument(
        "--output",
        type=Path,
        default=base_dir / "ddc_entry_counts.xlsx",
        help="Output XLSX path",
    )
    return parser.parse_args()


def read_ddc_codes(path: Path, sheet_name: str) -> pd.Series:
    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {path}")

    frame = pd.read_excel(
        path,
        sheet_name=sheet_name,
        usecols=["DDC"],
        dtype={"DDC": str},
        keep_default_na=False,
    )
    codes = frame["DDC"].astype(str).str.strip()
    valid = codes.str.fullmatch(r"\d{1,3}")
    if not valid.all():
        examples = codes.loc[~valid].head(10).tolist()
        raise ValueError(
            "DDC must contain only 1-3 digit codes; invalid examples: "
            f"{examples}"
        )
    return codes


def count_by_level(codes: pd.Series):
    level_tables = {}
    summary_rows = []

    for level in (1, 2, 3):
        level_codes = codes.loc[codes.str.len().eq(level)]
        counts = (
            level_codes.value_counts(sort=False)
            .rename_axis("DDC")
            .reset_index(name="entry_count")
            .sort_values("DDC", kind="stable")
            .reset_index(drop=True)
        )
        # DDC hierarchy display convention: pad on the right to three digits.
        # Examples: DL1 1 -> 100, DL2 12 -> 120, DL3 123 -> 123.
        counts["DDC"] = counts["DDC"].str.pad(3, side="right", fillchar="0")
        counts["percentage_of_level"] = (
            counts["entry_count"] / len(level_codes) * 100.0
        )
        level_tables[level] = counts

        distribution = counts["entry_count"]
        summary_rows.append(
            {
                "level": f"DL{level}",
                "ddc_digits": level,
                "node_count": len(counts),
                "entry_count": len(level_codes),
                "min_entries_per_node": int(distribution.min()),
                "max_entries_per_node": int(distribution.max()),
                "mean_entries_per_node": float(distribution.mean()),
                "median_entries_per_node": float(distribution.median()),
            }
        )

    return pd.DataFrame(summary_rows), level_tables


def save_report(summary: pd.DataFrame, level_tables, output: Path):
    output.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for level in (1, 2, 3):
            level_tables[level].to_excel(
                writer, sheet_name=f"DL{level}", index=False
            )


def main():
    args = parse_args()
    codes = read_ddc_codes(args.input, args.sheet)
    summary, level_tables = count_by_level(codes)
    save_report(summary, level_tables, args.output)

    print(summary.to_string(index=False))
    for level in (1, 2, 3):
        print(f"\nDL{level} counts")
        print(level_tables[level].to_string(index=False))
    print(f"\nSaved: {args.output.resolve()}")


if __name__ == "__main__":
    main()
