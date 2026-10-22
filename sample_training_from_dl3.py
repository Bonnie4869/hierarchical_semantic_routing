"""Build large_model training sheets from the raw DL3 sheet only.

DL2 is sampled from the full eligible DL3 pool, DL1 from sampled DL2,
and final DL3 prototypes use 15 texts per class. Raw descriptions are kept.
"""
import argparse
import html
import unicodedata
from pathlib import Path

import pandas as pd


def text_key(value):
    if pd.isna(value):
        return ""
    return " ".join(
        unicodedata.normalize("NFKC", html.unescape(str(value))).casefold().split()
    )


def codes(series):
    values = pd.to_numeric(series, errors="coerce")
    valid = values.notna() & values.between(0, 999) & values.eq(values.round())
    if not valid.all():
        raise ValueError(f"Invalid DDC labels: {series[~valid].head().tolist()}")
    return values.astype(int).astype(str).str.zfill(3)


def build(pool, test, seed=42, dl1_n=500, dl2_n=250, dl3_n=15):
    if any(n <= 0 for n in (dl1_n, dl2_n, dl3_n)):
        raise ValueError("Sample counts must be positive")
    for frame in (pool, test):
        if "description" not in frame:
            raise ValueError("Missing description column")
    test_label = "DL3" if "DL3" in test else "DDC"
    if test_label not in test:
        raise ValueError("Test file needs DL3 or DDC")
    test_classes = set(codes(test[test_label]))
    test_keys = set(test.description.map(text_key)) - {""}
    if not test_keys:
        raise ValueError("No nonempty test descriptions")

    pool = pool.copy()
    pool["DDC"] = codes(pool.DDC)
    pool["_key"] = pool.description.map(text_key)
    overlap = pool["_key"].isin(test_keys)
    print(f"Excluded test-overlap rows: {int(overlap.sum())}")
    pool = pool[~overlap & pool["_key"].ne("")].copy()
    pool = pool[pool.DDC.isin(test_classes)].copy()
    # Record deduplication does not change the text supplied to spaCy.
    pool = pool.drop_duplicates("_key").sort_values(["DDC", "_key"])
    counts = pool.groupby("DDC").size()
    insufficient = sorted(test_classes - set(counts[counts >= dl3_n].index))
    if insufficient:
        raise ValueError(f"Test categories cannot supply {dl3_n} unique rows: {insufficient}")
    pool["source_DDC"] = pool.DDC
    pool["DL3"] = pool.DDC
    pool["DL2"] = pool.DDC.str[:2] + "0"
    pool["DL1"] = pool.DDC.str[0] + "00"
    report = []

    def sample(frame, level, target):
        parts = []
        for node, group in frame.groupby(level, sort=True):
            selected = min(target, len(group))
            parts.append(group.sample(n=selected, replace=False, random_state=seed))
            report.append(dict(level=level, node=node, available=len(group),
                               target=target, selected=selected, shortage=target-selected))
            if selected < target:
                print(f"{level} {node}: {selected}/{target}; no replacement")
        return pd.concat(parts, ignore_index=True)

    dl2 = sample(pool, "DL2", dl2_n)
    dl1 = sample(dl2, "DL1", dl1_n)
    dl3 = sample(pool, "DL3", dl3_n)
    if not set(dl1["_key"]).issubset(set(dl2["_key"])):
        raise RuntimeError("DL1 is not a subset of DL2")
    tables = {}
    for name, frame in {"DL1": dl1, "DL2": dl2, "DL3": dl3}.items():
        if frame["_key"].duplicated().any() or set(frame["_key"]) & test_keys:
            raise RuntimeError(f"Duplicate or test overlap: {name}")
        frame = frame.copy()
        # large_model groups by DDC within each sheet: use the parent code.
        frame["DDC"] = frame[name]
        frame = frame.drop(columns=["_key", "clean_description"], errors="ignore")
        tables[name] = frame
        print(f"{name}: {len(frame)} rows, {frame.DDC.nunique()} nodes")
    return tables, pd.DataFrame(report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--output", default="trainingset.xlsx")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dl1", type=int, default=500)
    parser.add_argument("--dl2", type=int, default=250)
    parser.add_argument("--dl3", type=int, default=15)
    args = parser.parse_args()
    output = Path(args.output)
    report_path = output.with_name(output.stem + "_sampling_report.csv")
    for path in (output, report_path):
        if path.exists():
            raise FileExistsError(f"Will not overwrite: {path}")
    if output.resolve() in {Path(args.input).resolve(), Path(args.test).resolve()}:
        raise ValueError("Output must differ from inputs")
    # Original DL1/DL2 sheets are never read.
    pool = pd.read_excel(args.input, sheet_name="DL3", dtype=str, keep_default_na=False)
    test = pd.read_excel(args.test, dtype=str, keep_default_na=False)
    tables, report = build(pool, test, args.seed, args.dl1, args.dl2, args.dl3)
    output.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for sheet, frame in tables.items():
            frame.to_excel(writer, sheet_name=sheet, index=False)
    report.to_csv(report_path, index=False, encoding="utf-8-sig")
    print(f"Saved {output}; raw description retained; test set unchanged")


if __name__ == "__main__":
    main()
