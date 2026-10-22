"""Build independent DL1/DL2/DL3 DDC reference features from three sheets.

Input schema:
  - workbook ``trainingset.xlsx``
  - sheets ``DL1``, ``DL2`` and ``DL3``
  - ``DDC``: three-digit node code in every sheet (100, 120, 123, ...)
  - ``description``: document text

The sheet name determines the DDC level. DDC string length is not used to infer
the level. Rows already sampled into each sheet are used exactly as provided.
"""

import os

import pandas as pd

from feature_builder_node import aggregate_category_features, save_category_features


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(BASE_DIR, "trainingset.xlsx")
LABEL_COLUMN = "DDC"
TEXT_COLUMN = "description"
OUTPUT_DIR = os.path.join(BASE_DIR, "ZSL", "Category_features", "DDC")

LEVEL_CONFIG = {
    1: {"sheet": "DL1", "subdir": "level1", "level_name": "DL1"},
    2: {"sheet": "DL2", "subdir": "level2", "level_name": "DL2"},
    3: {"sheet": "DL3", "subdir": "level3", "level_name": "DL3"},
}


def read_training_data() -> pd.DataFrame:
    """Read DL1/DL2/DL3 sheets and assign level from the sheet name."""
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"找不到输入文件: {INPUT_FILE}")

    available_sheets = pd.ExcelFile(INPUT_FILE).sheet_names
    required_sheets = [LEVEL_CONFIG[level]["sheet"] for level in (1, 2, 3)]
    missing_sheets = [s for s in required_sheets if s not in available_sheets]
    if missing_sheets:
        raise ValueError(
            f"缺少 Sheet {missing_sheets!r}; 可用 Sheet: {available_sheets}"
        )

    level_frames = []
    for level in (1, 2, 3):
        sheet = LEVEL_CONFIG[level]["sheet"]
        frame = pd.read_excel(
            INPUT_FILE,
            sheet_name=sheet,
            usecols=[LABEL_COLUMN, TEXT_COLUMN],
            dtype={LABEL_COLUMN: str},
            keep_default_na=False,
        )
        frame[LABEL_COLUMN] = frame[LABEL_COLUMN].astype(str).str.strip()
        frame[TEXT_COLUMN] = frame[TEXT_COLUMN].astype(str).str.strip()

        # All sheets use the normalized three-digit node representation.
        invalid_code = ~frame[LABEL_COLUMN].str.fullmatch(r"\d{3}")
        if invalid_code.any():
            examples = frame.loc[invalid_code, LABEL_COLUMN].head(10).tolist()
            raise ValueError(
                f"Sheet {sheet}: DDC必须是三位数字编码；异常值示例: "
                + repr(examples)
            )

        empty_text = frame[TEXT_COLUMN].eq("")
        if empty_text.any():
            print(
                f"警告: Sheet {sheet} 丢弃 "
                f"{int(empty_text.sum())} 条空 description"
            )
            frame = frame.loc[~empty_text].copy()

        frame["ddc_level"] = level
        level_frames.append(frame)

    return pd.concat(level_frames, ignore_index=True)


def build_level_features(frame: pd.DataFrame, level: int) -> None:
    """Build node-local features from rows read from the corresponding sheet."""
    config = LEVEL_CONFIG[level]
    level_frame = frame.loc[frame["ddc_level"].eq(level)]
    groups = level_frame.groupby(LABEL_COLUMN, sort=True)

    print("\n" + "=" * 60)
    print(
        f"构建 {config['level_name']}：Sheet={config['sheet']}，"
        f"{len(level_frame)} 条记录，{groups.ngroups} 个类别"
    )
    print("=" * 60)

    output_dir = os.path.join(OUTPUT_DIR, config["subdir"])
    os.makedirs(output_dir, exist_ok=True)

    for index, (label, group) in enumerate(groups, 1):
        # Use exactly the rows already sampled into this level's sheet.
        corpus = group[TEXT_COLUMN].tolist()
        print(
            f"\n  ({index}/{groups.ngroups}) 类别 {label}: "
            f"{len(corpus)} 篇独立文档"
        )
        features_data = aggregate_category_features(corpus)
        output_path = os.path.join(output_dir, f"category_{label}.pkl")
        save_category_features(features_data, output_path)

    print(
        f"\n  {config['level_name']} 完成："
        f"{len(level_frame)} 条记录，{groups.ngroups} 个类别"
    )


def main() -> None:
    print("\n" + "=" * 60)
    print("按 DL1/DL2/DL3 Sheet 构建三套独立节点特征")
    print("=" * 60)
    print(f"输入文件: {INPUT_FILE}")
    print("输入 Sheets: DL1, DL2, DL3")

    frame = read_training_data()
    counts = frame.groupby("ddc_level").size().to_dict()
    unique_counts = frame.groupby("ddc_level")[LABEL_COLUMN].nunique().to_dict()
    for level in (1, 2, 3):
        print(
            f"  DL{level}: {counts.get(level, 0)} 条记录，"
            f"{unique_counts.get(level, 0)} 个类别"
        )

    for level in (1, 2, 3):
        build_level_features(frame, level)

    print("\n" + "=" * 60)
    print("训练完成！")
    print(f"输出目录: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
