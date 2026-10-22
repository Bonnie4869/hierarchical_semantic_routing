# embed_reference_bows.py
"""
只生成参考词袋（用于 WMD/Sinkhorn 精确计算）
输出:
  - reference_bows.pkl (分层存储，包含 name, ddc_level, semantic)
  - reference_metadata.pkl (和矩阵版本共用)
"""

import os
import pickle
import time
from typing import Dict

# ===== 配置 =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ZSL_DIR = os.path.join(BASE_DIR, "ZSL")

# ===== 输入输出路径 =====
LEVEL1_DIR = os.path.join(ZSL_DIR, "Category_features", "DDC", "level1")
LEVEL2_DIR = os.path.join(ZSL_DIR, "Category_features", "DDC", "level2")
LEVEL3_DIR = os.path.join(ZSL_DIR, "Category_features", "DDC", "level3")

OUTPUT_DIR = os.path.join(ZSL_DIR, "Reference_embeddings_v2")
OUTPUT_BOW_PATH = os.path.join(OUTPUT_DIR, "reference_bows.pkl")
OUTPUT_META_PATH = os.path.join(OUTPUT_DIR, "reference_metadata.pkl")


def load_category_files(directory: str) -> Dict[str, Dict]:
    """加载目录下所有 category_*.pkl 文件"""
    categories = {}
    if not os.path.exists(directory):
        print(f"警告: 目录不存在: {directory}")
        return categories
    for filename in os.listdir(directory):
        if filename.startswith("category_") and filename.endswith(".pkl"):
            filepath = os.path.join(directory, filename)
            with open(filepath, "rb") as f:
                data = pickle.load(f)
            category_name = filename[len("category_") : -len(".pkl")]
            categories[category_name] = data
    return categories


def extract_bow_from_category(category_data: Dict) -> Dict[int, Dict[str, float]]:
    """
    分层提取词袋
    返回: {1: {词: 权重}, 2: {词对: 权重}, 3: {三元组: 权重}}
    """
    bow = {1: {}, 2: {}, 3: {}}

    for level in [1, 2, 3]:
        features = category_data["category_features"].get(level, {})
        for term, weight in features.items():
            if isinstance(term, str):
                bow[level][term] = bow[level].get(term, 0) + weight
            elif isinstance(term, tuple):
                key = "||".join(term)
                bow[level][key] = bow[level].get(key, 0) + weight

    return bow


def build_reference_bows() -> tuple:
    """构建所有类别的参考词袋"""
    print("\n" + "=" * 60)
    print("构建 Reference 词袋 (用于 WMD/Sinkhorn)")
    print("=" * 60)

    print("\n[1/3] 加载类别文件...")
    level1_cats = load_category_files(LEVEL1_DIR)
    level2_cats = load_category_files(LEVEL2_DIR)
    level3_cats = load_category_files(LEVEL3_DIR)

    print(f"  Level 1: {len(level1_cats)} 个类别")
    print(f"  Level 2: {len(level2_cats)} 个类别")
    print(f"  Level 3: {len(level3_cats)} 个类别")

    print("\n[2/3] 合并类别列表...")
    all_categories = []
    all_metadata = []

    for cat_name in sorted(level1_cats.keys()):
        all_categories.append((cat_name, level1_cats[cat_name], 1))
        all_metadata.append({"name": cat_name, "level": 1})

    for cat_name in sorted(level2_cats.keys()):
        all_categories.append((cat_name, level2_cats[cat_name], 2))
        all_metadata.append({"name": cat_name, "level": 2})

    for cat_name in sorted(level3_cats.keys()):
        all_categories.append((cat_name, level3_cats[cat_name], 3))
        all_metadata.append({"name": cat_name, "level": 3})

    print(f"  总共 {len(all_categories)} 个参考点")

    print("\n[3/3] 提取词袋...")
    all_bows = []

    total = len(all_categories)
    for i, (cat_name, data, level) in enumerate(all_categories):
        semantic_bow = extract_bow_from_category(data)
        # ★ 修改：包装成 classify_zsl.py 需要的格式
        all_bows.append(
            {"name": cat_name, "ddc_level": level, "semantic": semantic_bow}
        )
        all_metadata[i]["ddc_level"] = level  # ★ 同时给 metadata 添加 ddc_level

        if (i + 1) % 100 == 0:
            print(f"  已处理 {i + 1}/{total} 个类别")

    print(f"  处理完成！共 {total} 个类别")

    return all_bows, all_metadata


def save_reference_bows(bows: list, metadata: list):
    """保存词袋和元数据"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(OUTPUT_BOW_PATH, "wb") as f:
        pickle.dump(bows, f)
    print(f"\n✅ 词袋已保存至: {OUTPUT_BOW_PATH}")

    with open(OUTPUT_META_PATH, "wb") as f:
        pickle.dump(metadata, f)
    print(f"✅ 元数据已保存至: {OUTPUT_META_PATH}")


def main():
    start_time = time.time()
    try:
        bows, metadata = build_reference_bows()
        save_reference_bows(bows, metadata)

        elapsed = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"✅ 词袋生成完成！")
        print(f"   总耗时: {elapsed:.2f} 秒")
        print(f"   参考点总数: {len(metadata)}")
        print(f"   输出目录: {OUTPUT_DIR}")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback

        traceback.print_exc()


def load_reference_bows():
    """加载词袋（供测试阶段使用）"""
    if not os.path.exists(OUTPUT_BOW_PATH):
        raise FileNotFoundError(f"词袋文件不存在: {OUTPUT_BOW_PATH}")

    with open(OUTPUT_BOW_PATH, "rb") as f:
        bows = pickle.load(f)

    with open(OUTPUT_META_PATH, "rb") as f:
        metadata = pickle.load(f)

    return bows, metadata


if __name__ == "__main__":
    main()
