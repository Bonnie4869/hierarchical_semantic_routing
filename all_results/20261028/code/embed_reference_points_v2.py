# embed_reference_points.py
"""
将 category_*.pkl 文件中的"词+权重"字典，转换成三个独立的数字向量矩阵
支持多种词向量模型: Word2Vec (bin/txt), GloVe (txt), FastText (bin/vec)
输出:
  - L1_reference.npy: 语义 L1 参考向量 (N, d)，N = 所有 DDC 类别数
  - L2_reference.npy: 语义 L2 参考向量 (N, d)
  - L3_reference.npy: 语义 L3 参考向量 (N, d)
  - reference_metadata.pkl: 类别元数据 (name, ddc_level)
  - reference_bows.pkl: 结构化词袋 (用于 WMD/Sinkhorn)
      item = {"name", "ddc_level", "semantic": {1: L1词袋, 2: L2词袋, 3: L3词袋}}
      约定: ddc_level ∈ {1,2,3} = DDC 层级 DL1/DL2/DL3（分类标签粒度）；
            semantic 键 1/2/3 = 语义特征层 L1(名词)/L2(名词对)/L3(三元组)。
            两个"层"维度正交，勿混淆。
"""

import os
import pickle
import numpy as np
import time
from typing import Dict, Union, Tuple, List, Optional

# ===== 配置 =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ZSL_DIR = os.path.join(BASE_DIR, "ZSL")

# ===== 词向量模型配置 =====
# 使用 FastText .vec 文本格式（推荐，支持 OOV）
EMBEDDING_PATH = os.path.join(BASE_DIR, "wiki-news-300d-1M-subword.vec")
EMBEDDING_FORMAT = (
    "fasttext_vec"  # fasttext_vec, fasttext_bin, word2vec_bin, word2vec_txt, glove_txt
)

# ===== 输入输出路径 =====
LEVEL1_DIR = os.path.join(ZSL_DIR, "Category_features", "DDC", "level1")
LEVEL2_DIR = os.path.join(ZSL_DIR, "Category_features", "DDC", "level2")
LEVEL3_DIR = os.path.join(ZSL_DIR, "Category_features", "DDC", "level3")

OUTPUT_DIR = os.path.join(ZSL_DIR, "Reference_embeddings_v2")
L1_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "L1_reference.npy")
L2_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "L2_reference.npy")
L3_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "L3_reference.npy")
OUTPUT_META_PATH = os.path.join(OUTPUT_DIR, "reference_metadata.pkl")
OUTPUT_BOW_PATH = os.path.join(OUTPUT_DIR, "reference_bows.pkl")  # ★ 新增：词袋保存路径


# ============================================================
# 第一部分：加载不同格式的词向量
# ============================================================


def load_embeddings(model_path: str, format_type: str):
    """
    加载词向量模型
    支持: fasttext_vec, fasttext_bin, word2vec_bin, word2vec_txt, glove_txt
    返回: (model_data, format_type, vector_dim)
    """
    print(f"正在加载词向量: {model_path}")
    print(f"格式: {format_type}")
    start_time = time.time()

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"词向量文件未找到: {model_path}")

    if format_type == "fasttext_vec":
        word_vectors, vector_dim = load_fasttext_vec(model_path)
        print(f"   耗时: {time.time() - start_time:.2f} 秒")
        return word_vectors, format_type, vector_dim

    elif format_type == "fasttext_bin":
        model, vector_dim = load_fasttext_bin(model_path)
        print(f"   耗时: {time.time() - start_time:.2f} 秒")
        return model, format_type, vector_dim

    elif format_type == "word2vec_bin":
        model, vector_dim = load_word2vec_bin(model_path)
        print(f"   耗时: {time.time() - start_time:.2f} 秒")
        return model, format_type, vector_dim

    elif format_type == "word2vec_txt":
        model, vector_dim = load_word2vec_txt(model_path)
        print(f"   耗时: {time.time() - start_time:.2f} 秒")
        return model, format_type, vector_dim

    elif format_type == "glove_txt":
        word_vectors, vector_dim = load_glove_txt(model_path)
        print(f"   耗时: {time.time() - start_time:.2f} 秒")
        return word_vectors, format_type, vector_dim

    else:
        raise ValueError(f"不支持的格式: {format_type}")


def load_fasttext_vec(vec_path: str):
    """
    加载 FastText .vec 文本格式
    格式: 第一行 "词汇数 维度"，后面每行 "词 向量(空格分隔)"
    返回: (word_vectors_dict, vector_dim)
    """
    print("加载 FastText .vec 格式...")
    word_vectors = {}
    vector_dim = None
    start_time = time.time()

    with open(vec_path, "r", encoding="utf-8") as f:
        first_line = f.readline().strip()
        parts = first_line.split()

        if len(parts) == 2:
            vocab_size = int(parts[0])
            vector_dim = int(parts[1])
            print(f"  词汇量: {vocab_size}, 维度: {vector_dim}")
        else:
            f.seek(0)
            vector_dim = None

        count = 0
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue

            word = parts[0]
            if vector_dim is None:
                vector_dim = len(parts) - 1
                print(f"  自动检测维度: {vector_dim}")

            try:
                vec = np.array(
                    [float(x) for x in parts[1 : vector_dim + 1]], dtype=np.float32
                )
                if len(vec) == vector_dim:
                    word_vectors[word] = vec
                    count += 1
            except ValueError:
                continue

            if count % 100000 == 0:
                print(f"  已加载 {count} 个词向量...")

    print(f"[OK] 加载完成，共 {len(word_vectors)} 个词向量")
    print(f"   耗时: {time.time() - start_time:.2f} 秒")

    return word_vectors, vector_dim


def load_fasttext_bin(bin_path: str):
    """
    加载 FastText .bin 二进制格式
    返回: (model, vector_dim)
    """
    print("加载 FastText .bin 格式...")
    try:
        import fasttext

        model = fasttext.load_model(bin_path)
        vector_dim = model.get_dimension()
        print(f"[OK] 加载完成 (fasttext 库)，维度: {vector_dim}")
        return model, vector_dim
    except:
        try:
            from gensim.models.fasttext import FastText as GensimFastText

            model = GensimFastText.load_fasttext_format(bin_path)
            vector_dim = model.wv.vector_size
            print(f"[OK] 加载完成 (gensim 库)，维度: {vector_dim}")
            return model, vector_dim
        except Exception as e:
            raise ImportError(f"无法加载 .bin: {e}\n请安装: pip install fasttext")


def load_word2vec_bin(bin_path: str):
    """加载 Word2Vec .bin 二进制格式 (GoogleNews)"""
    print("加载 Word2Vec .bin 格式...")
    from gensim.models import KeyedVectors

    model = KeyedVectors.load_word2vec_format(bin_path, binary=True)
    vector_dim = model.vector_size
    print(f"[OK] 加载完成，词表: {len(model.key_to_index)}，维度: {vector_dim}")
    return model, vector_dim


def load_word2vec_txt(txt_path: str):
    """加载 Word2Vec .txt 文本格式"""
    print("加载 Word2Vec .txt 格式...")
    from gensim.models import KeyedVectors

    model = KeyedVectors.load_word2vec_format(txt_path, binary=False)
    vector_dim = model.vector_size
    print(f"[OK] 加载完成，词表: {len(model.key_to_index)}，维度: {vector_dim}")
    return model, vector_dim


def load_glove_txt(txt_path: str):
    """
    加载 GloVe .txt 格式
    格式: 每行 "词 向量(空格分隔)"
    返回: (word_vectors_dict, vector_dim)
    """
    print("加载 GloVe .txt 格式...")
    word_vectors = {}
    vector_dim = None
    start_time = time.time()

    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue

            word = parts[0]
            if vector_dim is None:
                vector_dim = len(parts) - 1
                print(f"  检测维度: {vector_dim}")

            try:
                vec = np.array([float(x) for x in parts[1:]], dtype=np.float32)
                if len(vec) == vector_dim:
                    word_vectors[word] = vec
            except ValueError:
                continue

    print(f"[OK] 加载完成，共 {len(word_vectors)} 个词向量")
    print(f"   耗时: {time.time() - start_time:.2f} 秒")
    return word_vectors, vector_dim


# ============================================================
# 第二部分：获取词向量的统一接口
# ============================================================


class EmbeddingWrapper:
    """统一词向量获取接口"""

    def __init__(self, model_data, format_type: str, vector_dim: int):
        self.model_data = model_data
        self.format_type = format_type
        self.vector_dim = vector_dim
        self._cache = {}  # 缓存已查询的词向量

    def get_vector(self, word: str) -> np.ndarray:
        """获取单个词的向量，支持 OOV"""
        # 检查缓存
        if word in self._cache:
            return self._cache[word]

        word_lower = word.lower()
        if word_lower in self._cache:
            vec = self._cache[word_lower]
            self._cache[word] = vec
            return vec

        vec = self._get_vector_impl(word)
        self._cache[word] = vec
        return vec

    def _get_vector_impl(self, word: str) -> np.ndarray:
        """实际获取向量的实现"""
        # 1. 直接查找
        vec = self._raw_get(word)
        if vec is not None and np.linalg.norm(vec) > 0:
            return vec

        # 2. 尝试小写
        vec = self._raw_get(word.lower())
        if vec is not None and np.linalg.norm(vec) > 0:
            return vec

        # 3. 尝试首字母大写
        vec = self._raw_get(word.capitalize())
        if vec is not None and np.linalg.norm(vec) > 0:
            return vec

        # 4. 找不到，返回零向量
        return np.zeros(self.vector_dim)

    def _raw_get(self, word: str) -> Optional[np.ndarray]:
        """底层获取向量"""
        if self.format_type in ["fasttext_vec", "glove_txt"]:
            # dict 类型
            return self.model_data.get(word)

        elif self.format_type == "fasttext_bin":
            # fasttext 库
            try:
                return self.model_data.get_word_vector(word)
            except:
                return None

        elif self.format_type in ["word2vec_bin", "word2vec_txt"]:
            # gensim KeyedVectors
            try:
                return self.model_data[word]
            except:
                return None

        return None

    def get_vector_for_term(self, term: Union[str, Tuple[str, ...]]) -> np.ndarray:
        """获取词或词组的向量（元组取平均）"""
        if isinstance(term, str):
            return self.get_vector(term)
        elif isinstance(term, tuple):
            vectors = [self.get_vector(w) for w in term]
            valid_vecs = [v for v in vectors if np.linalg.norm(v) > 0]
            if valid_vecs:
                return np.mean(valid_vecs, axis=0)
            else:
                return np.zeros(self.vector_dim)
        else:
            return np.zeros(self.vector_dim)


# ============================================================
# 第三部分：类别特征转换
# ============================================================


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


def extract_bow_from_category(
    category_data: Dict,
) -> Dict[int, Dict[Union[str, Tuple[str, ...]], float]]:
    """
    ★ 从 category_data 中提取【按语义层分层】的结构化词袋（用于 WMD/Sinkhorn）。

    说明：DLk 是 DDC 层级（分类标签），L1/L2/L3 是语义特征层级（文档表示），
    二者正交——本函数返回的是后者，即一个 DDC 类在三套语义粒度下的词袋。

    返回: {
        1: {单词(str): 权重, ...},              # L1 语义层：名词
        2: {(word_a, word_b): 权重, ...},        # L2 语义层：名词对（保留 tuple 键）
        3: {(word_a, word_b, word_c): 权重, ...} # L3 语义层：三元组（保留 tuple 键）
    }
    - L2/L3 不再提前 "||".join 成字符串：词袋保留结构信息，词向量阶段由
      EmbeddingWrapper.get_vector_for_term 对 tuple 的组分词取平均，query 侧
      也生成同样的结构化词袋即可严格对齐，避免字符串拼接/解析不一致。
    - 权重沿用 category_features 每层的原始归一化值（每个语义层内部和≈1）。
    """
    semantic_bows: Dict[int, Dict[Union[str, Tuple[str, ...]], float]] = {1: {}, 2: {}, 3: {}}
    features_map = category_data.get("category_features", {})
    for level in (1, 2, 3):
        for term, weight in features_map.get(level, {}).items():
            # L1 键是 str，L2/L3 键是 tuple，都原样保留
            if isinstance(term, (str, tuple)):
                semantic_bows[level][term] = float(weight)
    return semantic_bows


def category_to_three_vectors(
    category_data: Dict, embed_wrapper: EmbeddingWrapper
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    将 category_data 转换成三个独立的向量
    返回: (L1_vector, L2_vector, L3_vector)
    """
    vector_dim = embed_wrapper.vector_dim
    L1_vector = np.zeros(vector_dim)
    L2_vector = np.zeros(vector_dim)
    L3_vector = np.zeros(vector_dim)

    # Level 1: 单个词
    for term, weight in category_data["category_features"].get(1, {}).items():
        vec = embed_wrapper.get_vector_for_term(term)
        L1_vector += vec * weight

    # Level 2: 词对
    for term, weight in category_data["category_features"].get(2, {}).items():
        vec = embed_wrapper.get_vector_for_term(term)
        L2_vector += vec * weight

    # Level 3: 三元组
    for term, weight in category_data["category_features"].get(3, {}).items():
        vec = embed_wrapper.get_vector_for_term(term)
        L3_vector += vec * weight

    return L1_vector, L2_vector, L3_vector


def build_reference_embeddings(embed_wrapper: EmbeddingWrapper) -> tuple:
    """
    构建所有类别的参考向量
    返回: (L1_matrix, L2_matrix, L3_matrix, metadata, bows)
    """
    print("\n" + "=" * 60)
    print("开始构建 Reference Embeddings (三矩阵版本 + 词袋)")
    print(f"向量维度: {embed_wrapper.vector_dim}")
    print("=" * 60)

    # 1. 加载所有类别文件
    print("\n[1/4] 加载类别文件...")
    level1_cats = load_category_files(LEVEL1_DIR)
    level2_cats = load_category_files(LEVEL2_DIR)
    level3_cats = load_category_files(LEVEL3_DIR)

    print(f"  Level 1: {len(level1_cats)} 个类别")
    print(f"  Level 2: {len(level2_cats)} 个类别")
    print(f"  Level 3: {len(level3_cats)} 个类别")

    # 2. 合并所有类别
    print("\n[2/4] 合并类别列表...")
    all_categories = []
    all_metadata = []

    # 注：这里的 1/2/3 是 DDC 层级 DL1/DL2/DL3（与语义层 L1/L2/L3 无关），
    # 统一用键名 ddc_level 表达，避免与 semantic bows 的语义层键 1/2/3 混淆。
    for cat_name in sorted(level1_cats.keys()):
        all_categories.append((cat_name, level1_cats[cat_name], 1))
        all_metadata.append({"name": cat_name, "ddc_level": 1})

    for cat_name in sorted(level2_cats.keys()):
        all_categories.append((cat_name, level2_cats[cat_name], 2))
        all_metadata.append({"name": cat_name, "ddc_level": 2})

    for cat_name in sorted(level3_cats.keys()):
        all_categories.append((cat_name, level3_cats[cat_name], 3))
        all_metadata.append({"name": cat_name, "ddc_level": 3})

    print(f"  总共 {len(all_categories)} 个参考点")

    # 3. 转换向量 + 提取词袋
    print("\n[3/4] 转换类别特征为三个独立向量 + 提取词袋...")
    L1_list = []
    L2_list = []
    L3_list = []
    all_bows = []  # ★ 新增：存储每个类别的词袋

    total = len(all_categories)
    for i, (cat_name, data, ddc_level) in enumerate(all_categories):
        # 生成三个向量
        L1_vec, L2_vec, L3_vec = category_to_three_vectors(data, embed_wrapper)
        L1_list.append(L1_vec)
        L2_list.append(L2_vec)
        L3_list.append(L3_vec)

        # ★ 词袋条目自描述化，不再依赖与 metadata 的隐式顺序对齐。
        #    semantic 内层：{1: L1名词词袋, 2: L2名词对词袋, 3: L3三元组词袋}
        semantic = extract_bow_from_category(data)
        bow_item = {"name": cat_name, "ddc_level": ddc_level, "semantic": semantic}
        all_bows.append(bow_item)

        if (i + 1) % 100 == 0:
            print(f"  已处理 {i + 1}/{total} 个类别")

    print(f"  处理完成！共 {total} 个类别")

    # 4. 堆叠矩阵
    print("\n[4/4] 堆叠成矩阵...")
    L1_matrix = np.vstack(L1_list)
    L2_matrix = np.vstack(L2_list)
    L3_matrix = np.vstack(L3_list)

    print(f"  L1 矩阵形状: {L1_matrix.shape}")
    print(f"  L2 矩阵形状: {L2_matrix.shape}")
    print(f"  L3 矩阵形状: {L3_matrix.shape}")
    print(f"  词袋数量: {len(all_bows)}")

    return L1_matrix, L2_matrix, L3_matrix, all_metadata, all_bows


def save_reference_embeddings(
    L1: np.ndarray, L2: np.ndarray, L3: np.ndarray, metadata: list, bows: list
):
    """保存三个参考矩阵、元数据和词袋"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 保存三个矩阵
    np.save(L1_OUTPUT_PATH, L1)
    np.save(L2_OUTPUT_PATH, L2)
    np.save(L3_OUTPUT_PATH, L3)

    print(f"\n[OK] L1 参考向量已保存至: {L1_OUTPUT_PATH}")
    print(f"[OK] L2 参考向量已保存至: {L2_OUTPUT_PATH}")
    print(f"[OK] L3 参考向量已保存至: {L3_OUTPUT_PATH}")

    # 保存元数据
    with open(OUTPUT_META_PATH, "wb") as f:
        pickle.dump(metadata, f)
    print(f"[OK] 元数据已保存至: {OUTPUT_META_PATH}")

    # ★ 新增：保存词袋
    with open(OUTPUT_BOW_PATH, "wb") as f:
        pickle.dump(bows, f)
    print(f"[OK] 词袋已保存至: {OUTPUT_BOW_PATH}")


# ============================================================
# 第四部分：主函数和加载函数
# ============================================================


def main():
    """主函数"""
    start_time = time.time()

    try:
        # 1. 加载词向量模型
        model_data, format_type, vector_dim = load_embeddings(
            EMBEDDING_PATH, EMBEDDING_FORMAT
        )
        print(f"向量维度: {vector_dim}")

        # 2. 创建统一包装器
        embed_wrapper = EmbeddingWrapper(model_data, format_type, vector_dim)

        # 3. 构建参考向量 + 词袋
        L1, L2, L3, metadata, bows = build_reference_embeddings(embed_wrapper)

        # 4. 保存结果
        save_reference_embeddings(L1, L2, L3, metadata, bows)

        # 5. 打印统计
        elapsed = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"[OK] 全部完成！")
        print(f"   总耗时: {elapsed:.2f} 秒")
        print(f"   参考点总数: {len(metadata)}")
        print(f"   向量维度: {vector_dim}")
        print(f"   词袋数量: {len(bows)}")
        print(f"   词向量格式: {EMBEDDING_FORMAT}")
        print(f"   输出目录: {OUTPUT_DIR}")
        print("=" * 60)

    except Exception as e:
        print(f"\n[ERR] 错误: {e}")
        import traceback

        traceback.print_exc()


def load_reference_embeddings():
    """加载已保存的三个参考矩阵（供测试阶段使用）"""
    if not all(
        [os.path.exists(p) for p in [L1_OUTPUT_PATH, L2_OUTPUT_PATH, L3_OUTPUT_PATH]]
    ):
        raise FileNotFoundError(
            "参考向量文件不存在，请先运行 embed_reference_points.py"
        )

    L1 = np.load(L1_OUTPUT_PATH, allow_pickle=True)
    L2 = np.load(L2_OUTPUT_PATH, allow_pickle=True)
    L3 = np.load(L3_OUTPUT_PATH, allow_pickle=True)

    with open(OUTPUT_META_PATH, "rb") as f:
        metadata = pickle.load(f)

    return L1, L2, L3, metadata


def load_reference_with_bows():
    """★ 新增：加载参考矩阵 + 词袋（用于 WMD/Sinkhorn）"""
    L1, L2, L3, metadata = load_reference_embeddings()

    if not os.path.exists(OUTPUT_BOW_PATH):
        raise FileNotFoundError(f"词袋文件不存在: {OUTPUT_BOW_PATH}")

    with open(OUTPUT_BOW_PATH, "rb") as f:
        bows = pickle.load(f)

    # 词袋必须与矩阵/元数据一一对应（行序一致）
    if len(bows) != len(metadata):
        raise ValueError(
            f"reference_bows.pkl({len(bows)} 项) 与 "
            f"reference_metadata.pkl({len(metadata)} 项) 长度不一致"
        )

    return L1, L2, L3, metadata, bows


if __name__ == "__main__":
    main()
