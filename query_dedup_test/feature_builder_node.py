# feature_builder_node.py
"""
节点内DF的特征构建器
每个DDC节点独立计算DF和IDF
"""

import math
import pickle
import os
from collections import Counter, defaultdict
from itertools import combinations
from typing import List, Dict, Tuple, Union

import spacy

# 加载spaCy模型（使用小模型，速度更快）
nlp = spacy.load("en_core_web_sm")


# ============================================================
# [问题3修复] 单遍解析基础设施
# ============================================================
# 改动原因：此前 precompute_dfs（算DF）与 aggregate_category_features（算TF-IDF）
#           各自对同一批文本调用 nlp(text)，每篇文档被 spaCy 解析两遍，且是逐条
#           for 调用，没有利用 spaCy 的批处理能力。
# 改动方案：所有 spaCy 解析只发生在 aggregate_category_features 开头的一次
#           nlp.pipe(...) 批处理里；下面两个纯 Python helper 只在已解析的
#           Doc 对象上工作，不触发任何额外的模型调用。

# nlp.pipe 的批大小（越大吞吐越高，内存占用也越大）
_BATCH_SIZE = 64


def _extract_doc_features(
    doc,
) -> Tuple[
    Tuple[set, set, set],
    Tuple[List[str], List[Tuple[str, ...]], List[Tuple[str, ...]]],
]:
    """
    从一篇【已解析】的 Doc 中一次遍历提取全部中间量（不调用 nlp）。

    DF 与 TF-IDF 共用的“名词/术语提取、L2/L3 组合”逻辑都集中在这里，
    确保两轮计算基于完全相同的特征定义，不再各写一份。

    Args:
        doc: spaCy 解析后的文档对象（由外部 nlp.pipe 产出）

    Returns:
        (df_terms, tf_terms):
            df_terms = (doc_nouns, doc_pairs, doc_triples)
                       —— 文档级去重集合，供 DF 使用（相同组合在文档内只计 1 次）
            tf_terms = (all_nouns, all_pairs, all_triples)
                       —— 文档级可重复序列，供 TF-IDF 使用。L2/L3 在组合前
                          保留重复词，与 Dataset/feature_builder.py 一致；因此
                          重复组合和 (term, term) 自组合会参与 TF。
    """
    # ---- 文档级 DF 集合 ----
    doc_nouns = set()
    doc_pairs = set()
    doc_triples = set()

    # ---- 文档级 TF 序列 ----
    all_nouns: List[str] = []
    all_pairs: List[Tuple[str, ...]] = []
    all_triples: List[Tuple[str, ...]] = []

    for sent in doc.sents:
        # 提取名词（保留含重复的原始词流：L1 词频需要真实出现次数）
        raw_nouns = [
            token.lemma_.lower()
            for token in sent
            if token.pos_ in {"NOUN", "PROPN"}
        ]
        all_nouns.extend(raw_nouns)

        # 提取所有术语（名词 + 动词，排除 be/have/do；原始词流）
        raw_terms = [
            token.lemma_.lower()
            for token in sent
            if token.pos_ in {"NOUN", "PROPN"}
            or (
                token.pos_ == "VERB"
                and token.lemma_ not in {"be", "have", "do"}
            )
        ]

        # 与 Dataset/feature_builder.py 保持一致：组合前不对词流做 set 去重。
        # DF 仍由下面的 doc_pairs/doc_triples 集合保证“同一组合每篇文档只计一次”；
        # TF 则保留重复组合，并允许重复词产生 (term, term) 等自组合。
        nouns = sorted(raw_nouns)
        terms = sorted(raw_terms)

        # ---- 更新文档级 DF 集合（集合负责消除同一文档内的重复项）----
        doc_nouns.update(nouns)
        if len(nouns) >= 2:
            doc_pairs.update(combinations(nouns, 2))
        if len(terms) >= 3:
            doc_triples.update(combinations(terms, 3))

        # ---- 更新文档级 TF 序列（跨句子累计，保留每句内的组合次数）----
        if len(nouns) >= 2:
            all_pairs.extend(combinations(nouns, 2))
        if len(terms) >= 3:
            all_triples.extend(combinations(terms, 3))

    return (doc_nouns, doc_pairs, doc_triples), (all_nouns, all_pairs, all_triples)


def _accumulate_dfs(
    df_counts: Dict[int, Counter],
    doc_features: List[Tuple[Tuple[set, set, set], Tuple]],
) -> Dict[int, Counter]:
    """
    把每篇文档的 DF 集合累加进 df_counts（每个术语出现在几篇文档就计几）。

    doc_features 由 _extract_doc_features 对每篇文档各调用一次得到；
    累加本身是纯内存操作，不再触发 spaCy。
    """
    for (doc_nouns, doc_pairs, doc_triples), _ in doc_features:
        # 更新DF：每个术语只要出现在该文档中，就+1
        df_counts[1].update(doc_nouns)
        df_counts[2].update(doc_pairs)
        df_counts[3].update(doc_triples)
    return df_counts



def precompute_dfs(corpus: List[str]) -> Dict[int, Counter]:
    """
    预计算当前节点语料库的文档频率(DF)
    ===== 这是"节点内DF" =====

    [问题3修复] 本函数保留为“单独使用”时的兼容入口：
    内部改用 nlp.pipe 批处理解析一遍，不再逐条 nlp(text)。

    ⚠️ 训练主流程 aggregate_category_features 不要再调用本函数，
    否则 corpus 会被再解析一遍（等于退回“解析两遍”）；主流程应自己
    nlp.pipe 一次后复用 _extract_doc_features + _accumulate_dfs。

    Args:
        corpus: 当前节点的所有文档列表

    Returns:
        df_counts: {
            1: Counter(1-gram名词),
            2: Counter(2-gram名词对),
            3: Counter(3-gram名词+动词三元组)
        }
    """
    # [问题3修复] 一次批处理解析全部文档
    docs = list(nlp.pipe(corpus, batch_size=_BATCH_SIZE))
    doc_features = [_extract_doc_features(doc) for doc in docs]
    df_counts = _accumulate_dfs(
        {
            1: Counter(),  # 1-gram（单个实体/名词）
            2: Counter(),  # 2-gram（实体对）
            3: Counter(),  # 3-gram（实体-谓词-实体三元组）
        },
        doc_features,
    )
    return df_counts


# def aggregate_category_features(
#     corpus: List[str],
# ) -> Tuple[
#     Dict[int, Dict[Union[str, Tuple[str, ...]], float]], Dict[int, Counter], int
# ]:
#     """
#     聚合整个类别的特征向量（节点内TF-IDF）
#     ===== 核心：IDF基于当前节点的DF（节点内DF） =====

#     Args:
#         corpus: 当前节点的所有文档列表

#     Returns:
#         category_features: {
#             1: {"quantum": 0.30, "physics": 0.15, ...},
#             2: {("quantum","physics"): 0.12, ...},
#             3: {("quantum","physics","study"): 0.08, ...}
#         }
#         df_counts: 当前节点的DF统计
#         total_docs: 当前节点的文档总数
#     """
#     # ===== [问题3修复] Step 1: 只解析一遍（nlp.pipe 批处理）=====
#     # 改动原因：此前这里调用 precompute_dfs(corpus)，下面 Step 3 又对同一批文本
#     #           逐个 nlp(text)，每篇文档被 spaCy 解析两遍。
#     # 改动方案：把解析收敛为一次 nlp.pipe(...) 批处理，产出 Doc 列表；
#     #           再用 _extract_doc_features 每篇只遍历一次，同时缓存
#     #           “DF 集合 + TF 序列”，Step 3 直接复用缓存，不再接触 nlp。
#     total_docs = len(corpus)
#     docs = list(nlp.pipe(corpus, batch_size=_BATCH_SIZE))
#     doc_features = [_extract_doc_features(doc) for doc in docs]

#     # 用缓存的文档级集合累加“节点内 DF”（节点内DF！）
#     df_counts = _accumulate_dfs(
#         {
#             1: Counter(),  # 1-gram（单个实体/名词）
#             2: Counter(),  # 2-gram（实体对）
#             3: Counter(),  # 3-gram（实体-谓词-实体三元组）
#         },
#         doc_features,
#     )

#     # Step 2: 初始化特征向量
#     category_features = {
#         1: defaultdict(float),
#         2: defaultdict(float),
#         3: defaultdict(float),
#     }

#     # ===== [问题3修复] Step 3: 遍历【已解析】文档，计算TF-IDF并累加 =====
#     # 不再对每篇文档重复 nlp(text)，也不在此处重复句子级提取：
#     # 直接消费 Step 1 缓存好的 (all_nouns, all_pairs, all_triples)。
#     for _, (all_nouns, all_pairs, all_triples) in doc_features:
#         # ========== Level 1: TF-IDF（节点内DF） ==========
#         noun_counts = Counter(all_nouns)
#         total_nouns = len(all_nouns) if all_nouns else 1
#         for term, freq in noun_counts.items():
#             tf = freq / total_nouns
#             df = df_counts[1].get(term, 0)
#             idf = math.log((total_docs + 1) / (df + 1))  # 平滑处理
#             category_features[1][term] += tf * idf

#         # ========== Level 2: TF-IDF（节点内DF） ==========
#         pair_counts = Counter(all_pairs)
#         total_pairs = len(all_pairs) if all_pairs else 1
#         for term, freq in pair_counts.items():
#             tf = freq / total_pairs
#             df = df_counts[2].get(term, 0)
#             idf = math.log((total_docs + 1) / (df + 1))
#             category_features[2][term] += tf * idf

#         # ========== Level 3: TF-IDF（节点内DF） ==========
#         triple_counts = Counter(all_triples)
#         total_triples = len(all_triples) if all_triples else 1
#         for term, freq in triple_counts.items():
#             tf = freq / total_triples
#             df = df_counts[3].get(term, 0)
#             idf = math.log((total_docs + 1) / (df + 1))
#             category_features[3][term] += tf * idf

#     # Step 4: 归一化（每个级别独立归一化）
#     for level in category_features:
#         total_score = sum(category_features[level].values())
#         if total_score > 0:
#             for term in category_features[level]:
#                 category_features[level][term] /= total_score

#     return category_features, df_counts, total_docs
# feature_builder_node.py
# 完整的 aggregate_category_features 函数（包含过滤）

def aggregate_category_features(
    corpus: List[str],
) -> Tuple[
    Dict[int, Dict[Union[str, Tuple[str, ...]], float]], Dict[int, Counter], int
]:
    """
    聚合整个类别的特征向量（节点内TF-IDF）
    过滤权重 < 0.000001 的特征
    """
    total_docs = len(corpus)
    docs = list(nlp.pipe(corpus, batch_size=_BATCH_SIZE))
    doc_features = [_extract_doc_features(doc) for doc in docs]

    df_counts = _accumulate_dfs(
        {
            1: Counter(),
            2: Counter(),
            3: Counter(),
        },
        doc_features,
    )

    category_features = {
        1: defaultdict(float),
        2: defaultdict(float),
        3: defaultdict(float),
    }

    # 计算 TF-IDF
    for _, (all_nouns, all_pairs, all_triples) in doc_features:
        noun_counts = Counter(all_nouns)
        total_nouns = len(all_nouns) if all_nouns else 1
        for term, freq in noun_counts.items():
            tf = freq / total_nouns
            df = df_counts[1].get(term, 0)
            idf = math.log((total_docs + 1) / (df + 1))
            category_features[1][term] += tf * idf

        pair_counts = Counter(all_pairs)
        total_pairs = len(all_pairs) if all_pairs else 1
        for term, freq in pair_counts.items():
            tf = freq / total_pairs
            df = df_counts[2].get(term, 0)
            idf = math.log((total_docs + 1) / (df + 1))
            category_features[2][term] += tf * idf

        triple_counts = Counter(all_triples)
        total_triples = len(all_triples) if all_triples else 1
        for term, freq in triple_counts.items():
            tf = freq / total_triples
            df = df_counts[3].get(term, 0)
            idf = math.log((total_docs + 1) / (df + 1))
            category_features[3][term] += tf * idf

    # 归一化
    for level in category_features:
        total_score = sum(category_features[level].values())
        if total_score > 0:
            for term in category_features[level]:
                category_features[level][term] /= total_score

    # ===== ★ 过滤低权重特征（MIN_WEIGHT = 0.000001）=====
    MIN_WEIGHT = 0.000001
    
    for level in [1, 2, 3]:
        before = len(category_features[level])
        category_features[level] = {
            term: weight for term, weight in category_features[level].items()
            if weight >= MIN_WEIGHT
        }
        after = len(category_features[level])
        
        # 打印过滤信息
        if before > 0:
            print(f"    Level {level}: {before:,} → {after:,} (过滤 {before-after:,}, {100*(1-after/before):.1f}%)")

    return category_features, df_counts, total_docs

def save_category_features(features_data: Tuple, output_path: str):
    """保存类别特征到文件"""
    category_features, df_counts, total_docs = features_data
    data_to_save = {
        "category_features": category_features,
        "df_counts": df_counts,
        "total_docs": total_docs,
    }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(data_to_save, f)
    print(f"    已保存: {os.path.basename(output_path)}")


def load_category_features(file_path: str) -> Dict:
    """加载类别特征文件"""
    with open(file_path, "rb") as f:
        return pickle.load(f)
