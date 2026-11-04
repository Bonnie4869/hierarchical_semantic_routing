**有。按你要的“有正文＋已提供类别层级＋能用于现有 pipeline”，我更推荐下面两个正式发布的研究数据集。** 如果重点是保留三层结构，优先看 **BlurbGenreCollection**。

| 数据集 | 发布来源 | 输入文本 | 已提供的层级 | 对你的适用性 |
|---|---|---|---|---|
| **BlurbGenreCollection-EN（BGC）** | 汉堡大学研究团队 | 英文图书简介 | 最深4层，附父子关系文件 | 文本与现有图书实验最接近，可筛选三级路径 |
| **Web of Science（WOS-46985）** | HDLTex 论文作者团队 | 英文论文摘要 | 7个大领域→134个细分领域 | 两层结构清楚，适合验证方法在其他文本领域的效果 |

上述文本和层级均由发布方提供，不需要你自己从零整理类别树。[inf.uni-hamburg.de](https://www.inf.uni-hamburg.de/en/inst/ab/lt/resources/data/blurb-genre-collection.html?utm_source=chatgpt.com)

**1. BGC：与你当前实验最接近**

它包含 **91,892 条英文图书简介**，并提供训练、验证、测试划分，以及一个独立的**类别父子关系文件**。层级最深4层，但不同图书的标签深度不一定相同。[inf.uni-hamburg.de](https://www.inf.uni-hamburg.de/en/inst/ab/lt/resources/data/blurb-genre-collection.html?utm_source=chatgpt.com)

官方展示的《2001: A Space Odyssey》样本包含：

| 内容 | 样本信息 |
|---|---|
| 文本 | 介绍太空探索、宇宙飞船及人工智能 HAL 9000 的图书简介 |
| 第一级标签 | `Fiction` |
| 第二级标签 | `Science Fiction`、`Mystery & Suspense` |
| 第三级标签 | `Suspense & Thriller` |

**同一本书可以有多个类别。** 发布者根据出版社的细分类别整理了层级，并补齐祖先标签，所以这里的“官方”指数据集研究团队正式发布，而不是出版社原样提供完整分类树。[inf.uni-hamburg.de](https://www.inf.uni-hamburg.de/en/inst/ab/lt/resources/data/blurb-genre-collection.html?utm_source=chatgpt.com)

迁移到你的三层 pipeline，可以有两种做法：

- **先保持单路径分类：**筛选存在完整三级路径、且只有一条三级目标路径的样本；超过三级的标签截到第三级，并报告筛选规则。
- **保留多标签：**允许一条 query 对应多个正确类别路径，调整正确性判断与 Recall@K。

[官方介绍和下载入口](https://www.inf.uni-hamburg.de/en/inst/ab/lt/resources/data/blurb-genre-collection.html)

**2. WOS：更适合先把迁移实验跑通**

它提供 **46,985 篇论文摘要**，标签是两层。官方列出的路径例子是：

**Computer Science → Computer Graphics**

数据文件直接包括：

| 文件 | 内容 |
|---|---|
| `X.txt` | 输入文本 |
| `YL1.txt` | 一级类别标签 |
| `YL2.txt` | 二级类别标签 |

**好，就用 BGC。** 我已经下载并检查了官方压缩包，确实包含简介、分层标签和独立的层级文件。

[官方下载 ZIP](https://fiona.uni-hamburg.de/ca89b3cf/blurbgenrecollectionen.zip)

| 文件 | 内容 | 实际样本数 |
|---|---|---:|
| `BlurbGenreCollection_EN_train.txt` | 训练集 | 58,715 |
| `BlurbGenreCollection_EN_dev.txt` | 验证集 | 14,785 |
| `BlurbGenreCollection_EN_test.txt` | 测试集 | 18,394 |
| `hierarchy.txt` | 类别父子关系 | — |

数据记录使用 XML 风格标签，与你的 pipeline 可以这样对应：

| BGC 字段 | 用途 |
|---|---|
| `<body>` | 图书简介，作为输入文本 |
| `<d0>` | DL1 标签 |
| `<d1>` | DL2 标签 |
| `<d2>` | DL3 标签 |
| `<d3>` | 第四层，第一轮截断到第三级 |
| `<isbn>` | 样本标识及重复检查 |

层级文件中的真实路径包括：

**Nonfiction → Business → Marketing**  
**Nonfiction → Popular Science → Technology**

**第一轮建议先做“唯一三级路径”实验**，这样最容易对齐你现在的单路径分类代码。我按层级文件匹配样本标签，初步统计如下：

| 划分 | 含至少一条完整三级路径 | 恰好一条三级路径 |
|---|---:|---:|
| 训练集 | 28,729 | **18,925** |
| 验证集 | 7,270 | **4,771** |
| 测试集 | 9,006 | **5,945** |

这些是初步路径统计，尚未做空文本、重复简介和跨集合重复检查。“唯一三级路径”也不代表完全没有其他较浅层标签。

接入时，用训练集简介构建各级 reference，验证集调权重、Top-K 和正则参数，测试集评估；**保留你的 centroid 初筛＋Sinkhorn／WMD 精排，类别树改为读取 `hierarchy.txt`。** 这里不能再依赖 DDC 编码前缀推断父子关系。

元数据还包含领域、细分领域、关键词和摘要。因此不需要另外获取正文或推断类别归属。**注意：它只有两层，不能原样作为三层实验。**[Mendeley Data](https://data.mendeley.com/datasets/9rw3vkcfy4/6?utm_source=chatgpt.com)

[作者发布的数据与下载入口](https://data.mendeley.com/datasets/9rw3vkcfy4/6)

**针对你的需求，我会优先选 BGC：它保留了图书简介这种输入形式，提供现成层级，也能考察一本书跨多个类别的情况。** 不过它的标签是图书体裁／主题类别，与 DDC 不同；正式比较时应分别报告结果。
