好，就用 BGC。 我已经下载并检查了官方压缩包，确实包含简介、分层标签和独立的层级文件。
官方下载 ZIP
文件	内容	实际样本数
BlurbGenreCollection_EN_train.txt	训练集	58,715
BlurbGenreCollection_EN_dev.txt	验证集	14,785
BlurbGenreCollection_EN_test.txt	测试集	18,394
hierarchy.txt	类别父子关系	—


数据记录使用 XML 风格标签，与你的 pipeline 可以这样对应：
BGC 字段	用途
<body>	图书简介，作为输入文本
<d0>	DL1 标签
<d1>	DL2 标签
<d2>	DL3 标签
<d3>	第四层，第一轮截断到第三级
<isbn>	样本标识及重复检查


层级文件中的真实路径包括：
Nonfiction → Business → Marketing
Nonfiction → Popular Science → Technology
第一轮建议先做“唯一三级路径”实验，这样最容易对齐你现在的单路径分类代码。我按层级文件匹配样本标签，初步统计如下：
划分	含至少一条完整三级路径	恰好一条三级路径
训练集	28,729	18,925
验证集	7,270	4,771
测试集	9,006	5,945


这些是初步路径统计，尚未做空文本、重复简介和跨集合重复检查。“唯一三级路径”也不代表完全没有其他较浅层标签。
接入时，用训练集简介构建各级 reference，验证集调权重、Top-K 和正则参数，测试集评估；保留你的 centroid 初筛＋Sinkhorn／WMD 精排，类别树改为读取 hierarchy.txt。 这里不能再依赖 DDC 编码前缀推断父子关系。
