这个 `single_layer` 目录里有 **4组实验，均为898条Query**。它们是**单层分类**：直接在指定DL层级的全部节点中排序，不逐层进入，也不使用Lookahead。

**现有结果**

| 实验 | 候选节点数 | Top-1 | Top-3 | Top-5 | 平均分类耗时 |
|---|---:|---:|---:|---:|---:|
| DL1纯Sinkhorn，单元组 | 10 | **40.65%（365）** | 62.03%（557） | 75.17%（675） | 0.552秒 |
| DL3 Centroid，三种语义特征 | 898 | 13.36%（120） | 汇总未记录 | 27.28%（245） | **0.000424秒** |
| DL3纯Sinkhorn，单元组 | 898 | 19.38%（174） | 29.62%（266） | 36.75%（330） | 5.084秒 |
| DL3纯WMD，单元组 | 898 | **19.93%（179）** | **30.51%（274）** | **36.86%（331）** | 5.523秒 |

其中两组纯OT分类均使用：

- **仅名词／专有名词单元组**；
- Cosine ground cost，最多1000 terms；
- 直接按OT距离排序，**不乘Centroid分数**；
- 没有Centroid初筛、没有层级门控；
- Sinkhorn为普通 `sinkhorn`，`reg=0.05`。

**DL3的Sinkhorn与WMD可以直接比较**

WMD相较Sinkhorn：

- Top-1多正确 **5条**；
- Top-3多命中 **8条**；
- Top-5多命中 **1条**；
- 每条分类慢约 **0.44秒**。

因此，这组单层实验中，**WMD准确率略高，Sinkhorn略快**，差距不大。

**Centroid对照需要注意特征不同**

Centroid使用名词、名词对、三元组，权重为 **0.12／0.21／0.67**；两组OT只使用单元组。因此，虽然纯OT的DL3结果更高，**不能把差异全部归因于距离算法**。

这也不能直接证明“纯Sinkhorn比层级融合更好”：单层DL3对所有898条Query直接分类，避免了前层初筛和门控造成的丢失；层级流程还包含不同的特征组合、融合与Lookahead。

代码对应关系：

| 文件 | 用途 |
|---|---|
| [flat_dl3_centroid_top5.py](/Users/sinkhorn/Bonnie/large_model/single_layer/flat_dl3_centroid_top5.py) | 直接分类全部DL3节点，使用Centroid |
| [flat_dl3_unigram_ot.py](/Users/sinkhorn/Bonnie/large_model/single_layer/flat_dl3_unigram_ot.py) | 仅单元组，直接进行DL3 Sinkhorn或WMD分类 |
| [flat_level_unigram_ot.py](/Users/sinkhorn/Bonnie/large_model/single_layer/flat_level_unigram_ot.py) | 可选择DL1／DL2／DL3，使用单元组OT分类 |

耗时均为分类阶段，不包含模型加载和文本特征提取。来源：[single_layer结果目录](/Users/sinkhorn/Bonnie/large_model/single_layer/)。
