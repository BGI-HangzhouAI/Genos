# Genos: Genomic Foundation Model

<p align="center">
  <a href="README.md">English</a> | 中文
</p>

## 1. 模型名字

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/AJdl65AJbjamvOke/img/a5308220-330d-4e76-84e5-d773856897e6.png)

Genos-1.2B / Genos-10B

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/NpQlK5jmkj0ADqDv/img/7e85331a-d2f9-493d-936c-fa922448c47e.png)

## 2. 简要说明

Genos：人类基因组基础模型

【模型架构与技术突破】

Genos作为人类基因组领域的基础模型，依托数百个的高质量基因组基准数据进行训练，实现了对人类基因组序列长达百万碱基的上下文建模能力。通过单碱基级的分辨率学习，该模型具备了识别基因组中隐含的深层序列规律与功能特征的能力，为科学家构建起连接遗传信息与生命活动的新研究方法。本次发布包含12亿参数与100亿参数两个版本，均采用混合专家（MoE）架构，通过动态路由机制实现计算资源的优化配置，显著提升模型在复杂调控网络解析中的表现。

【功能模块与科学价值】 作为基因组的"语言学家"，GenOS具备精准识别关键功能元件的核心能力，能够深入解析微小基因变异对转录调控网络的级联效应。其创新性体现在对非编码区调控元件的预测精度突破传统方法局限，可动态模拟变异位点对RNA表达谱的潜在影响，并追踪至表型形成的分子路径。在此基础上，研究团队开发了模块化应用接口，构建起"预测-解释-验证"的全链条研究体系。通过引入可解释性增强机制，该模型不仅提供高置信度的预测结果，更揭示调控网络中的关键节点与作用通路，为分子机制解析提供新的研究范式。

【开放生态与临床转化】 秉承开放科学理念，Genos在Github和Huggingface提供开源模型，并同时在DCS Cloud平台部署云端推理服务。研究者可下载模型进行部署及推理，或选择在DCS Cloud云端进行部署，我们还为使用者提供了从变异功能注释到表型预测的全流程分析示例代码，帮助使用者更快熟悉模型使用方法及功能。模型权重将进行持续更新，其在精准医学、群体健康、监测及发育生物学等领域的应用潜力将进一步释放。

【科学哲学与未来展望】

GenOS为科学家研究基因的复杂调控及对功能的影响提供了新的可能性。通过融合多组学数据与表型信息，该模型正在重塑我们理解基因-环境互作的认知框架。未来随着跨模态学习能力的提升，GenOS有望成为连接遗传密码与生命现象的"翻译器"，在疾病预警、药物靶点发现及合成生物学等领域开启全新研究维度，目标实现从"基因组学"到"功能组学"的范式跨越。

## 3. 性能对比图（待定稿后绘图）

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/eYVOL5jwEmY8Glpz/img/6b844694-626c-41a5-bc02-4c4ba0e64970.png)

## 4. 模型说明

**数据基座：构建全球基因组多样性图谱** 本研究整合了人类泛基因组参考联盟（HPRC）、人类基因组结构变异图谱计划（HGSVC）等国际顶级基因组学队列的标准化公开数据，构建起覆盖全球欧亚非美多样性族群的数百例全基因组近端粒到端粒（nearly telomere-to-telomere）组装的高质量基因组数据集。通过实施严格的质量控制，确保数据集在单核苷酸分辨率（single nucleotide resolution）上达到高质量高精度，为跨族群泛化能力奠定坚实基础。

**架构创新：混合专家网络的基因组解码革命**

Genos基于Transformer架构，采用分层混合专家网络（Hierarchical Mixture-of-Experts, HMoE），主要技术点包括：

1.  **超长序列单核苷酸分辨率建模** 通过引入超长序列参数化策略、多维张量并行计算与多尺度注意力机制，成功攻克百万级碱基序列的建模挑战。创新性的分形注意力（fractal attention）模块有效缓解了超长上下文（10^6 bp）中的局部-全局依赖衰减问题，实现了单碱基变异（SNV）与结构变异（SV）的协同解析。
    
2.  **训练稳定性优化体系** 针对基因组数据特有的低熵特征分布，采用专家负载均衡机制。通过梯度裁剪与专家选择策略的协同优化，避免小词汇表规模（4碱基）导致的专家模块负载失衡问题。
    
3.  **动态专家激活架构** 此次发布的两个模型：12亿参数版本与100亿参数版本，均支持百万级超长序列推理。动态路由算法（Dynamic Routing Algorithm）可根据输入序列的特征，实时激活相关专家模块。
    

| **Version(Parameters)** | **1.2B** | **10B** |
| --- | --- | --- |
| **Active Patameters** |  |  |
| **Number of Layers** | 12 |  |
| **MoE Hidden Dimension (per Expert)** | 1024 | 4096 |
| **Number of Attention Heads** | 4096 | 8192 |
| **Tokenization** | Single-base tokenization |  |
| **Vocabulary Size** | 128(padded) | 256(padded) |
| **Trained Tokens** | 1500 B | 2000 B |

## 5. 部署及使用

*   ~~**当前状态**~~~~: 研发与优化阶段，支持内部科研使用。~~
    
*   **硬件要求**: 支持主流GPU环境，无特殊硬件限制
    
*   **使用方式**: 提供模型权重用于下游任务微调，支持序列嵌入生成、多组学数据生成、变异效应预测等任务。
    

## 6. 性能测评

GenOS 基因基座模型评测体系

本评测体系旨在系统化评估 GenOS 模型在基因组序列分析、转录效应预测以及生物医学下游应用中的2综合能力。我们的评测不仅关注模型在标准基准数据集上的得分，更注重其解决真实世界生物医学问题的潜力。分为三个类型：**短序列评测**、**长序列评测、变异热点预测。**三个任务分别测评模型对基因元件的识别和理解、对长程调控的理解、以及检验模型能否仅凭序列特征捕捉局部变异的易发性，进一步考察模型是否具备刻画人群分化与演化历史相关信号的能力。

|  | 说明 | **task** | **1.2b-8k** | **10b-1M** | **GENERator-3b** | **HyenaDNA-1M** | **NT-2.5b-multi** | **Evo2-7b** | **Evo2-40b** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 短序列测评<br>（序列长度200-600bp) | 评价模型对基因元件的识别和理解能力，如预测序列是否是外显子、启动子、增强子、甲基化等基因元件 | demo\_coding\_vs\_intergenomic\_seqs | 0.9717 | **0.9907** | 0.9855 | 0.9127 | 0.9763 | 0.9824 | **0.9886** |
|  |  | human\_enhancers\_cohn | **0.8723** | **0.8806** | 0.8181 | 0.7799 | 0.7873 | 0.7733 | 0.7756 |
|  |  | human\_ocr\_ensembl | **0.7730** | **0.7785** | 0.7270 | 0.6916 | 0.7285 | 0.7505 | 0.7635 |
|  |  | splice\_sites\_all | 0.7704 | 0.8064 | 0.8071 | 0.7110 | 0.8603 | **0.8747** | **0.9138** |
|  |  | H3 | **0.9413** | **0.9394** | 0.9163 | 0.8722 | 0.9371 | 0.9140 | 0.9311 |
|  |  | H3K36me3 | 0.7949 | 0.8300 | 0.8247 | 0.6787 | 0.8288 | **0.8615** | **0.8823** |
| 变异热点测评<br>（序列长度：8k～128k） | 评价模型对基因组整体规律的理解能力，如识别人类基因组序列上的变异热点区域 | CPC\_131072 | 0.9600 | **0.9886** | 0.9620 | **0.9735** |  |  |  |
|  |  | CPC\_32768 | 0.9331 | **0.9720** | 0.9237 | 0.9064 |  | 0.9504 | **0.9611** |
|  |  | CPC\_8192 | **0.9437** | **0.9547** | 0.9315 | 0.8914 |  | 0.9425 | 0.9401 |
| 长序列测评<br>（序列长度：8k） | 评价模型对基因互作、调控作用的识别和理解能力；如识别长序列上的启动子、增强子等长程调控作用 | regulatory\_element\_enhancer\_8K | **0.7532** | **0.7536** | 0.7390 | 0.7282 |  | 0.7454 | 0.7527 |
|  |  | regulatory\_element\_promoter\_8K | 0.9252 | **0.9291** | 0.9195 | 0.8890 |  | **0.9255** | 0.9227 |
|  |  | variant\_effect\_causal\_eqtl\_8K | **0.7078** | 0.6973 | 0.6920 | 0.6887 |  | 0.7039 | **0.7054** |
|  |  | variant\_effect\_pathogenic\_clinvar\_8K | 0.6613 | **0.9298** | 0.7206 | 0.6117 |  | 0.7308 | **0.9167** |

变异热点预测的数据处理说明。

测评任务的引用出处

## 7. 应用场景案例说明

### 案例1：RNA-seq数据生成

*   [ ] ![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/NpQlK5jmkj0ADqDv/img/6c011ea2-11cb-4304-83d1-4099d9f48909.png)
    

## 训练流程

### 1.1 数据准备预处理

1.  本案例训练数据目前来源于Encode和Gtex（与AlphaGenome相同），获取~~667组~~样本单碱基转录组数据，将数据按不同细胞类型、正负链定义为不同batch并给予标签。每个batch内多个样本表达量进行平均，得到细胞类型均一化的单碱基表达量数据，保持为bigwig格式。
    
2.  基于模型支持长度，参考基因组与.bw的RNA-seq数据，按32kb长度窗口进行截取，两个窗口间重叠16kb，保证训练数据的重复性。
    
3.  最终将成对的32KB基因组和转录组数据，分别作为模型输入与输出提供给基座模型，并将预测vs真实表达量的皮尔森相关性系数作为主要loss进行微调训练。
    

### 1.2 网络架构

```apl
+---------------------+
|   Input: input_ids  |
|     [Batch, Length] |
+---------------------+
           ↓
+-----------------------------+
|      Base Model (gLM)       |
|    → last_hidden_state      |
|     [B, L, D]               |
+-----------------------------+
           ↓ (transpose)
+-----------------------------+
|   Transpose: [B, L, D] →    |
|       [B, D, L]             |
+-----------------------------+
           ↓
+--------------------------------------------------+
|                   CNN Head                       |
|  Sequential:                                     |
|  1. Conv1d(D → D/4, k=3, p=1)                    |
|     → BatchNorm → GELU → Dropout                 |
|                                                  |
|  2. Conv1d(D/4 → D/16, k=3, p=2, dilation=2)     |
|     → BatchNorm → GELU → Dropout                 |
|                                                  |
|  3. Conv1d(D/16 → 1, k=1) → squeeze               |
|                                                  |
|  Output: logits [B, L]                           |
+--------------------------------------------------+
           ↓
+-----------------------------------------+
|    Softplus Scaling:                    |
|    logits = F.softplus(logits) *        |
|             F.softplus(scale_param)     |
|    → Ensures output > 0                 |
+-----------------------------------------+
           ↓
     ╭─────────────╮
     │ Training?   │
     ╰─────┬───────╯
           │ Yes
           ↓
+-----------------------------------------+
|   Loss Computation                      |
|   scaled_labels = target_scaling(       |
            labels, track_mean)           |
|                                         |
|   Supported:                            |
|   - MSE (selected)                      |
|   - Poisson                             |
|   - Tweedie                             |
|   - Poisson-Multinomial                 |
|                                         |
|   loss = L(logits, labels)              |
+-----------------------------------------+
           │
     ┌─────┴──────┐
     │ No / Eval  │
     ╰────────────╯
           ↓
+--------------------------------------+
|   Inference                          |
|   logits = predictions_scaling(      |   
|              logits, track_mean)     |
+--------------------------------------+
           ↓
+-----------------------------+
|         Output Dict         |
| { "loss": loss,             |
|   "logits": logits }        |
+-----------------------------+
```

## 评价指标

### 2.1 轨迹级/单碱基精度指标

[请至钉钉文档查看「电子表格」](https://alidocs.dingtalk.com/i/nodes/2Amq4vjg89gGdom7FMBxj2j1V3kdP0wQ?iframeQuery=anchorId%3DX02mg3hpixkf97ys221l6&rnd=0.5594786719483986)

### 2.2 基因级指标

| pearson | log1p pearson | spearman |
| --- | --- | --- |
| 0.8867 | 0.9553 | 0.6422 |

### 2.3 差异基因表达：预测轨迹 vs 真实轨迹

显著差异的基因数（|log2(FoldChange)|>2）为0。（zscore标准化处理后）

### ![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/1wvqreb4KwLGenak/img/8188e6d0-2420-4c61-9cdf-b0f5722b22ee.png)![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/1wvqreb4KwLGenak/img/cd20eb99-3bee-4db7-8189-469a8da8d5b5.png)

### 2.4 轨迹可视化

*   **预测准确一致性**
    

19号染色体全局：蓝色为真实，红色为预测

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/1wvqreb4KwLGenak/img/8df97a24-138c-419b-a283-e0edf879e9c5.png)

ZFP36：

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/1wvqreb4KwLGenak/img/3ae87bb5-5d4e-405b-af01-2999b018e80d.png)

JUNB：

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/1wvqreb4KwLGenak/img/8cbc4d93-f453-459b-821d-1a292d19ada6.png)

chr19: 10587331-11635907 （AlphaGenome在HepG2 Cell Line中选取的区域）

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/1wvqreb4KwLGenak/img/0acce3fd-9dc3-4087-bfd5-110673d740ab.png)

### 案例2： 组学+文本交互式疾病诊断

## 项目概述

本项目是一个基于DNA模型+文本模型的DNA序列分析系统，主要用于基因变异效应预测和疾病关联分析。项目结合了DNA序列编码器和大型语言模型，实现了对DNA序列的深度理解和分析。

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/NpQlK5jmkj0ADqDv/img/e23b7011-3e63-4d2d-8c6d-e879b288d30b.png)

##  输入输出分析

### 1.1 输入数据格式

#### 1.1.1 KEGG数据集

*   **文件格式**: JSON格式
    
*   **数据结构**:
    

```markdown
  ```json
  {
    "question": "染色体信息和通路网络定义的问题描述",
    "answer": "疾病名称（如：cushing syndrome, parkinson's disease, amyotrophic lateral sclerosis）",
    "reasoning": "详细的推理步骤，包含10个步骤的生物学分析",
    "reference_sequence": "参考DNA序列（大写字母，去除空格）",
    "variant_sequence": "变异DNA序列（大写字母，去除空格）"
  }
  ```
```

#### 1.1.2 VEP（Variant Effect Prediction）数据集

*   **文件格式**: JSON格式
    
*   **数据结构**:
    

```markdown
  ```json
  {
    "question": "变异效应预测问题",
    "answer": "变异效应分类结果",
    "reference_sequence": "参考序列",
    "variant_sequence": "变异序列"
  }
  ```
```

### 1.2 输出数据格式

#### 1.2.1 模型输出

*   **文本生成**: 模型生成包含推理过程和最终答案的文本
    
*   **格式**:
    

```markdown
  ```
  <|im_start|>assistant
  [推理内容]
  Answer: [最终答案]<|im_end|>
  ```
```

#### 1.2.2 评估结果

*   **CSV格式**: 包含预测结果和真实标签的对比
    
*   **列结构**:
    
    *   `ground_truth`: 真实标签
        
    *   `pred_label`: 预测标签
        
    *   `generated_text`: 完整生成文本
        

## 数据预处理流程

### 2.1 DNA序列预处理

#### 2.1.1 序列标准化

*   **大小写转换**: 所有DNA序列转换为大写字母
    
*   **空格去除**: 移除序列中的空白字符
    
*   **序列截断**: 使用`truncate_dna`函数从两端截断序列
    

```markdown
 ```
  def truncate_dna(example, truncate_dna_per_side=1024):
      # 从序列两端各截断1024个碱基对
      # 如果序列太短，返回中间部分
  ```
```

#### 2.1.2 序列标记化

*   **DNA标记器**: 使用字符级标记器处理DNA序列
    
*   **特殊标记**: 添加`<|dna_start|>`, `<|dna_pad|>`, `<|dna_end|>`标记
    
*   **序列长度限制**: 最大长度2048个token
    

### 2.2 文本预处理

#### 2.2.1 对话格式转换

*   **多模态格式**: 结合DNA序列和文本的对话格式
    
*   **角色定义**:
    
    *   `user`: 包含DNA序列和问题
        
    *   `assistant`: 包含推理过程和答案
        

#### 2.2.2 模板应用

*   **聊天模板**: 使用自定义的聊天模板格式化输入
    
*   **特殊标记处理**: 正确处理`<|im_start|>`和`<|im_end|>`标记
    

### 2.3 数据加载和批处理

#### 2.3.1 数据集分割

*   **训练集**: 80%
    
*   **验证集**: 10%
    
*   **测试集**: 10%
    

#### 2.3.2 批处理函数

*   **qwen\_dna\_collate\_fn**: 专门为Qwen DNA模型设计的批处理函数
    
*   **标签掩码**: 只对助手回复部分计算损失
    
*   **填充处理**: 左填充策略，添加特殊标记
    

## 测评指标方案

### 3.1 分类指标

#### 3.1.1 基础指标

*   **准确率 (Accuracy)**: 正确预测的样本占总样本的比例
    
*   **精确率 (Precision)**: 宏平均精确率，所有类别精确率的平均值
    
*   **召回率 (Recall)**: 宏平均召回率，所有类别召回率的平均值
    
*   **F1分数 (F1-Score)**: 宏平均F1分数，精确率和召回率的调和平均
    

#### 3.1.2 计算方式

```python
# 使用sklearn的classification_report
report_dict = classification_report(
    y_true, y_pred, 
    labels=labels, 
    output_dict=True, 
    zero_division=1
)

# 提取宏平均指标
macro_metrics = report_dict['macro avg']
Accuracy = accuracy_score(ground_truth, pred_label)
Precision = macro_metrics['precision']
Recall = macro_metrics['recall'] 
F1_score = macro_metrics['f1-score']

```

### 3.2 评估流程

#### 3.2.1 验证阶段评估

*   **实时生成**: 在验证集上生成文本
    
*   **答案提取**: 从生成文本中提取答案部分
    
*   **指标计算**: 计算分类指标并记录到日志
    

#### 3.2.2 测试阶段评估

*   **完整评估**: 在测试集上进行完整评估
    
*   **结果保存**: 将预测结果保存为CSV文件
    
*   **指标汇总**: 生成详细的分类报告
    

### 3.3 特殊处理

#### 3.3.1 答案提取

*   **正则匹配**: 使用`extract_single_entry`函数提取答案
    
*   **格式处理**: 处理包含`<think>`前缀的预测结果
    
*   **错误处理**: 对无法提取的答案返回NaN
    

#### 3.3.2 多类别支持

*   **动态标签**: 根据真实标签动态确定类别
    
*   **零除处理**: 使用`zero_division=1`处理无预测的情况
    
*   **宏平均**: 使用宏平均确保所有类别平等对待
    

##  模型架构

### 4.1 核心组件

*   **DNA编码器**: 基于Genos获取用于DNA序列编码
    
*   **文本模型**: Qwen用于文本理解和生成
    
*   **投影层**: 将DNA特征投影到文本嵌入空间
    
*   **LoRA适配**: 使用LoRA进行参数高效微调
    

### 4.2 训练策略

*   **冻结策略**: 可选择冻结DNA编码器或文本模型
    
*   **混合精度**: 使用DeepSpeed策略进行高效训练
    
*   **梯度累积**: 支持大批次训练
    

## 结果数据对比

不同模型在KEGG数据集上的结果对比情况如下

模型说明：

NT：InstaDeepAI/nucleotide-transformer-v2-500m-multi-species

our\_gene：Mixtral\_onehot\_mix\_1b\_16n\_8k293B\_eod\_111\_pai\_0805

| model\_type | Model | Source | Accuracy | F1-score | Precision | Recall |
| --- | --- | --- | --- | --- | --- | --- |
| dna | NT | Paper | 86.55% | 69.76% | 73.23% | 66.62% |
| dna | NT | Code | 89.31% | 68.99% | 74.20% | 68.14% |
| dna | our\_gene | Code | 91.72% | 78.75% | 81.09% | 81.64% |
| dna | evo2 | Code | 76.90% | 44.94% | 47.58% | 45.99% |
| dna | evo2 | Paper | 88.28% | 72.43% | 75.23% | 69.83% |
| dna-llm | NT+Qwen1B | Paper | 88.42% | 72.13% | 75.42% | 71.91% |
| dna-llm | NT+Qwen4B | Paper | 96.90% | 89.03% | 90.99% | 89.38% |
| dna-llm | NT+Qwen1B | Code | 93.45% | 90.26% | 97.88% | 88.47% |
| dna-llm | NT+Qwen4B | Code | 97.24% | 95.35% | 100.00% | 94.07% |
| dna-llm | our\_gene+Qwen1B | Code | 91.72% | 86.39% | 99.33% | 82.49% |
| dna-llm | our\_gene+Qwen4B | Code | 99.31% | 92.93% | 98.48% | 93.94% |
| dna-llm | evo2+Qwen1B | Code | 85.97% | 74.72% | 94.29% | 70.51% |
| dna-llm | evo2+Qwen4B | Code | 92.63% | 86.29% | 97.17% | 84.29% |
| dna-llm | evo2+Qwen3-1B | Paper | 90.42% | 75.62% | 77.42% | 73.91% |
| dna-llm | evo2+Qwen3-4B | Paper | 97.24% | 86.30% | 86.75% | 87.25% |
| llm | Qwen1B | Paper | 85.17% | 65.71% | 71.39% | 64.19% |
| llm | Qwen4B | Paper | 93.48% | 85.44% | 88.31% | 86.72% |
| llm | Qwen1B | Code | 88.62% | 82.56% | 93.03% | 80.77% |
| llm | Qwen4B | Code | 97.93% | 93.06% | 98.99% | 93.41% |

**8. 数据可用性**

## 8. Licence 说明

本工作倡导生物AI社区的开放共享，支持MIT License。详见`MIT © 2025 <Your Name or Company>`

## 9. 联系我们

如果您疑问或合作意向，欢迎联系我们。 邮箱: bgi-Genos@genomics.cn
