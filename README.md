# Genos：人类基因组基础模型

![model](images/Genos_LOGO.png)

## 1. 目录

- [Genos：人类基因组基础模型](#genos人类基因组基础模型)
  - [1. 目录](#1-目录)
  - [2. 简要说明](#2简要说明)
  - [3. 模型说明](#3模型说明)
    - [数据](#数据)
    - [架构](#架构)
  - [4. 部署及使用](#4部署及使用)
  - [5. 性能测评](#5性能测评)
  - [6. 应用场景案例说明](#6应用场景案例说明)
    - [案例1：RNA-seq数据生成](#案例1rna-seq数据生成)
    - [案例2： 基因模型+文本模型疾病预测](#案例2基因模型文本模型疾病预测)
      - [项目概述](#项目概述)
      - [结果数据对比](#结果数据对比)
  - [7. 数据可用性](#7数据可用性)
  - [8. Licence 说明](#8licence说明)
  - [9. 联系我们](#9联系我们)
  - [References](#references)

## 2. 简要说明

Genos：人类基因组基础模型

【模型架构与技术突破】

Genos作为人类基因组领域的基础模型，依托数百个高质量人类基因组基准数据进行训练，实现了对人类基因组序列长达百万碱基的上下文建模能力。通过单碱基级的分辨率学习，该模型具备了识别基因组中隐含的深层序列规律与功能特征的能力，为科学家构建起连接遗传信息与生命活动的新研究方法。本次发布包含12亿参数与100亿参数两个版本，均采用混合专家（MoE）架构，通过动态路由机制实现计算资源的优化配置，显著提升模型在复杂调控网络解析中的表现。

【功能模块与科学价值】 Genos具备精准识别关键功能元件的核心能力，能够深入解析微小基因变异对转录调控网络的级联效应，突破现在对非编码区调控元件的预测精度的传统方法局限，动态模拟变异位点对RNA表达谱的潜在影响，并追踪至表型形成的分子路径。在此基础上，研究团队开发了模块化应用接口，构建起"预测-解释-验证"的全链条研究体系。通过引入可解释性增强机制，该模型不仅提供高置信度的预测结果，更揭示调控网络中的关键节点与作用通路，为分子机制解析提供新的研究范式。

【开放生态与临床转化】 秉承开放科学理念，Genos在Github和Huggingface提供开源模型，并同时在DCS Cloud平台部署云端推理服务。研究者可下载模型进行部署及推理，或选择在DCS Cloud云端进行部署，我们还为使用者提供了从变异功能注释到表型预测的全流程分析示例代码，帮助使用者更快熟悉模型使用方法及功能。模型权重将进行持续更新，其在精准医学、群体健康、监测及发育生物学等领域的应用潜力将进一步释放。

【科学哲学与未来展望】

Genos为科学家研究基因的复杂调控及对功能的影响提供了新的可能性。未来随着跨模态学习能力的提升，Genos有望成为连接遗传密码与生命现象的"翻译器"，在疾病预警、药物靶点发现及合成生物学等领域开启全新研究维度，目标实现从"基因组学"到"功能组学"的范式跨越。

## 3. 模型说明

### 数据

本研究整合了人类泛基因组参考联盟（HPRC）、人类基因组结构变异图谱计划（HGSVC）等国际顶级基因组学队列的标准化公开数据，构建起覆盖全球欧亚非美多样性族群的数百例全基因组近端粒到端粒（nearly telomere-to-telomere）组装的高质量基因组数据集。通过实施严格的质量控制，确保数据集在单核苷酸分辨率（single nucleotide resolution）上达到高质量高精度，为跨族群泛化能力奠定坚实基础。

![/Users/ALLEN_1/Documents/work/模型结构图(2)/居中.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/NpQlK5jmkj0ADqDv/img/b63bb8a7-3894-44be-8e9f-9e22f07f35d3.png)

### 架构

Genos基于Transformer架构，采用混合专家网络（Mixture-of-Experts, MoE），主要技术点包括：

1.  **超长序列单核苷酸分辨率建模** 通过引入超长序列参数化策略、多维张量并行计算与多尺度注意力机制，成功攻克百万级碱基序列的建模挑战。
    
2.  **训练稳定性优化体系** 针对基因组数据特有的低熵特征分布，采用专家负载均衡机制。通过梯度裁剪与专家选择策略的协同优化，避免小词汇表规模（4碱基）导致的专家模块负载失衡问题。
    
3.  **动态专家激活架构** 此次发布的两个模型：12亿参数版本与100亿参数版本，均支持百万级超长序列推理。动态路由算法（Dynamic Routing Algorithm）可根据输入序列的特征，实时激活相关专家模块。
    
| **Model Specification** | **Genos 1.2B** | **Genos 10B** |
| --- | --- | --- |
| **Version** | Genos 1.2B | Genos 10B |
| ++**Model Scale**++ |  |  |
| Total Parameters | 1.2B | 10B |
| Activated Parameters | 0.33B | 2.87B |
| Trained Tokens | 1600 B | 2200 B |
| ++**Architecture**++ |  |  |
| Architecture | MoE | MoE |
| Number of Experts | 8 | 8 |
| Selected Experts per Token | 2 | 2 |
| Number of Layers | 12 | 12 |
| Attention Hidden Dimension | 1024 | 4096 |
| Number of Attention Heads | 16 | 16 |
| MoE Hidden Dimension (per Expert) | 4096 | 8192 |
| Vocabulary Size | 128 (padded) | 256 (padded) |
| Context Length | up to 1M | up to 1M |


## 4. 部署及使用

### docker部署
我们强烈建议使用docker部署我们的模型，我们训练模型使用的镜像已经在dockerhub上面可以获取。

1. 拉取环境
   ```bash
   docker pull zhanxiaoai/mega:v1 
   ```
2. 启动容器
   ```bash
   docker run -it --gpus all --shm-size 32g mega:v1 /bin/bash
   ```
3. 下载权重
4. 使用
   可以参考Notebooks里面的案例调用

   - [embedding获取](Notebooks/zh/01.embedding_zh.ipynb)
   - [种群预测](Notebooks/zh/02.Population_classify_Demo.ipynb)

### API接口调用
1. 安装Genos的SDK的包
   ```bash
   pip install genos-client
   ```
2. 接口使用详细参考[SDK使用介绍](sdk/README.md)

模型权重：  
| Model Name        | Parameters | Huggingface ckpt | Megatron ckpt |
|-------------------|------------|----------------|---------------|
| `Genos-1.2B`  | 1.2B       |  [Genos-1.2B](https://huggingface.co/BGI-HangzhouAI/Genos-1.2B) |  [Genos-Megatron-1.2B](https://huggingface.co/BGI-HangzhouAI/Genos-Megatron-1.2B) |
| `Genos-10B`       | 10B        |  [Genos-10B](https://huggingface.co/BGI-HangzhouAI/Genos-10B)   |  [Genos-Megatron-10B](https://huggingface.co/BGI-HangzhouAI/Genos-Megatron-10B)   |


## 5. 性能测评

Genos 基因基座模型评测体系

本评测体系旨在系统化评估 Genos 模型在基因组序列分析、转录效应预测以及生物医学下游应用中的2综合能力。我们采用多个标准基准数据集对Genos进行评估,包括基因组学基准（GB）、核苷酸转换器基准（NTB）和基因组学长程基准（LRB）数据集\[3-5\]，为确保可比较性，数据集划分遵循官方配置，若无官方标准则采用基于染色体的分区方案。我们对Genos 和主要同类模型的三大能力进行横向评估：

*   **长序列评测:** 测评模型对长程调控、以及更复杂基因互作的识别和理解。针对长程建模能力评估，采用四项LRB任务\[5\]，涵盖增强子与启动子检测（regulatory element enhancer 8K、regulatory element promoter 8K），以及变异对表达影响（variant effect causal eqtl 8K）和疾病致病性预测（variant effect pathogenic clinvar 8K）。我们为LRB任务构建了8,192 bp（8K）序列的测试数据集。其中LRB任务将22号染色体留作验证集。
    
*   **短序列评测:** 测评模型对基因元件的识别和理解，我们从GB中选取了三类代表性分类任务：编码与非编码序列区分（demo coding vs intergenomic seqs）、增强子检测（human enhancers cohn）及开放染色质区域识别（human ocr ensembl）\[3\]。NTB任务包含剪接位点识别（splice sites all）和组蛋白修饰分类（H3、H3K36me3）\[4\]。
    
*   **人群分类评测:** 测试模型是否能有效利用更长序列中包含的更多的基因信息进行更为准确的推断。我们基于人类泛基因组联盟（**H**uman **P**angenome **R**eference **C**onsortium, _BioProject ID: PRJNA730823_）数据，在非洲、东亚、欧洲三组人群上设计了人群分类任务。我们根据样本的VCF文件与参考基因组序列生成样本伪序列，基于VCF文件记录的变异位点信息截取样本9号染色体变异密集区域，采用8,192 bp（8K）、32,768 bp（32K）和131,072 bp（128K）三种长度序列，通过xgboost分类器对单条序列做分类预测。
    
*   **变异热点预测:** 测评模型能否仅凭序列特征捕捉局部变异的易发性，进一步考察模型是否具备刻画人群分化与演化历史相关信号的能力。我们基于中国泛基因组联盟数据（**C**hinese **P**angenome **C**onsortium ）\[6\]设计了突变热点分类任务。采用8,192 bp（8K）、32,768 bp（32K）和131,072 bp（128K）三种长度序列。通过泊松右尾检验识别突变热点，将各序列突变计数与同一染色体所有片段的背景均值比较，以错误发现率FDR < 0.05判定显著性。最终数据集由全部热点序列与等量随机选取的非热点序列合并构成。
    

![model](images/评测结果.20251013.png)


*   HyenaDNA, Nucleotide Transformer (NT), 以及GENERator系列的其他版本公开模型也进行了测试，因篇幅有限未能罗列已测评的模型还包括GENERator-1.2b，HyenaDNA-32k，HyenaDNA-450k，NT-500M，以及 Evo2-1b，这里仅展示了各系列中表现最好的模型。
    
*   NT公开模型中由于最大可接受输入长度为6000，在8K及以上长度输入的任务中均不可用
    
*   Evo2 1b公开模型中只有8K输入长度的base版本，无法用在更长输入的任务中
    
*   Evo2 7b，40b模型在HuggingFace框架下由于资源占用问题无法对128K或更长的序列进行推理
    

## 6. 应用场景案例说明

### 案例1：RNA-seq数据生成

1.  **任务描述**
    
    本任务基于Genos大模型对于基因组的大量预训练积累，通过微调实现从DNA序列直接预测单碱基分辨率的RNA-seq表达谱，覆盖多种细胞类型和组织。其科学意义在于构建了基因组序列与转录组表达之间的直接映射关系，为理解基因调控机制和加速转录组学研究提供了创新工具
    
2.  **任务输入与输出**
    
    本任务属于**回归任务**。任务输入为hg38参考基因组部分序列（目前版本以32 kb为窗口），输出为不同细胞类型、基因组正负链对应序列位置的平均归一化RNA-seq信号值（单碱基精度）。核心是通过学习序列到表达的复杂映射，预测连续的转录组表达水平。
    
3.  **数据来源**
    
    训练数据来源于公共数据库ENCODE (ENCODE Consortium, 2012) 和GTEx (Kim-Hellmuth et al., 2020)。
    
    数据经过整合后，共包含667个元数据组的单碱基转录组bigwig文件，同时输入hg38参考基因组。模型训练使用染色体1-22上的所有位置序列及其对应的平均RNA-seq谱作为配对数据。 目前公开版本使用了4例人B淋巴细胞和13例NK自然杀伤细胞进行微调，为去除样本间个体差异，两种细胞类型数据表达量取样本间平均表达量后均一化为各一组数据。
    
    \*目前公开版本已可实现其中人B淋巴细胞和NK自然杀伤细胞两种细胞类型1-22号染色体预测推理
    
4.  **模型设计**
    

*   下游模型架构设计​
    

模型在预训练的Genos-1.2B基础之上，替换原始输出头为任务专用的卷积模块。该模块由三层一维卷积层构成，通道维度逐层递减（1024→256→64→1），每层后接批归一化、GELU激活和丢弃正则化（dropout=0.1）。最终输出通过可学习权重参数缩放并经过Softplus激活，确保预测值为非负连续值，契合RNA-seq信号的回归特性。这种设计增强了模型对局部序列模式的捕捉能力，同时通过卷积的平移不变性优化了计算效率。

*   微调策略与训练优化​
    

采用全参数微调（full fine-tuning），以均方误差（MSE）作为回归任务的损失函数。为解决RNA-seq信号值分布偏斜的问题，训练时引入平方根平滑裁剪和幂变换进行数值压缩，推理时执行逆操作以还原信号尺度。优化器选用Adafactor，配合余弦退火学习率调度器和线性预热（预热步数占总数5%），全局批量大小设为256，训练60个周期。这一策略在保证稳定收敛的同时，显著降低了长序列训练中的梯度波动风险。

*   基因组序列处理与上下文建模​
    

为平衡长程依赖学习与计算成本，输入序列窗口长度设置为32 kb，相邻窗口重叠16 kb，覆盖染色体1-22的所有位点

（1）.  **指标及指标定义**
    

评估核心指标为log1p变换后的皮尔逊相关系数（log1p-transformed Pearson correlation coefficient）​。该指标用于衡量模型预测的RNA-seq谱与实验测得的真实谱在不同基因组范围内的一致性，具体计算范围包括：

*   全基因组（Whole genome）​：全基因组范围内，单碱基精度
    
*   基因表达量（Gene expression）​：基因表达量矩阵相关性，基因精度
    

\*皮尔逊相关系数经过log1p（即log(1 + r)）变换，以更好地评估信号预测的全局相关性。

（2）.  **测评指标**
Genos模拟生成多细胞类型RNA-seq表达量与真实测序结果结果相关性0.9+。
![model](images/RNA_exp.benchmark_res.20251020.png)

    
（3）.  **输出示例**
    

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/NpQlK5jmkj0ADqDv/img/fff8c7df-66fe-4022-98f5-6c10531bb460.png)

### 案例2： 基因模型+文本模型疾病预测

#### 项目概述

**简介：**

该项目为了验证多模态模型（基因模型+文本模型）在基因变异导致的疾病预测任务上，能够处理原始DNA序列，同时利用大语言模型的推理能力，生成具有生物学一致性的解释与预测。

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/NpQlK5jmkj0ADqDv/img/35006876-15a3-42be-9f1c-34167d2819cd.png)

**数据**：

数据来源于论文Bioreason\[7\]当中的kegg任务，该任务数据通过多阶段整合KEGG通路与临床数据库变异信息，采用标准化符号系统表征分子网络中的各类相互作用，提供参考序列与变异序列的比。KEGG数据集包含1449条共计37种疾病信息，其中数据分配训练：验证：测试= 8:1:1。数据输入包括问题描述，参考基因序列与变异基因序列。输出包括推理和疾病分类预测。

**模型设计：**

*   架构设计：DNA模型、文本模型（Qwen3-1b、Qwen3-4b）、DNA embeding到文本embeding的投影层。
    
*   模型训练：本次模型训练旨在实现DNA序列与自然语言的高效对齐。训练当中冻结了基因模型（支持Evo2，Genos模型，），训练DNA embeding 到文本embeding的投影层，Lora微调文本模型，使用deepspeed策略进行高效训练。
    
*   基因序列处理：总共有两条基因序列：1、参考基因序列。2、变异基因序列。以变异基因为中心上下游，共计1024bp的基因序列。
    

**评价指标：**

*   **准确率 (Accuracy)**: 正确预测的样本占总样本的比例。
    

#### 结果数据对比

不同模型在KEGG数据集上的结果对比情况如下表，其中基因模型中 Genos-10B 性能领先，文本-基因融合模型性能远超单独模态的模型，其中 021-8B与 Genos-1.2B融合模型准确率高达 98.28%，比单独用 Genos-1.2B高出 7%。

![model](images/text_gLM.benchmark_res.20251020.png)

模型说明：

NT-2.5b-multi：InstaDeepAI/nucleotide-transformer-v2-500m-multi-species

Evo2-1b：arcinstitute/evo2\_1b\_base

HyenaDna-1m: LongSafari/hyenadna-large-1m-seqlen

Genos-1.2B: BGI-HangzhouAI/Genos-1.2B

Genos-10B: BGI-HangzhouAI/Genos-10B

021-8B: 021 Science Foundation Model-8B is a large language model trained on extensive scientific corpora with profound scientific cognition. It is scheduled to be released at a later date. 

## 7. 数据可用性

我们收集的公开测评数据均已标注出处，其中，我们谨此感谢人类泛基因组参考联盟HRPC（生物项目编号：PRJNA730823）及其资助机构——美国国家人类基因组研究所（NHGRI）。

模型训练和评测均基于021科学基础模型和[zero2x.org](https://zero2x.org.cn)平台开展。

我们测评中所用的数据集正在整理中，即将上线，敬请期待。

## 8. Licence 说明

本工作倡导生物AI社区的开放共享，支持MIT License。

## 9. 联系我们

如果您疑问或合作意向，欢迎联系我们。 邮箱: Genos@genomics.cn

## References

\[1\] The Human Genome Project. (2003). Finishing the euchromatic sequence of the human genome. Nature, 431(7011), 931 - 945.

\[2\] 1000 Genomes Project Consortium. (2010). A map of human genome variation from population - scale sequencing. Nature, 467(7319), 1061 - 1073.

\[3\] Grešová, K., Martinek, V., Čechák, D., Šimeček, P. & Alexiou, P. Genomic benchmarks: a collection of datasets for genomic sequence classification. BMC Genomic Data 24, (2023).

\[4\] Dalla-Torre, H. et al. Nucleotide Transformer: building and evaluating robust foundation models for human genomics. Nature Methods (2024) doi:10.1038/s41592-024-02523-z.

\[5\] Trop, E. et al. The Genomics Long-Range Benchmark: Advancing DNA Language Models. OpenReview [https://openreview.net/forum?id=8O9HLDrmtq.](https://openreview.net/forum?id=8O9HLDrmtq.)

\[6\] Gao, Y. et al. A pangenome reference of 36 Chinese populations. Nature 619, 112–121 (2023).

\[7\] Fallahpour, Adibvafa, et al. BioReason: Incentivizing Multimodal Biological Reasoning within a DNA-LLM Model. arXiv preprint arXiv:2505.23579 (2025).

\[8\] Liao, W.W. et al. A draft human pangenome reference. Nature 617, 312–324 (2023).
