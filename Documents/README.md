# Genos Genomic Foundation Model

<p align="center">
  English | <a href="README.zh.md">中文</a>
</p>

## 1. Model Name

Genos-1.2B-32K / Genos-10B

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/AJdl65AJbjamvOke/img/a5308220-330d-4e76-84e5-d773856897e6.png)

![image](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/meonarb8PVbQ0qXx/img/ff899db6-5f29-4e54-b3a9-8cf1d0e4ef0f.png)

## 2.Brief Description

Human Genome Intelligence "System": Genos Very Large Scale Pre-Training Human Genome Base Model

**\[Model architecture and technology breakthrough\]** as the leading basic model in the field of human genome, Genos relies on massive high-quality genomic benchmark data for in-depth training,Its innovation lies in the breakthrough realization of the human genome sequence up to million base pairs of Context modeling ability. Through single-base resolution learning, the model successfully analyzes the deep sequence rules and functional characteristics hidden in the genome, and constructs an intelligent bridge connecting genetic information and life activities. The current version includes two configurations of 1.2 billion parameters and 10 billion parameters,The advanced hybrid expert (MoE) architecture is adopted to realize the optimal allocation of computing resources through dynamic routing mechanism, which significantly improves the performance of the model in complex regulatory network analysis.

**\[Function module and scientific value\]** as the "linguist" of genome, Genos has the core ability to accurately identify key functional elements,It can deeply analyze the cascade effect of micro gene variation on transcriptional regulatory network. Its innovation is reflected in the prediction accuracy of regulatory elements in the non-coding region, which breaks through the limitations of traditional methods, can dynamically simulate the potential impact of mutation sites on RNA expression profiles, and trace the molecular path of phenotype formation. On this basis, the research team developed a modular application interface,Build a full-chain research system of "prediction-interpretation-verification. By introducing an interpretable enhancement mechanism, the model not only provides high confidence prediction results, but also reveals the key nodes and pathways in the regulatory network, providing a new research paradigm for molecular mechanism analysis.

**\[Open Ecology and Clinical Transformation\]** Adhering to the concept of open science,Genos has deployed cloud reasoning services on the Huada DCSCloud platform to build a "cloud lab" for genomic intelligence analysis ". Researchers can upload data through an intuitive interface to obtain a full-process analysis from mutation function annotation to phenotype prediction, completely free from the constraints of local computing power and algorithm deployment. This decentralized computing power support model,It enables researchers worldwide to share the predictive power of leading models and accelerate the transition from genomic discovery to clinical applications. As the model continues to optimize and iterate, its application potential in precision medicine, group health monitoring and developmental biology will continue to be released, laying a key foundation for building an active health management system covering the whole life cycle.

**\[Philosophy of Science and Future Prospects\]** The birth of Genos marks a major innovation in the research paradigm of genomics. It is not only a breakthrough tool for computational biology, but also an intelligent carrier for decoding the mysteries of life. By fusing multi-omics data with phenotypic information, the model is reshaping our cognitive framework for understanding gene-environment interactions.In the future, with the improvement of cross-modal learning ability, Genos is expected to become a "translator" connecting genetic code and life phenomena, opening up a new research dimension in the fields of disease early warning, drug target discovery and synthetic biology, and finally realizing the paradigm leap from "genomics" to "functional omics. This landmark technological breakthrough,It is pushing mankind steadily towards a new era of precision medicine.

## 3. Performance comparison chart

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/eYVOL5jwEmY8Glpz/img/6b844694-626c-41a5-bc02-4c4ba0e64970.png)

## 4. Model Description

Genos: Paradigm Innovation of Genomic Intelligence Analysis Engine\]

**Data Base: construct global genome diversity map** this study integrates the standardized data of top international genomics cohorts such as human pan-Genome Reference Consortium (HPRC), Human genome structural variation mapping project (HGSVC), etc,A multi-source Heterogeneous genome data set covering five continents of Europe, Asia, Africa, the United States and Australia and containing hundreds of cases of genome-wide telomere to Telomere (telomere-to-telomere) assembly was constructed. By implementing a strict quality control process-including base quality segmentation filtering, structural variation standardized mapping and population genetic background correction-to ensure that the data set achieves clinical-level accuracy at single nucleotide resolution (single nucleotide resolution),Lay a solid foundation for cross-ethnic generalization.

**Architecture Innovation: Genome Decoding Revolution of Hybrid Expert Network** Based on the deep evolution of Transformer architecture, this system innovatively constructs Hierarchical Hybrid expert network (HMoE),Its core breakthrough is reflected in:

1.  **Single nucleotide resolution modeling of ultra-long sequences** Successfully overcome the modeling challenge of million-base sequences by introducing ultra-long sequence parameterization strategies, multi-dimensional tensor parallel computing and multi-scale attention mechanisms. The innovative fractal attention module effectively alleviates the local-global dependency decay problem in very long contexts (10 ^ 6 bp),The cooperative analysis of single base variation (SNV) and structural variation (SV) was realized.
    
2.  **Training Stability Optimization System** An expert load balancing mechanism has been developed for the distribution of low entropy characteristics unique to genomic data. Through the collaborative optimization of gradient clipping and expert selection strategy, the problem of expert module load imbalance caused by small vocabulary size (4 bases) is successfully solved.
    
3.  **Dynamic Expert Activation Architecture** released two versions of the intelligent engine: The Elite version (1.2 billion parameters) and the flagship version (10 billion parameters), both of which support Million-level ultra-long sequence inference. Its original Dynamic Routing Algorithm (Dynamic Routing Algorithm) can be based on the epigenetic characteristics of the input sequence, population genetic background and other meta-information,Activate the relevant expert module in real time.
    

The birth of Genos marks the qualitative change of genomics research paradigm from "data accumulation" to "intelligent analysis", and provides a revolutionary intelligent decision support platform for the design of individualized diagnosis and treatment plan, disease risk prediction and precise intervention strategy formulation. This breakthrough,It is gradually pushing human interpretation of the life code to a new dimension of single-base accuracy.

| **Parameters** | **1.2B** | **10B** |
| --- | --- | --- |
| **Number of Layers** <br>**(Dense layer included)** | 1.25B | 10.27B |
| **MoE Hidden Dimension (per Expert)** | 1024 | 4096 |
| **Number of Attention Heads** | 4096 | 8192 |
| **Vocabulary Size** | 128(padded) | 256(padded) |
| **Trained Tokens** | 1500 B | 2000 B |

## 5. Deployment and Use

*   **Current status**: R & D and optimization phase, supporting internal scientific research.
    
*   **Hardware requirements**: Support mainstream GPU environment, no special hardware restrictions
    

**Usage**: provides model weights for downstream task tuning,Support sequence embedding generation, multi-group data generation, variation effect prediction and other tasks.

## 6. Performance evaluation

Genos gene pedestal model evaluation system

The purpose of this evaluation system is to systematically evaluate the comprehensive ability of Genos Model in genome sequence analysis, transcription effect prediction and biomedical downstream applications. Our evaluation focuses not only on the model's score on the standard benchmark dataset,Pay more attention to its potential to solve real-world biomedical problems. It is divided into three types: **short sequence evaluation**, \*\*long sequence evaluation, and mutation hot spot prediction. \*\*The three tasks were to assess the model's identification and understanding of gene elements, its understanding of long-range regulation, and its ability to capture the susceptibility of local variation based on sequence features alone,Further examine whether the model has the ability to depict the signal related to population differentiation and evolutionary history.

|  | **task** | **1.2b-8k** | **10b-1M** | **GENERator-3b** | **HyenaDNA-1M** | **NT-2.5b-multi** | **Evo2-7b** | **Evo2-40b** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Short sequence evaluation   <br>(sequence length 200-600bp) | demo\_coding\_vs\_intergenomic\_seqs | 0.9717 | **0.9907** | 0.9855 | 0.9127 | 0.9763 | 0.9824 | **0.9886** |
|  | human\_enhancers\_cohn | **0.8723** | **0.8806** | 0.8181 | 0.7799 | 0.7873 | 0.7733 | 0.7756 |
|  | human\_ocr\_ensembl | **0.7730** | **0.7785** | 0.7270 | 0.6916 | 0.7285 | 0.7505 | 0.7635 |
|  | splice\_sites\_all | 0.7704 | 0.8064 | 0.8071 | 0.7110 | 0.8603 | **0.8747** | **0.9138** |
|  | H3 | **0.9413** | **0.9394** | 0.9163 | 0.8722 | 0.9371 | 0.9140 | 0.9311 |
|  | H3K36me3 | 0.7949 | 0.8300 | 0.8247 | 0.6787 | 0.8288 | **0.8615** | **0.8823** |
| Mutation hot spot evaluation   <br>(sequence length: 8K ~ 128K) | CPC\_131072 | 0.9600 | **0.9886** | 0.9620 | **0.9735** |  |  |  |
|  | CPC\_32768 | 0.9331 | **0.9720** | 0.9237 | 0.9064 |  | 0.9504 | **0.9611** |
|  | CPC\_8192 | **0.9437** | **0.9547** | 0.9315 | 0.8914 |  | 0.9425 | 0.9401 |
| Long sequence evaluation   <br>(sequence length: 8K) | regulatory\_element\_enhancer\_8K | **0.7532** | **0.7536** | 0.7390 | 0.7282 |  | 0.7454 | 0.7527 |
|  | regulatory\_element\_promoter\_8K | 0.9252 | **0.9291** | 0.9195 | 0.8890 |  | **0.9255** | 0.9227 |
|  | variant\_effect\_causal\_eqtl\_8K | **0.7078** | 0.6973 | 0.6920 | 0.6887 |  | 0.7039 | **0.7054** |
|  | variant\_effect\_pathogenic\_clinvar\_8K | 0.6613 | **0.9298** | 0.7206 | 0.6117 |  | 0.7308 | **0.9167** |

## 7. Case Description

### Case 1:RNA-seq Data Generation

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/meonarb8PVbQ0qXx/img/e12686fb-bc3f-4849-ac4c-6aa59f1fe859.png)

## Training process

### 1.1 Preprocessing of Data Preparation

1.  The training data in this case is currently derived from Encode and Gtex (the same as AlphaGenome). Obtain single-base transcriptome data of 667 groups of samples, define the data as different batches according to different cell types and positive and negative strands, and give labels.The expression of multiple samples in each batch is averaged to obtain single-base expression data of cell type normalization, which is maintained in bigwig format.
    
2.  Based on the model support length, refer to the RNA-seq data of genome and. bw, intercept according to the 32kb length window,The two windows overlap 16kb to ensure the repeatability of the training data.
    
3.  Finally, the pairs of 32KB genome and transcriptome data are provided to the base model as model input and output, and the Pearson correlation coefficient of the prediction vs true expression is used as the main loss for fine-tuning training.
    

### 1.2 Network Architecture

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

## Training parameters

*   Complete the training of 1~22 chromosomes and four trajectories modeling separately.
    
*   1.2 b-128k model full amount fine adjustment
    
*   8 nodes 64 cards (H100) training 40 rounds
    

```python
num_train_epochs=1,
per_device_train_batch_size=1,
per_device_eval_batch_size=1,
gradient_accumulation_steps=32,

dataloader_num_workers = 4,

learning_rate=5e-5,
lr_scheduler_type="cosine",
warmup_ratio=0.05,
weight_decay=0.01,
max_grad_norm=1.0,
optim="adafactor",

eval_strategy="epoch",
save_strategy="epoch",
eval_accumulation_steps=10,
save_total_limit=5,
save_safetensors=True,

fp16=False,
bf16=True,
half_precision_backend="auto",

logging_steps=1,
report_to="wandb",

ddp_find_unused_parameters=True,
seed=42
```
```markdown
09/23/2025 01:29:48 - INFO - root - ✅ [Distributed] 初始化成功！ rank=0, world_size=8, local_rank=0
09/23/2025 01:29:48 - INFO - root - ✅ [Distributed] 初始化成功！ rank=1, world_size=8, local_rank=1
09/23/2025 01:29:48 - INFO - root - ✅ [Distributed] 初始化成功！ rank=2, world_size=8, local_rank=2
09/23/2025 01:29:48 - INFO - root - ✅ Logging system initialized. Log file: output3__dlc/log/training_20250923_012945.log
09/23/2025 01:29:48 - INFO - root - ✅ [Distributed] 初始化成功！ rank=3, world_size=8, local_rank=3
09/23/2025 01:29:51 - INFO - root - 🌐 wandb: Logged in as: beneldor-zhejiang-lab
09/23/2025 01:29:51 - INFO - root - 📊 Project: RNA-Seq_Coverage_Track | Run Name: bs256_chr19_track27_epoch20_one-hot-mix-1.2b-128k_flash_attn-pai_mse_0923_dlc
09/23/2025 01:29:51 - INFO - root - 🚀 Run URL: None
09/23/2025 01:29:51 - INFO - root - 💾 Local Dir: /your/path/RNASeqCoverageTrackPrediction_single-track/wandb/offline-run-20250923_012951-w0sac55n/files
09/23/2025 01:29:51 - INFO - root - 🚀 加载预训练模型和分词器...
09/23/2025 01:29:51 - INFO - root - ⚡ 使用 Flash Attention
09/23/2025 01:29:51 - INFO - root - 🧬 训练染色体: ['chr19']
09/23/2025 01:29:51 - INFO - root - 🧬 验证染色体: ['chr19']
09/23/2025 01:29:51 - INFO - root - 🏷️ 获取数据索引...
09/23/2025 01:29:51 - INFO - root - 🏷️ 索引文件已存在，直接加载: data/processed/all_chroms_train_index.csv
09/23/2025 01:29:51 - INFO - root - 🏷️ 索引文件已存在，直接加载: data/processed/all_chroms_val_index.csv
09/23/2025 01:29:51 - INFO - root - 📈 训练信号轨道（1个）: ['batch27']
09/23/2025 01:29:51 - INFO - root - 📈 验证信号轨道（1个）: ['batch27']
09/23/2025 01:29:51 - INFO - root - 🧩 创建训练数据集...
09/23/2025 01:29:51 - INFO - root - 🧩 创建验证数据集...
09/23/2025 01:29:51 - INFO - root - ✅ 训练: 3,663 样本, 验证: 37 样本
09/23/2025 01:29:51 - INFO - root - 🧠 构建下游预测头...
09/23/2025 01:29:53 - INFO - root - gcc -pthread -B /opt/miniconda3/compiler_compat -fno-strict-overflow -DNDEBUG -O2 -Wall -fPIC -O2 -isystem /opt/miniconda3/include -fPIC -O2 -isystem /opt/miniconda3/include -fPIC -c /tmp/tmphymsgmt6/test.c -o /tmp/tmphymsgmt6/test.o
09/23/2025 01:29:53 - INFO - root - gcc -pthread -B /opt/miniconda3/compiler_compat /tmp/tmphymsgmt6/test.o -laio -o /tmp/tmphymsgmt6/a.out
09/23/2025 01:29:53 - INFO - root - gcc -pthread -B /opt/miniconda3/compiler_compat -fno-strict-overflow -DNDEBUG -O2 -Wall -fPIC -O2 -isystem /opt/miniconda3/include -fPIC -O2 -isystem /opt/miniconda3/include -fPIC -c /tmp/tmpknvpfro1/test.c -o /tmp/tmpknvpfro1/test.o
09/23/2025 01:29:53 - INFO - root - gcc -pthread -B /opt/miniconda3/compiler_compat /tmp/tmpknvpfro1/test.o -laio -o /tmp/tmpknvpfro1/a.out
09/23/2025 01:29:53 - INFO - root - 📊 模型参数量: 1,246,799,874 (可训练: 1,246,799,874 → 100.0%)
09/23/2025 01:29:53 - INFO - root - ⚙️ 配置训练参数...
09/23/2025 01:29:53 - INFO - root - gcc -pthread -B /opt/miniconda3/compiler_compat -fno-strict-overflow -DNDEBUG -O2 -Wall -fPIC -O2 -isystem /opt/miniconda3/include -fPIC -O2 -isystem /opt/miniconda3/include -fPIC -c /tmp/tmpsn0vqu54/test.c -o /tmp/tmpsn0vqu54/test.o
09/23/2025 01:29:53 - INFO - root - gcc -pthread -B /opt/miniconda3/compiler_compat -fno-strict-overflow -DNDEBUG -O2 -Wall -fPIC -O2 -isystem /opt/miniconda3/include -fPIC -O2 -isystem /opt/miniconda3/include -fPIC -c /tmp/tmpdsl483us/test.c -o /tmp/tmpdsl483us/test.o
09/23/2025 01:29:53 - INFO - root - gcc -pthread -B /opt/miniconda3/compiler_compat /tmp/tmpdsl483us/test.o -laio -o /tmp/tmpdsl483us/a.out
09/23/2025 01:29:53 - INFO - root - gcc -pthread -B /opt/miniconda3/compiler_compat /tmp/tmpsn0vqu54/test.o -laio -o /tmp/tmpsn0vqu54/a.out
09/23/2025 01:29:53 - INFO - root - gcc -pthread -B /opt/miniconda3/compiler_compat -fno-strict-overflow -DNDEBUG -O2 -Wall -fPIC -O2 -isystem /opt/miniconda3/include -fPIC -O2 -isystem /opt/miniconda3/include -fPIC -c /tmp/tmpafd605pa/test.c -o /tmp/tmpafd605pa/test.o
09/23/2025 01:29:53 - INFO - root - gcc -pthread -B /opt/miniconda3/compiler_compat -fno-strict-overflow -DNDEBUG -O2 -Wall -fPIC -O2 -isystem /opt/miniconda3/include -fPIC -O2 -isystem /opt/miniconda3/include -fPIC -c /tmp/tmpb36s7oer/test.c -o /tmp/tmpb36s7oer/test.o
09/23/2025 01:29:53 - INFO - root - gcc -pthread -B /opt/miniconda3/compiler_compat -fno-strict-overflow -DNDEBUG -O2 -Wall -fPIC -O2 -isystem /opt/miniconda3/include -fPIC -O2 -isystem /opt/miniconda3/include -fPIC -c /tmp/tmpzpoqhvoy/test.c -o /tmp/tmpzpoqhvoy/test.o
09/23/2025 01:29:53 - INFO - root - gcc -pthread -B /opt/miniconda3/compiler_compat /tmp/tmpafd605pa/test.o -laio -o /tmp/tmpafd605pa/a.out
09/23/2025 01:29:53 - INFO - root - gcc -pthread -B /opt/miniconda3/compiler_compat /tmp/tmpzpoqhvoy/test.o -laio -o /tmp/tmpzpoqhvoy/a.out
09/23/2025 01:29:53 - INFO - root - gcc -pthread -B /opt/miniconda3/compiler_compat /tmp/tmpb36s7oer/test.o -laio -o /tmp/tmpb36s7oer/a.out
09/23/2025 01:29:53 - INFO - root - gcc -pthread -B /opt/miniconda3/compiler_compat -fno-strict-overflow -DNDEBUG -O2 -Wall -fPIC -O2 -isystem /opt/miniconda3/include -fPIC -O2 -isystem /opt/miniconda3/include -fPIC -c /tmp/tmpd9vmcr2a/test.c -o /tmp/tmpd9vmcr2a/test.o
09/23/2025 01:29:53 - INFO - root - gcc -pthread -B /opt/miniconda3/compiler_compat /tmp/tmpd9vmcr2a/test.o -L/usr/local/cuda -L/usr/local/cuda/lib64 -lcufile -o /tmp/tmpd9vmcr2a/a.out
09/23/2025 01:29:53 - INFO - root - gcc -pthread -B /opt/miniconda3/compiler_compat -fno-strict-overflow -DNDEBUG -O2 -Wall -fPIC -O2 -isystem /opt/miniconda3/include -fPIC -O2 -isystem /opt/miniconda3/include -fPIC -c /tmp/tmppx24iks_/test.c -o /tmp/tmppx24iks_/test.o
09/23/2025 01:29:53 - INFO - root - gcc -pthread -B /opt/miniconda3/compiler_compat /tmp/tmppx24iks_/test.o -L/usr/local/cuda -L/usr/local/cuda/lib64 -lcufile -o /tmp/tmppx24iks_/a.out
09/23/2025 01:29:53 - INFO - root - gcc -pthread -B /opt/miniconda3/compiler_compat -fno-strict-overflow -DNDEBUG -O2 -Wall -fPIC -O2 -isystem /opt/miniconda3/include -fPIC -O2 -isystem /opt/miniconda3/include -fPIC -c /tmp/tmp5s7u_767/test.c -o /tmp/tmp5s7u_767/test.o
09/23/2025 01:29:53 - INFO - root - gcc -pthread -B /opt/miniconda3/compiler_compat /tmp/tmp5s7u_767/test.o -laio -o /tmp/tmp5s7u_767/a.out
09/23/2025 01:29:53 - INFO - root - gcc -pthread -B /opt/miniconda3/compiler_compat -fno-strict-overflow -DNDEBUG -O2 -Wall -fPIC -O2 -isystem /opt/miniconda3/include -fPIC -O2 -isystem /opt/miniconda3/include -fPIC -c /tmp/tmpimkf0kco/test.c -o /tmp/tmpimkf0kco/test.o
09/23/2025 01:29:53 - INFO - root - gcc -pthread -B /opt/miniconda3/compiler_compat -fno-strict-overflow -DNDEBUG -O2 -Wall -fPIC -O2 -isystem /opt/miniconda3/include -fPIC -O2 -isystem /opt/miniconda3/include -fPIC -c /tmp/tmpwgjranuw/test.c -o /tmp/tmpwgjranuw/test.o
09/23/2025 01:29:53 - INFO - root - gcc -pthread -B /opt/miniconda3/compiler_compat /tmp/tmpimkf0kco/test.o -L/usr/local/cuda -L/usr/local/cuda/lib64 -lcufile -o /tmp/tmpimkf0kco/a.out
09/23/2025 01:29:53 - INFO - root - gcc -pthread -B /opt/miniconda3/compiler_compat /tmp/tmpwgjranuw/test.o -L/usr/local/cuda -L/usr/local/cuda/lib64 -lcufile -o /tmp/tmpwgjranuw/a.out
09/23/2025 01:29:54 - INFO - root - gcc -pthread -B /opt/miniconda3/compiler_compat -fno-strict-overflow -DNDEBUG -O2 -Wall -fPIC -O2 -isystem /opt/miniconda3/include -fPIC -O2 -isystem /opt/miniconda3/include -fPIC -c /tmp/tmp3tg9pdxh/test.c -o /tmp/tmp3tg9pdxh/test.o
09/23/2025 01:29:54 - INFO - root - gcc -pthread -B /opt/miniconda3/compiler_compat /tmp/tmp3tg9pdxh/test.o -L/usr/local/cuda -L/usr/local/cuda/lib64 -lcufile -o /tmp/tmp3tg9pdxh/a.out
09/23/2025 01:29:54 - INFO - root - gcc -pthread -B /opt/miniconda3/compiler_compat -fno-strict-overflow -DNDEBUG -O2 -Wall -fPIC -O2 -isystem /opt/miniconda3/include -fPIC -O2 -isystem /opt/miniconda3/include -fPIC -c /tmp/tmpwab9_il_/test.c -o /tmp/tmpwab9_il_/test.o
09/23/2025 01:29:54 - INFO - root - gcc -pthread -B /opt/miniconda3/compiler_compat -fno-strict-overflow -DNDEBUG -O2 -Wall -fPIC -O2 -isystem /opt/miniconda3/include -fPIC -O2 -isystem /opt/miniconda3/include -fPIC -c /tmp/tmpnoe9e7u5/test.c -o /tmp/tmpnoe9e7u5/test.o
09/23/2025 01:29:54 - INFO - root - gcc -pthread -B /opt/miniconda3/compiler_compat /tmp/tmpwab9_il_/test.o -L/usr/local/cuda -L/usr/local/cuda/lib64 -lcufile -o /tmp/tmpwab9_il_/a.out
09/23/2025 01:29:54 - INFO - root - gcc -pthread -B /opt/miniconda3/compiler_compat /tmp/tmpnoe9e7u5/test.o -L/usr/local/cuda -L/usr/local/cuda/lib64 -lcufile -o /tmp/tmpnoe9e7u5/a.out
09/23/2025 01:29:54 - INFO - root - gcc -pthread -B /opt/miniconda3/compiler_compat -fno-strict-overflow -DNDEBUG -O2 -Wall -fPIC -O2 -isystem /opt/miniconda3/include -fPIC -O2 -isystem /opt/miniconda3/include -fPIC -c /tmp/tmp6juysz7q/test.c -o /tmp/tmp6juysz7q/test.o
09/23/2025 01:29:54 - INFO - root - gcc -pthread -B /opt/miniconda3/compiler_compat /tmp/tmp6juysz7q/test.o -L/usr/local/cuda -L/usr/local/cuda/lib64 -lcufile -o /tmp/tmp6juysz7q/a.out
09/23/2025 01:29:55 - INFO - root - 🏋️‍♂️ 启动训练...
09/23/2025 03:03:45 - INFO - root - ✅ 训练完成！
09/23/2025 03:03:45 - INFO - root - 🧹 资源已释放: train_dataset（LazyGenomicDataset）
09/23/2025 03:03:45 - INFO - root - 🧹 资源已释放: val_dataset（LazyGenomicDataset）
09/23/2025 03:03:45 - INFO - root - 🧹 wandb run 已结束
09/23/2025 03:03:45 - INFO - root - 🎉 主流程执行完毕！

```

## Evaluation Index

### 2.1 trajectory level/single base accuracy index

[请至钉钉文档查看「电子表格」](https://alidocs.dingtalk.com/i/nodes/QG53mjyd80RbdBYktbZpMBm4V6zbX04v?iframeQuery=anchorId%3DX02mg3hpixkf97ys221l6&utm_medium=dingdoc_doc_plugin_card&utm_scene=person_space&utm_source=dingdoc_doc)

### 2.2 Gene Level Indicators

| pearson | log1p pearson | spearman |
| --- | --- | --- |
| 0.8867 | 0.9553 | 0.6422 |

### 2.3 differential gene expression: predicted versus true trajectories

The number of genes with significant difference (| log2(FoldChange)|>2) was 0. (After zscore standardization)

### ![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/1wvqreb4KwLGenak/img/8188e6d0-2420-4c61-9cdf-b0f5722b22ee.png)![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/1wvqreb4KwLGenak/img/cd20eb99-3bee-4db7-8189-469a8da8d5b5.png)

### 2.4 trajectory visualization

*   **Accurate consistency of forecasts**
    

Chromosome 19 global: True in blue, predicted in Red

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/1wvqreb4KwLGenak/img/8df97a24-138c-419b-a283-e0edf879e9c5.png)

ZFP36：

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/1wvqreb4KwLGenak/img/3ae87bb5-5d4e-405b-af01-2999b018e80d.png)

JUNB：

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/1wvqreb4KwLGenak/img/8cbc4d93-f453-459b-821d-1a292d19ada6.png)

chr19: 10587331-11635907 （AlphaGenome在HepG2 Cell Line中选取的区域）

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/1wvqreb4KwLGenak/img/0acce3fd-9dc3-4087-bfd5-110673d740ab.png)

### Case 2: Oomics + Text Interactive Disease Diagnosis

## Project Overview

This project is a DNA sequence analysis system based on DNA model + text model, which is mainly used for gene variation effect prediction and disease association analysis. The project combines a DNA sequence encoder with a large-scale language model,Achieve a deep understanding and analysis of DNA sequences.

![image](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/meonarb8PVbQ0qXx/img/e26295ae-704f-47e4-a304-d81961de6dbf.jpg)

## Input and output analysis

### 1.1 Input data format

#### 1.1.1 KEGG data set

*   **File Format**: JSON format
    
*   **Data structure**:
    

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

#### 1.1.2 VEP(Variant Effect Prediction) data set

*   **File Format**: JSON format
    
*   **Data structure**:
    

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

### 1.2 output data format

#### 1.2.1 Model Output

*   **Text Generation**: The model generates text that contains the reasoning process and the final answer.
    
*   **Format**:
    

```markdown
  ```
  <|im_start|>assistant
  [推理内容]
  Answer: [最终答案]<|im_end|>
  ```
```

#### 1.2.2 Evaluation Results

**CSV format**: contains the comparison between the prediction result and the real label.

*   **Column structure**:
    
*   'Ground\_truth': Real label
    
*   'Pred\_Label': prediction label
    
*   'Generated\_text': Complete generated text
    

## Data preprocessing process

### 2.1 DNA sequence preprocessing

#### 2.1.1 Sequence Normalization

**Case conversion**: convert all DNA sequences to uppercase letters

*   **Space removal**: Removes white space characters from the sequence
    
*   **Sequence truncation**: Use the 'truncate\_DNA 'function to truncate the sequence from both ends
    

```markdown
 ```python
  def truncate_dna(example, truncate_dna_per_side=1024):
      # 从序列两端各截断1024个碱基对
      # 如果序列太短，返回中间部分
  ```
```

#### 2.1.2 Sequence Tokenization

*   **OURGEN \*\*\*\* taggers**: Use character-level taggers to process DNA sequences
    
*   **Special tag**: Add '<| dna\_start |>', '<| dna\_pad |>', '<| dna\_end |>' tags
    
*   **Sequence length limit**: The maximum length is 2048 tokens.
    

### 2.2 Text preprocessing

#### 2.2.1 Conversation Format Conversion

*   **Multimodal Format**: Conversation format that combines DNA sequences and text
    
*   **Role Definitions**:
    
*   'user': contains DNA sequences and questions
    
*   'assistant': contains reasoning process and answers
    

#### 2.2.2 Template Application

*   **Chat Template**: Use a custom chat template to format input
    
*   **Special tag handling**: Correctly handle '<| im\_start |>' and '<| im\_end |>' tags
    

### 2.3 Data loading and batch processing

#### 2.3.1 Data Set Segmentation

*   **Training set**: 80%
    
*   **Validation Set**: 10%
    
*   **Test set**: 10%
    

#### 2.3.2 Batch Functions

*   **qwen\_DNA \_collate\_FN**: a batch function designed specifically for Qwen DNA Models
    
*   **Label mask**: Calculate the loss only for the helper reply part
    
*   **Fill processing**: left fill policy,Add Special Marker
    

## Evaluation indicator scheme

### 3.1 classification indicators

#### 3.1.1 Basic indicators

*   **Accuracy (Accuracy)**: the proportion of correctly predicted samples to the total sample
    
*   **Precision**: Macro average Precision,Average of accuracy rates for all categories
    
*   **Recall rate (Recall)**: macro average Recall rate, the average of all categories of Recall rate
    

**F1 score (F1-Score)**: Macro average F1 score, reconciliation average of precision and recall

#### 3.1.2 calculation method

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

### 3.2 Assessment Process

#### 3.2.1 Validation Phase Assessment

*   **Real-time generation**: Generate text on the validation set
    

**Answer Extraction**: Extract The Answer part from the generated text

*   **Indicator calculation**: calculates category indicators and records them in the log.
    

#### 3.2.2 Test Phase Assessment

*   **Complete evaluation**: Complete evaluation on the test set
    
*   **Save Results**: Save the prediction results as a CSV file.
    
*   **Indicator Summary**: Generate a detailed classification report
    

### 3.3 Special treatment

#### 3.3.1 Answer extraction

*   **Regular Matching**: Use the 'extract\_single\_entry 'function to extract the answer
    
*   **Format Processing**: Processing the prediction result containing the '
    

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

## 8. Licence description

This work advocates the open sharing of biological AI community and supports MIT License. See 'MIT©2025 '

## 9. Contact Us

If you have questions or cooperation intention,Feel free to contact us. Email: bgi-Genos@genomics.cn
