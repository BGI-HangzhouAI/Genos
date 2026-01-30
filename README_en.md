# Genos: A Human-Centric Genomic Foundation Model

![model](images/Genos_LOGO.gif)

<div align="center" style="line-height: 2;">

[![Documentation Status](https://readthedocs.org/projects/genos-client/badge/?version=latest)](https://genos-client.readthedocs.io/en/latest/?badge=latest)  [![PyPI version](https://badge.fury.io/py/genos-client.svg)](https://badge.fury.io/py/genos-client)  [![Collection](https://img.shields.io/badge/🤗-Genos%20%20Collection-blue)](https://huggingface.co/collections/BGI-HangzhouAI/genos)
 <a href="https://cloud.stomics.tech/#/inferance-web?type=model" target="_blank">
      <img alt="DCS" src="https://img.shields.io/badge/☁️%20DCS-Inference Services%20-6f42c1"/>
  </a>
  <a href="https://www.zero2x.org/genos" target="_blank">
      <img alt="Homepage" src="https://img.shields.io/badge/🌐%20Homepage-zero2x%20-536af5"/>
  </a>
  <a href="https://academic.oup.com/gigascience/advance-article/doi/10.1093/gigascience/giaf132/8296738?login=false" target="_blank">
      <img alt="Technical Report" src="https://img.shields.io/badge/📜%20Technical Report-GigaSience-brightgreen?logo=Linkedin&logoColor=white"/>
  </a>
  <a href="https://github.com/BGI-HangzhouAI/Genos/blob/main/LICENSE" target="_blank">
       <img alt="License" src="https://img.shields.io/badge/📑%20License- Apache 2.0-f5de53"/> 
![Badge](https://hitscounter.dev/api/hit?url=https%3A%2F%2Fgithub.com%2FBGI-HangzhouAI%2FGenos&label=visitors&icon=github&color=%6ec044&message=&style=flat&tz=UTC)


 [English](README_en.md) ｜ [中文](README_zh.md) 

</div>




## Table of Contents

- [Updates](#updates)
- [Model Introduction](#model-introduction)
- [Models and Data](#models-and-data)
  - [Training Data](#training-data)
  - [Model Architecture](#model-architecture)
  - [Genos-10B-v2 Highlights](#genos-10b-v2-highlights)
- [Performance Evaluation](#performance-evaluation)
- [Deployment and Usage](#deployment-and-usage)
  - [Docker Deployment](#docker-deployment)
  - [Model Download](#model-download)
  - [API/SDK](#apisdk)
  - [Notebook Examples](#notebook-examples)
- [Application Cases](#application-cases)
  - [Case 1: RNA-seq Data Generation](#case-1-rna-seq-data-generation)
  - [Case 2: Genomic Model + Text Model Disease Prediction](#case-2-genomic-model--text-model-disease-prediction)
- [Inference Optimization and Adaptation](#inference-optimization-and-adaptation)
- [Data Availability](#data-availability)
- [License](#license)
- [Citation](#citation)
- [Contact Us](#contact-us)

## Updates

- **2026.01.30** | Released [Gengram](https://github.com/BGI-HangzhouAI/Gengram) — a retrieval-augmented genomic foundation model featuring explicit motif memory, efficient hash-based k-mer lookup, and enhanced biological interpretability. 🚀 Explore it and see how it transforms genomic modeling！
- **2025.12.25** | Added [NTv3-650M-Pre](https://huggingface.co/InstaDeepAI/NTv3_650M_pre) as a new competitor model in the Performance Evaluation section. Stay tuned to see how it performs against our Genos-10B-v2 model!
- **2025.12.18** | Released Genos-10B-v2 model: `Genos-10B-v2`.
  - v2 updates: Introduced non-human primate and multiple mammalian genomes, adopting a phased, 1:1 mixing strategy to enhance cross-species generalization and evolutionary context modeling.
  - New evaluations: (1) Cross-species generalization capability; (2) Ultra-long context tasks added.
  
  - Inference optimization provides vLLM image; optional adaptation for domestic hardware such as Huawei and MuXi.

## Model Introduction


**【Model Architecture and Technical Breakthrough】**

As a foundation model in the human genomics field, Genos is trained on hundreds of high-quality human genome reference datasets, achieving context modeling capabilities for human genomic sequences up to millions of base pairs. Through single-base resolution learning, the model has acquired the ability to identify deep sequence patterns and functional features hidden in the genome, building a new research approach for scientists to connect genetic information with life activities. This release includes two versions with 1.2 billion and 10 billion parameters, both adopting a Mixture of Experts (MoE) architecture, achieving optimal computational resource allocation through dynamic routing mechanisms, significantly improving the model's performance in complex regulatory network analysis.

**【Functional Modules and Scientific Value】** 

Genos possesses the core capability to accurately identify key functional elements, enabling in-depth analysis of the cascading effects of minute genetic variations on transcriptional regulatory networks, breaking through the limitations of traditional methods in predicting regulatory elements in non-coding regions, dynamically simulating the potential impact of variant sites on RNA expression profiles, and tracing to the molecular pathways of phenotype formation. Based on this, the research team has developed modular application interfaces, constructing a full-chain research system of "prediction-explanation-verification". By introducing interpretability enhancement mechanisms, the model not only provides high-confidence prediction results but also reveals key nodes and action pathways in regulatory networks, providing a new research paradigm for molecular mechanism analysis.

**【Open Ecosystem and Clinical Translation】**

Adhering to the principles of open science, Genos provides open-source models on GitHub and Hugging Face, while simultaneously deploying cloud inference services on the DCS Cloud platform. Researchers can download models for deployment and inference, or choose to deploy on DCS Cloud. We also provide users with full-process analysis example code from variant functional annotation to phenotype prediction, helping users become familiar with model usage methods and functions more quickly. Model weights will be continuously updated, and their application potential in precision medicine, population health, monitoring, and developmental biology will be further released.

**【Scientific Philosophy and Future Prospects】**

Genos provides new possibilities for scientists to study the complex regulation of genes and their functional impacts. In the future, with the enhancement of cross-modal learning capabilities, Genos is expected to become a "translator" connecting genetic codes with life phenomena, opening new research dimensions in disease early warning, drug target discovery, and synthetic biology, aiming to achieve a paradigm shift from "genomics" to "functional genomics".

## Models and Data

### Training Data

- Human core corpus: The core human dataset consists of haplotype-resolved and reference assemblies provided by internationally recognized consortia, including 231 assemblies from the Human Pangenome Reference Consortium (HPRC, V2), 65 assemblies from the Human Genome Structural Variation Consortium (HGSVC), 21 genomes from the Centre d'Étude du Polymorphisme Humain (CEPH) cohort, as well as the GRCh38 and CHM13 reference genomes. After strict quality control, this core dataset contains 636 high-quality human genomes, totaling approximately 244.35 billion base pairs (corresponding to over 150 billion tokens), representing diverse global populations.
- Genos-10B-v2 additions: Approximately 60 billion base pairs of high-coverage East Asian human genomes generated by BGI's CycloneSeq platform, 950.1 billion base pairs from RefSeq non-human primate genomes, and 48.485 billion base pairs from RefSeq non-primate mammalian genomes, mixed in a phased 1:1 ratio with the core human corpus.
- Quality control: Strict filtering and standardization, covering diverse global populations, ensuring single-base precision and cross-population generalization.

### Model Architecture

- Transformer-based Mixture of Experts network, Top-2 routing, 25% FFN sparsity.
- Ultra-long context: RoPE base 50M, multi-dimensional tensor/pipeline/context/data/expert parallelism; supports up to 1M tokens.
- Training stability: Gradient clipping, expert load balancing (aux loss + z-loss), GQA 50% KV cache compression, Flash Attention.
- Inference: Dynamic expert activation, single-sequence million-base inference available.

Model Parameters:

<div align="center">

|  | Genos-1.2B | Genos-10B | Genos-10B-v2 |
| --- | --- | --- | --- |
| Total Parameters | 1.2B | 10B | 10B |
| Active Parameters | 0.33B | 2.87B | 2.87B |
| Training Tokens | 1600B | 2200B | 6286B |
| Architecture | MoE | MoE | MoE |
| Number of Experts | 8 | 8 | 8 |
| Top-k | 2 | 2 | 2 |
| Layers | 12 | 12 | 12 |
| Attention Hidden | 1024 | 4096 | 4096 |
| Attention Heads | 16 | 16 | 16 |
| MoE FFN Hidden | 4096 | 8192 | 8192 |
| Vocabulary | 128 (pad) | 256 (pad) | 256 (pad) |
| Max Context Length | 1M | 1M | 1M |

</div>

### Genos-10B-v2 Highlights

- Broader species coverage: Introduced non-human primate and multiple mammalian sequences, enhancing evolutionary and conservation modeling.
- Phased balanced mixing: New data gradually mixed 1:1 with human core corpus, maintaining human signal dominance while expanding diversity.
- Long-range task enhancement: Achieved leading or co-leading performance on DNALongBench long-range tasks and cross-species classification.

## Performance Evaluation

- Long sequences: DNALongBench (enhancer-promoter associations, eQTL, etc.).
- Short sequences: Genomic element/open chromatin/splice site classification tasks maintain leading performance.
- Cross-species: Multi-species sequence and genomic element classification tasks validate v2's evolutionary generalization capability.
- Variant hotspots and population classification: Consistently outperforms similar open models at 8K/32K/128K sequence lengths.

---
**New Evaluation Results**
<div align="center">
<img src="images\Evaluation_results.png" width="90%" title="Evaluation">
</div>

---

**Initial Evaluation Results**
<div align="center">
<img src="images/评测结果.20251013.png" width="90%" title="Evaluation">
</div>

## Deployment and Usage

### Docker Environment Deployment

```bash
docker pull bgigenos/mega:v1
docker run -it --gpus all --shm-size 32g bgigenos/mega:v1 /bin/bash
```

### Model Weight Download

| Model | Total Parameters | Hugging Face | Megatron ckpt |
| --- | --- | --- | --- |
| Genos-1.2B | 1.2B | [Genos-1.2B](https://huggingface.co/BGI-HangzhouAI/Genos-1.2B) | [Genos-Megatron-1.2B](https://huggingface.co/BGI-HangzhouAI/Genos-Megatron-1.2B) |
| Genos-10B | 10B | [Genos-10B](https://huggingface.co/BGI-HangzhouAI/Genos-10B) | [Genos-Megatron-10B](https://huggingface.co/BGI-HangzhouAI/Genos-Megatron-10B) |
| Genos-10B-v2 | 10B | [Genos-10B-v2](https://huggingface.co/BGI-HangzhouAI/Genos-10B-v2) | [Genos-Megatron-10B-v2](https://huggingface.co/BGI-HangzhouAI/Genos-Megatron-10B-v2) |

### API/SDK

```bash
pip install genos-client
```

For specific usage, see [SDK Documentation](sdk/README.md).

### Notebook Usage Examples

- [Embedding Extraction](Notebooks/en/01.embedding_en.ipynb)
- [Population Classification](Notebooks/en/02.Population_classify_Demo.ipynb)
- [Variant Effect Prediction](Notebooks/en/03.ClinVar_variant_predict_en.ipynb)
- [RNA Coverage Track Prediction](Notebooks/en/04.RNASeqConvTrack_en.ipynb)


## Application Cases

### Case 1: RNA-seq Data Generation

1.  **Task Description**
    
    This task is based on Genos' extensive pre-training accumulation on genomic data, achieving direct prediction of single-base resolution RNA-seq expression profiles from DNA sequences through fine-tuning, covering multiple cell types and tissues. Its scientific significance lies in constructing a direct mapping relationship between genomic sequences and transcriptomic expression, providing an innovative tool for understanding gene regulatory mechanisms and accelerating transcriptomic research.
    
2.  **Task Input and Output**
    
    This task is a **regression task**. The task input is a partial sequence of the hg38 reference genome (currently using 32 kb as the window), and the output is the average normalized RNA-seq signal values (single-base precision) for different cell types, corresponding to sequence positions on the positive and negative strands of the genome. The core is to predict continuous transcriptomic expression levels by learning the complex mapping from sequence to expression.
    
3.  **Data Sources**
    
    Training data comes from public databases ENCODE (ENCODE Consortium, 2012) and GTEx (Kim-Hellmuth et al., 2020).
    
    After integration, the data contains single-base transcriptomic bigwig files from 667 metadata groups, along with the hg38 reference genome as input. Model training uses all positional sequences on chromosomes 1-22 and their corresponding average RNA-seq profiles as paired data. The current public version uses 4 human B lymphocyte samples and 13 NK natural killer cell samples for fine-tuning. To remove inter-sample individual differences, expression data for both cell types are averaged across samples and normalized to one dataset each.
    
    \*The current public version can already perform prediction inference for human B lymphocytes and NK natural killer cells on chromosomes 1-22.
    
4.  **Model Design**
    

*   Downstream Model Architecture Design​
    

The model replaces the original output head with a task-specific convolutional module based on the pre-trained Genos-1.2B. This module consists of three one-dimensional convolutional layers with decreasing channel dimensions (1024→256→64→1), each followed by batch normalization, GELU activation, and dropout regularization (dropout=0.1). The final output is scaled by learnable weight parameters and passed through Softplus activation to ensure predicted values are non-negative continuous values, matching the regression characteristics of RNA-seq signals. This design enhances the model's ability to capture local sequence patterns while optimizing computational efficiency through the translation invariance of convolutions.

*   Fine-tuning Strategy and Training Optimization​
    

Full parameter fine-tuning is adopted, using mean squared error (MSE) as the loss function for the regression task. To address the skewed distribution of RNA-seq signal values, square root smooth clipping and power transformation are introduced during training for numerical compression, with inverse operations performed during inference to restore signal scale. The optimizer uses Adafactor, combined with a cosine annealing learning rate scheduler and linear warmup (warmup steps account for 5% of total steps), with global batch size set to 256, training for 60 epochs. This strategy ensures stable convergence while significantly reducing the risk of gradient fluctuations in long-sequence training.

*   Genomic Sequence Processing and Context Modeling​
    

To balance long-range dependency learning with computational cost, the input sequence window length is set to 32 kb, with adjacent windows overlapping by 16 kb, covering all sites on chromosomes 1-22.

(1).  **Metrics and Metric Definitions**
    

The core evaluation metric is the log1p-transformed Pearson correlation coefficient​. This metric is used to measure the consistency between model-predicted RNA-seq profiles and experimentally measured true profiles across different genomic ranges, with specific calculation ranges including:

*   Whole genome​: Genome-wide, single-base precision
    
*   Gene expression​: Gene expression matrix correlation, gene-level precision
    

\*The Pearson correlation coefficient is transformed by log1p (i.e., log(1 + r)) to better evaluate the global correlation of signal prediction.

(2).  **Evaluation Metrics**
Genos simulates and generates multi-cell type RNA-seq expression levels with correlation of 0.9+ compared to real sequencing results.
![model](images/RNA_exp.benchmark_res.20251020.png)

    
(3).  **Output Example**
    

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/NpQlK5jmkj0ADqDv/img/fff8c7df-66fe-4022-98f5-6c10531bb460.png)

### Case 2: Genomic Model + Text Model Disease Prediction

#### Project Overview

**Introduction:**

This project aims to validate that multimodal models (genomic model + text model) can process raw DNA sequences in disease prediction tasks caused by genetic variants, while leveraging the reasoning capabilities of large language models to generate biologically consistent explanations and predictions.

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/NpQlK5jmkj0ADqDv/img/35006876-15a3-42be-9f1c-34167d2819cd.png)

**Data:**

Data comes from the KEGG task in the Bioreason paper [7], which integrates KEGG pathway and clinical database variant information through multiple stages, using standardized symbol systems to represent various interactions in molecular networks, providing comparisons between reference sequences and variant sequences. The KEGG dataset contains 1449 entries with 37 disease types, with data allocation of training:validation:test = 8:1:1. Data input includes problem descriptions, reference gene sequences, and variant gene sequences. Output includes reasoning and disease classification prediction.

**Model Design:**

*   Architecture design: DNA model, text model (Qwen3-1b, Qwen3-4b), projection layer from DNA embedding to text embedding.
    
*   Model training: This model training aims to achieve efficient alignment between DNA sequences and natural language. During training, the genomic model is frozen (supports Evo2, Genos models), the projection layer from DNA embedding to text embedding is trained, the text model is fine-tuned with LoRA, and DeepSpeed strategy is used for efficient training.
    
*   Genomic sequence processing: There are two gene sequences in total: 1. Reference gene sequence. 2. Variant gene sequence. A total of 1024 bp genomic sequence centered on the variant gene with upstream and downstream regions.
    

**Evaluation Metrics:**

*   **Accuracy**: The proportion of correctly predicted samples to total samples.
    

#### Results Comparison

The comparison of different models on the KEGG dataset is shown in the table below. Among genomic models, Genos-10B leads in performance, and text-genomic fusion models far outperform single-modal models. The 021-8B and Genos-1.2B fusion model achieves an accuracy of 98.28%, 7% higher than using Genos-1.2B alone.

<div align="center">
  <img src="images/text_gLM.benchmark_res.20251020.png" alt="model" style="width:50%;">
</div>

Model Descriptions:

NT-2.5b-multi: [InstaDeepAI/nucleotide-transformer-2.5b-multi-species](https://huggingface.co/InstaDeepAI/nucleotide-transformer-2.5b-multi-species)

Evo2-1b: [arcinstitute/evo2\_1b\_base](https://huggingface.co/arcinstitute/evo2_1b_base)

HyenaDna-1m: [LongSafari/hyenadna-large-1m-seqlen](https://huggingface.co/LongSafari/hyenadna-large-1m-seqlen)

Genos-1.2B: [BGI-HangzhouAI/Genos-1.2B](https://huggingface.co/BGI-HangzhouAI/Genos-1.2B)

Genos-10B: [BGI-HangzhouAI/Genos-10B](https://huggingface.co/BGI-HangzhouAI/Genos-10B)

021-8B: 021 Science Foundation Model-8B is a large language model trained on extensive scientific corpora with profound scientific cognition. It is scheduled to be released at a later date. 

## Inference Optimization and Adaptation

- vLLM Optimization: We conducted inference optimization experiments on Genos using the vLLM framework. This solution significantly improved throughput and reduced inference latency. By leveraging vLLM's innovative PagedAttention algorithm and efficient memory management mechanisms, we achieved more than 7x throughput improvement compared to traditional inference methods.
  - Pull the image
  ```bash
  docker pull bgigenos/vllm:v1

  docker run -it --entrypoint /bin/bash --gpus all --shm-size 32g bgigenos/vllm:v1
  ```
  - For embedding inference using vLLM, please refer to [vllm example](Notebooks/en/05.vllm_example_en.ipynb)

- For other hardware adaptations, please refer to [Adaptation](Adaptation)
  - Huawei Ascend NPU
  - MuXi GPU 


## Data Availability

- All training data sources are indicated. Core human corpus comes from HPRC, HGSVC, CEPH, GRCh38, CHM13, etc.
<div align="center">

| **Dataset** | **Data License** | **Source** |
|:---:|:---:|:---:| 
| HPRC Data Release 2 | MIT |🌐 [HPRC](https://humanpangenome.org/hprc-data-release-2/)|
| HGSVC | Public website | 🌐 [HGSVC](https://www.internationalgenome.org/data-portal/data-collection/structural-variation) |
| CEPH | Public website |🌐 [CEPH](https://uofuhealth.utah.edu/center-genomic-medicine/research/ceph-resources)  |
| GRCh38 | Public website |🌐 [GRCh38](https://www.ncbi.nlm.nih.gov/datasets/genome/GCF_000001405.26/)|
| CHM13 | Public website |🌐 [CHM13](https://www.ncbi.nlm.nih.gov/datasets/genome/GCF_009914755.1/)|
| High-coverage East Asian human genomes | Internal data | Contact via email | 
| RefSeq non-human primate genomes | Public website |🌐 [NCBI RefSeq](https://ftp.ncbi.nlm.nih.gov/refseq/release/)  | 
| RefSeq non-primate mammalian genomes| Public website |🌐 [NCBI RefSeq](https://ftp.ncbi.nlm.nih.gov/refseq/release/)|

</div>

- Evaluation datasets are being organized and will be continuously updated on the [Hugging Face project homepage](https://huggingface.co/BGI-HangzhouAI/datasets).
  - [Human Population Classification Task Dataset](https://huggingface.co/datasets/BGI-HangzhouAI/Benchmark_Dataset-Human_population_classification)
  - [Variant Hotspot Prediction Task Dataset](https://huggingface.co/datasets/BGI-HangzhouAI/Benchmark_Dataset-variant_hotspot)
  - [Genomic Element Classification Task Dataset](https://huggingface.co/datasets/BGI-HangzhouAI/Benchmark_Dataset-Genomic_element_classification)
  - [Primate Mammal Species Classification Task Dataset](https://huggingface.co/datasets/BGI-HangzhouAI/Benchmark_Dataset-Primate_mammal_species_classification)

## License

- Models and code follow [Apache License 2.0](LICENSE).

## Citation

```
@article{10.1093/gigascience/giaf132,
    author = {Genos Team, Hangzhou, China},
    title = {Genos: A Human-Centric Genomic Foundation Model},
    journal = {GigaScience},
    pages = {giaf132},
    year = {2025},
    month = {10},
    issn = {2047-217X},
    doi = {10.1093/gigascience/giaf132},
    url = {https://doi.org/10.1093/gigascience/giaf132},
    eprint = {https://academic.oup.com/gigascience/advance-article-pdf/doi/10.1093/gigascience/giaf132/64848789/giaf132.pdf},
}
```

## Contact Us

- Email: [Genos@genomics.cn](mailto:Genos@genomics.cn)
- Issues and Suggestions: Welcome to submit Issues.

## References
\[1\] The Human Genome Project. (2003). Finishing the euchromatic sequence of the human genome. Nature, 431(7011), 931 - 945.

\[2\] 1000 Genomes Project Consortium. (2010). A map of human genome variation from population - scale sequencing. Nature, 467(7319), 1061 - 1073.

\[3\] Grešová, K., Martinek, V., Čechák, D., Šimeček, P. & Alexiou, P. Genomic benchmarks: a collection of datasets for genomic sequence classification. BMC Genomic Data 24, (2023).

\[4\] Dalla-Torre, H. et al. Nucleotide Transformer: building and evaluating robust foundation models for human genomics. Nature Methods (2024) doi:10.1038/s41592-024-02523-z.

\[5\] Trop, E. et al. The Genomics Long-Range Benchmark: Advancing DNA Language Models. OpenReview [https://openreview.net/forum?id=8O9HLDrmtq.](https://openreview.net/forum?id=8O9HLDrmtq.)

\[6\] Gao, Y. et al. A pangenome reference of 36 Chinese populations. Nature 619, 112–121 (2023).

\[7\] Fallahpour, Adibvafa, et al. BioReason: Incentivizing Multimodal Biological Reasoning within a DNA-LLM Model. arXiv preprint arXiv:2505.23579 (2025).

\[8\] Liao, W.W. et al. A draft human pangenome reference. Nature 617, 312–324 (2023).
