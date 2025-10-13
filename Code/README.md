# Genos: Genomic Language Model with MoE

Genos is a Mixture-of-Experts (MoE) Transformer for ultra-long DNA sequence modeling (up to 1M tokens), built on Megatron-LM.

## Model Architecture

| Component | Genos-1.2B | Genos-10B |
|-----------|------------|-----------|
| Layers | 12 | 12 |
| Hidden Size | 1024 | 4096 |
| Moe Hidden Size | 4096 | 8192 |
| Attention Heads | 16 | 16 |
| Query Groups (GQA) | 8 | 8 |
| MoE Experts | 8 (Top-2 routing) | 8 (Top-2 routing) |
| Max Context | 1M tokens | 1M tokens |
| Active Params | 0.33B | 2.87B |
| Total Params | 1.2B | 10B |

### Key Features

- **MoE**: 8 experts, Top-2 routing, 25% FFN sparsity
- **GQA**: 50% KV cache reduction
- **RoPE**: Base 50M for ultra-long context 
- **Modern Stack**: RMSNorm, SwiGLU, Flash Attention

## Training

### Pre-Training Strategy
- **Objective**: Next Token Prediction (NTP) with self-supervised learning
- **Progressive Context Scaling**: 8K → 32K → 128K → 1M tokens across training stages
- **Data**:  high-quality, chromosome-scale de novo assemblies from publicly available resources such as HPRC and HGSVC
- **Tokenizer**: One-hot optimized for DNA bases (A, T, C, G and N)

### Infrastructure
- **Framework**: Megatron-LM on 256 GPUs
- **Parallelism**: 5D strategy (TP, CP, DP, PP, EP)
- **Batch**: Global 1024, Micro 1
- **Optimizer**: AdamW (distributed sharded)
- **Learning Rate**: up to 1e-4, cosine decay, 5-10% warmup
- **Precision**: BF16 for most compute, FP32 for softmax/gradients/routing

### Key Optimizations
- **MoE Load Balancing**: Aux loss (1e-3) + Z-loss (1e-3)
- **Communication**: Grouped GEMM, AllToAll dispatch, overlapped gradient reduction
- **Memory**: Flash Attention, sequence parallelism, distributed optimizer



## Citation
If you use Genos in your research, please cite:

```bibtex
@article{Genos2025,
  title={Genos: A Human-Centric Genomic Foundation Model},
  author={Genos team，Hangzhou，China},
  journal={Preprint},
  year={2025}
}
```

Built with [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)


