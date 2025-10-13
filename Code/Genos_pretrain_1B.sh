#!/bin/bash

# Runs Genos 1.2B model

export CUDA_DEVICE_MAX_CONNECTIONS=1

export WANDB_API_KEY=$1
export WANDB_NAME=$2
SEQ_LENGTH=8192
MAX_LENGTH=1048576
TRAIN_SAMPLES=$3
LR_DECAY_SAMPLES=$((TRAIN_SAMPLES * 80 / 100))
CHECKPOINT_PATH=$4

TOKENIZER_TYPE=SentencePieceTokenizer
TOKENIZER_MODEL=$5
MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=1024

DATA_PATH=$6



DISTRIBUTED_ARGS=" \
    --nnodes=$WORLD_SIZE \
    --nproc_per_node=8 \
    --node_rank=$RANK  \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT"

MODEL_ARGS=" \
    --use-mcore-models \
    --disable-bias-linear \
    --seq-length $SEQ_LENGTH \
    --max-position-embeddings $MAX_LENGTH \
    --num-layers 12 \
    --hidden-size 1024 \
    --ffn-hidden-size 4096 \
    --num-attention-heads 16 \
    --init-method-std 0.01 \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --normalization RMSNorm \
    --position-embedding-type rope \
    --swiglu \
    --untie-embeddings-and-output-weights \
    --group-query-attention \
    --num-query-groups 8 \
    --no-masked-softmax-fusion \
    --no-position-embedding \
    --rotary-base 50000000"

MOE_ARGS=" \
    --num-experts 8 \
    --moe-router-topk 2 \
    --moe-router-load-balancing-type aux_loss \
    --moe-aux-loss-coeff 1e-3 \
    --moe-grouped-gemm \
    --moe-token-dispatcher-type alltoall \
    --overlap-param-gather \
    --overlap-grad-reduce \
    --moe-router-dtype fp32 \
    --moe-z-loss-coeff 1e-3"

DATA_ARGS=" \
    --num-workers 8 \
    --dataloader-type cyclic \
    --tokenizer-type ${TOKENIZER_TYPE} \
    --tokenizer-model ${TOKENIZER_MODEL} \
    --data-path ${DATA_PATH} \
    --split 1000,0,0 \
    --no-create-attention-mask-in-dataloader"


TRAINING_ARGS=" \
    --micro-batch-size ${MICRO_BATCH_SIZE} \
    --global-batch-size ${GLOBAL_BATCH_SIZE} \
    --lr 1e-4 \
    --train-samples ${TRAIN_SAMPLES} \
    --lr-decay-samples ${LR_DECAY_SAMPLES} \
    --lr-decay-style cosine \
    --min-lr 1e-5 \
    --weight-decay 0.1 \
    --lr-warmup-fraction 0.1 \
    --clip-grad 1.0 \
    --bf16 \
    --use-flash-attn \
    --attention-softmax-in-fp32 \
    --accumulate-allreduce-grads-in-fp32 \
    --disable-bf16-reduced-precision-matmul"


MODEL_PARALLEL_ARGS=" \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --expert-model-parallel-size 1 \
    --sequence-parallel \
    --use-distributed-optimizer"



LOGGING_ARGS=" \
    --moe-per-layer-logging \
    --log-interval 10 \
    --save-interval 10000 \
    --eval-interval 50000000 \
    --eval-iters 0 \
    --save $CHECKPOINT_PATH \
    --tensorboard-dir "${CHECKPOINT_PATH}/tensorboard" \
    --no-load-optim \
    --no-load-rng \
    --wandb-project ${WANDB_PROJECT:-"Genos"} \
    --wandb-exp-name ${WANDB_NAME:-"Genos-1.2B"} \
    --log-throughput"





torchrun ${DISTRIBUTED_ARGS} pretrain_gpt.py \
    ${MODEL_ARGS} \
    ${MOE_ARGS} \
    ${DATA_ARGS} \
    ${TRAINING_ARGS} \
    ${MODEL_PARALLEL_ARGS} \
    ${LOGGING_ARGS}
