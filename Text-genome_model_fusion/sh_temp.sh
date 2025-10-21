#!/bin/bash

WANDB_PROJECT=kegg  

     
export CUDA_VISIBLE_DEVICES=0
python train_dna_qwen_kegg_hard_new_metrics_without_wandb.py \
    --cache_dir model_weights \
    --text_model_name model_weights/Qwen/Qwen3-1___7B \
    --dna_model_name hyenadna-large-1m-seqlen \
    --wandb_project $WANDB_PROJECT \
    --strategy deepspeed_stage_2 \
    --max_epochs 5 \
    --num_gpus 1 \
    --batch_size 1 \
    --model_type dna-llm \
    --dataset_type kegg \
    --max_length_dna 1024 \
    --max_length_text 8192 \
    --truncate_dna_per_side 1024 \
    --merge_val_test_set True \
    --dna_is_evo2 False \
    --return_answer_in_batch True \
    --gradient_accumulation_steps 8 \
    --dna_embedding_layer blocks.20.mlp.l3   # set to blocks.40.mlp.l3 for evo2_40b

