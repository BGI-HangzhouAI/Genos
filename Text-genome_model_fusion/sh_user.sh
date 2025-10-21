#!/bin/bash

WANDB_PROJECT=kegg  

     
export CUDA_VISIBLE_DEVICES=0
python train_dna_qwen_kegg_hard_new_metrics_without_wandb.py \
    --cache_dir ###cache_dir### \
    --text_model_name ###text_model_name### \
    --dna_model_name ###dna_model_name### \
    --wandb_project $WANDB_PROJECT \
    --strategy deepspeed_stage_2 \
    --max_epochs ###max_epochs### \
    --num_gpus 1 \
    --batch_size 1 \
    --model_type dna-llm \
    --dataset_type ###dataset_type### \
    --max_length_dna ###max_length_dna### \
    --max_length_text ###max_length_text### \
    --truncate_dna_per_side 1024 \
    --merge_val_test_set True \
    --dna_is_evo2 ###dna_is_evo2### \
    --return_answer_in_batch True \
    --gradient_accumulation_steps ###gradient_accumulation_steps### \
    --dna_embedding_layer ###dna_embedding_layer###   # set to blocks.40.mlp.l3 for evo2_40b

