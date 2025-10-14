#!/bin/bash

PROJECT_NAME=kegg_test  

# export MASTER_PORT=7799      
# export CUDA_VISIBLE_DEVICES=7
# python train_dna_qwen.py \
#     --cache_dir model_weights/arcinstitute/evo2_1b_base/evo2_1b_base.pt \
#     --text_model_name model_weights/Qwen/Qwen3-1___7B \
#     --dna_model_name evo2_1b_base \
#     --project_name $PROJECT_NAME \
#     --strategy deepspeed_stage_2 \
#     --max_epochs 5 \
#     --num_gpus 1 \
#     --batch_size 1 \
#     --model_type dna-llm \
#     --dataset_type kegg \
#     --max_length_dna 1024 \
#     --max_length_text 8192 \
#     --truncate_dna_per_side 1024 \
#     --merge_val_test_set True \
#     --dna_is_evo2 True \
#     --return_answer_in_batch True \
#     --gradient_accumulation_steps 8 \
#     --dna_embedding_layer blocks.20.mlp.l3   # set to blocks.40.mlp.l3 for evo2_40b
    
# export CUDA_VISIBLE_DEVICES=6
# python train_dna_qwen.py \
#     --cache_dir model_weights/arcinstitute/evo2_1b_base/evo2_1b_base.pt \
#     --text_model_name model_weights/Qwen/Qwen3-4B \
#     --dna_model_name evo2_1b_base \
#     --project_name $PROJECT_NAME \
#     --strategy deepspeed_stage_2 \
#     --max_epochs 5 \
#     --num_gpus 1 \
#     --batch_size 1 \
#     --model_type dna-llm \
#     --dataset_type kegg \
#     --max_length_dna 1024 \
#     --max_length_text 8192 \
#     --truncate_dna_per_side 1024 \
#     --merge_val_test_set True \
#     --dna_is_evo2 True \
#     --return_answer_in_batch True \
#     --gradient_accumulation_steps 8 \
#     --dna_embedding_layer blocks.20.mlp.l3   # set to blocks.40.mlp.l3 for evo2_40b

# export MASTER_PORT=7798      
# export CUDA_VISIBLE_DEVICES=0,1,2,3
# python train_dna_qwen.py \
#     --cache_dir model_weights/arcinstitute/evo2_40b/evo2_40b.pt \
#     --text_model_name model_weights/Qwen/Qwen3-1___7B \
#     --dna_model_name evo2_40b \
#     --project_name $PROJECT_NAME \
#     --strategy ddp \
#     --max_epochs 5 \
#     --num_gpus 4 \
#     --batch_size 1 \
#     --model_type dna-llm \
#     --dataset_type kegg \
#     --max_length_dna 1024 \
#     --max_length_text 8192 \
#     --truncate_dna_per_side 1024 \
#     --merge_val_test_set True \
#     --dna_is_evo2 True \
#     --return_answer_in_batch True \
#     --gradient_accumulation_steps 8 \
#     --dna_embedding_layer blocks.40.mlp.l3   # set to blocks.40.mlp.l3 for evo2_40b

# export MASTER_PORT=17990      
# export CUDA_VISIBLE_DEVICES=0
# python train_dna_qwen.py \
#     --cache_dir model_weights \
#     --text_model_name model_weights/Qwen/Qwen3-1___7B \
#     --dna_model_name hyenadna-large-1m-seqlen \
#     --project_name $PROJECT_NAME \
#     --strategy deepspeed_stage_2 \
#     --max_epochs 5 \
#     --num_gpus 1 \
#     --batch_size 1 \
#     --model_type dna-llm \
#     --dataset_type kegg \
#     --max_length_dna 1024 \
#     --max_length_text 8192 \
#     --truncate_dna_per_side 1024 \
#     --merge_val_test_set True \
#     --return_answer_in_batch True \
#     --gradient_accumulation_steps 8 
    
export MASTER_PORT=30002
export CUDA_VISIBLE_DEVICES=3
# Genos-1b KEGG
python train_dna_only.py \
    --cache_dir None \
    --dna_model_name model_weights/Genos-1b \
    --strategy deepspeed_stage_2 \
    --max_epochs 10 \
    --project_name $PROJECT_NAME \
    --num_gpus 1 \
    --batch_size 1 \
    --dataset_type kegg \
    --max_length_dna 2048 \
    --truncate_dna_per_side 1024 \
    --merge_val_test_set True \
    --gradient_accumulation_steps 8 \