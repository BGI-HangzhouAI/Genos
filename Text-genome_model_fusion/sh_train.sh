#!/bin/bash

WANDB_PROJECT=kegg_test  

# export MASTER_PORT=7799      
# export CUDA_VISIBLE_DEVICES=7
# python train_dna_qwen_kegg_hard_new_metrics_without_wandb.py \
#     --cache_dir model_weights/arcinstitute/evo2_1b_base/evo2_1b_base.pt \
#     --text_model_name model_weights/Qwen/Qwen3-1___7B \
#     --dna_model_name evo2_1b_base \
#     --wandb_project $WANDB_PROJECT \
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
# python train_dna_qwen_kegg_hard_new_metrics_without_wandb.py \
#     --cache_dir model_weights/arcinstitute/evo2_1b_base/evo2_1b_base.pt \
#     --text_model_name model_weights/Qwen/Qwen3-4B \
#     --dna_model_name evo2_1b_base \
#     --wandb_project $WANDB_PROJECT \
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
# python train_dna_qwen_kegg_hard_new_metrics_without_wandb.py \
#     --cache_dir model_weights/arcinstitute/evo2_40b/evo2_40b.pt \
#     --text_model_name model_weights/Qwen/Qwen3-1___7B \
#     --dna_model_name evo2_40b \
#     --wandb_project $WANDB_PROJECT \
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
# python train_dna_qwen_kegg_hard_new_metrics_without_wandb.py \
#     --cache_dir model_weights \
#     --text_model_name model_weights/Qwen/Qwen3-1___7B \
#     --dna_model_name hyenadna-large-1m-seqlen \
#     --wandb_project $WANDB_PROJECT \
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
#     --gradient_accumulation_steps 8 \
    
export MASTER_PORT=17994     
export CUDA_VISIBLE_DEVICES=0,2,3,4,5,6
python train_dna_qwen_kegg_hard_new_metrics_no_fit.py \
    --cache_dir model_weights \
    --text_model_name model_weights/Qwen/Qwen3-4B \
    --dna_model_name model_weights/onehot_mix_10b_12L_1M140B_cpt_8k298B_cpt_32k200B_stage2_1_2112_0915 \
    --wandb_project $WANDB_PROJECT \
    --strategy deepspeed_stage_2 \
    --max_epochs 5 \
    --num_gpus 1 \
    --batch_size 8 \
    --model_type dna-llm \
    --dataset_type kegg \
    --max_length_dna 1024 \
    --max_length_text 8192 \
    --truncate_dna_per_side 1024 \
    --merge_val_test_set True \
    --return_answer_in_batch True \
    --gradient_accumulation_steps 8 \
    --ckpt_path checkpoints/kegg-kegg-Qwen3-4B-20250926-084417/kegg-kegg-Qwen3-4B-epoch=03-val_loss_epoch=0.3710.ckpt
    # --ckpt_path checkpoints/kegg-kegg-Qwen3-4B-20250926-084417/kegg-kegg-Qwen3-4B-epoch=03-val_loss_epoch=0.3710.ckpt

