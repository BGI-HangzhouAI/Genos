"""
简化版FastAPI服务 - DNA序列分析问答
只需输入DNA序列和问题，模型给出答案
"""

import gc
import os
import logging
import glob
from typing import Optional
from xmlrpc.client import Boolean
from sklearn import base
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
import pandas as pd
import pytorch_lightning as pl
from datasets import Dataset, DatasetDict
from argparse import ArgumentParser
from pytorch_lightning.strategies import DeepSpeedStrategy

from bioreason.models.dna_llm import DNALLMModel
from bioreason.models.evo2_tokenizer import register_evo2_tokenizer
from bioreason.models.hyenaDNA_tokenizer import register_hyena_tokenizer
from bioreason.dataset.kegg import get_format_kegg_function, qwen_dna_collate_fn
from bioreason.dataset.utils import truncate_dna
from bioreason.models.dl.processing_dl import DLProcessor
from pytorch_lightning.loggers import TensorBoardLogger 

from functools import partial
from torch.utils.data import DataLoader
from transformers.tokenization_utils_base import BatchEncoding
from deploy_dna_llm import DNALLMFineTuner



# 注册Evo2 tokenizer
register_evo2_tokenizer()
register_hyena_tokenizer()

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 全局变量
model = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 创建FastAPI应用
app = FastAPI(
    title="DNA序列问答API",
    description="简单的DNA序列分析问答服务",
    version="1.0.0"
)

# 添加CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 数据模型
class QuestionRequest(BaseModel):
    """问题请求模型"""
    reference_sequence: str = Field(..., description="参考DNA序列", min_length=1)
    variant_sequence: str = Field(..., description="变异DNA序列", min_length=1)
    question: str = Field(..., description="你的问题", min_length=1)
    max_length: int = Field(default=512, description="最大生成长度")
    temperature: float = Field(default=0.7, description="生成温度")

class AnswerResponse(BaseModel):
    """回答响应模型"""
    answer: str
    processing_time: float
    model_loaded: bool

def find_latest_checkpoint(checkpoint_dir: str = "checkpoints/kegg-kegg-Qwen3-1___7B-20250926-070747") -> Optional[str]:
    """查找最新的checkpoint文件"""
    try:
        pattern = os.path.join(checkpoint_dir, "**/last.ckpt")
        checkpoint_files = glob.glob(pattern, recursive=True)
        
        if not checkpoint_files:
            logger.warning(f"未找到checkpoint文件: {pattern}")
            return None
        
        latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
        logger.info(f"找到checkpoint: {latest_checkpoint}")
        return latest_checkpoint
    except Exception as e:
        logger.error(f"查找checkpoint失败: {e}")
        return None


def load_checkpoint_model(args: ArgumentParser):
    """加载checkpoint模型"""
    global model
    
    try:
        logger.info("正在加载模型...")
        base_model = DNALLMFineTuner(args)
        llm_logger = TensorBoardLogger(
            save_dir=args.log_dir,
            name=args.wandb_project, 
        
        )
        # 查找checkpoint
        trainer = pl.Trainer(
            max_epochs=args.max_epochs,
            accelerator="gpu",
            devices=args.num_gpus,
            strategy=(
                "ddp"
                if args.strategy == "ddp"
                else DeepSpeedStrategy(stage=2, offload_optimizer=False, allgather_bucket_size=5e8, reduce_bucket_size=5e8)
            ),
            precision="bf16-mixed",
            # callbacks=callbacks,
            logger=llm_logger,
            deterministic=False,
            enable_checkpointing=True,
            enable_progress_bar=True,
            enable_model_summary=True,
            log_every_n_steps=5,
            accumulate_grad_batches=args.gradient_accumulation_steps,
            gradient_clip_val=1.0,
            # val_check_interval=1 / 3,
            check_val_every_n_epoch=2,
        )
   
        trainer.test(base_model, ckpt_path=args.ckpt_path if args.ckpt_path else "best",)

        base_model.eval()
        base_model.to(device)
        
        model = base_model
        logger.info("✅ 模型加载完成！")
        return True
        
    except Exception as e:
        logger.error(f"❌ 模型加载失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False
def get_dataloader(ref_sequence: str, var_sequence: str, question: str):
    """获取数据加载器"""
    questions = [question]
    answers = ["1"]
    reference_sequences = [ref_sequence]
    variant_sequences = [var_sequence]
    reasonings = ["推理1"]
    # 步骤1：创建字典格式的数据
    data_dict = {
        "question": questions,
        "answer": answers,
        "reference_sequence": reference_sequences,
        "variant_sequence": variant_sequences,
        "reasoning": reasonings
    }

    # 步骤2：转换为 Pandas DataFrame
    df = pd.DataFrame(data_dict)

    # 步骤3：创建 Dataset 对象
    val_dataset = Dataset.from_pandas(df)

    # 组合成 DatasetDict
    dataset = DatasetDict({
        "val": val_dataset
    })
    dataset = dataset.map(get_format_kegg_function("dna-llm"))
   
    val_dataset = dataset["val"]
    labels = []
    for split, data in dataset.items():
        labels.extend(data["answer"])
    global_labels = sorted(list(set(labels)))
    
    val_dataset = val_dataset.map(
        truncate_dna, fn_kwargs={"truncate_dna_per_side": 1024}
    )
    processor = DLProcessor(
                tokenizer=model.model.text_tokenizer,
                dna_tokenizer=model.model.dna_tokenizer,
            )
    # Create partial function with all required arguments except the batch
    collate_fn = partial(
        qwen_dna_collate_fn,
        processor=processor,
        max_length_text=8192,
        max_length_dna=1024,
        return_answer_in_batch=True,
    )
    return DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        persistent_workers=False,
        pin_memory=False,
    )

def generate_answer(ref_sequence: str, var_sequence: str, question: str) -> str:
    """生成答案 - 完全按照训练脚本的格式处理"""
    if model is None:
        raise ValueError("模型未加载")
    
    try:
        val_dataloader = get_dataloader(ref_sequence, var_sequence, question)
        total_batches = len(val_dataloader)
        
        # Storage
        generations = []
        all_preds = []
        all_labels = []
        glo_tokenizer = model.model.text_tokenizer
        for batch_idx, batch in enumerate(val_dataloader):
            
            print(f"Processing validation batch {batch_idx}/{total_batches}")
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            answer = batch["answer"]  # ground truth label (string)
            dna_tokenized = batch.get("dna_tokenized")
            if dna_tokenized is not None:
                dna_tokenized = dna_tokenized.to(device)
            batch_idx_map = batch.get("batch_idx_map")
            assistant_start_marker = "<|im_start|>assistant\n"
            assistant_marker_tokens = glo_tokenizer.encode(assistant_start_marker, add_special_tokens=False)
            marker_tensor = torch.tensor(assistant_marker_tokens, device=input_ids.device)
            marker_len = len(assistant_marker_tokens)
            # Log batch metadata to console 
            print(f"Batch {batch_idx} metadata - Batch Size: {input_ids.shape[0]}, Input Seq Len: {input_ids.shape[1]}")

            for example_idx in range(input_ids.size(0)):
                # Locate assistant marker
                non_pad = (input_ids[example_idx] != glo_tokenizer.pad_token_id).nonzero(as_tuple=True)[0]
                start_idx = non_pad[0].item() if len(non_pad) > 0 else 0
                assistant_pos = None
                for pos in range(start_idx, input_ids.size(1) - marker_len + 1):
                    if torch.all(input_ids[example_idx, pos:pos + marker_len] == marker_tensor):
                        assistant_pos = pos
                        break
                # Log to console if assistant marker was found 
                print(f"Assistant marker found for example {example_idx}: {assistant_pos is not None}")
                if assistant_pos is None:
                    continue
                # Prepare generation input
                gen_input_ids = input_ids[example_idx:example_idx + 1, start_idx:assistant_pos + marker_len]
                gen_attention_mask = attention_mask[example_idx:example_idx + 1, start_idx:assistant_pos + marker_len]
                example_dna_data = None
                example_batch_map = None
                if dna_tokenized is not None and batch_idx_map is not None:
                    example_indices = [i for i, idx in enumerate(batch_idx_map) if idx == example_idx]
                    if example_indices:
                        example_dna_data = BatchEncoding({
                            "input_ids": dna_tokenized.input_ids[example_indices].to(device),
                            "attention_mask": dna_tokenized.attention_mask[example_indices].to(device),
                        })
                        example_batch_map = [0] * len(example_indices)
                # Log generation start to console 
                print(f"Generating for example {example_idx} in batch {batch_idx}")
                with torch.no_grad():
                    generated = model.model.generate(
                        input_ids=gen_input_ids,
                        attention_mask=gen_attention_mask,
                        dna_tokenized=example_dna_data,
                        batch_idx_map=example_batch_map,
                        max_new_tokens=2000,
                        temperature=0.6,
                        top_p=0.95,
                        top_k=20,
                        do_sample=True,
                    )
                user_input = glo_tokenizer.decode(gen_input_ids[0], skip_special_tokens=False).strip()
                generation = glo_tokenizer.decode(generated[0], skip_special_tokens=False).strip()
                ground_truth = answer[example_idx]
                if ";" in ground_truth:
                    ground_truth = ground_truth.split(";")[0]
               
                if ground_truth.lower() in generation.lower():
                    pred_label = ground_truth
                else:
                    pred_label = generation.lower()
               
                all_labels.append(ground_truth)
                all_preds.append(pred_label)
                generations.append({
                    "batch_idx": batch_idx,
                    "example_idx": example_idx,
                    "user_input": user_input,
                    "generation": generation,
                    "ground_truth": ground_truth,
                    "pred_label": pred_label,
                    "contains_ground_truth": ground_truth.lower() in generation.lower()
                })
                torch.cuda.empty_cache()
                gc.collect()
        
        print(generations)
        
        return generations[0]['generation']
        
    except Exception as e:
        logger.error(f"生成答案失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise e

@app.on_event("startup")
async def startup_event():
    """启动时加载模型"""
    logger.info("🚀 启动服务...")
    parser = ArgumentParser()
    base_dir = './'
    # Model configuration
    parser.add_argument("--model_type", type=str, choices=["llm", "dna-llm"], default="dna-llm")
    parser.add_argument("--text_model_name", type=str, default=base_dir+"model_weights/Qwen/Qwen3-1___7B")
    parser.add_argument("--text_model_finetune", type=bool, default=True)
    parser.add_argument("--dna_model_finetune", type=bool, default=False)
    parser.add_argument("--dna_is_evo2", type=bool, default=False)
    parser.add_argument("--dna_embedding_layer", type=str, default=None)
    parser.add_argument("--dna_model_name",type=str,default=base_dir+"model_weights/onehot_mix_1b_128k364B_cpt_8k298B_cpt_1m140B_cpt_8k200B_stage1_1_1004")
    # Training parameters
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_epochs", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--max_length_dna", type=int, default=1024)
    parser.add_argument("--max_length_text", type=int, default=8192)
    parser.add_argument("--truncate_dna_per_side", type=int, default=1024)
    parser.add_argument("--return_answer_in_batch", type=bool, default=False)
    # LoRA parameters
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    # Infrastructure and paths
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--log_dir", type=str, default="tb_logs") 
    parser.add_argument("--cache_dir", type=str, default="model-weights")
    parser.add_argument("--ckpt_path", type=str, default=base_dir+"checkpoints/kegg-kegg-Qwen3-1___7B-20251011-134801/kegg-kegg-Qwen3-1___7B-epoch=01-val_loss_epoch=0.5006.ckpt")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--strategy", type=str, default="deepspeed_stage_2")
    # Dataset configuration
    parser.add_argument("--dataset_type", type=str,
                        choices=["kegg", "variant_effect_coding", "variant_effect_non_snv", "kegg_hard"],
                        default="kegg")
    parser.add_argument("--use_qwen_dna_collate_fn", type=bool, default=True)
    parser.add_argument("--kegg_data_dir_local", type=str, default="data/kegg")
    parser.add_argument("--kegg_data_dir_huggingface", type=str, default="wanglab/kegg")
    parser.add_argument("--variant_effect_coding_data_dir_huggingface", type=str,
                        default="wanglab/variant_effect_coding")
    parser.add_argument("--variant_effect_non_snv_data_dir_huggingface", type=str,
                        default="wanglab/variant_effect_non_snv")
    parser.add_argument("--merge_val_test_set", type=bool, default=False)
    # Logging and monitoring 
    parser.add_argument("--wandb_project", type=str, default="nt-500m-qwen3-1.7b-finetune")
    
    args = parser.parse_args()
    success = load_checkpoint_model(args)
    if not success:
        logger.warning("⚠️  模型加载失败，服务将以受限模式运行")

@app.get("/")
async def root():
    """根端点"""
    return {
        "message": "DNA序列问答API服务",
        "version": "1.0.0",
        "status": "模型已加载" if model is not None else "模型未加载",
        "usage": "POST /ask - 提问",
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device),
    }

@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """
    提问端点 - 输入DNA序列和问题，获得模型的回答
    
    示例：
    ```json
    {
        "reference_sequence": "ATCGATCGATCG...",
        "variant_sequence": "ATCGATCGATCG...",
        "question": "这个变异会导致什么疾病？",
        "max_length": 512,
        "temperature": 0.7
    }
    ```
    """
    import time
    start_time = time.time()
    
    if model is None:
        raise HTTPException(status_code=503, detail="模型未加载，请稍后重试")
    
    try:
        # 生成答案
        answer = generate_answer(
            request.reference_sequence,
            request.variant_sequence,
            request.question
           
        )
        
        processing_time = time.time() - start_time
        
        return AnswerResponse(
            answer=answer,
            processing_time=processing_time,
            model_loaded=True
        )
        
    except Exception as e:
        logger.error(f"处理请求失败: {e}")
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        "simple_app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
