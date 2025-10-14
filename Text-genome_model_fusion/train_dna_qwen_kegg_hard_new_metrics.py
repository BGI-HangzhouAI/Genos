import csv
import gc
import io
import multiprocessing
import os
import time
import traceback
from argparse import ArgumentParser
from functools import partial
from typing import *
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import pandas as pd
import numpy as np
# from sympy import im
import torch
# import wandb
from datasets import DatasetDict, concatenate_datasets, load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from transformers.tokenization_utils_base import BatchEncoding

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, Callback
# from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DeepSpeedStrategy

from bioreason.dataset.kegg import get_format_kegg_function, qwen_dna_collate_fn
from bioreason.dataset.utils import truncate_dna
from bioreason.dataset.variant_effect import (
    clean_variant_effect_example,
    clean_variant_effect_non_snv_example,
    get_format_variant_effect_function,
)
from bioreason.models.dl.processing_dl import DLProcessor
from bioreason.models.dna_llm import DNALLMModel
from bioreason.models.evo2_tokenizer import register_evo2_tokenizer
# from bioreason.models.zjgen_tokenizer import register_Zjgen_tokenizer

register_evo2_tokenizer()
# register_Zjgen_tokenizer()


# Set start method to 'spawn' for CUDA compatibility with multiprocessing
torch.multiprocessing.set_sharing_strategy("file_system")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class DataProcessor:
    @staticmethod
    def extract_single_entry(text: str) -> str:
        """
        从一段多行文本中提取：
        1. 仅包含 "answer:" 的那一整行（忽略其它行）；
        2. 该行中 "answer:" 之后、且位于同一行内的 "<|im_end|>" 之前的内容；
        3. 若该行没有 "<|im_end|>"，则把整行剩余部分都保留
        4. 如果找不到 "answer:"，返回 NaN。
        """
        if not isinstance(text, str):
            return np.nan

        # 逐行扫描
        for line in text.splitlines():
            if "answer:" not in line:
                continue  # 跳过不含 "answer:" 的行

            # 取出 "answer:" 之后的内容
            _, right = line.split("answer:", 1)

            # 取 "<|im_end|>" 之前的内容
            if "<|im_end|>" in right:
                result = right.split("<|im_end|>", 1)[0].strip()
            else:
                # 若该行没有 "<|im_end|>"，则把整行剩余部分都保留
                result = right.strip()

            return result if result else np.nan

        return np.nan

    @staticmethod
    def process_csv_to_df(csv_path, correction=False):
        input = open(csv_path, encoding='utf-8').read()

        # separator是$的表格整理
        df = pd.read_csv(
            io.StringIO(input),
            sep='$'
        )

        if correction:
            # 检查每个字符串是否以<think>前缀开头
            condition = df["pred_label"].str.startswith("<think>")

            # df["pred_label"]列如果以<think>前缀开头，则再抽提一下
            df.loc[condition, "pred_label"] = \
                df.loc[condition, "pred_label"].apply(DataProcessor.extract_single_entry)

        return df

    @staticmethod
    def generate_metrics_df(y_true, y_pred, labels, report_type):
        """
        生成分类报告，转换为DataFrame，并添加报告类型标识。
        """
        if not labels:
            print(f"警告：'{report_type}' 的 labels 列表为空，跳过报告生成。")
            return pd.DataFrame()

        report_dict = classification_report(
            y_true,
            y_pred,
            labels=labels,
            output_dict=True,
            zero_division=1
        )

        # 将字典转换为DataFrame并进行初步格式化
        report_df = pd.DataFrame(report_dict).transpose()
        report_df.index.name = 'Class/Metric'

        # 报告类型标识，为了区分不同情况下的调用（如，用true_label或者用label合集作为label的不同情况）
        report_df['Report Type'] = report_type
        return report_df

    @staticmethod
    def get_metrics_from_csv(full_csv_file_path):

        df_metrics_source = DataProcessor.process_csv_to_df(full_csv_file_path, correction=True)

        # 针对这一行对应的csv文件对应的df，找到labels
        ground_truth = df_metrics_source['ground_truth'].tolist()
        pred_label = df_metrics_source['pred_label'].tolist()

        # labels 只包含 ground_truth 中出现的所有唯一类别
        ground_truth_only_labels = sorted(list(set(ground_truth)))
        df_case2 = DataProcessor.generate_metrics_df(
            ground_truth,
            pred_label,
            ground_truth_only_labels,
            'Ground_Truth_Labels_Only'
        )

        Accuracy = accuracy_score(ground_truth, pred_label)

        macro_metrics = df_case2.loc['macro avg'] if 'macro avg' in df_case2.index else None
        Precision = macro_metrics['precision'] if macro_metrics is not None else 0
        Recall = macro_metrics['recall'] if macro_metrics is not None else 0
        F1_score = macro_metrics['f1-score'] if macro_metrics is not None else 0

        return Accuracy, Precision, Recall, F1_score


class PeriodicTestCallback(Callback):
    def __init__(self, test_every_n_epochs=2):
        super().__init__()
        self.test_every_n_epochs = test_every_n_epochs

    def on_train_epoch_end(self, trainer, pl_module):
        current_epoch = trainer.current_epoch + 1  # 注意：epoch 从 0 开始
        if current_epoch % self.test_every_n_epochs == 0:
            print(f"\n[Callback] Running test after epoch {current_epoch}...\n")
            trainer.test(pl_module)


class DNALLMFineTuner(pl.LightningModule):
    """
    PyTorch Lightning module for fine-tuning DNA-LLM models.
    """

    def __init__(self, hparams):
        """
        Initialize the DNALLMFineTuner.

        Args:
            hparams: Hyperparameters for the model and training
        """
        super().__init__()
        self.save_hyperparameters(hparams)

        self.text_model_name = self.hparams.text_model_name
        self.dna_model_name = self.hparams.dna_model_name
        self.cache_dir = self.hparams.cache_dir
        self.learning_rate = self.hparams.learning_rate
        self.weight_decay = self.hparams.weight_decay
        self.text_model_finetune = self.hparams.text_model_finetune
        self.dna_model_finetune = self.hparams.dna_model_finetune
        self.lora_rank = self.hparams.lora_rank
        self.lora_alpha = self.hparams.lora_alpha
        self.lora_dropout = self.hparams.lora_dropout
        self.max_length_dna = self.hparams.max_length_dna
        self.max_length_text = self.hparams.max_length_text
        self.dna_is_evo2 = self.hparams.dna_is_evo2
        self.dna_embedding_layer = self.hparams.dna_embedding_layer
        self.return_answer_in_batch = self.hparams.return_answer_in_batch
        self.merge_val_test_set = self.hparams.merge_val_test_set

        # Store dataset configuration
        self.dataset_type = self.hparams.dataset_type

        # Load model
        self.model = DNALLMModel(
            text_model_name=self.text_model_name,
            dna_model_name=self.dna_model_name,
            cache_dir=self.cache_dir,
            max_length_dna=self.max_length_dna,
            max_length_text=self.max_length_text,
            text_model_finetune=self.text_model_finetune,
            dna_model_finetune=self.dna_model_finetune,
            dna_is_evo2=self.dna_is_evo2,
            dna_embedding_layer=self.dna_embedding_layer,
        )

        self.text_model = self.model.text_model
        self.dna_model = self.model.dna_model
        self.dna_projection = self.model.dna_projection

        # Load tokenizer for target text
        self.tokenizer = self.model.text_tokenizer

        # Prepare model for training
        self.lora_config = self._prep_for_training()

    def _get_target_modules(self):
        # Apply LoRA to all linear layers in the text model
        target_modules = []

        # Get all unique linear layer names
        seen_names = set()
        for name, module in self.text_model.named_modules():
            if isinstance(module, torch.nn.Linear):
                names = name.split(".")
                target_name = names[-1]  # Use the last part of the name

                # Skip output head but include all other linear layers
                if target_name != "lm_head" and target_name not in seen_names:
                    target_modules.append(target_name)
                    seen_names.add(target_name)

        # Add attention-specific layers
        attention_patterns = [
            "q_proj",
            "k_proj",
            "v_proj",
            "out_proj",
            "query",
            "key",
            "value",
        ]
        for pattern in attention_patterns:
            if pattern not in seen_names:
                target_modules.append(pattern)

        # Return all unique layer names to apply LoRA to all layers
        return list(target_modules)

    def _prep_for_training(self) -> LoraConfig:
        """
        Load and configure the DNALLMModel.
        """

        # Freeze DNA encoder parameters
        if self.dna_model_finetune:
            pass
        else:
            if self.dna_is_evo2:
                for param in self.dna_model.model.parameters():
                    param.requires_grad = False
            else:
                for param in self.dna_model.parameters():
                    param.requires_grad = False

        if self.text_model_finetune:
            target_modules = self._get_target_modules()

            lora_config = LoraConfig(
                r=self.lora_rank,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                target_modules=target_modules,
                init_lora_weights="gaussian",
                bias="none",
                task_type="CAUSAL_LM",
            )

            # Prepare text model for training
            self.text_model = prepare_model_for_kbit_training(self.text_model)
            self.text_model = get_peft_model(self.text_model, lora_config)
        else:
            # Freeze text model parameters
            for param in self.text_model.parameters():
                param.requires_grad = False

        # Make projection layer trainable
        for param in self.dna_projection.parameters():
            param.requires_grad = True

        return lora_config

    def _step(self, batch: Dict, batch_idx: int, prefix: str) -> torch.Tensor:
        """
        Performs a single step for training, validation, or testing.

        Args:
            batch: Dictionary containing the batch data
            batch_idx: Integer indicating the batch index
            prefix: String indicating the step type ('train', 'val', or 'test')

        Returns:
            torch.Tensor: The computed loss for this batch
        """
        if prefix == "test":
            return {"loss": torch.tensor(0.0, device=self.device)}

        # Get batch data from the collate function
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device) if "labels" in batch else None
        dna_tokenized = batch.get("dna_tokenized")
        if dna_tokenized is not None:
            dna_tokenized = dna_tokenized.to(self.device)
        batch_idx_map = batch.get("batch_idx_map")

        # Forward pass through the model
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            dna_tokenized=dna_tokenized,
            batch_idx_map=batch_idx_map,
            labels=labels,
        )

        # Get the loss from model outputs
        loss = outputs.loss

        # Occasionally show generations for debugging purposes - ONLY during training/validation
        # You can reduce the frequency of generations by increasing the step size to make the model train faster
        if (prefix == "train" and (self.global_step % 100 == 0)) or (prefix == "val" and (batch_idx % 100 == 0)):
            try:
                # Select first example from batch for demonstration
                example_idx = 0

                print(
                    f"\n=== Sample Generation (step {self.global_step} / {self.trainer.estimated_stepping_batches}) ==="
                )

                # Get the tokens that define the assistant pattern
                assistant_start_marker = "<|im_start|>assistant\n"
                assistant_marker_tokens = self.tokenizer.encode(assistant_start_marker, add_special_tokens=False)
                marker_tensor = torch.tensor(assistant_marker_tokens, device=input_ids.device)
                marker_len = len(assistant_marker_tokens)

                # Find non-padding tokens in input
                non_pad = (input_ids[example_idx] != self.tokenizer.pad_token_id).nonzero(as_tuple=True)[0]
                if len(non_pad) > 0:
                    start_idx = non_pad[0].item()  # First non-padding token
                else:
                    start_idx = 0

                # For each position, check if the next marker_len tokens match the pattern
                matches = []
                for pos in range(start_idx, input_ids.size(1) - marker_len + 1):
                    if torch.all(input_ids[example_idx, pos: pos + marker_len] == marker_tensor):
                        matches.append(pos)
                        break  # Stop at first match

                assistant_pos = matches[0] if matches else None

                if assistant_pos is not None:
                    # Get input up to and including the assistant marker
                    gen_input_ids = input_ids[
                                    example_idx: example_idx + 1, start_idx: assistant_pos + marker_len
                                    ]
                    gen_attention_mask = attention_mask[
                                         example_idx: example_idx + 1, start_idx: assistant_pos + marker_len
                                         ]

                    # Extract DNA data for this example
                    example_dna_data = None
                    example_batch_map = None

                    if dna_tokenized is not None and batch_idx_map is not None:
                        # Find DNA sequences for this example
                        example_indices = [i for i, idx in enumerate(batch_idx_map) if idx == example_idx]

                        if len(example_indices) > 0:
                            # Extract just this example's DNA data
                            example_dna_data = BatchEncoding(
                                {
                                    "input_ids": dna_tokenized.input_ids[example_indices].to(self.device),
                                    "attention_mask": dna_tokenized.attention_mask[example_indices].to(self.device),
                                }
                            )

                            # For generation we need all sequences mapped to index 0
                            example_batch_map = [0] * len(example_indices)

                    # Generate text
                    with torch.no_grad():
                        generated = self.model.generate(
                            input_ids=gen_input_ids,
                            attention_mask=gen_attention_mask,
                            dna_tokenized=example_dna_data,
                            batch_idx_map=example_batch_map,
                            max_new_tokens=800,
                            temperature=0.6,
                            top_p=0.95,
                            top_k=20,
                            do_sample=True,
                        )

                    # Decode and display
                    user_input = self.tokenizer.decode(gen_input_ids[0], skip_special_tokens=False).strip()
                    generation = self.tokenizer.decode(generated[0], skip_special_tokens=False).strip()

                    # Free memory early
                    del generated, gen_input_ids, gen_attention_mask, example_dna_data, example_batch_map
                    gc.collect()

                    print(f"=====[Sample {prefix} {batch_idx}]=====")
                    print(f"=====[User input]=====\n{user_input}")
                    print(f"=====[Complete generation]=====\n{generation}")

                    # Get ground truth if available
                    ground_truth = ""
                    if labels is not None:
                        # Find all positions where we have valid labels (not -100)
                        valid_label_pos = (labels[example_idx] != -100).nonzero(as_tuple=True)[0]

                        if len(valid_label_pos) > 0:
                            # Check if valid labels start after assistant marker
                            if valid_label_pos[0] >= assistant_pos + marker_len:
                                ground_truth = self.tokenizer.decode(
                                    input_ids[example_idx, valid_label_pos], skip_special_tokens=False
                                ).strip()
                                print(f"=====[Ground truth]=====\n{ground_truth}")

                    # Log to wandb
                    # timestamp = time.time()
                    wandb_logger = self.logger.experiment
                    if not hasattr(self, "pred_table"):
                        self.pred_table = wandb.Table(
                            columns=["timestamp", "prefix", "batch_idx", "user_input", "generation", "ground_truth"])
                    self.pred_table.add_data(time.time(), prefix, batch_idx, user_input, generation, ground_truth)
                    wandb_logger.log({f"step_id": self.pred_table})

                    # step_id = f"gen_{self.global_step}-{timestamp}"
                    # wandb_logger = self.logger.experiment
                    # wandb_logger.log(
                    #     {
                    #         step_id: wandb.Table(
                    #             columns=["timestamp", "prefix", "batch_idx", "user_input", "generation", "ground_truth"],
                    #             data=[[timestamp, prefix, batch_idx, user_input, generation, ground_truth]],
                    #         )
                    #     }
                    # )

                    # Clean up memory
                    del user_input, generation, ground_truth
                    torch.cuda.empty_cache()
                    gc.collect()

                else:
                    print("No assistant marker found in the input sequence")

            except Exception as e:
                print(f"Error during sample generation: {str(e)}")
                traceback.print_exc()

        # Get current learning rate (skip during test as scheduler might not be available)
        if prefix != "test":
            current_lr = self.lr_schedulers().get_last_lr()[0]
        else:
            current_lr = 0

        # Logging metrics
        self.log(
            f"{prefix}_loss",
            loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
        )
        self.log(
            f"{prefix}_loss_epoch",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        # Only log learning rate during training/validation
        if prefix != "test":
            self.log(
                "lr",
                current_lr,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,
            )

        return loss

    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        """Perform a single training step."""
        return self._step(batch, batch_idx, prefix="train")

    def validation_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        """Perform a single validation step."""
        return self._step(batch, batch_idx, prefix="val")

    def test_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        """Perform a single test step."""
        return self._step(batch, batch_idx, prefix="test")

    def configure_optimizers(self):
        """
        Configure optimizers and learning rate schedulers.

        Returns:
            Tuple[List, List]: A tuple containing a list of optimizers and schedulers
        """
        optimizer = AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(0.1 * total_steps)

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def train_dataloader(self) -> DataLoader:
        """Create and return the training DataLoader."""
        # Load dataset based on type specified in hyperparameters

        if self.hparams.dataset_type == "kegg" or self.hparams.dataset_type == "kegg_hard":
            # Use Hugging Face dataset if provided
            dataset_type = self.hparams.dataset_type
            dataset = load_dataset('json', data_files={
                "train": f"/mnt/zzb/peixunban/hujie/BioReason/data/{dataset_type}/train.json",
                "val": f"/mnt/zzb/peixunban/hujie/BioReason/data/{dataset_type}/val.json",
                "test": f"/mnt/zzb/peixunban/hujie/BioReason/data/{dataset_type}/test.json"
            })
            dataset = dataset.map(get_format_kegg_function(self.hparams.model_type))

            labels = []
            for split, data in dataset.items():
                labels.extend(data["answer"])
            self.labels = sorted(list(set(labels)))

            train_dataset = dataset["train"]

            if self.hparams.truncate_dna_per_side:
                train_dataset = train_dataset.map(
                    truncate_dna, fn_kwargs={"truncate_dna_per_side": self.hparams.truncate_dna_per_side}
                )

            processor = DLProcessor(
                tokenizer=self.model.text_tokenizer,
                dna_tokenizer=self.model.dna_tokenizer,
            )

            # Create partial function with all required arguments except the batch
            collate_fn = partial(
                qwen_dna_collate_fn,
                processor=processor,
                max_length_text=self.max_length_text,
                max_length_dna=self.max_length_dna,
                return_answer_in_batch=self.return_answer_in_batch,
            )


        elif self.hparams.dataset_type == "variant_effect_coding":
            dataset = load_dataset("json", data_files={
                "train": "/mnt/zzb/peixunban/hujie/BioReason/data/vep/ve_coding_train.json",
                "test": "/mnt/zzb/peixunban/hujie/BioReason/data/vep/ve_coding_test.json"})
            cleaned_dataset = dataset.map(clean_variant_effect_example)
            dataset = dataset.map(get_format_variant_effect_function(self.hparams.model_type))

            labels = []
            for split, data in cleaned_dataset.items():
                labels.extend(data["answer"])
            self.labels = sorted(list(set(labels)))

            train_dataset = dataset["train"]

            if self.hparams.truncate_dna_per_side:
                train_dataset = train_dataset.map(
                    truncate_dna, fn_kwargs={"truncate_dna_per_side": self.hparams.truncate_dna_per_side}
                )

            processor = DLProcessor(
                tokenizer=self.model.text_tokenizer,
                dna_tokenizer=self.model.dna_tokenizer,
            )

            # Create partial function with all required arguments except the batch
            collate_fn = partial(
                qwen_dna_collate_fn,
                processor=processor,
                max_length_text=self.max_length_text,
                max_length_dna=self.max_length_dna,
                return_answer_in_batch=self.return_answer_in_batch,
            )

        elif self.hparams.dataset_type == "variant_effect_non_snv":
            dataset = load_dataset("json", data_files={
                "train": "/mnt/zzb/peixunban/hujie/BioReason/data/vep/ve_non_snv_train.json",
                "test": "/mnt/zzb/peixunban/hujie/BioReason/data/vep/ve_non_snv_test.json"})
            dataset = dataset.map(clean_variant_effect_non_snv_example)
            cleaned_dataset = dataset.map(clean_variant_effect_example)
            dataset = dataset.rename_column("mutated_sequence", "variant_sequence")

            labels = []
            for split, data in cleaned_dataset.items():
                labels.extend(data["answer"])
            self.labels = sorted(list(set(labels)))

            train_dataset = dataset["train"]

            if self.hparams.truncate_dna_per_side:
                train_dataset = train_dataset.map(
                    truncate_dna, fn_kwargs={"truncate_dna_per_side": self.hparams.truncate_dna_per_side}
                )
            train_dataset = train_dataset.map(get_format_variant_effect_function(self.hparams.model_type))

            processor = DLProcessor(
                tokenizer=self.model.text_tokenizer,
                dna_tokenizer=self.model.dna_tokenizer,
            )

            # Create partial function with all required arguments except the batch
            collate_fn = partial(
                qwen_dna_collate_fn,
                processor=processor,
                max_length_text=self.max_length_text,
                max_length_dna=self.max_length_dna,
                return_answer_in_batch=self.return_answer_in_batch,
            )

        else:
            raise ValueError(f"Unknown dataset type: {self.hparams.dataset_type}")

        return DataLoader(
            train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=self.hparams.num_workers,
            persistent_workers=False,
            pin_memory=False,
        )

    def val_dataloader(self) -> DataLoader:
        """Create and return the validation DataLoader."""

        if self.hparams.dataset_type == "kegg" or self.hparams.dataset_type == "kegg_hard":
            # Use Hugging Face dataset
            dataset_type = self.hparams.dataset_type
            dataset = load_dataset('json', data_files={
                "train": f"/mnt/zzb/peixunban/hujie/BioReason/data/{dataset_type}/train.json",
                "val": f"/mnt/zzb/peixunban/hujie/BioReason/data/{dataset_type}/val.json",
                "test": f"/mnt/zzb/peixunban/hujie/BioReason/data/{dataset_type}/test.json"
            })
            dataset = dataset.map(get_format_kegg_function(self.hparams.model_type))

            if self.hparams.merge_val_test_set:
                val_dataset = concatenate_datasets([dataset['test'], dataset['val']])
            else:
                val_dataset = dataset["val"]

            labels = []
            for split, data in dataset.items():
                labels.extend(data["answer"])
            self.labels = sorted(list(set(labels)))

            if self.hparams.truncate_dna_per_side:
                val_dataset = val_dataset.map(
                    truncate_dna, fn_kwargs={"truncate_dna_per_side": self.hparams.truncate_dna_per_side}
                )

            processor = DLProcessor(
                tokenizer=self.model.text_tokenizer,
                dna_tokenizer=self.model.dna_tokenizer,
            )

            # Create partial function with all required arguments except the batch
            collate_fn = partial(
                qwen_dna_collate_fn,
                processor=processor,
                max_length_text=self.max_length_text,
                max_length_dna=self.max_length_dna,
                return_answer_in_batch=self.return_answer_in_batch,
            )

        elif self.hparams.dataset_type == "variant_effect_coding":
            dataset = load_dataset("json", data_files={
                "train": "/mnt/zzb/peixunban/hujie/BioReason/data/vep/ve_coding_train.json",
                "test": "/mnt/zzb/peixunban/hujie/BioReason/data/vep/ve_coding_test.json"})
            cleaned_dataset = dataset.map(clean_variant_effect_example)
            dataset = dataset.map(get_format_variant_effect_function(self.hparams.model_type))

            labels = []
            for split, data in cleaned_dataset.items():
                labels.extend(data["answer"])
            self.labels = sorted(list(set(labels)))

            val_dataset = dataset["test"]

            if self.hparams.truncate_dna_per_side:
                val_dataset = val_dataset.map(
                    truncate_dna, fn_kwargs={"truncate_dna_per_side": self.hparams.truncate_dna_per_side}
                )

            processor = DLProcessor(
                tokenizer=self.model.text_tokenizer,
                dna_tokenizer=self.model.dna_tokenizer,
            )

            # Create partial function with all required arguments except the batch
            collate_fn = partial(
                qwen_dna_collate_fn,
                processor=processor,
                max_length_text=self.max_length_text,
                max_length_dna=self.max_length_dna,
                return_answer_in_batch=self.return_answer_in_batch,
            )

        elif self.hparams.dataset_type == "variant_effect_non_snv":
            dataset = load_dataset("json", data_files={
                "train": "/mnt/zzb/peixunban/hujie/BioReason/data/vep/ve_non_snv_train.json",
                "test": "/mnt/zzb/peixunban/hujie/BioReason/data/vep/ve_non_snv_test.json"})
            cleaned_dataset = dataset.map(clean_variant_effect_example)
            dataset = dataset.map(clean_variant_effect_non_snv_example)

            labels = []
            for split, data in cleaned_dataset.items():
                labels.extend(data["answer"])
            self.labels = sorted(list(set(labels)))

            dataset = dataset.rename_column("mutated_sequence", "variant_sequence")
            val_dataset = dataset["test"]

            if self.hparams.truncate_dna_per_side:
                val_dataset = val_dataset.map(
                    truncate_dna, fn_kwargs={"truncate_dna_per_side": self.hparams.truncate_dna_per_side}
                )
            val_dataset = val_dataset.map(get_format_variant_effect_function(self.hparams.model_type))

            processor = DLProcessor(
                tokenizer=self.model.text_tokenizer,
                dna_tokenizer=self.model.dna_tokenizer,
            )

            # Create partial function with all required arguments except the batch
            collate_fn = partial(
                qwen_dna_collate_fn,
                processor=processor,
                max_length_text=self.max_length_text,
                max_length_dna=self.max_length_dna,
                return_answer_in_batch=self.return_answer_in_batch,
            )

        else:
            raise ValueError(f"Unknown dataset type: {self.hparams.dataset_type}")

        return DataLoader(
            val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=self.hparams.num_workers,
            persistent_workers=False,
            pin_memory=False,
        )

    def test_dataloader(self) -> DataLoader:
        """Create and return the test DataLoader."""
        return self.val_dataloader()

    # Only for VEP datasets, for KEGG use the resulting generations in W&B
    def on_test_epoch_end(self):
        """
        Called at the end of test epoch to generate text for all test examples
        and calculate classification metrics (multi-class safe).
        """
        wandb_logger = self.logger.experiment
        wandb_logger.log({"test_progress": 0.0, "status": "starting test generation"})

        self.model.eval()
        test_dataloader = self.test_dataloader()
        total_batches = len(test_dataloader)

        pos_label = self.labels[1]  # Positive label (not used directly for multi-class)
        neg_label = self.labels[0]

        wandb_logger.log({
            "positive_label": pos_label,
            "negative_label": neg_label
        })

        print(f"Using labels - Positive: '{pos_label}', Negative: '{neg_label}'")

        # Storage
        generations = []
        all_preds = []
        all_labels = []

        for batch_idx, batch in enumerate(test_dataloader):
            wandb_logger.log({
                "test_progress": batch_idx / total_batches,
                "status": f"processing batch {batch_idx}/{total_batches}"
            })

            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            answer = batch["answer"]  # ground truth label (string)
            dna_tokenized = batch.get("dna_tokenized")
            if dna_tokenized is not None:
                dna_tokenized = dna_tokenized.to(self.device)
            batch_idx_map = batch.get("batch_idx_map")

            assistant_start_marker = "<|im_start|>assistant\n"
            assistant_marker_tokens = self.tokenizer.encode(assistant_start_marker, add_special_tokens=False)
            marker_tensor = torch.tensor(assistant_marker_tokens, device=input_ids.device)
            marker_len = len(assistant_marker_tokens)

            # Log batch metadata to wandb
            wandb_logger.log({
                "batch_size": input_ids.shape[0],
                "input_sequence_length": input_ids.shape[1]
            })

            for example_idx in range(input_ids.size(0)):

                # Locate assistant marker
                non_pad = (input_ids[example_idx] != self.tokenizer.pad_token_id).nonzero(as_tuple=True)[0]
                start_idx = non_pad[0].item() if len(non_pad) > 0 else 0
                assistant_pos = None
                for pos in range(start_idx, input_ids.size(1) - marker_len + 1):
                    if torch.all(input_ids[example_idx, pos:pos + marker_len] == marker_tensor):
                        assistant_pos = pos
                        break

                # Log to wandb if assistant marker was found
                wandb_logger.log({"assistant_marker_found": assistant_pos is not None})

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
                            "input_ids": dna_tokenized.input_ids[example_indices].to(self.device),
                            "attention_mask": dna_tokenized.attention_mask[example_indices].to(self.device),
                        })
                        example_batch_map = [0] * len(example_indices)
                # Log generation start to wandb
                wandb_logger.log({"status": f"generating for example {example_idx} in batch {batch_idx}"})

                with torch.no_grad():
                    generated = self.model.generate(
                        input_ids=gen_input_ids,
                        attention_mask=gen_attention_mask,
                        dna_tokenized=example_dna_data,
                        batch_idx_map=example_batch_map,
                        max_new_tokens=800,
                        temperature=0.6,
                        top_p=0.95,
                        top_k=20,
                        do_sample=True,
                    )

                user_input = self.tokenizer.decode(gen_input_ids[0], skip_special_tokens=False).strip()
                generation = self.tokenizer.decode(generated[0], skip_special_tokens=False).strip()

                ground_truth = answer[example_idx]
                if ";" in ground_truth:
                    ground_truth = ground_truth.split(";")[0]

                # === 预测类别解析逻辑（多分类关键点） ===
                # 你可以改成更严格的解析，比如匹配标签列表里的第一个匹配
                # pred_label = None
                # for lbl in self.labels:
                #     if lbl.lower() in generation.lower():
                #         pred_label = lbl
                #         break
                if ground_truth.lower() in generation.lower():
                    pred_label = ground_truth
                else:
                    pred_label = generation.lower()
                # if pred_label is None:
                #     pred_label = "UNKNOWN"

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

        # === 全局一次性计算多分类指标 ===

        # accuracy = accuracy_score(all_labels, all_preds)
        # precision_macro = precision_score(all_labels, all_preds, average="macro", zero_division=0)
        # recall_macro = recall_score(all_labels, all_preds, average="macro", zero_division=0)
        # f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        #
        # if self.hparams.dataset_type == "kegg":
        #     precision = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
        #     recall = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
        #     f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
        # else:
        #     precision = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
        #     recall = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
        #     f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
        #
        # self.log("test_acc_epoch", accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        # self.log("test_precision_epoch", precision, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        # self.log("test_recall_epoch", recall, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        # self.log("test_f1_epoch", f1, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        #
        # self.log("test_precision_epoch_macro", precision_macro, on_step=False, on_epoch=True, prog_bar=True,
        #          logger=True)
        # self.log("test_recall_epoch_macro", recall_macro, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        # self.log("test_f1_epoch_macro", f1_macro, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        # # {
        # #     "test_accuracy": accuracy,
        # #     "test_precision": precision_macro,
        # #     "test_recall": recall_macro,
        # #     "test_f1": f1_macro,
        # #     "test_precision_micro": precision_micro,
        # #     "test_recall_micro": recall_micro,
        # #     "test_f1_micro": f1_micro,
        # #     # "confusion_matrix": cm.tolist()
        # # })

        # Save generations to CSV
        model_name = self.hparams.text_model_name.split('/')[-1]
        if self.hparams.ckpt_path:
            csv_path = os.path.join(self.hparams.ckpt_path,
                                    f"{time.strftime('%Y%m%d-%H%M%S')}-test_generations_{model_name}.csv")
        else:
            csv_path = os.path.join(self.hparams.checkpoint_dir,
                                    f"{time.strftime('%Y%m%d-%H%M%S')}-test_generations_{model_name}.csv")

        try:
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                if generations:
                    writer = csv.DictWriter(f, fieldnames=generations[0].keys(), delimiter='$')
                    writer.writeheader()
                    writer.writerows(generations)
                wandb_logger.log({"csv_saved": True, "csv_path": csv_path})
        except Exception as e:
            wandb_logger.log({"csv_saved": False, "csv_path": csv_path, "error": str(e)})

        # 从csv计算acc 和 macro metrics

        print(f"Test End: {csv_path}")

        print(f"Generations length: {len(generations) if generations else 0}")
        if generations:
            print(f"First generation sample: {generations[0]}")
        else:
            print("Generations list is empty!")

        Accuracy, Precision, Recall, F1_score = DataProcessor.get_metrics_from_csv(csv_path)

        # log记录
        self.log("test_acc_epoch", Accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_precision_epoch_macro", Precision, on_step=False, on_epoch=True, prog_bar=True,
                 logger=True)
        self.log("test_recall_epoch_macro", Recall, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_f1_epoch_macro", F1_score, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # Print summary
        print(f"Test Summary:\n"
              f"Accuracy: {Accuracy:.4f}\n"
              f"Macro Precision: {Precision:.4f}, Recall: {Recall:.4f}, F1: {F1_score:.4f}\n"
              )

        return {
            "test_acc_epoch": Accuracy,
            "test_precision_epoch": Precision,
            "test_recall_epoch": Recall,
            "test_f1_epoch": F1_score,
        }

    def on_val_epoch_end(self):
        """
        Called at the end of validation epoch to generate text for all validation examples
        and calculate classification metrics (multi-class safe).
        """
        wandb_logger = self.logger.experiment
        wandb_logger.log({"val_progress": 0.0, "status": "starting validation generation"})

        self.model.eval()
        val_dataloader = self.val_dataloader()
        total_batches = len(val_dataloader)

        pos_label = self.labels[1]  # Positive label (not used directly for multi-class)
        neg_label = self.labels[0]
        wandb_logger.log({
            "positive_label": pos_label,
            "negative_label": neg_label
        })

        print(f"Using labels - Positive: '{pos_label}', Negative: '{neg_label}'")

        # Storage
        generations = []
        all_preds = []
        all_labels = []

        for batch_idx, batch in enumerate(val_dataloader):
            wandb_logger.log({
                "val_progress": batch_idx / total_batches,
                "status": f"processing batch {batch_idx}/{total_batches}"
            })

            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            answer = batch["answer"]  # ground truth label (string)
            dna_tokenized = batch.get("dna_tokenized")
            if dna_tokenized is not None:
                dna_tokenized = dna_tokenized.to(self.device)
            batch_idx_map = batch.get("batch_idx_map")

            assistant_start_marker = "<|im_start|>assistant\n"
            assistant_marker_tokens = self.tokenizer.encode(assistant_start_marker, add_special_tokens=False)
            marker_tensor = torch.tensor(assistant_marker_tokens, device=input_ids.device)
            marker_len = len(assistant_marker_tokens)

            # Log batch metadata to wandb
            wandb_logger.log({
                "batch_size": input_ids.shape[0],
                "input_sequence_length": input_ids.shape[1]
            })

            for example_idx in range(input_ids.size(0)):

                # Locate assistant marker
                non_pad = (input_ids[example_idx] != self.tokenizer.pad_token_id).nonzero(as_tuple=True)[0]
                start_idx = non_pad[0].item() if len(non_pad) > 0 else 0
                assistant_pos = None
                for pos in range(start_idx, input_ids.size(1) - marker_len + 1):
                    if torch.all(input_ids[example_idx, pos:pos + marker_len] == marker_tensor):
                        assistant_pos = pos
                        break

                # Log to wandb if assistant marker was found
                wandb_logger.log({"assistant_marker_found": assistant_pos is not None})

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
                            "input_ids": dna_tokenized.input_ids[example_indices].to(self.device),
                            "attention_mask": dna_tokenized.attention_mask[example_indices].to(self.device),
                        })
                        example_batch_map = [0] * len(example_indices)
                # Log generation start to wandb
                wandb_logger.log({"status": f"generating for example {example_idx} in batch {batch_idx}"})

                with torch.no_grad():
                    generated = self.model.generate(
                        input_ids=gen_input_ids,
                        attention_mask=gen_attention_mask,
                        dna_tokenized=example_dna_data,
                        batch_idx_map=example_batch_map,
                        max_new_tokens=800,
                        temperature=0.6,
                        top_p=0.95,
                        top_k=20,
                        do_sample=True,
                    )

                user_input = self.tokenizer.decode(gen_input_ids[0], skip_special_tokens=False).strip()
                generation = self.tokenizer.decode(generated[0], skip_special_tokens=False).strip()

                ground_truth = answer[example_idx]
                if ";" in ground_truth:
                    ground_truth = ground_truth.split(";")[0]

                # === 预测类别解析逻辑（多分类关键点） ===
                # 你可以改成更严格的解析，比如匹配标签列表里的第一个匹配
                # pred_label = None
                # for lbl in self.labels:
                #     if lbl.lower() in generation.lower():
                #         pred_label = lbl
                #         break
                if ground_truth.lower() in generation.lower():
                    pred_label = ground_truth
                else:
                    pred_label = generation.lower()
                # if pred_label is None:
                #     pred_label = "UNKNOWN"

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

        # === 全局一次性计算多分类指标 ===
        # accuracy = accuracy_score(all_labels, all_preds)
        # precision_macro = precision_score(all_labels, all_preds, average="macro", zero_division=0)
        # recall_macro = recall_score(all_labels, all_preds, average="macro", zero_division=0)
        # f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        #
        # precision_micro = precision_score(all_labels, all_preds, average="micro", zero_division=0)
        # recall_micro = recall_score(all_labels, all_preds, average="micro", zero_division=0)
        # f1_micro = f1_score(all_labels, all_preds, average="micro", zero_division=0)
        #
        # # Confusion matrix
        # # cm = confusion_matrix(all_labels, all_preds, labels=self.labels)
        #
        # # Log to wandb
        # self.log("val_acc_epoch", accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        # self.log("val_precision_epoch", precision_macro, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        # self.log("val_recall_epoch", recall_macro, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        # self.log("val_f1_epoch", f1_macro, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        # # {
        # #     "test_accuracy": accuracy,
        # #     "test_precision": precision_macro,
        # #     "test_recall": recall_macro,
        # #     "test_f1": f1_macro,
        # #     "test_precision_micro": precision_micro,
        # #     "test_recall_micro": recall_micro,
        # #     "test_f1_micro": f1_micro,
        # #     # "confusion_matrix": cm.tolist()
        # # })

        # Save generations to CSV
        model_name = self.hparams.text_model_name.split('/')[-1]
        if self.hparams.ckpt_path:
            csv_path = os.path.join(self.hparams.ckpt_path,
                                    f"{time.strftime('%Y%m%d-%H%M%S')}-test_generations_{model_name}.csv")
        else:
            csv_path = os.path.join(self.hparams.checkpoint_dir,
                                    f"{time.strftime('%Y%m%d-%H%M%S')}-test_generations_{model_name}.csv")

        try:
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                if generations:
                    writer = csv.DictWriter(f, fieldnames=generations[0].keys(), delimiter='$')
                    writer.writeheader()
                    writer.writerows(generations)
                wandb_logger.log({"csv_saved": True, "csv_path": csv_path})
        except Exception as e:
            wandb_logger.log({"csv_saved": False, "csv_path": csv_path, "error": str(e)})

        # 从csv计算acc 和 macro metrics

        print(f"Validation End: {csv_path}")

        print(f"Generations length: {len(generations) if generations else 0}")
        if generations:
            print(f"First generation sample: {generations[0]}")
        else:
            print("Generations list is empty!")

        Accuracy, Precision, Recall, F1_score = DataProcessor.get_metrics_from_csv(csv_path)

        # log记录
        self.log("val_acc_epoch", Accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_precision_epoch_macro", Precision, on_step=False, on_epoch=True, prog_bar=True,
                 logger=True)
        self.log("val_recall_epoch_macro", Recall, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_f1_epoch_macro", F1_score, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # Print summary
        print(f"Val Summary:\n"
              f"Accuracy: {Accuracy:.4f}\n"
              f"Macro Precision: {Precision:.4f}, Recall: {Recall:.4f}, F1: {F1_score:.4f}\n"
              )

        return {
            "val_acc_epoch": Accuracy,
            "val_precision_epoch": Precision,
            "val_recall_epoch": Recall,
            "val_f1_epoch": F1_score,
        }


def main(args: ArgumentParser):
    """
    Main function to run the DNA-Text fine-tuning process.

    Args:
        args (ArgumentParser): Parsed command-line arguments
    """
    # Set random seed and environment variables
    pl.seed_everything(args.seed)
    torch.cuda.empty_cache()
    torch.set_float32_matmul_precision("medium")

    # Setup directories
    run_name = f"{args.wandb_project}-{args.dataset_type}-{args.text_model_name.split('/')[-1]}"
    args.checkpoint_dir = f"{args.checkpoint_dir}/{run_name}-{time.strftime('%Y%m%d-%H%M%S')}"

    # Initialize model
    model = DNALLMFineTuner(args)

    # Setup callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=args.checkpoint_dir,
            filename=f"{run_name}-" + "{epoch:02d}-{val_loss_epoch:.4f}",
            save_top_k=2,
            monitor="val_loss_epoch",
            mode="min",
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="step"),
        # PeriodicTestCallback(test_every_n_epochs=2)
    ]

    # Setup logger
    is_resuming = args.ckpt_path is not None
    logger = WandbLogger(
        project=args.wandb_project,
        entity=args.wandb_entity,
        save_dir=args.log_dir,
        name=run_name,
        resume="allow" if is_resuming else None,  # Allow resuming existing run
    )

    # Initialize the PyTorch Lightning Trainer
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
        callbacks=callbacks,
        logger=logger,
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

    # Start the training process
    trainer.fit(model, ckpt_path=args.ckpt_path)
    trainer.test(model, ckpt_path=args.ckpt_path if args.ckpt_path else "best")


if __name__ == "__main__":
    print("Starting training...")
    parser = ArgumentParser()

    # Model configuration
    parser.add_argument("--model_type", type=str, choices=["llm", "dna-llm"], default="dna-llm")
    parser.add_argument("--text_model_name", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--dna_model_name", type=str,
                        default="InstaDeepAI/nucleotide-transformer-v2-500m-multi-species")
    parser.add_argument("--text_model_finetune", type=bool, default=True)
    parser.add_argument("--dna_model_finetune", type=bool, default=False)
    parser.add_argument("--dna_is_evo2", type=bool, default=False)
    parser.add_argument("--dna_embedding_layer", type=str, default=None)

    # Training parameters
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_epochs", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--max_length_dna", type=int, default=1024)
    parser.add_argument("--max_length_text", type=int, default=1024)
    parser.add_argument("--truncate_dna_per_side", type=int, default=1024)
    parser.add_argument("--return_answer_in_batch", type=bool, default=False)

    # LoRA parameters
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    # Infrastructure and paths
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--cache_dir", type=str, default="/model-weights")
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--strategy", type=str, default="ddp")

    # Dataset configuration
    parser.add_argument("--dataset_type", type=str,
                        choices=["kegg", "variant_effect_coding", "variant_effect_non_snv", "kegg_hard"],
                        default="kegg_hard")
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
    parser.add_argument("--wandb_entity", type=str)

    args = parser.parse_args()

    main(args)